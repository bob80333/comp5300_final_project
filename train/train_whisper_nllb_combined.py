import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    WhisperProcessor,
    DataCollatorWithPadding,
)
from datasets import load_dataset, interleave_datasets
from tqdm.auto import tqdm
import os

# Set environment variable for parallel tokenization, since it detects the dataloader multiprocessing and throws an error
os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == "__main__":
    whisper_name = "openai/whisper-tiny"
    nllb_name = "facebook/nllb-200-distilled-600M"


    # Load Whisper model
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_name)
    whisper_processor = WhisperProcessor.from_pretrained(whisper_name)


    # Load Facebook's NLLB-200 decoder
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name)
    nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_name)

    # Freeze parameters of both pre-trained models
    for param in whisper_model.parameters():
        param.requires_grad = False

    for param in nllb_model.parameters():
        param.requires_grad = False


    # Define a linear adapter layer
    class LinearAdapter(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.linear(x)


    # Assume output dimension of Whisper and input dimension of NLLB Decoder are known
    input_dim = whisper_model.config.hidden_size  # e.g., 768 for Whisper
    output_dim = nllb_model.config.d_model  # e.g., 1024 for NLLB

    adapter = LinearAdapter(input_dim, output_dim)


    # Combine models
    class WhisperWithNLLBDecoder(nn.Module):
        def __init__(self, whisper, nllb_decoder, nllb_lm_head, adapter):
            super().__init__()
            self.whisper = whisper
            self.adapter = adapter
            self.nllb_decoder = nllb_decoder
            self.lm_head = nllb_lm_head

        def forward(self, audio_input, target_nllb_tokens):
            # encode audio input using Whisper model
            encoder_outputs = self.whisper.model.encoder(audio_input)
            hidden_states = encoder_outputs.last_hidden_state
            adapted = self.adapter(hidden_states)
            hidden = self.nllb_decoder(
                input_ids=target_nllb_tokens, encoder_hidden_states=adapted
            ).last_hidden_state
            logits = self.lm_head(hidden)
            return logits


    # Instantiate the combined model
    combined_model = WhisperWithNLLBDecoder(whisper_model, nllb_model.model.decoder, nllb_model.lm_head, adapter)


    # Load datasets
    #latvian_dataset = load_dataset("covost2", "lv_en", split="train", data_dir="data/lv")
    #mongolian_dataset = load_dataset("covost2", "mn_en", split="train", data_dir="data/mn")
    #tamil_dataset = load_dataset("covost2", "ta_en", split="train", data_dir="data/ta")

    #combined_dataset = interleave_datasets([latvian_dataset, mongolian_dataset, tamil_dataset])

    combined_dataset = load_dataset("covost2", "pt_en", split="test", data_dir="data/pt").select(range(1024))
    print(combined_dataset)


    # Preprocess and create dataloader
    def preprocess(example):
        # Assume dataset structure and perform necessary handling like text conversion
        audio = example["audio"]["array"]
        sample_rate = example["audio"]["sampling_rate"]
        input_values = whisper_processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features[0]
        #print(input_values.shape)
        
        target_ids = nllb_tokenizer(example["translation"], return_tensors="pt").input_ids[0]
        #print(target_ids.shape)
        return {"input_values": input_values, "input_ids": target_ids}


    processed_dataset = combined_dataset.map(preprocess, remove_columns=combined_dataset.column_names)
    processed_dataset.set_format(type="torch", columns=["input_values", "input_ids"])

    collator_fn = DataCollatorWithPadding(nllb_tokenizer)

    loader = DataLoader(processed_dataset, batch_size=16, shuffle=True, num_workers=1, collate_fn=collator_fn)

    # Training
    optimizer = torch.optim.Adam(combined_model.adapter.parameters(), lr=3e-4)
    combined_model.train()

    for epoch in range(1):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            audio_input = batch["input_values"]
            tokens = batch["input_ids"]
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            outputs = combined_model(audio_input, input_tokens)
            # Assuming a simple loss calculation for demonstration
            loss = torch.nn.functional.cross_entropy(outputs.transpose(1, 2), target_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

    print("Finished fine-tuning!")
