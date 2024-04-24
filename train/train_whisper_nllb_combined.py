import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    WhisperProcessor
)
from datasets import load_dataset, interleave_datasets
from tqdm.auto import tqdm


# Load Whisper model
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")


# Load Facebook's NLLB-200 decoder
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")

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
    def __init__(self, whisper, nllb_decoder, adapter):
        super().__init__()
        self.whisper = whisper
        self.adapter = adapter
        self.nllb_decoder = nllb_decoder

    def forward(self, audio_input, target_nllb_tokens):
        # encode audio input using Whisper model
        encoder_outputs = self.whisper.model.encoder(audio_input)
        hidden_states = encoder_outputs.last_hidden_state
        adapted = self.adapter(hidden_states)
        logits = self.nllb_decoder(
            input_ids=target_nllb_tokens, encoder_hidden_states=adapted
        ).logits
        return logits


# Instantiate the combined model
combined_model = WhisperWithNLLBDecoder(whisper_model, nllb_model, adapter)


# Load datasets
latvian_dataset = load_dataset("covost2", "lv_en", split="train", data_dir="data/lv")
mongolian_dataset = load_dataset("covost2", "mn_en", split="train", data_dir="data/mn")
tamil_dataset = load_dataset("covost2", "ta_en", split="train", data_dir="data/ta")

combined_dataset = interleave_datasets([latvian_dataset, mongolian_dataset, tamil_dataset])


# Preprocess and create dataloader
def preprocess(example):
    # Assume dataset structure and perform necessary handling like text conversion
    audio = example["audio"]
    input_values = whisper_processor(audio, return_tensors="pt").input_values
    
    target_ids = nllb_tokenizer(example["translation"], return_tensors="pt").input_ids
    return {"input_values": input_values, "labels": target_ids}


processed_dataset = combined_dataset.map(preprocess, remove_columns=combined_dataset.column_names)
loader = DataLoader(processed_dataset, batch_size=8, shuffle=True, num_workers=4)

# Training
optimizer = torch.optim.Adam(combined_model.adapter.parameters(), lr=3e-4)
combined_model.train()

for epoch in range(3):
    loop = tqdm(loader, leave=True)
    for batch in loop:
        inputs = batch["input_values"]
        labels = batch["labels"]

        outputs = combined_model(inputs)
        # Assuming a simple loss calculation for demonstration
        loss = torch.nn.functional.cross_entropy(outputs.transpose(1, 2), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

print("Finished fine-tuning!")
