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
from datasets import load_dataset, interleave_datasets, load_metric
from tqdm.auto import tqdm
import os
import numpy as np

# Set environment variable for parallel tokenization, since it detects the dataloader multiprocessing and throws an error
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class LinearAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


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

    def generate(self, audio_input, max_length):
        encoder_outputs = self.whisper.model.encoder(audio_input)
        hidden_states = encoder_outputs.last_hidden_state
        adapted = self.adapter(hidden_states)
        tokens = (
            torch.tensor([[nllb_tokenizer.bos_token_id]])
            .to(device)
            .repeat(audio_input.shape[0], 1)
        )
        for _ in range(max_length):
            hidden = self.nllb_decoder(
                input_ids=tokens, encoder_hidden_states=adapted
            ).last_hidden_state
            logits = self.lm_head(hidden)[:, -1, :]
            predicted_token = torch.argmax(logits, dim=-1)
            tokens = torch.cat([tokens, predicted_token.unsqueeze(-1)], dim=-1)
        return tokens

    def eval(self):
        self.whisper.eval()
        self.nllb_decoder.eval()
        self.lm_head.eval()
        self.adapter.eval()

    def train(self):
        self.lm_head.train()
        self.adapter.train()


def evaluate(model, dataloader, tokenizer, max_length, bleu_metric, bertscore_metric):
    references = []
    predictions = []
    for batch in tqdm(dataloader):
        audio_input = batch["input_values"].to(device)
        target_tokens = batch["input_ids"].to(device)

        generated_tokens = model.generate(audio_input, max_length)
        generated_text = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        target_text = tokenizer.batch_decode(target_tokens, skip_special_tokens=True)

        references.extend(target_text)
        predictions.extend(generated_text)

    bleu_result = bleu_metric.compute(predictions=predictions, references=references)
    bertscore_result = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        model_type="microsoft/deberta-xlarge-mnli",
    )

    bleu_score = bleu_result["score"]
    bertscore_score = np.mean(bertscore_result["f1"])

    return bleu_score, bertscore_score


if __name__ == "__main__":
    whisper_name = "openai/whisper-tiny"
    nllb_name = "facebook/nllb-200-distilled-600M"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Assume output dimension of Whisper and input dimension of NLLB Decoder are known
    input_dim = whisper_model.config.hidden_size  # e.g., 768 for Whisper
    output_dim = nllb_model.config.d_model  # e.g., 1024 for NLLB

    adapter = LinearAdapter(input_dim, output_dim)

    # Instantiate the combined model
    combined_model = WhisperWithNLLBDecoder(
        whisper_model, nllb_model.model.decoder, nllb_model.lm_head, adapter
    )
    combined_model.to(device)

    # Load datasets
    # latvian_dataset = load_dataset("covost2", "lv_en", split="train", data_dir="data/lv")
    # mongolian_dataset = load_dataset("covost2", "mn_en", split="train", data_dir="data/mn")
    # tamil_dataset = load_dataset("covost2", "ta_en", split="train", data_dir="data/ta")

    combined_dataset = load_dataset(
        "covost2", "pt_en", split="test", data_dir="data/pt"
    ).select(range(64))
    print(combined_dataset)

    # val_latvian_dataset = load_dataset("covost2", "lv_en", split="validation", data_dir="data/lv")
    # val_mongolian_dataset = load_dataset("covost2", "mn_en", split="validation", data_dir="data/mn")
    # val_tamil_dataset = load_dataset("covost2", "ta_en", split="validation", data_dir="data/ta")

    # combined_val = interleave_datasets([val_latvian_dataset, val_mongolian_dataset, val_tamil_dataset]).select(range(256))

    combined_val = load_dataset(
        "covost2", "pt_en", split="validation", data_dir="data/pt"
    ).select(range(32))
    print(combined_val)

    # Preprocess and create dataloader
    def preprocess(example):
        # Assume dataset structure and perform necessary handling like text conversion
        audio = example["audio"]["array"]
        sample_rate = example["audio"]["sampling_rate"]
        input_values = whisper_processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features[0]

        target_ids = nllb_tokenizer(
            example["translation"], return_tensors="pt"
        ).input_ids[0]
        return {"input_values": input_values, "input_ids": target_ids}

    processed_dataset = combined_dataset.map(
        preprocess, remove_columns=combined_dataset.column_names
    )
    processed_dataset.set_format(type="torch", columns=["input_values", "input_ids"])

    collator_fn = DataCollatorWithPadding(nllb_tokenizer)

    loader = DataLoader(
        processed_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=collator_fn,
    )


    # Training
    combined_model.adapter.train()
    combined_model.lm_head.train()

    params = list(combined_model.adapter.parameters()) + list(
        combined_model.lm_head.parameters()
    )
    optimizer = torch.optim.Adam(params, lr=3e-4)

    bleu_metric = load_metric("sacrebleu")
    bertscore_metric = load_metric("bertscore")

    with open("logs/train_log2.txt", "w") as f:
        for epoch in range(3):
            loop = tqdm(loader, leave=True)
            for batch in loop:
                audio_input = batch["input_values"].to(device)
                tokens = batch["input_ids"].to(device)
                input_tokens = tokens[:, :-1]
                target_tokens = tokens[:, 1:]

                outputs = combined_model(audio_input, input_tokens)
                # Assuming a simple loss calculation for demonstration
                loss = torch.nn.functional.cross_entropy(
                    outputs.transpose(1, 2),
                    target_tokens,
                    ignore_index=nllb_tokenizer.pad_token_id,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loop.set_description(f"Epoch {epoch + 1}")
                loop.set_postfix(loss=loss.item())

        combined_model.eval()
        bleu_result, bertscore_result = evaluate(
            combined_model, loader, nllb_tokenizer, 128, bleu_metric, bertscore_metric
        )
        print(f"BLEU: {bleu_result}, BERTScore: {bertscore_result}")
        f.write(f"BLEU: {bleu_result}, BERTScore: {bertscore_result}\n")
        f.flush()
        combined_model.train()

    print("Finished fine-tuning!")
