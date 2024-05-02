import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    DataCollatorWithPadding,
    MaxLengthCriteria
)
from datasets import load_dataset, interleave_datasets
import evaluate
from tqdm.auto import tqdm
import os
import numpy as np
from adamw_bfloat16 import LR, AdamW_BF16

# Set environment variable for parallel tokenization, since it detects the dataloader multiprocessing and throws an error
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def evaluate_model(model, dataloader, tokenizer, max_length, bleu_metric, bertscore_metric):
    with torch.no_grad():
        references = []
        predictions = []
        for batch in tqdm(dataloader):
            audio_input = batch["input_values"].to(device)
            audio_input = audio_input.to(torch.bfloat16)
            target_tokens = batch["input_ids"].to(device)
    
            generated_tokens = model.generate(audio_input, language='en', max_new_tokens=max_length)
            generated_text = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            target_text = tokenizer.batch_decode(target_tokens, skip_special_tokens=True)
        
            # nested list for references as BLEU requires it
            target_text = [[x] for x in target_text]
        
            references.extend(target_text)
            predictions.extend(generated_text)
        
        print("Predictions:", generated_text)
        print("GT:", target_text)

        bleu_result = bleu_metric.compute(predictions=predictions, references=references)
        bertscore_result = bertscore_metric.compute(
            predictions=predictions,
            references=references,
            model_type="microsoft/deberta-xlarge-mnli",
            batch_size=16,
        )

        bleu_score = bleu_result["score"]
        bertscore_score = np.mean(bertscore_result["f1"])

    return bleu_score, bertscore_score


if __name__ == "__main__":
    whisper_name = "openai/whisper-large-v2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Whisper model
    model = WhisperForConditionalGeneration.from_pretrained(whisper_name, device_map=device, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    whisper_processor = WhisperProcessor.from_pretrained(whisper_name, language='en', task='transcribe')


    # Load datasets
    latvian_dataset = load_dataset("covost2", "lv_en", split="train", data_dir="data/lv")
    mongolian_dataset = load_dataset("covost2", "mn_en", split="train", data_dir="data/mn")
    tamil_dataset = load_dataset("covost2", "ta_en", split="train", data_dir="data/ta")
    
    combined_dataset = interleave_datasets([latvian_dataset, mongolian_dataset, tamil_dataset]).select(range(32))

    #combined_dataset = load_dataset(
    #    "covost2", "pt_en", split="test", data_dir="data/pt"
    #).select(range(64))
    print(combined_dataset)

    val_latvian_dataset = load_dataset("covost2", "lv_en", split="validation", data_dir="data/lv")
    val_mongolian_dataset = load_dataset("covost2", "mn_en", split="validation", data_dir="data/mn")
    val_tamil_dataset = load_dataset("covost2", "ta_en", split="validation", data_dir="data/ta")

    combined_val = interleave_datasets([val_latvian_dataset, val_mongolian_dataset, val_tamil_dataset]).select(range(16))

    #combined_val = load_dataset(
    #    "covost2", "pt_en", split="validation", data_dir="data/pt"
    #).select(range(32))
    print(combined_val)

    # Preprocess and create dataloader
    def preprocess(example):
        # Assume dataset structure and perform necessary handling like text conversion
        audio = example["audio"]["array"]
        sample_rate = example["audio"]["sampling_rate"]
        input_values = whisper_processor(
            audio=audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features[0]

        target_ids = whisper_processor(
            text=example["translation"], return_tensors="pt"
        ).input_ids[0]
        return {"input_values": input_values, "input_ids": target_ids}

    processed_dataset = combined_dataset.map(
        preprocess, remove_columns=combined_dataset.column_names
    )
    processed_dataset.set_format(type="torch", columns=["input_values", "input_ids"])
    
    processed_val = combined_val.map(
        preprocess, remove_columns=combined_val.column_names
    )
    processed_val.set_format(type="torch", columns=["input_values", "input_ids"])

    collator_fn = DataCollatorWithPadding(whisper_processor.tokenizer)

    loader = DataLoader(
        processed_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=6,
        collate_fn=collator_fn,
    )
    
    val_loader = DataLoader(
        processed_val,
        batch_size=8,
        shuffle=False,
        num_workers=3,
        collate_fn=collator_fn
    )


    # Training
    model = model.to(torch.bfloat16)
    model.train()
    model.config.dropout = 0.1 # add dropout to reduce overfitting
    model.gradient_checkpointing_enable()

    optimizer = AdamW_BF16(model.parameters(), lr_function=LR(lr=5e-6, preheat_steps=100))

    bleu_metric = evaluate.load("sacrebleu")
    bertscore_metric = evaluate.load("bertscore")

    with open("logs/train_log7.txt", "w") as f:
        for epoch in range(5):
            loop = tqdm(loader, leave=True)
            for batch in loop:
                audio_input = batch["input_values"].to(device)
                audio_input = audio_input.to(torch.bfloat16)
                tokens = batch["input_ids"].to(device)
                input_tokens = tokens[:, :-1]
                target_tokens = tokens[:, 1:]

                logits = model(input_features=audio_input, decoder_input_ids=input_tokens).logits
                _, _, classes = logits.shape
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, classes),
                    target_tokens.contiguous().view(-1),
                    ignore_index=whisper_processor.tokenizer.pad_token_id,
                )

                loss.backward()
                optimizer.step(zero_grad=True)

                loop.set_description(f"Epoch {epoch + 1}")
                loop.set_postfix(loss=loss.item())

            model.eval()
            bleu_result, bertscore_result = evaluate_model(
                model, val_loader, whisper_processor, 50, bleu_metric, bertscore_metric
            )
            print(f"BLEU: {bleu_result}, BERTScore: {bertscore_result}")
            f.write(f"BLEU: {bleu_result}, BERTScore: {bertscore_result}\n")
            f.flush()
            model.save_pretrained(f"checkpoints/whisper_large_v2_ft_epoch_{epoch}/")
            whisper_processor.save_pretrained(f"checkpoints/whisper_large_v2_ft_epoch_{epoch}/")
            model.train()

    print("Finished fine-tuning!")
