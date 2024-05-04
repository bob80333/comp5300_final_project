import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    WhisperProcessor,
    DataCollatorWithPadding,
)
from datasets import load_dataset, interleave_datasets, load_metric
from tqdm import tqdm
import os
import numpy as np

import time

from datasets import load_dataset, load_metric

covost2_langs = {
    #"ar_en": "arabic",
    #"ca_en": "catalan",
    #"cy_en": "welsh",
    #"de_en": "german",
    #"es_en": "spanish",
    #"et_en": "estonian",
    #"fa_en": "persian",
    #"fr_en": "french",
    #"id_en": "indonesian",
    #"it_en": "italian",
    #"ja_en": "japanese",
    "lv_en": "latvian",
    "mn_en": "mongolian",
    #"nl_en": "dutch",
    #"pt_en": "portuguese",
    #"ru_en": "russian",
    #"sl_en": "slovenian",
    #"sv-SE_en": "swedish",
    "ta_en": "tamil",
    #"tr_en": "turkish",
    #"zh-CN_en": "chinese",
}




class LinearAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class WhisperWithNLLBDecoder(nn.Module):
    def __init__(self, whisper_encoder, nllb_decoder, nllb_lm_head, adapter):
        super().__init__()
        self.whisper_encoder = whisper_encoder
        self.adapter = adapter
        self.nllb_decoder = nllb_decoder
        self.lm_head = nllb_lm_head

    def forward(self, audio_input, target_nllb_tokens):
        # encode audio input using Whisper model
        encoder_outputs = self.whisper_encoder(audio_input)
        hidden_states = encoder_outputs.last_hidden_state
        adapted = self.adapter(hidden_states)
        hidden = self.nllb_decoder(
            input_ids=target_nllb_tokens, encoder_hidden_states=adapted
        ).last_hidden_state
        logits = self.lm_head(hidden)
        return logits

    def generate(self, audio_input, max_length):
        encoder_outputs = self.whisper_encoder(audio_input)
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
        self.whisper_encoder.eval()
        self.nllb_decoder.eval()
        self.lm_head.eval()
        self.adapter.eval()

    def train(self):
        self.lm_head.train()
        self.adapter.train()
        self.nllb_decoder.train()

splits = ["test"]

bleu = load_metric("sacrebleu")
bertscore = load_metric("bertscore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = "mps"
    
    
model_kwargs = {}
if device == "cuda":
    model_kwargs["torch_dtype"] = torch.bfloat16
    model_kwargs["attn_implementation"] = "flash_attention_2"

whisper_name = "openai/whisper-large-v2"
whisper_processor = WhisperProcessor.from_pretrained(whisper_name)

model = torch.load("checkpoints/combined/combined_epoch_0.pt")
model.to(device)
model.to(torch.bfloat16)


nllb_name = "facebook/nllb-200-1.3B"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_name, tgt_lang="eng_Latn")


with open("logs/combined_v2_finetuned.txt", "w") as f:
    for lang in tqdm(covost2_langs.keys()):
        dataset = load_dataset("covost2", lang, data_dir="data/"+lang.split("_")[0])
        
        for split in tqdm(splits):

        
            dataset_split = dataset[split]
            model_translated = []
            original = []


            for item in tqdm(dataset_split):
                audio = item["audio"]["array"]
                sample_rate = item["audio"]["sampling_rate"]
                input_values = whisper_processor(
                    audio, sampling_rate=sample_rate, return_tensors="pt"
                ).input_features[0]
                audio_input = input_values.to(device).to(torch.bfloat16).unsqueeze(0)
                generated_tokens = model.generate(audio_input, 50)
                generated = nllb_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                model_translated.extend(generated)
                original.append([item["translation"]])


            bleu_result = bleu.compute(predictions=model_translated, references=original)
            bertscore_result = bertscore.compute(
                predictions=model_translated,
                references=original,
                model_type="microsoft/deberta-xlarge-mnli",  # this model correlates best with human judgement according to BERTScore github
            )
            bleu_score = bleu_result["score"]
            bertscore_val = np.mean(bertscore_result["f1"])
            print(lang, "BERTScore", bertscore_val, "BLEU", bleu_score)
            f.write(
                f"BLEU Result for {lang} test split: {bleu_score}, BERTScore: {bertscore_val}\n\n"
            )

