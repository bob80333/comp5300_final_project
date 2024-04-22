from datasets import load_dataset, load_metric
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

import torch
from tqdm import tqdm

covost2_langs = {
    "ar_en": "arabic",
    "ca_en": "catalan",
    "cy_en": "welsh",
    "de_en": "german",
    "es_en": "spanish",
    "et_en": "estonian",
    "fa_en": "persian",
    "fr_en": "french",
    "id_en": "indonesian",
    "it_en": "italian",
    "ja_en": "japanese",
    "lv_en": "latvian",
    "mn_en": "mongolian",
    "nl_en": "dutch",
    "pt_en": "portuguese",
    "ru_en": "russian",
    "sl_en": "slovenian",
    "sv-SE_en": "swedish",
    "ta_en": "tamil",
    "tr_en": "turkish",
    "zh-CN_en": "chinese",
}

splits = ["test"]

bleu = load_metric("sacrebleu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = "mps"
    
    
model_kwargs = {}
if device == "cuda":
    model_kwargs["torch_dtype"] = torch.bfloat16
    model_kwargs["attn_implementation"] = "flash_attention_2"


# model loaded
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-large-v3", device=device, model_kwargs=model_kwargs
)
with open("logs/whisper_large_v3_baseline.txt", "w") as f:
    for lang in tqdm(covost2_langs.keys()):
        dataset = load_dataset("covost2", lang, data_dir="data/"+lang.split("_")[0])
        
        for split in tqdm(splits):

        
            dataset_split = dataset[split]
            model_translated = []
            original = []
            
            for out in tqdm(pipe(KeyDataset(KeyDataset(dataset_split, "audio"), "array"), generate_kwargs={"task": "translate", "language": covost2_langs[lang]}, batch_size=8)):
                model_translated.append(out["text"])
                
            for item in tqdm(dataset_split):
                original.append([item["translation"]])


            result = bleu.compute(predictions=model_translated, references=original)
            print(f"BLEU Result for {lang} {split} split: {result}")
            f.write(f"BLEU Result for {lang} {split} split: {result}\n\n")
