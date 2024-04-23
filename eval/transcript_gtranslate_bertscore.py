from datasets import load_dataset, load_metric
from transformers import pipeline
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

if __name__ == "__main__":

    # Configure torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = "mps"

    # Define dataset languages

    # reduce down to Whisper-large-v2 worst 3 languages
    # for speed and reducing character count (GTranslate has 500k / month limit for free use, $20/1M chars after that)

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
        #"lv_en": "latvian",
        #"mn_en": "mongolian",
        #"nl_en": "dutch",
        #"pt_en": "portuguese",
        #"ru_en": "russian",
        #"sl_en": "slovenian",
        #"sv-SE_en": "swedish",
        "ta_en": "tamil",
        #"tr_en": "turkish",
        #"zh-CN_en": "chinese",
    }


    # Load evaluation metric
    bertscore = load_metric("bertscore")

    with open("logs/gtranslate_tamil.txt", "r", encoding='utf-8') as f:
        text = f.read()
        predictions = [[x] for x in text.split("\n")]
        references = []
        dataset = load_dataset("covost2", "ta_en", data_dir="data/ta", split="test")
        dataloader = DataLoader(dataset, num_workers=3)
        for item in tqdm(dataloader):
            references.append([item["translation"]])
            
        result = bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli")
        print(np.mean(result["f1"]))
        