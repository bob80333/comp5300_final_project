from datasets import load_dataset, load_metric
from transformers import pipeline
import torch
from tqdm import tqdm
from google.cloud import translate_v2 as translate
from torch.utils.data import DataLoader

if __name__ == "__main__":

    # Configure torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = "mps"


    # Initialize Google Translate API
    client = translate.Client()

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

    # Process each language dataset
    
    with open("logs/samples.txt", "w", encoding='utf-8') as f:

        for lang in tqdm(covost2_langs.keys()):
            # Get language name
            lang_name = lang.split("_")[0]
            # Load the dataset
            dataset = load_dataset("covost2", lang, data_dir="data/"+lang_name, split="test", download_mode='force_redownload')
            dataloader = DataLoader(dataset, num_workers=1)
            i = 0
            print(lang_name)
            for item in tqdm(dataloader):
                print(item)
                f.write(str(item))
                i += 1
                if i > 5:
                    break
