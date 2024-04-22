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


    # Load evaluation metric
    bleu = load_metric("sacrebleu")

    # Process each language dataset
    length = 0
    #with open("logs/transcript_gtranslate_baseline.txt", "w") as f:

    for lang in tqdm(covost2_langs.keys()):
        # Load the dataset
        dataset = load_dataset("covost2", lang, data_dir="data/"+lang.split("_")[0], split="test")
        dataloader = DataLoader(dataset, num_workers=8)
        # Collect predictions and references
        predictions, references = [], []
        lang_length = 0
        for item in tqdm(dataloader):
            length += len(item["sentence"][0])
            lang_length += len(item["sentence"][0])
                # Translate using Google Translate API
                #translation = client.translate(item["sentence"], source_language=lang.split("_")[0], target_language='en')['translatedText']
                
                # Collecting translated text and original references
                #predictions.append(translation)
                #references.append([item["translation"]])

            # Compute BLEU score
            #result = bleu.compute(predictions=predictions, references=references)
            #print(f"BLEU Result for {lang} test split: {result}")
            #f.write(f"BLEU Result for {lang} test split: {result}\n\n")
            
        print(lang, lang_length)
            
            
    print("Total length in characters", length)