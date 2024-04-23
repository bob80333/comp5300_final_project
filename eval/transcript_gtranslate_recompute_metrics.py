from datasets import load_dataset, load_metric
from transformers import pipeline
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np


# recompute metrics using saved logs, allowing for adding metrics without re-running the translation and using up the free GTranslate limit
if __name__ == "__main__":

    # Define dataset languages

    # reduce down to Whisper-large-v2 worst 3 languages
    # for speed and reducing character count (GTranslate has 500k / month limit for free use, $20/1M chars after that)

    covost2_langs = {
        # "ar_en": "arabic",
        # "ca_en": "catalan",
        # "cy_en": "welsh",
        # "de_en": "german",
        # "es_en": "spanish",
        # "et_en": "estonian",
        # "fa_en": "persian",
        # "fr_en": "french",
        # "id_en": "indonesian",
        # "it_en": "italian",
        # "ja_en": "japanese",
        "lv_en": "latvian",
        "mn_en": "mongolian",
        # "nl_en": "dutch",
        # "pt_en": "portuguese",
        # "ru_en": "russian",
        # "sl_en": "slovenian",
        # "sv-SE_en": "swedish",
        "ta_en": "tamil",
        # "tr_en": "turkish",
        # "zh-CN_en": "chinese",
    }

    # Load evaluation metrics
    bertscore = load_metric("bertscore")
    bleu = load_metric("sacrebleu")

    with open("logs/gtranslate_sentences.txt", "r", encoding="utf-8") as f, open(
        "logs/transcript_gtranslate_recompute.txt", "w"
    ) as f2:
        text = f.read()
        # split by \n\n to get languages, remove last language as it is empty
        languages = text.split("\n\n")[:-1]
        # get language name:
        for language_data in languages:
            lang_sentences = language_data.split("\n")

            lang = lang_sentences[0][:-1]  # remove the colon
            lang_name = lang.split("_")[0]

            sentences = lang_sentences[1:]
            predictions = [x for x in sentences]

            references = []
            dataset = load_dataset(
                "covost2", lang, data_dir="data/" + lang_name, split="test"
            )
            dataloader = DataLoader(dataset, num_workers=3)

            for item in tqdm(dataloader):
                references.append([item["translation"]])

            bertscore_result = bertscore.compute(
                predictions=predictions,
                references=references,
                model_type="microsoft/deberta-xlarge-mnli",  # this model correlates best with human judgement according to BERTScore github
            )
            # use f1 score as that is what best correlates with human judgement according to the BERTScore paper
            bertscore_val = np.mean(bertscore_result["f1"])

            bleu_result = bleu.compute(predictions=predictions, references=references)
            bleu_val = bleu_result["score"]

            print(lang_name, "BERTScore", bertscore_val, "BLEU", bleu_val)
            f2.write(
                f"BLEU Result for {lang} test split: {bleu_val}, BERTScore: {bertscore_val}\n\n"
            )
