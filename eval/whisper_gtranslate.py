from datasets import load_dataset
from transformers import pipeline
import torch
from tqdm import tqdm
from google.cloud import translate_v2 as translate

# Configure torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = "mps"

model_kwargs = {}
if device == "cuda":
    model_kwargs["torch_dtype"] = torch.bfloat16

# Initialize Whisper model for transcription
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-large-v3", device=device, model_kwargs=model_kwargs
)

# Initialize Google Translate API
client = translate.Client()

# Define dataset languages

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
# Load evaluation metric
bleu = load_metric("sacrebleu")

# Process each language dataset
results = {}
for lang in tqdm(covost2_langs.keys()):
    # Load the dataset
    dataset = load_dataset("covost2", lang, data_dir="data/"+lang.split("_")[0], split="test")

    # Collect predictions and references
    predictions, references = [], []
    for item in tqdm(dataset):
        # Transcribe audio
        transcription_result = pipe(item["audio"]["array"])
        transcription = transcription_result["text"]
        
        # Translate using Google Translate API
        translation = client.translate(transcription, target_language='en')['translatedText']
        
        # Collecting translated text and original references
        predictions.append(translation)
        references.append([item["translation"]])

    # Compute BLEU score
    bleu_result = bleu.compute(predictions=predictions, references=references)
    results[lang] = bleu_result['score']

# Output results
for lang, score in results.items():
    print(f"{lang}: BLEU Score = {score}")