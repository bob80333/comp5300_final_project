import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm.auto import tqdm


# Load Whisper model
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

# Load Facebook's NLLB-200 decoder
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

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
    
    def forward(self, input_values):
        encoder_outputs = self.whisper.model.encoder(input_values)
        hidden_states = encoder_outputs.last_hidden_state
        adapted = self.adapter(hidden_states)
        logits = self.nllb_decoder(inputs_embeds=adapted).logits
        return logits

# Instantiate the combined model
combined_model = WhisperWithNLLBDecoder(whisper_model, nllb_model, adapter)


# Load dataset
dataset = load_dataset("covost2", "en", split='train')

# Preprocess and create dataloader
def preprocess(example):
    # Assume dataset structure and perform necessary handling like text conversion
    audio = example['audio']
    input_values = whisper_model.feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_values
    # Stub target token IDs, in practice, you'd tokenize target text
    target_ids = [0] * 50  # Dummy token IDs
    return {'input_values': input_values[0], 'labels': target_ids}

processed_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
loader = DataLoader(processed_dataset, batch_size=8, shuffle=True)

# Training
optimizer = torch.optim.Adam(combined_model.adapter.parameters(), lr=1e-3)
combined_model.train()

for epoch in range(3):
    loop = tqdm(loader, leave=True)
    for batch in loop:
        inputs = batch['input_values'].to(torch.float32)  # Ensure correct type
        labels = batch['labels']

        outputs = combined_model(inputs)
        # Assuming a simple loss calculation for demonstration
        loss = torch.nn.functional.cross_entropy(outputs.transpose(1, 2), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

print("Finished fine-tuning!")