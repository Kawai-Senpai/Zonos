import os
import torch
import torchaudio
from config import device

# Set up model cache directory in the project folder for speech-to-text models
models_dir = os.path.abspath("models")
os.environ["TORCH_HOME"] = models_dir
torch.hub.set_dir(models_dir)
os.makedirs(models_dir, exist_ok=True)

# Load the pre-trained ASR pipeline from torchaudio.
# Here we use the WAV2VEC2_ASR_BASE_960H model.
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

# Get the target sample rate from the bundle.
target_sample_rate = bundle.sample_rate

# Load your audio file.
# Replace 'sample.wav' with your audio file path.
waveform, sample_rate = torchaudio.load("sample.wav")

# If the audio sample rate is different from what the model expects, resample it.
if sample_rate != target_sample_rate:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    sample_rate = target_sample_rate

# If the audio has more than one channel, average them to get a single channel.
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Run the model in inference mode.
with torch.no_grad():
    # The model outputs emissions (logits over tokens).
    emissions, _ = model(waveform.to(device))
    # emissions shape: (batch, time, num_tokens)

# Greedy decoding: for each time step, choose the token with the highest score.
predicted_ids = torch.argmax(emissions, dim=-1)
predicted_ids = predicted_ids[0].tolist()  # take the first (and only) batch element

# Get the vocabulary (list of tokens) from the bundle.
labels = bundle.get_labels()

# Convert token IDs to symbols.
# Note: The blank token is usually represented by "|" or a similar marker.
transcription = "".join([labels[i] for i in predicted_ids]).replace("|", " ").strip()

print("Transcription:", transcription)
