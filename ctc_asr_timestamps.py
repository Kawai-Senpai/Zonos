import os
import torch
import torchaudio

# Set up model cache directory in the project folder
models_dir = os.path.abspath("models")
os.environ["TORCH_HOME"] = models_dir
torch.hub.set_dir(models_dir)
os.makedirs(models_dir, exist_ok=True)

# Choose device: GPU if available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ASR pipeline (Wav2Vec2 ASR Base 960H) from torchaudio.
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

# Get the target sample rate from the bundle.
target_sample_rate = bundle.sample_rate

# Load your audio file (replace 'sample.wav' with your audio file path).
waveform, sample_rate = torchaudio.load("sample.wav")
# Resample if necessary.
if sample_rate != target_sample_rate:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    sample_rate = target_sample_rate
# If multi-channel, convert to mono.
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Run the model to get emissions.
with torch.no_grad():
    emissions, _ = model(waveform.to(device))
# emissions shape: (batch, time, num_tokens)

# Greedy decoding: choose the token with highest score at each time step.
predicted_ids = torch.argmax(emissions, dim=-1)[0].tolist()  # use first batch item

# Get the vocabulary from the bundle.
labels = bundle.get_labels()
id2sym = {i: token for i, token in enumerate(labels)}

# Define the blank token. For many CTC models, token ID 0 is the blank.
blank = 0

# Determine frame duration.
# The Wav2Vec2 model in this bundle uses a hop_length of 320 samples.
hop_length = 320
frame_duration = hop_length / sample_rate

def ctc_decode_with_timestamps(pred_ids, frame_duration, blank=0, id2sym=None):
    """
    Decode a sequence of token IDs (from CTC) by collapsing consecutive identical non-blank tokens,
    and record the start and end times (in seconds) for each group.
    Returns a list of dicts: {"token": ..., "start_time": ..., "end_time": ...}.
    """
    spans = []
    T = len(pred_ids)
    current_token = None
    start_idx = None
    for i, token in enumerate(pred_ids):
        if token == blank:
            if current_token is not None:
                spans.append({
                    "token": id2sym[current_token],
                    "start_time": start_idx * frame_duration,
                    "end_time": i * frame_duration,
                })
                current_token = None
                start_idx = None
        else:
            if current_token is None:
                current_token = token
                start_idx = i
            elif token != current_token:
                spans.append({
                    "token": id2sym[current_token],
                    "start_time": start_idx * frame_duration,
                    "end_time": i * frame_duration,
                })
                current_token = token
                start_idx = i
    if current_token is not None:
        spans.append({
            "token": id2sym[current_token],
            "start_time": start_idx * frame_duration,
            "end_time": T * frame_duration,
        })
    return spans

# Get token-level time spans.
time_spans = ctc_decode_with_timestamps(predicted_ids, frame_duration, blank=blank, id2sym=id2sym)

# (Optional) You can further group token spans into words based on a space symbol if needed.

# Output the token-level aligned spans.
print("Time spans:")
for span in time_spans:
    print(span)
