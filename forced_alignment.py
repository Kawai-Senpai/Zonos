import os
import torch
import torchaudio
import torchaudio.functional as F

# Set up model cache directory in the project folder
models_dir = os.path.abspath("models")
os.environ["TORCH_HOME"] = models_dir
torch.hub.set_dir(models_dir)

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Load a pre-trained forced alignment pipeline
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model().to("cpu")  # or use "cuda" if available

# Load your audio file
waveform, sample_rate = torchaudio.load("sample.wav")

# Get emissions from the model
with torch.no_grad():
    emission, _ = model(waveform)

# Retrieve the full vocabulary from the model; assume get_labels() returns a list of symbols.
labels = bundle.get_labels()  # e.g. ['-', 'a', 'b', ..., 'z', ...]
id2sym = {i: token for i, token in enumerate(labels)}
sym2id = {token: i for i, token in id2sym.items()}

# Debug: Print vocabulary information
print("\nVocabulary information:")
print(f"Vocabulary: {labels}")
print(f"Total tokens: {len(labels)}")

# Use the full transcript (including spaces) instead of splitting by whitespace.
transcript = "hello world"
# Tokenize character‐by‐character, preserving spaces if present in sym2id.
tokenized_transcript = [sym2id[char] for char in transcript.lower() if char in sym2id]

# Prepare target tensor for forced alignment
targets = torch.tensor([tokenized_transcript], dtype=torch.int32)

# Run forced alignment (blank=0 is assumed)
alignments, scores = F.forced_align(emission, targets, blank=0)

print("Alignments:", alignments)
print("Scores:", scores.exp())

# Calculate frame duration (hop_length / sample_rate)
# MMS_FA model typically uses hop_length of 320
hop_length = 320
frame_duration = hop_length / sample_rate

# Merge the frame-level tokens into token spans
token_spans = F.merge_tokens(alignments[0], scores[0])  # Use first batch item

# Convert spans to time intervals
time_spans = [
    {
        "token": id2sym.get(span.token, f"<UNK_{span.token}>"),  # Should now show actual symbol
        "start_time": span.start * frame_duration,
        "end_time": span.end * frame_duration,
        "score": float(span.score)
    }
    for span in token_spans
]

# Debug: Print token information mapping
print("\nToken mapping debug:")
for span in token_spans:
    token_str = id2sym.get(span.token, f"<UNK_{span.token}>")
    print(f"Token ID: {span.token}, Mapped to: {token_str}")

print("\nTime spans:")
for span in time_spans:
    print(f"Token: {span['token']}, "
          f"Start: {span['start_time']:.3f}s, "
          f"End: {span['end_time']:.3f}s, "
          f"Score: {span['score']:.3f}")

# (Optional) Post-process the aligned tokens to reinsert spaces exactly where they occur in the original transcript.
# forced alignment produced spans for non-space tokens only.
orig_chars = list(transcript.lower())  # e.g. ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
reconstructed_spans = []
span_idx = 0
for char in orig_chars:
    if char != " ":
        # Append the next available span from forced alignment.
        reconstructed_spans.append(time_spans[span_idx])
        span_idx += 1
    else:
        # Insert a space using the gap between the previous and next non-space tokens.
        if reconstructed_spans and span_idx < len(time_spans):
            prev_span = reconstructed_spans[-1]
            next_span = time_spans[span_idx]
            space_span = {
                "token": " ",
                "start_time": prev_span["end_time"],
                "end_time": next_span["start_time"],
                "score": None,  # spaces don't have a score
            }
            reconstructed_spans.append(space_span)
        else:
            # In case the space is at the beginning or end; here we simply skip.
            pass

print("\nProcessed time spans matching original transcript:")
for span in reconstructed_spans:
    print(span)
