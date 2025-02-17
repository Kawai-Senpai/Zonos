import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import time
from config import device

start_time = time.time()
print("Loading model...")
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Loading audio file...")
wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
print(f"Audio file loaded in {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Creating speaker embedding...")
speaker = model.make_speaker_embedding(wav, sampling_rate)
print(f"Speaker embedding created in {time.time() - start_time:.2f} seconds")

torch.manual_seed(421)

input_string = input("Enter a string: ")

start_time = time.time()
print("Creating conditioning dictionary...")
cond_dict = make_cond_dict(text=input_string, speaker=speaker, language="en-us")
conditioning = model.prepare_conditioning(cond_dict)
print(f"Conditioning dictionary created in {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Generating codes...")
codes = model.generate(conditioning)
print(f"Codes generated in {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Decoding audio...")
wavs = model.autoencoder.decode(codes).cpu()
print(f"Audio decoded in {time.time() - start_time:.2f} seconds")

start_time = time.time()
print("Saving audio file...")
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
print(f"Audio file saved in {time.time() - start_time:.2f} seconds")
