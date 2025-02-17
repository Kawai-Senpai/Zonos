import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from config import device

def get_input(prompt, default=None, conv=str):
    inp = input(f"{prompt} [{default if default is not None else ''}]: ")
    if inp.strip() == "":
        return default
    try:
        return conv(inp)
    except Exception as e:
        print(f"Invalid input. Using default value {default}.")
        return default

def main_menu():
    while True:
        print("\n--- Zonos Audio Synthesis Menu ---")
        print("1. Synthesize Audio")
        print("2. Exit")
        choice = input("Select an option: ")
        if choice == "1":
            synthesize_audio()
        elif choice == "2":
            print("Exiting.")
            break
        else:
            print("Invalid option.")

def synthesize_audio():
    # Model/general settings
    model_repo = get_input("Enter model repo id", "Zyphra/Zonos-v0.1-transformer", str)
    text = get_input("Enter text to synthesize", None, str)
    language = get_input("Enter language code", "en-us", str)
    
    # New: Model cache directory ‒ where models are downloaded locally.
    cache_dir = get_input("Enter model download directory (cache)", "models", str)
    
    # Optional audio files
    speaker_audio_path = get_input("Enter path to speaker audio (optional)", "", str)
    prefix_audio_path = get_input("Enter path to prefix audio (optional)", "", str)
    
    # Conditioning parameters
    dnsmos_ovrl = get_input("Enter DNSMOS Overall (1.0 - 5.0)", 4.0, float)
    fmax = get_input("Enter Fmax (Hz) (0 - 24000)", 24000, float)
    vq_single = get_input("Enter VQ Score (0.5 - 0.8)", 0.78, float)
    pitch_std = get_input("Enter Pitch Std (0.0 - 300.0)", 45.0, float)
    speaking_rate = get_input("Enter Speaking Rate (5.0 - 30.0)", 15.0, float)
    
    # Generation parameters
    cfg_scale = get_input("Enter CFG Scale (1.0 - 5.0)", 2.0, float)
    min_p = get_input("Enter Min P (0.0 - 1.0)", 0.15, float)
    seed = get_input("Enter seed", 420, int)
    randomize_seed = get_input("Randomize seed? (yes/no)", "yes", str).lower() in ("yes", "y")
    
    # Advanced parameters
    unconditional_raw = get_input("Enter unconditional keys (comma separated; options: speaker, emotion, vqscore_8, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised)", "", str)
    unconditional_keys = [k.strip() for k in unconditional_raw.split(",") if k.strip()] if unconditional_raw else []
    
    # Emotion slider values
    print("\nEnter emotion slider values (0.0 to 1.0). Press Enter to accept default.")
    emotion1 = get_input("Happiness", 1.0, float)
    emotion2 = get_input("Sadness", 0.05, float)
    emotion3 = get_input("Disgust", 0.05, float)
    emotion4 = get_input("Fear", 0.05, float)
    emotion5 = get_input("Surprise", 0.05, float)
    emotion6 = get_input("Anger", 0.05, float)
    emotion7 = get_input("Other", 0.1, float)
    emotion8 = get_input("Neutral", 0.2, float)
    
    speaker_noised = get_input("Denoise speaker? (yes/no)", "no", str).lower() in ("yes", "y")
    
    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    torch.manual_seed(seed)
    
    print(f"\nLoading model '{model_repo}' on device '{device}'...")
    model = Zonos.from_pretrained(model_repo, device=device, cache_dir=cache_dir)
    model.requires_grad_(False).eval()
    
    # Process speaker audio (if given)
    speaker_embedding = None
    if speaker_audio_path:
        try:
            wav, sr = torchaudio.load(speaker_audio_path)
            speaker_embedding = model.make_speaker_embedding(wav, sr)
            speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)
        except Exception as e:
            print(f"Error processing speaker audio: {e}")
    
    # Process prefix audio (if given)
    audio_prefix_codes = None
    if prefix_audio_path:
        try:
            wav_prefix, sr_prefix = torchaudio.load(prefix_audio_path)
            wav_prefix = wav_prefix.mean(0, keepdim=True)
            wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
            wav_prefix = wav_prefix.to(device, dtype=torch.float32)
            audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))
        except Exception as e:
            print(f"Error processing prefix audio: {e}")
    
    # Create emotion and VQ tensors
    emotion_tensor = torch.tensor([emotion1, emotion2, emotion3, emotion4, emotion5, emotion6, emotion7, emotion8], device=device)
    vq_tensor = torch.tensor([vq_single] * 8, device=device).unsqueeze(0)
    
    # Build conditioning dictionary with all parameters
    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=speaker_embedding,
        emotion=emotion_tensor,
        vqscore_8=vq_tensor,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised,
        device=device,
        unconditional_keys=unconditional_keys,
    )
    conditioning = model.prepare_conditioning(cond_dict)
    
    max_new_tokens = 86 * 30
    print("Generating audio, please wait...")
    codes = model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=1,
        sampling_params=dict(min_p=min_p)
    )
    wav_out = model.autoencoder.decode(codes).cpu().detach()
    if wav_out.dim() == 2 and wav_out.size(0) > 1:
        wav_out = wav_out[0:1, :]
    output_file = get_input("Enter output WAV file name", "output.wav", str)
    torchaudio.save(output_file, wav_out.squeeze().unsqueeze(0), model.autoencoder.sampling_rate)
    print(f"Audio has been saved to {output_file}\n")

if __name__ == "__main__":
    main_menu()
