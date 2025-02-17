import argparse
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

def main():
    parser = argparse.ArgumentParser(description="Zonos CLI Synthesizer")
    parser.add_argument("--repo", type=str, default="Zyphra/Zonos-v0.1-torch", help="Model repo id")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--language", type=str, default="en-us", help="Language code")
    parser.add_argument("--speaker_audio", type=str, help="Path to speaker audio (optional)")
    parser.add_argument("--prefix_audio", type=str, help="Path to prefix audio (optional)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="CFG scale")
    parser.add_argument("--min_p", type=float, default=0.15, help="Minimum p for sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Loading model {args.repo} on device {args.device}...")
    model = Zonos.from_pretrained(args.repo, device=args.device, cache_dir="models")
    model.requires_grad_(False).eval()

    cond_dict = make_cond_dict(text=args.text, language=args.language)
    conditioning = model.prepare_conditioning(cond_dict)

    audio_prefix_codes = None
    if args.prefix_audio:
        wav_prefix, sr_prefix = torchaudio.load(args.prefix_audio)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
        wav_prefix = wav_prefix.to(args.device, dtype=torch.float32)
        audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))

    print("Generating audio...")
    codes = model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=86 * 30,
        cfg_scale=args.cfg_scale,
        batch_size=1,
        sampling_params=dict(min_p=args.min_p)
    )
    wav_out = model.autoencoder.decode(codes).cpu().detach()
    sr_out = model.autoencoder.sampling_rate
    if wav_out.dim() == 2 and wav_out.size(0) > 1:
        wav_out = wav_out[0:1, :]
    torchaudio.save(args.output, wav_out.squeeze().unsqueeze(0), sr_out)
    print(f"Audio saved to {args.output}")

if __name__ == "__main__":
    main()
