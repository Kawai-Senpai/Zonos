import torch
from zonos.model import Zonos

def download_model():
    print("Downloading model, please wait...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cpu", cache_dir="models")
    print("Model downloaded and saved in 'models/' directory.")

if __name__ == "__main__":
    download_model()
