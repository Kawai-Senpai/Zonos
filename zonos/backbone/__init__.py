import torch
from config import device

USING_CPU = device.type == 'cpu'

if not USING_CPU:
    try:
        from ._mamba_ssm import MambaSSMZonosBackbone
    except ImportError:
        print("Mamba-SSM not installed, MambaSSMZonosBackbone will be unavailable.")
else:
    print("Running on CPU, MambaSSMZonosBackbone will be unavailable.")

from ._torch import TorchZonosBackbone

BACKBONES = {
    "torch": TorchZonosBackbone,
}

if not USING_CPU:
    try:
        BACKBONES["mamba_ssm"] = MambaSSMZonosBackbone
    except:
        pass

DEFAULT_BACKBONE_CLS = TorchZonosBackbone
