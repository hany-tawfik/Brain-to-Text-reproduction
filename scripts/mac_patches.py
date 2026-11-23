"""
Run the upstream Brain-to-Text trainer on a Mac without editing upstream files.

The upstream trainer (model_training/rnn_trainer.py) assumes CUDA:
  - picks only CUDA or CPU as the device
  - wraps the model in torch.compile (unreliable on MPS)
  - uses AdamW(fused=True)
  - uses bfloat16 autocast with device_type="cuda"

build_trainer() works around all four:
  - disables torch.compile and fused AdamW before the trainer is built
  - forces use_amp = False (float32)
  - moves the model to MPS after construction (optimizer keeps the same parameter objects)

Import from the repo's model_training folder (the scripts add it to sys.path).
"""

import os
import sys
import warnings

import torch

sys.path.insert(0, os.getcwd())


def pick_device(name: str = "auto") -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda:0")
    if name != "auto":
        warnings.warn(f"Requested device '{name}' not available, using CPU")
    return torch.device("cpu")


def _apply_patches() -> None:
    # 1. No torch.compile
    torch.compile = lambda model, *a, **k: model  # type: ignore[assignment]

    # 2. No fused AdamW (CUDA-only in some torch versions)
    _AdamW = torch.optim.AdamW

    class AdamWNoFused(_AdamW):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs.pop("fused", None)
            super().__init__(*args, **kwargs)

    torch.optim.AdamW = AdamWNoFused  # type: ignore[misc]

    # 3. Quiet the harmless autocast/pin_memory warnings on non-CUDA devices
    warnings.filterwarnings("ignore", message=".*autocast.*")
    warnings.filterwarnings("ignore", message=".*pin_memory.*")


def build_trainer(args, device: torch.device):
    """args: OmegaConf config already adjusted by the caller."""
    _apply_patches()
    args.use_amp = False
    from rnn_trainer import BrainToTextDecoder_Trainer  # imported after patching

    trainer = BrainToTextDecoder_Trainer(args)
    trainer.device = device
    trainer.model.to(device)
    trainer.logger.info(f"Mac patches applied. Training device: {device}")
    return trainer


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
