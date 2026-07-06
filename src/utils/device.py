"""Shared PyTorch device resolution."""

import os


def resolve_torch_device(preferred: str) -> str:
    """
    Resolve a preferred device ("auto", "mps", "cpu") to an available one.

    Sets PYTORCH_ENABLE_MPS_FALLBACK before importing torch so unsupported
    MPS ops fall back to CPU instead of erroring.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch

    if preferred == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
