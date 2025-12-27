"""
Shared Silero VAD singleton.
Loads the model once and shares it across VoiceActivityDetector and SpeechToText.
"""

import os
import threading
from typing import Optional, Tuple, Callable

from rich.console import Console

console = Console()

# Module-level singleton state
_vad_model = None
_vad_utils = None
_vad_device = None
_load_lock = threading.Lock()
_is_loaded = False


def get_vad_model() -> Tuple[any, Callable, str]:
    """
    Get the shared Silero VAD model and utility functions.

    Returns:
        Tuple of (model, get_speech_timestamps function, device string)

    Raises:
        RuntimeError: If model fails to load
    """
    global _vad_model, _vad_utils, _vad_device, _is_loaded

    # Fast path: already loaded
    if _is_loaded:
        return _vad_model, _vad_utils, _vad_device

    # Thread-safe loading
    with _load_lock:
        # Double-check after acquiring lock
        if _is_loaded:
            return _vad_model, _vad_utils, _vad_device

        console.print("[yellow]Loading Silero VAD model (shared)...[/yellow]")

        try:
            # Enable MPS fallback for any unsupported ops
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            import torch

            # Use MPS (Metal) for GPU acceleration on Apple Silicon
            device = "mps" if torch.backends.mps.is_available() else "cpu"

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )

            # Move model to GPU if available
            model = model.to(device)

            _vad_model = model
            _vad_utils = utils[0]  # get_speech_timestamps function
            _vad_device = device
            _is_loaded = True

            console.print(f"[green]Shared VAD model ready on {device.upper()}[/green]")
            return _vad_model, _vad_utils, _vad_device

        except Exception as e:
            console.print(f"[red]Failed to load Silero VAD: {e}[/red]")
            raise RuntimeError(f"Failed to load VAD model: {e}")


def is_vad_loaded() -> bool:
    """Check if VAD model is already loaded."""
    return _is_loaded
