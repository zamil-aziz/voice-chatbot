"""
Shared Silero VAD singleton.
Loads the model once and shares it across VoiceActivityDetector and SpeechToText.
"""

import threading
from typing import Optional, Tuple, Callable

from rich.console import Console

console = Console()

# Module-level singleton state
_vad_model = None
_vad_utils = None
_load_lock = threading.Lock()
_is_loaded = False


def get_vad_model() -> Tuple[any, Callable]:
    """
    Get the shared Silero VAD model and utility functions.

    Returns:
        Tuple of (model, get_speech_timestamps function)

    Raises:
        RuntimeError: If model fails to load
    """
    global _vad_model, _vad_utils, _is_loaded

    # Fast path: already loaded
    if _is_loaded:
        return _vad_model, _vad_utils

    # Thread-safe loading
    with _load_lock:
        # Double-check after acquiring lock
        if _is_loaded:
            return _vad_model, _vad_utils

        console.print("[yellow]Loading Silero VAD model (shared)...[/yellow]")

        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )

            _vad_model = model
            _vad_utils = utils[0]  # get_speech_timestamps function
            _is_loaded = True

            console.print("[green]Shared VAD model ready[/green]")
            return _vad_model, _vad_utils

        except Exception as e:
            console.print(f"[red]Failed to load Silero VAD: {e}[/red]")
            raise RuntimeError(f"Failed to load VAD model: {e}")


def is_vad_loaded() -> bool:
    """Check if VAD model is already loaded."""
    return _is_loaded
