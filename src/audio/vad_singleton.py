"""
Shared Silero VAD singleton.
Loads the model once and shares it across VoiceActivityDetector and SpeechToText.
"""

import threading
from typing import Tuple, Callable

from rich.console import Console

from config.settings import settings
from ..utils import resolve_torch_device

console = Console()

# Module-level singleton state
_vad_model = None
_vad_utils = None
_vad_device = None
_load_lock = threading.Lock()
_is_loaded = False


def get_vad_model(preferred_device: str = "cpu") -> Tuple[any, Callable, str]:
    """
    Get the shared Silero VAD model and utility functions.

    Uses the silero-vad pip package (no torch.hub network fetch). With
    settings.vad.use_onnx (default) the model runs on onnxruntime, which is
    faster than the JIT torch model for the tiny per-chunk workload.

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
            from silero_vad import load_silero_vad, get_speech_timestamps

            use_onnx = settings.vad.use_onnx
            if use_onnx:
                # ONNX runs on CPU; callers keep tensors on CPU
                model = load_silero_vad(onnx=True)
                device = "cpu"
            else:
                device = resolve_torch_device(preferred_device)
                model = load_silero_vad(onnx=False).to(device)

            _vad_model = model
            _vad_utils = get_speech_timestamps
            _vad_device = device
            _is_loaded = True

            backend = "ONNX" if use_onnx else "TORCH"
            console.print(f"[green]Shared VAD model ready ({backend}, {device.upper()})[/green]")
            return _vad_model, _vad_utils, _vad_device

        except Exception as e:
            console.print(f"[red]Failed to load Silero VAD: {e}[/red]")
            raise RuntimeError(f"Failed to load VAD model: {e}")
