"""
Speech-to-Text module using Whisper via MLX.
Optimized for Apple Silicon.
"""

import time
from pathlib import Path
from typing import Optional
import numpy as np

from rich.console import Console

console = Console()


class SpeechToText:
    """Whisper-based speech recognition using MLX."""

    def __init__(
        self,
        model_name: str = "mlx-community/whisper-large-v3-mlx",
        language: str = "en",
    ):
        self.model_name = model_name
        self.language = language
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Whisper model. Downloads if not cached."""
        console.print(f"[yellow]Loading Whisper model: {self.model_name}[/yellow]")
        start = time.time()

        try:
            import mlx_whisper

            # Model will be downloaded on first use
            self.model = mlx_whisper
            console.print(
                f"[green]Whisper model ready in {time.time() - start:.2f}s[/green]"
            )
        except ImportError as e:
            console.print(f"[red]Failed to import mlx_whisper: {e}[/red]")
            console.print("[yellow]Run: pip install mlx-whisper[/yellow]")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate in Hz (default 16000)

        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start = time.time()

        # Ensure audio is in correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        # Transcribe using mlx_whisper
        result = self.model.transcribe(
            audio,
            path_or_hf_repo=self.model_name,
            language=self.language,
            fp16=True,  # Use FP16 for faster inference on M-series
        )

        text = result.get("text", "").strip()
        elapsed = time.time() - start

        console.print(f"[dim]STT ({elapsed:.2f}s): {text}[/dim]")

        return text

    def transcribe_file(self, audio_path: str | Path) -> str:
        """
        Transcribe audio from a file.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)

        Returns:
            Transcribed text
        """
        result = self.model.transcribe(
            str(audio_path),
            path_or_hf_repo=self.model_name,
            language=self.language,
            fp16=True,
        )

        return result.get("text", "").strip()


# Quick test
if __name__ == "__main__":
    import scipy.io.wavfile as wav

    stt = SpeechToText()

    # Test with a sample file if exists
    test_file = Path("test_audio.wav")
    if test_file.exists():
        text = stt.transcribe_file(test_file)
        console.print(f"[green]Transcription: {text}[/green]")
    else:
        console.print("[yellow]No test_audio.wav found. Create one to test.[/yellow]")
        console.print("[dim]You can record with: arecord -d 5 -f S16_LE test_audio.wav[/dim]")
