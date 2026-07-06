"""
Speech-to-Text module using Whisper via MLX.
Optimized for Apple Silicon.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from rich.console import Console

from config.settings import settings

console = Console()


class SpeechToText:
    """Whisper-based speech recognition using MLX."""

    def __init__(
        self,
        model_name: str = "mlx-community/whisper-large-v3-turbo",
        language: str = "en",
    ):
        self.model_name = model_name
        self.language = language
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Whisper model with timeout. Downloads if not cached."""
        console.print(f"[yellow]Loading Whisper model: {self.model_name}[/yellow]")
        start = time.time()

        def do_load():
            import mlx_whisper
            return mlx_whisper

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(do_load)
                self.model = future.result(timeout=settings.model_load_timeout)

            console.print(
                f"[green]Whisper model ready in {time.time() - start:.2f}s[/green]"
            )
        except FuturesTimeoutError:
            raise RuntimeError(
                f"Whisper model loading timed out after {settings.model_load_timeout}s"
            )
        except ImportError as e:
            console.print(f"[red]Failed to import mlx_whisper: {e}[/red]")
            console.print("[yellow]Run: pip install mlx-whisper[/yellow]")
            raise

    def _is_hallucination(self, text: str) -> bool:
        """Detect common hallucination patterns like repetition loops."""
        words = text.lower().split()
        if len(words) < 5:
            return False

        # Check if same word repeated many times (>50% of text)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        max_repeat = max(word_counts.values())
        if max_repeat > len(words) * 0.5:
            console.print(f"[yellow]Hallucination detected (repetition)[/yellow]")
            return True

        return False

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as numpy array (float32, mono), already
                trimmed to speech boundaries by the pipeline VAD
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

        if len(audio) < sample_rate * 0.1:  # Less than 100ms of audio
            console.print("[dim]Audio too short, skipping[/dim]")
            return ""

        # Transcribe using mlx_whisper
        result = self.model.transcribe(
            audio,
            path_or_hf_repo=self.model_name,
            language=self.language,
            fp16=True,  # Use FP16 for faster inference on M-series
            condition_on_previous_text=False,  # Prevents hallucination on short phrases
            compression_ratio_threshold=2.4,  # Detect repetition loops
            no_speech_threshold=0.6,  # Better silence detection
        )

        text = result.get("text", "").strip()
        elapsed = time.time() - start

        # Check for hallucination patterns
        if self._is_hallucination(text):
            console.print(f"[dim]STT ({elapsed:.2f}s): [rejected hallucination][/dim]")
            return ""

        console.print(f"[dim]STT ({elapsed:.2f}s): {text}[/dim]")

        return text

    def warmup(self) -> None:
        """Warm up the model to avoid cold-start latency on first real transcription."""
        if self.model is None:
            return

        console.print("[dim]Warming up Whisper...[/dim]")
        start = time.time()
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        _ = self.model.transcribe(
            dummy_audio,
            path_or_hf_repo=self.model_name,
            language=self.language,
            fp16=True,
        )
        console.print(f"[dim]Whisper warm-up done in {time.time() - start:.2f}s[/dim]")

# Quick test
if __name__ == "__main__":
    import scipy.io.wavfile as wav

    stt = SpeechToText()

    # Test with a sample file if exists
    test_file = Path("test_audio.wav")
    if test_file.exists():
        rate, data = wav.read(test_file)
        audio = data.astype(np.float32)
        if data.dtype == np.int16:
            audio /= 32768.0
        text = stt.transcribe(audio, sample_rate=rate)
        console.print(f"[green]Transcription: {text}[/green]")
    else:
        console.print("[yellow]No test_audio.wav found. Create one to test.[/yellow]")
