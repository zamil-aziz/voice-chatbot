"""
Audio playback module for playing synthesized speech.
"""

import numpy as np
import threading
import queue
from typing import Optional

from rich.console import Console

console = Console()


class AudioPlayer:
    """
    Audio playback with support for streaming and interruption.

    Supports both blocking playback and background streaming.
    """

    def __init__(
        self,
        sample_rate: int = 24000,  # Kokoro outputs 24kHz
        device: Optional[int] = None,
    ):
        """
        Initialize audio player.

        Args:
            sample_rate: Sample rate in Hz
            device: Audio device index (None for default)
        """
        self.sample_rate = sample_rate
        self.device = device

        self.is_playing = False
        self._stop_flag = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue()

    def play(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Play audio (blocking).

        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate (uses default if None)
        """
        import sounddevice as sd

        sr = sample_rate or self.sample_rate

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        self.is_playing = True
        try:
            sd.play(audio, sr, device=self.device)
            sd.wait()
        finally:
            self.is_playing = False

    def play_async(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Play audio in background (non-blocking).

        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate (uses default if None)
        """
        import sounddevice as sd

        sr = sample_rate or self.sample_rate

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        self.is_playing = True
        sd.play(audio, sr, device=self.device)

        # Start a thread to wait for completion
        def wait_for_completion():
            sd.wait()
            self.is_playing = False

        thread = threading.Thread(target=wait_for_completion, daemon=True)
        thread.start()

    def stop(self) -> None:
        """Stop any playing audio immediately (for barge-in)."""
        import sounddevice as sd

        self._stop_flag.set()
        sd.stop()
        self.is_playing = False

    def start_streaming(self) -> None:
        """Start background streaming playback."""
        self._stop_flag.clear()
        self._audio_queue = queue.Queue()
        self.is_playing = True  # Mark as playing immediately for barge-in detection

        def stream_worker():
            import sounddevice as sd

            while not self._stop_flag.is_set():
                try:
                    audio, sr = self._audio_queue.get(timeout=0.1)
                    if audio is None:  # Sentinel to stop
                        break

                    self.is_playing = True
                    sd.play(audio, sr, device=self.device)
                    sd.wait()
                    self.is_playing = False

                except queue.Empty:
                    continue

        self._playback_thread = threading.Thread(target=stream_worker, daemon=True)
        self._playback_thread.start()

    def queue_audio(
        self, audio: np.ndarray, sample_rate: Optional[int] = None
    ) -> None:
        """
        Queue audio for streaming playback.

        Args:
            audio: Audio samples
            sample_rate: Sample rate
        """
        sr = sample_rate or self.sample_rate

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        self._audio_queue.put((audio, sr))

    def stop_streaming(self) -> None:
        """Stop streaming playback."""
        self._stop_flag.set()
        self._audio_queue.put((None, None))  # Sentinel

        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None

        import sounddevice as sd

        sd.stop()
        self.is_playing = False

    def wait(self) -> None:
        """Wait for current playback to complete."""
        import sounddevice as sd

        sd.wait()
        self.is_playing = False


# Quick test
if __name__ == "__main__":
    import time

    player = AudioPlayer(sample_rate=24000)

    # Generate a test tone
    duration = 1.0  # seconds
    frequency = 440  # Hz (A4)
    t = np.linspace(0, duration, int(24000 * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    console.print("[bold]Playing test tone (440Hz)...[/bold]")
    player.play(audio)

    console.print("[bold]Playing async with early stop...[/bold]")
    player.play_async(audio)
    time.sleep(0.3)
    player.stop()
    console.print("[green]Stopped early![/green]")

    console.print("[green]Playback test complete![/green]")
