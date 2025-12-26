"""
Audio capture module for real-time microphone input.
"""

import numpy as np
import threading
import queue
from collections import deque
from typing import Callable, Optional

from rich.console import Console

console = Console()


class AudioCapture:
    """
    Real-time audio capture from microphone.

    Uses a background thread to continuously capture audio
    and push chunks to a queue for processing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,
        device: Optional[int] = None,
    ):
        """
        Initialize audio capture.

        Args:
            sample_rate: Sample rate in Hz
            channels: Number of channels (1 for mono)
            chunk_size: Samples per chunk
            device: Audio device index (None for default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device

        self.stream = None
        self.is_running = False
        # Max queue size: ~3 seconds of audio at typical chunk sizes
        self.audio_queue: queue.Queue = queue.Queue(maxsize=100)

        # Pre-buffer to capture audio before speech detection (500ms worth)
        # This ensures we don't lose the beginning of utterances while VAD confirms speech
        pre_buffer_chunks = int((sample_rate * 0.5) / chunk_size) + 1  # ~500ms of chunks
        self.pre_buffer: deque = deque(maxlen=pre_buffer_chunks)

        # Accumulated audio for recording
        self.recording = False
        self.recorded_chunks: list = []

    def start(self) -> None:
        """Start audio capture."""
        if self.is_running:
            return

        import sounddevice as sd

        console.print(
            f"[yellow]Starting audio capture "
            f"({self.sample_rate}Hz, {self.chunk_size} samples)[/yellow]"
        )

        def callback(indata, frames, time_info, status):
            if status:
                console.print(f"[red]Audio status: {status}[/red]")

            # Get mono audio as float32 (single copy, reuse for all destinations)
            audio = indata[:, 0].astype(np.float32)

            # Queue for processing (non-blocking to avoid audio glitches)
            try:
                self.audio_queue.put_nowait(audio)
            except queue.Full:
                pass  # Drop chunk if queue full (consumer too slow)

            # Always keep recent audio in pre-buffer (for capturing speech start)
            self.pre_buffer.append(audio)

            # Store if recording (need copy here since we're accumulating)
            if self.recording:
                self.recorded_chunks.append(audio.copy())

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            device=self.device,
            callback=callback,
            dtype=np.float32,
        )

        self.stream.start()
        self.is_running = True
        console.print("[green]Audio capture started[/green]")

    def stop(self) -> None:
        """Stop audio capture."""
        if not self.is_running:
            return

        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        console.print("[yellow]Audio capture stopped[/yellow]")

    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the next audio chunk from the queue.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Audio chunk as numpy array, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def start_recording(self) -> None:
        """Start accumulating audio, including pre-buffer for speech already spoken."""
        # Include pre-buffered audio from before speech detection was confirmed
        # This captures the beginning of the utterance that would otherwise be lost
        self.recorded_chunks = list(self.pre_buffer)
        self.recording = True

    def stop_recording(self) -> np.ndarray:
        """
        Stop recording and return accumulated audio.

        Returns:
            Complete audio recording as numpy array
        """
        self.recording = False
        if self.recorded_chunks:
            return np.concatenate(self.recorded_chunks)
        return np.array([], dtype=np.float32)

    def clear_queue(self) -> None:
        """Clear any pending audio chunks."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def list_devices() -> None:
        """Print available audio devices."""
        import sounddevice as sd

        console.print("[bold]Available audio devices:[/bold]")
        console.print(sd.query_devices())

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Quick test
if __name__ == "__main__":
    import time

    AudioCapture.list_devices()
    console.print()

    with AudioCapture() as capture:
        console.print("[bold]Recording for 3 seconds...[/bold]")
        capture.start_recording()

        start = time.time()
        chunks_received = 0

        while time.time() - start < 3:
            chunk = capture.get_chunk(timeout=0.1)
            if chunk is not None:
                chunks_received += 1

        audio = capture.stop_recording()

        console.print(f"[green]Received {chunks_received} chunks[/green]")
        console.print(f"[green]Total audio: {len(audio) / 16000:.2f}s[/green]")
        console.print(f"[green]Audio range: {audio.min():.3f} to {audio.max():.3f}[/green]")
