"""
Audio playback module for playing synthesized speech.
"""

import collections
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

    # Max audio chunks in queue (~2 seconds of audio at typical chunk sizes)
    MAX_QUEUE_SIZE = 50

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

        self._is_playing = False
        self._playing_lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)

    @property
    def is_playing(self) -> bool:
        """Thread-safe getter for is_playing flag."""
        with self._playing_lock:
            return self._is_playing

    @is_playing.setter
    def is_playing(self, value: bool) -> None:
        """Thread-safe setter for is_playing flag."""
        with self._playing_lock:
            self._is_playing = value

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

        # Clear the ring buffer if it exists (for streaming mode)
        if hasattr(self, '_stream_buffer') and hasattr(self, '_buffer_lock'):
            with self._buffer_lock:
                self._stream_buffer.clear()

    def start_streaming(self) -> None:
        """Start background streaming playback."""
        self._stop_flag.clear()
        self._audio_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.is_playing = True  # Mark as playing immediately for barge-in detection

        # Ring buffer for gapless playback
        self._stream_buffer: collections.deque = collections.deque()
        self._buffer_lock = threading.Lock()
        self._stream_started = threading.Event()

        def stream_worker():
            import sounddevice as sd

            def audio_callback(outdata, frames, time_info, status):
                with self._buffer_lock:
                    available = len(self._stream_buffer)
                    if available >= frames:
                        # Fill output from buffer
                        for i in range(frames):
                            outdata[i, 0] = self._stream_buffer.popleft()
                    elif available > 0:
                        # Partial buffer - use what we have, pad with silence
                        for i in range(available):
                            outdata[i, 0] = self._stream_buffer.popleft()
                        outdata[available:, 0] = 0
                    else:
                        # Buffer empty - output silence
                        outdata[:, 0] = 0

            # Open continuous output stream
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=audio_callback,
                device=self.device,
                blocksize=1024,  # ~42ms at 24kHz
            ):
                self._stream_started.set()

                while not self._stop_flag.is_set():
                    try:
                        audio, sr = self._audio_queue.get(timeout=0.1)
                        if audio is None:  # Sentinel to stop
                            # Wait for buffer to drain before exiting
                            while len(self._stream_buffer) > 0 and not self._stop_flag.is_set():
                                threading.Event().wait(0.05)
                            break

                        # Add audio to ring buffer
                        if audio.dtype != np.float32:
                            audio = audio.astype(np.float32)
                        flat_audio = audio.flatten()

                        with self._buffer_lock:
                            self._stream_buffer.extend(flat_audio)

                    except queue.Empty:
                        # Check if buffer is empty and we should stop
                        with self._buffer_lock:
                            if len(self._stream_buffer) == 0:
                                self.is_playing = False
                            else:
                                self.is_playing = True
                        continue

        self._playback_thread = threading.Thread(target=stream_worker, daemon=True)
        self._playback_thread.start()

    def queue_audio(
        self, audio: np.ndarray, sample_rate: Optional[int] = None
    ) -> bool:
        """
        Queue audio for streaming playback.

        Args:
            audio: Audio samples
            sample_rate: Sample rate

        Returns:
            True if queued successfully, False if queue is full
        """
        sr = sample_rate or self.sample_rate

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        try:
            self._audio_queue.put((audio, sr), timeout=1.0)
            return True
        except queue.Full:
            console.print("[yellow]Audio queue full, dropping chunk[/yellow]")
            return False

    def stop_streaming(self) -> None:
        """Stop streaming playback after draining the queue."""
        # Send sentinel to signal end of stream (worker will finish current queue)
        self._audio_queue.put((None, None))

        # Wait for thread to finish playing all queued audio
        if self._playback_thread:
            self._playback_thread.join(timeout=30.0)  # Allow time for audio to play
            self._playback_thread = None

        # Only now set flags - audio has finished naturally
        self._stop_flag.set()
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
