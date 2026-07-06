"""
Audio playback module for playing synthesized speech.
"""

import collections
import time
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
        # Ring buffer for gapless streaming playback: deque of numpy chunks
        # plus a read offset into the head chunk
        self._stream_buffer: collections.deque = collections.deque()
        self._read_offset = 0
        self._buffered_samples = 0
        self._buffer_lock = threading.Lock()
        self._stream_started = threading.Event()

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

    def stop(self) -> None:
        """Stop any playing audio immediately and discard queued audio."""
        import sounddevice as sd

        self._stop_flag.set()
        sd.stop()

        with self._buffer_lock:
            self._stream_buffer.clear()
            self._read_offset = 0
            self._buffered_samples = 0

        # Join the worker so a later start_streaming() never overlaps
        # two OutputStreams on the same device
        if self._playback_thread:
            self._playback_thread.join(timeout=2.0)
            self._playback_thread = None

        self.is_playing = False

    def start_streaming(self) -> None:
        """Start background streaming playback."""
        self._stop_flag.clear()
        self._audio_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        # Playing from the moment a response starts until stop()/stop_streaming();
        # a briefly drained buffer between sentences does NOT mean the response
        # is over, so nothing else may flip this flag
        self.is_playing = True

        self._stream_buffer = collections.deque()
        self._read_offset = 0
        self._buffered_samples = 0
        self._stream_started.clear()

        def stream_worker():
            import sounddevice as sd

            def audio_callback(outdata, frames, time_info, status):
                out = outdata[:, 0]
                filled = 0
                with self._buffer_lock:
                    while filled < frames and self._stream_buffer:
                        chunk = self._stream_buffer[0]
                        take = min(frames - filled, len(chunk) - self._read_offset)
                        out[filled:filled + take] = chunk[
                            self._read_offset:self._read_offset + take
                        ]
                        filled += take
                        self._read_offset += take
                        if self._read_offset >= len(chunk):
                            self._stream_buffer.popleft()
                            self._read_offset = 0
                    self._buffered_samples -= filled
                if filled < frames:
                    out[filled:] = 0

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
                            while not self._stop_flag.is_set():
                                with self._buffer_lock:
                                    if self._buffered_samples <= 0:
                                        break
                                time.sleep(0.05)
                            break

                        # Add audio to ring buffer
                        if audio.dtype != np.float32:
                            audio = audio.astype(np.float32)
                        flat_audio = audio.flatten()

                        with self._buffer_lock:
                            self._stream_buffer.append(flat_audio)
                            self._buffered_samples += len(flat_audio)

                    except queue.Empty:
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
            True if queued successfully, False if playback was stopped
        """
        sr = sample_rate or self.sample_rate

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Short put timeouts so a stopped player never blocks the caller
        while not self._stop_flag.is_set():
            try:
                self._audio_queue.put((audio, sr), timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def stop_streaming(self) -> None:
        """Stop streaming playback after draining the queue."""
        # Send sentinel to signal end of stream (worker will finish current queue)
        try:
            self._audio_queue.put((None, None), timeout=1.0)
        except queue.Full:
            pass

        # Wait for thread to finish playing all queued audio
        if self._playback_thread:
            self._playback_thread.join(timeout=10.0)
            if self._playback_thread.is_alive():
                console.print(
                    "[yellow]Playback worker did not stop within 10s[/yellow]"
                )
            self._playback_thread = None

        # Only now set flags - audio has finished naturally
        self._stop_flag.set()
        self.is_playing = False


# Quick test
if __name__ == "__main__":
    player = AudioPlayer(sample_rate=24000)

    # Generate a test tone
    duration = 1.0  # seconds
    frequency = 440  # Hz (A4)
    t = np.linspace(0, duration, int(24000 * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    console.print("[bold]Playing test tone (440Hz)...[/bold]")
    player.play(audio)

    console.print("[bold]Streaming the same tone in chunks...[/bold]")
    player.start_streaming()
    for start_idx in range(0, len(audio), 4800):
        player.queue_audio(audio[start_idx:start_idx + 4800])
    player.stop_streaming()

    console.print("[green]Playback test complete![/green]")
