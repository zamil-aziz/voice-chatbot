"""
Voice Activity Detection using Silero VAD.
Determines when the user is speaking vs silent.
"""

import numpy as np
from typing import Tuple
from collections import deque

from rich.console import Console

from .vad_singleton import get_vad_model
from config.settings import settings

console = Console()


class VoiceActivityDetector:
    """
    Silero VAD wrapper for detecting speech in audio.

    Uses a sliding window approach with configurable thresholds
    for detecting speech start/end.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        sample_rate: int = 16000,
        window_size_samples: int = 512,
        smoothing_window: int = 4,
    ):
        """
        Initialize VAD.

        Args:
            threshold: Speech probability threshold (0-1)
            min_speech_duration_ms: Minimum speech duration to trigger
            min_silence_duration_ms: Silence duration to end speech
            sample_rate: Audio sample rate (8000 or 16000)
            window_size_samples: Samples per VAD window (512 for 16kHz)
            smoothing_window: Chunks averaged for onset detection
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate
        self.window_size_samples = window_size_samples

        # State tracking
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None

        # Probability history for smoothing speech onset
        self.prob_history = deque(maxlen=smoothing_window)
        self.last_raw_prob = 0.0

        # Use shared VAD model singleton (includes device for GPU acceleration)
        self.model, self.get_speech_timestamps, self.device = get_vad_model(settings.vad.device)

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Get speech probability for an audio chunk.

        Args:
            audio_chunk: Audio samples (float32, mono)

        Returns:
            Probability that the chunk contains speech (0-1)
        """
        import torch

        # Ensure correct shape and type
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Normalize
        if np.abs(audio_chunk).max() > 1.0:
            audio_chunk = audio_chunk / np.abs(audio_chunk).max()

        # Convert to tensor and move to GPU if available
        audio_tensor = torch.from_numpy(audio_chunk).to(self.device)

        # Get probability
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        self.last_raw_prob = speech_prob

        # Smooth probability
        self.prob_history.append(speech_prob)
        smoothed_prob = np.mean(self.prob_history)

        return smoothed_prob

    def process(
        self, audio_chunk: np.ndarray, timestamp_ms: float
    ) -> Tuple[bool, bool, bool]:
        """
        Process an audio chunk and detect speech events.

        Args:
            audio_chunk: Audio samples
            timestamp_ms: Current timestamp in milliseconds

        Returns:
            Tuple of (is_speech, speech_started, speech_ended)
        """
        smoothed_prob = self.get_speech_probability(audio_chunk)
        # Asymmetric detection: onset uses the smoothed probability to reject
        # one-chunk noise spikes; while speaking, the raw probability decides
        # so silence registers immediately instead of waiting for the window
        # mean to decay (the min_silence timer already debounces short pauses)
        if self.is_speaking:
            is_speech = self.last_raw_prob >= self.threshold
        else:
            is_speech = smoothed_prob >= self.threshold

        speech_started = False
        speech_ended = False

        if is_speech:
            if not self.is_speaking:
                # Potential speech start
                if self.speech_start_time is None:
                    self.speech_start_time = timestamp_ms
                elif (
                    timestamp_ms - self.speech_start_time >= self.min_speech_duration_ms
                ):
                    # Confirmed speech start
                    self.is_speaking = True
                    speech_started = True
                    console.print("[cyan]Speech started[/cyan]")

            # Reset silence tracking
            self.silence_start_time = None
        else:
            if self.is_speaking:
                # Potential speech end
                if self.silence_start_time is None:
                    self.silence_start_time = timestamp_ms
                elif (
                    timestamp_ms - self.silence_start_time
                    >= self.min_silence_duration_ms
                ):
                    # Confirmed speech end
                    self.is_speaking = False
                    speech_ended = True
                    self.speech_start_time = None
                    console.print("[cyan]Speech ended[/cyan]")

            # Reset speech start tracking
            if not self.is_speaking:
                self.speech_start_time = None

        return is_speech, speech_started, speech_ended

    def set_profile(self, threshold: float, min_speech_duration_ms: int) -> None:
        """Switch detection strictness, e.g. a stricter profile during playback."""
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms

    def reset(self) -> None:
        """Reset VAD state."""
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.prob_history.clear()
        self.last_raw_prob = 0.0


# Quick test
if __name__ == "__main__":
    import sounddevice as sd
    import time

    vad = VoiceActivityDetector()

    console.print("[bold]Testing VAD - speak into the microphone[/bold]")
    console.print("Press Ctrl+C to stop\n")

    sample_rate = 16000
    chunk_size = 512

    def callback(indata, frames, time_info, status):
        if status:
            console.print(f"[red]{status}[/red]")

        audio = indata[:, 0].astype(np.float32)
        prob = vad.get_speech_probability(audio)

        # Visual indicator
        bar_length = int(prob * 30)
        bar = "=" * bar_length + " " * (30 - bar_length)
        speech = "[green]SPEECH[/green]" if prob > 0.5 else "[dim]silence[/dim]"
        console.print(f"\r[{bar}] {prob:.2f} {speech}", end="")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            blocksize=chunk_size,
            callback=callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped[/yellow]")
