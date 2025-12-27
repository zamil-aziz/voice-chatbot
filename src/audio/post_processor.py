"""
Audio Post-Processor for natural voice enhancement.
Applies subtle effects to make TTS output sound more human-like.
"""

from typing import Optional
import numpy as np

from config.settings import PostProcessingSettings


class AudioPostProcessor:
    """
    Apply post-processing effects to TTS audio for naturalness.

    Effects include:
    - Pitch micro-variations (simulates human pitch instability)
    - Dynamics processing (evens out volume)
    - Warmth enhancement (adds low-frequency body)
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        config: Optional[PostProcessingSettings] = None
    ):
        self.sample_rate = sample_rate
        self.config = config or PostProcessingSettings()

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply all enabled processing steps to audio.

        Args:
            audio: Input audio as float32 numpy array

        Returns:
            Processed audio as float32 numpy array
        """
        if not self.config.enabled:
            return audio

        if len(audio) == 0:
            return audio

        # Ensure float32
        audio = audio.astype(np.float32)

        if self.config.pitch_variation_enabled:
            audio = self._add_pitch_variation(audio)

        if self.config.dynamics_enabled:
            audio = self._apply_dynamics(audio)

        if self.config.warmth_enabled:
            audio = self._add_warmth(audio)

        return audio

    def _add_pitch_variation(self, audio: np.ndarray) -> np.ndarray:
        """
        Add subtle pitch micro-variations for naturalness.

        Human speech has tiny random pitch fluctuations that synthetic
        speech lacks. This adds a slow, subtle random modulation.

        Args:
            audio: Input audio

        Returns:
            Audio with pitch micro-variations
        """
        depth = self.config.pitch_variation_depth

        if depth <= 0 or len(audio) < 100:
            return audio

        # Create slow random modulation curve
        # Use about 5 Hz modulation rate for subtle pitch drift
        duration = len(audio) / self.sample_rate
        num_points = max(2, int(duration * 5))

        # Generate smooth random curve
        np.random.seed(None)  # Use true randomness
        random_points = np.random.randn(num_points) * depth

        # Smooth the random curve using a simple moving average
        if num_points >= 3:
            kernel_size = min(3, num_points)
            kernel = np.ones(kernel_size) / kernel_size
            random_points = np.convolve(random_points, kernel, mode='same')

        # Interpolate to audio length
        modulation = np.interp(
            np.linspace(0, 1, len(audio)),
            np.linspace(0, 1, num_points),
            random_points
        )

        # Apply pitch shift via time-domain stretching
        # Create new sample indices with subtle variations
        indices = np.arange(len(audio), dtype=np.float64)
        indices = indices * (1 + modulation)
        indices = np.clip(indices, 0, len(audio) - 1)

        # Interpolate audio at new positions
        result = np.interp(np.arange(len(audio)), indices, audio)

        return result.astype(np.float32)

    def _apply_dynamics(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply gentle compression for consistent loudness.

        Evens out volume differences between words and phrases,
        making speech easier to listen to while preserving dynamics.

        Args:
            audio: Input audio

        Returns:
            Audio with dynamics processing
        """
        if len(audio) < 100:
            return audio

        ratio = self.config.compression_ratio
        threshold = 0.3  # Start compressing above this level

        # Calculate envelope using RMS in small windows
        window_size = int(0.01 * self.sample_rate)  # 10ms windows
        if window_size < 1:
            window_size = 1

        # Pad audio for even window division
        pad_length = (window_size - len(audio) % window_size) % window_size
        padded = np.pad(audio, (0, pad_length), mode='constant')

        # Calculate RMS envelope
        reshaped = padded.reshape(-1, window_size)
        rms = np.sqrt(np.mean(reshaped ** 2, axis=1) + 1e-10)

        # Expand envelope back to audio length
        envelope = np.repeat(rms, window_size)[:len(audio)]

        # Apply soft-knee compression
        gain = np.ones_like(envelope)
        mask = envelope > threshold
        gain[mask] = (threshold + (envelope[mask] - threshold) / ratio) / envelope[mask]

        # Smooth gain changes (100ms attack/release)
        smooth_samples = int(0.1 * self.sample_rate)
        if smooth_samples > 1:
            kernel = np.ones(smooth_samples) / smooth_samples
            gain = np.convolve(gain, kernel, mode='same')

        # Apply gain
        result = audio * gain

        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 0.99:
            result = result * (0.99 / max_val)

        return result.astype(np.float32)

    def _add_warmth(self, audio: np.ndarray) -> np.ndarray:
        """
        Add subtle low-frequency boost for vocal warmth.

        Enhances the bass frequencies slightly to add body to the voice,
        making it sound fuller and more natural.

        Args:
            audio: Input audio

        Returns:
            Audio with warmth enhancement
        """
        boost_db = self.config.warmth_boost_db

        if boost_db <= 0 or len(audio) < 100:
            return audio

        # Use a simple first-order IIR lowpass filter
        # Cutoff around 200Hz for warmth frequencies
        cutoff_hz = 200
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        dt = 1.0 / self.sample_rate
        alpha = dt / (rc + dt)

        # Apply lowpass filter to extract low frequencies
        low_freq = np.zeros_like(audio)
        low_freq[0] = audio[0]
        for i in range(1, len(audio)):
            low_freq[i] = low_freq[i-1] + alpha * (audio[i] - low_freq[i-1])

        # Calculate boost multiplier
        boost_linear = 10 ** (boost_db / 20) - 1.0

        # Blend: original + boosted low frequencies
        result = audio + boost_linear * low_freq * 0.5

        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 0.99:
            result = result * (0.99 / max_val)

        return result.astype(np.float32)


# Quick test
if __name__ == "__main__":
    import sounddevice as sd

    print("Audio Post-Processor Demo")
    print("=" * 60)

    # Generate test tone
    duration = 2.0
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create a simple voice-like signal (formant-like tones)
    signal = (
        0.5 * np.sin(2 * np.pi * 150 * t) +  # Fundamental
        0.3 * np.sin(2 * np.pi * 500 * t) +  # First formant
        0.2 * np.sin(2 * np.pi * 1500 * t)   # Second formant
    )
    # Add amplitude envelope
    envelope = np.exp(-t * 0.5) * (1 - np.exp(-t * 10))
    signal = (signal * envelope).astype(np.float32)

    processor = AudioPostProcessor(sample_rate=sample_rate)

    print("\nPlaying original signal...")
    sd.play(signal, sample_rate)
    sd.wait()

    print("Playing processed signal...")
    processed = processor.process(signal)
    sd.play(processed, sample_rate)
    sd.wait()

    print("\nDone!")
