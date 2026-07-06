"""
Text-to-Speech module using Kokoro on MLX.
Produces realistic, natural-sounding speech fully in-process.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, List, Tuple
import numpy as np

from rich.console import Console

console = Console()


def _log_stream_stats(start: float, first_chunk_time, chunk_count: int, total_audio_samples: int, sample_rate: int) -> None:
    """Log one line of streamed-synthesis timing detail."""
    total_time = time.time() - start
    audio_duration = total_audio_samples / sample_rate if total_audio_samples > 0 else 0
    rtf = total_time / audio_duration if audio_duration > 0 else 0  # Real-time factor
    first_chunk_time = first_chunk_time or 0
    console.print(
        f"[dim]TTS detail: first={first_chunk_time*1000:.0f}ms, "
        f"{chunk_count} chunks, {audio_duration:.2f}s audio, "
        f"RTF={rtf:.2f}x[/dim]"
    )


def _audio_to_numpy(audio_chunk) -> np.ndarray:
    audio = np.asarray(audio_chunk, dtype=np.float32)
    return audio.reshape(-1)


def _patch_mlx_audio_sinegen() -> None:
    """Fix a length mismatch in mlx-audio's Kokoro vocoder (as of 0.4.4).

    SineGen._f02sine down- then up-samples the phase by the hop size (300);
    the rounding can yield one extra frame, so the sine excitation ends up
    300 samples longer than the f0-derived unvoiced mask and the two fail to
    broadcast (e.g. on inputs like "Your AI voice assistant"). The reference
    PyTorch implementation keeps the sine length equal to the f0 length, so
    reconcile to that.
    """
    from mlx_audio.tts.models.kokoro import istftnet
    import mlx.core as mx

    if getattr(istftnet.SineGen, "_f02sine_length_patched", False):
        return

    original = istftnet.SineGen._f02sine

    def _f02sine_fixed(self, f0_values):
        sines = original(self, f0_values)
        target = f0_values.shape[1]
        if sines.shape[1] > target:
            sines = sines[:, :target, :]
        elif sines.shape[1] < target:
            pad = mx.zeros(
                (sines.shape[0], target - sines.shape[1], sines.shape[2]),
                dtype=sines.dtype,
            )
            sines = mx.concatenate([sines, pad], axis=1)
        return sines

    istftnet.SineGen._f02sine = _f02sine_fixed
    istftnet.SineGen._f02sine_length_patched = True


def _create_blended_voice_tensor(
    pipeline,
    voice_blend: Optional[List[Tuple[str, float]]],
    known_voices: Optional[dict[str, str]] = None,
):
    """Create a voice blend tensor, skipping invalid blend entries."""
    if not voice_blend:
        return None

    console.print(f"[dim]Creating voice blend: {voice_blend}[/dim]")

    tensors = []
    weights = []
    voice_names = []

    for voice_name, weight in voice_blend:
        if known_voices is not None and voice_name not in known_voices:
            console.print(f"[yellow]Warning: Unknown voice '{voice_name}' in blend[/yellow]")
            continue
        try:
            tensors.append(pipeline.load_voice(voice_name))
            weights.append(weight)
            voice_names.append(voice_name)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load voice '{voice_name}': {e}[/yellow]")

    if len(tensors) < 2:
        console.print("[yellow]Voice blend requires at least 2 voices, using default[/yellow]")
        return None

    total_weight = sum(weights)
    if total_weight <= 0:
        console.print("[yellow]Voice blend weights must be positive, using default[/yellow]")
        return None

    normalized_weights = [weight / total_weight for weight in weights]
    blended = sum(
        tensor * normalized_weight
        for tensor, normalized_weight in zip(tensors, normalized_weights)
    )

    voice_desc = ", ".join(
        f"{voice_name}:{weight:.0%}"
        for voice_name, weight in zip(voice_names, normalized_weights)
    )
    console.print(f"[green]Voice blend created: {voice_desc}[/green]")
    return blended


class TextToSpeech:
    """Kokoro-based text-to-speech synthesis on MLX."""

    # Available Kokoro English voices (28 total, sorted by quality grade)
    VOICES = {
        # American Female (11 voices)
        "af_heart": "American Female - Heart [A]",
        "af_bella": "American Female - Bella (warm, friendly) [A-]",
        "af_nicole": "American Female - Nicole (soft, calm) [B-]",
        "af_aoede": "American Female - Aoede [C+]",
        "af_kore": "American Female - Kore [C+]",
        "af_sarah": "American Female - Sarah (clear, professional) [C+]",
        "af_nova": "American Female - Nova [C]",
        "af_alloy": "American Female - Alloy [C]",
        "af_sky": "American Female - Sky (young, energetic) [C-]",
        "af_jessica": "American Female - Jessica [D]",
        "af_river": "American Female - River [D]",
        # American Male (9 voices)
        "am_fenrir": "American Male - Fenrir [C+]",
        "am_michael": "American Male - Michael (friendly, casual) [C+]",
        "am_puck": "American Male - Puck [C+]",
        "am_echo": "American Male - Echo [D]",
        "am_eric": "American Male - Eric [D]",
        "am_liam": "American Male - Liam [D]",
        "am_onyx": "American Male - Onyx [D]",
        "am_santa": "American Male - Santa [D-]",
        "am_adam": "American Male - Adam (deep, confident) [F+]",
        # British Female (4 voices)
        "bf_emma": "British Female - Emma (elegant, refined) [B-]",
        "bf_isabella": "British Female - Isabella (warm, articulate) [C]",
        "bf_alice": "British Female - Alice [D]",
        "bf_lily": "British Female - Lily [D]",
        # British Male (4 voices)
        "bm_george": "British Male - George (distinguished, clear) [C]",
        "bm_fable": "British Male - Fable [C]",
        "bm_lewis": "British Male - Lewis (friendly, approachable) [D+]",
        "bm_daniel": "British Male - Daniel [D]",
    }

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        sample_rate: int = 24000,
        voice_blend: Optional[List[Tuple[str, float]]] = None,
        model_name: str = "mlx-community/Kokoro-82M-bf16",
        load_timeout: int = 300,
    ):
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self.voice_blend = voice_blend  # e.g., [("af_bella", 0.6), ("af_heart", 0.4)]
        self.model_name = model_name
        self.load_timeout = load_timeout
        self.pipeline = None
        self._blended_voice_tensor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the MLX Kokoro model and pipeline with timeout."""
        console.print(f"[yellow]Loading TTS model (voice: {self.voice})[/yellow]")
        start = time.time()

        def do_load():
            from mlx_audio.tts.models.kokoro import KokoroPipeline
            from mlx_audio.tts.utils import get_model_path, load_model

            _patch_mlx_audio_sinegen()
            model_path = get_model_path(self.model_name)
            model = load_model(model_path)
            # 'a' = American English, 'b' = British English
            return KokoroPipeline(
                lang_code=self.voice[0],
                model=model,
                repo_id=self.model_name,
            )

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(do_load)
                self.pipeline = future.result(timeout=self.load_timeout)
        except FuturesTimeoutError:
            raise RuntimeError(
                f"TTS model loading timed out after {self.load_timeout}s"
            )
        except ImportError as e:
            console.print(f"[red]Failed to import mlx_audio: {e}[/red]")
            console.print("[yellow]Run: pip install mlx-audio[/yellow]")
            raise

        if self.voice_blend:
            self._blended_voice_tensor = _create_blended_voice_tensor(
                self.pipeline,
                self.voice_blend,
                self.VOICES,
            )

        console.print(f"[green]TTS ready in {time.time() - start:.2f}s[/green]")

    def synthesize(self, text: str, speed: Optional[float] = None) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            speed: Optional speed override (uses instance speed if None)

        Returns:
            Tuple of (audio samples as float32 numpy array, sample rate)
        """
        start = time.time()
        use_speed = speed if speed is not None else self.speed

        audio_chunks = [
            audio_chunk
            for _, _, audio_chunk in self.synthesize_stream(text, speed=use_speed)
        ]

        if audio_chunks:
            audio = np.concatenate(audio_chunks)
        else:
            audio = np.array([], dtype=np.float32)

        elapsed = time.time() - start
        duration = len(audio) / self.sample_rate if len(audio) > 0 else 0

        console.print(
            f"[dim]TTS ({elapsed:.2f}s): {duration:.2f}s audio for "
            f"{len(text)} chars[/dim]"
        )

        return audio, self.sample_rate

    def synthesize_stream(self, text: str, speed: Optional[float] = None):
        """
        Synthesize speech with streaming output.

        Args:
            text: Text to synthesize
            speed: Optional speed override (uses instance speed if None)

        Yields:
            Tuples of (graphemes, phonemes, audio_chunk)
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        use_speed = speed if speed is not None else self.speed
        voice = (
            self._blended_voice_tensor
            if self._blended_voice_tensor is not None
            else self.voice
        )

        start = time.time()
        first_chunk_time = None
        chunk_count = 0
        total_audio_samples = 0

        for graphemes, phonemes, audio_chunk in self.pipeline(
            text,
            voice=voice,
            speed=use_speed,
        ):
            if audio_chunk is None:
                continue

            chunk_count += 1
            if first_chunk_time is None:
                first_chunk_time = time.time() - start

            audio_chunk = _audio_to_numpy(audio_chunk)

            total_audio_samples += len(audio_chunk)
            yield graphemes, phonemes, audio_chunk

        _log_stream_stats(start, first_chunk_time, chunk_count, total_audio_samples, self.sample_rate)

    def warmup(self) -> None:
        """Warm up TTS to avoid cold-start latency on first real synthesis."""
        if self.pipeline is None:
            return

        console.print("[dim]Warming up TTS...[/dim]")
        start = time.time()
        # Generate a short phrase to warm up the model
        for _ in self.synthesize_stream("Hi", speed=1.0):
            pass
        elapsed = time.time() - start
        console.print(f"[dim]TTS warm-up done in {elapsed:.2f}s[/dim]")


# Quick test
if __name__ == "__main__":
    import sounddevice as sd

    tts = TextToSpeech(voice="af_bella")

    # Synthesize and play
    text = "Hello! I'm your AI voice assistant. How can I help you today?"
    console.print(f"\n[bold]Synthesizing:[/bold] {text}")

    audio, sr = tts.synthesize(text)

    console.print(f"[green]Playing audio ({len(audio) / sr:.2f}s)...[/green]")
    sd.play(audio, sr)
    sd.wait()

    console.print("[green]Done![/green]")
