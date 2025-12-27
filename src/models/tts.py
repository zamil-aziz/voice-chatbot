"""
Text-to-Speech module using Kokoro.
Produces realistic, natural-sounding speech.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional, List, Tuple, Union
import numpy as np

from rich.console import Console

from config.settings import settings

console = Console()


class TextToSpeech:
    """Kokoro-based text-to-speech synthesis."""

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
        voice: str = "af_nicole",
        speed: float = 1.0,
        sample_rate: int = 24000,
        voice_blend: Optional[List[Tuple[str, float]]] = None,
    ):
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self.voice_blend = voice_blend  # e.g., [("af_bella", 0.6), ("af_heart", 0.4)]
        self.pipeline = None
        self._blended_voice_tensor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Kokoro TTS model with timeout."""
        console.print(f"[yellow]Loading TTS model (voice: {self.voice})[/yellow]")
        start = time.time()

        def do_load():
            import os
            # Enable MPS fallback BEFORE importing torch to avoid meta tensor issues
            # This allows unsupported ops to fall back to CPU while keeping GPU for the rest
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            import torch
            from kokoro import KPipeline

            # 'a' = American English, 'b' = British English
            lang_code = self.voice[0]  # 'a' or 'b'

            # Use MPS (Metal) for GPU acceleration on Apple Silicon
            device = "mps" if torch.backends.mps.is_available() else "cpu"

            try:
                pipeline = KPipeline(lang_code=lang_code, device=device)
                console.print(f"[green]TTS loaded on {device.upper()} (GPU accelerated)[/green]")
                return pipeline
            except RuntimeError as e:
                # If MPS still fails, fall back to CPU
                if device == "mps":
                    console.print(f"[yellow]MPS failed ({str(e)[:50]}...), using CPU[/yellow]")
                    return KPipeline(lang_code=lang_code, device="cpu")
                raise

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(do_load)
                self.pipeline = future.result(timeout=settings.model_load_timeout)

            # Create blended voice if configured
            if self.voice_blend:
                self._create_blended_voice()

            console.print(
                f"[green]TTS ready in {time.time() - start:.2f}s[/green]"
            )
        except FuturesTimeoutError:
            raise RuntimeError(
                f"TTS model loading timed out after {settings.model_load_timeout}s"
            )
        except ImportError as e:
            console.print(f"[red]Failed to import kokoro: {e}[/red]")
            console.print("[yellow]Run: pip install kokoro[/yellow]")
            raise

    def _create_blended_voice(self) -> None:
        """
        Create a blended voice tensor from multiple voices.

        Voice blending allows mixing characteristics from different voices
        to create unique, more expressive voice profiles.
        """
        import torch

        if not self.voice_blend or len(self.voice_blend) == 0:
            return

        console.print(f"[dim]Creating voice blend: {self.voice_blend}[/dim]")

        tensors = []
        weights = []

        for voice_name, weight in self.voice_blend:
            if voice_name not in self.VOICES:
                console.print(f"[yellow]Warning: Unknown voice '{voice_name}' in blend[/yellow]")
                continue
            try:
                # Load voice tensor using Kokoro's internal method
                # Kokoro stores voice tensors that can be averaged
                voice_tensor = self.pipeline.load_voice(voice_name)
                tensors.append(voice_tensor)
                weights.append(weight)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load voice '{voice_name}': {e}[/yellow]")

        if len(tensors) < 2:
            console.print("[yellow]Voice blend requires at least 2 voices, using default[/yellow]")
            return

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Weighted blend of voice tensors
        blended = sum(t * w for t, w in zip(tensors, normalized_weights))
        self._blended_voice_tensor = blended

        voice_desc = ", ".join(f"{v}:{w:.0%}" for v, w in zip(
            [vb[0] for vb in self.voice_blend],
            normalized_weights
        ))
        console.print(f"[green]Voice blend created: {voice_desc}[/green]")

    def _get_voice_for_synthesis(self) -> Union[str, "torch.Tensor"]:
        """Get the voice to use for synthesis (blended tensor or voice name)."""
        if self._blended_voice_tensor is not None:
            return self._blended_voice_tensor
        return self.voice

    def synthesize(self, text: str, speed: Optional[float] = None) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            speed: Optional speed override (uses instance speed if None)

        Returns:
            Tuple of (audio samples as float32 numpy array, sample rate)
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        start = time.time()
        use_speed = speed if speed is not None else self.speed

        # Generate audio (use blended voice if available)
        generator = self.pipeline(
            text,
            voice=self._get_voice_for_synthesis(),
            speed=use_speed,
        )

        # Collect all audio chunks (convert tensors to numpy)
        audio_chunks = []
        for _, _, audio_chunk in generator:
            # Kokoro returns tensors, convert to numpy
            if hasattr(audio_chunk, 'numpy'):
                audio_chunk = audio_chunk.numpy()
            elif hasattr(audio_chunk, '__array__'):
                audio_chunk = np.asarray(audio_chunk)
            audio_chunks.append(audio_chunk)

        # Concatenate all chunks
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

        start = time.time()
        first_chunk_time = None
        chunk_count = 0
        total_audio_samples = 0

        for graphemes, phonemes, audio_chunk in self.pipeline(
            text,
            voice=self._get_voice_for_synthesis(),
            speed=use_speed,
        ):
            chunk_count += 1
            if first_chunk_time is None:
                first_chunk_time = time.time() - start

            # Convert tensor to numpy if needed
            if hasattr(audio_chunk, 'numpy'):
                audio_chunk = audio_chunk.numpy()
            elif hasattr(audio_chunk, '__array__'):
                audio_chunk = np.asarray(audio_chunk)

            total_audio_samples += len(audio_chunk)
            yield graphemes, phonemes, audio_chunk

        # Log detailed TTS timing
        total_time = time.time() - start
        audio_duration = total_audio_samples / self.sample_rate if total_audio_samples > 0 else 0
        rtf = total_time / audio_duration if audio_duration > 0 else 0  # Real-time factor
        console.print(
            f"[dim]TTS detail: first={first_chunk_time*1000:.0f}ms, "
            f"{chunk_count} chunks, {audio_duration:.2f}s audio, "
            f"RTF={rtf:.2f}x[/dim]"
        )

    def list_voices(self) -> dict[str, str]:
        """List available voices with descriptions."""
        return self.VOICES

    def set_voice(self, voice: str) -> None:
        """
        Change the voice.

        Args:
            voice: Voice identifier (e.g., 'af_bella')
        """
        if voice not in self.VOICES:
            available = ", ".join(self.VOICES.keys())
            raise ValueError(f"Unknown voice: {voice}. Available: {available}")

        # Check if we need to reload (language changed)
        old_lang = self.voice[0]
        new_lang = voice[0]

        self.voice = voice

        if old_lang != new_lang:
            console.print(f"[yellow]Switching language, reloading model...[/yellow]")
            self._load_model()

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

    # List voices
    console.print("[bold]Available voices:[/bold]")
    for voice_id, description in tts.list_voices().items():
        console.print(f"  {voice_id}: {description}")

    # Synthesize and play
    text = "Hello! I'm your AI voice assistant. How can I help you today?"
    console.print(f"\n[bold]Synthesizing:[/bold] {text}")

    audio, sr = tts.synthesize(text)

    console.print(f"[green]Playing audio ({len(audio) / sr:.2f}s)...[/green]")
    sd.play(audio, sr)
    sd.wait()

    console.print("[green]Done![/green]")
