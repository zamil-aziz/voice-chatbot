"""
Text-to-Speech module using Kokoro.
Produces realistic, natural-sounding speech.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional
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
        voice: str = "af_heart",
        speed: float = 1.0,
        sample_rate: int = 24000,
    ):
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self.pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Kokoro TTS model with timeout."""
        console.print(f"[yellow]Loading TTS model (voice: {self.voice})[/yellow]")
        start = time.time()

        def do_load():
            from kokoro import KPipeline
            # 'a' = American English, 'b' = British English
            lang_code = self.voice[0]  # 'a' or 'b'
            return KPipeline(lang_code=lang_code)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(do_load)
                self.pipeline = future.result(timeout=settings.model_load_timeout)

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

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (audio samples as float32 numpy array, sample rate)
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        start = time.time()

        # Generate audio
        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
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

    def synthesize_stream(self, text: str):
        """
        Synthesize speech with streaming output.

        Args:
            text: Text to synthesize

        Yields:
            Tuples of (graphemes, phonemes, audio_chunk)
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")

        for graphemes, phonemes, audio_chunk in self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
        ):
            # Convert tensor to numpy if needed
            if hasattr(audio_chunk, 'numpy'):
                audio_chunk = audio_chunk.numpy()
            elif hasattr(audio_chunk, '__array__'):
                audio_chunk = np.asarray(audio_chunk)
            yield graphemes, phonemes, audio_chunk

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
