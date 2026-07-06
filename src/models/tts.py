"""
Text-to-Speech module using Kokoro.
Produces realistic, natural-sounding speech.
"""

import time
import itertools
import multiprocessing as mp
import os
import queue
import traceback
from typing import Optional, List, Tuple, Union
import numpy as np

from rich.console import Console

from ..utils import resolve_torch_device

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


def _audio_to_numpy(audio_chunk):
    if hasattr(audio_chunk, "numpy"):
        return audio_chunk.numpy()
    if hasattr(audio_chunk, "__array__"):
        return np.asarray(audio_chunk)
    return audio_chunk


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


def _tts_worker_main(
    request_queue,
    response_queue,
    voice: str,
    speed: float,
    voice_blend: Optional[List[Tuple[str, float]]],
    device: str,
) -> None:
    """Run Kokoro in an isolated process to avoid native shutdown crashes."""
    request_id = None
    try:
        from kokoro import KPipeline

        lang_code = voice[0]
        resolved_device = resolve_torch_device(device)
        pipeline = KPipeline(lang_code=lang_code, device=resolved_device)

        blended_voice = None
        if voice_blend:
            blended_voice = _create_blended_voice_tensor(
                pipeline,
                voice_blend,
                TextToSpeech.VOICES,
            )

        response_queue.put(("ready", resolved_device))

        while True:
            request = request_queue.get()
            command = request[0]
            if command == "close":
                break

            if command != "synthesize":
                continue

            _, request_id, text, request_speed = request
            use_speed = request_speed if request_speed is not None else speed
            selected_voice = blended_voice if blended_voice is not None else voice

            for graphemes, phonemes, audio_chunk in pipeline(
                text,
                voice=selected_voice,
                speed=use_speed,
            ):
                response_queue.put((
                    "chunk",
                    request_id,
                    graphemes,
                    phonemes,
                    _audio_to_numpy(audio_chunk),
                ))
            response_queue.put(("done", request_id))
    except BaseException:
        response_queue.put(("error", request_id, traceback.format_exc()))
    finally:
        # Kokoro imports sentencepiece, whose Abseil flag cleanup can SIGBUS on
        # this macOS/Python combination. Exit directly so the parent stays clean.
        try:
            response_queue.close()
            response_queue.join_thread()
        except Exception:
            pass
        os._exit(0)


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
        voice_blend: Optional[List[Tuple[str, float]]] = None,
        device: str = "cpu",
        isolated: bool = True,
        load_timeout: int = 300,
    ):
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self.voice_blend = voice_blend  # e.g., [("af_bella", 0.6), ("af_heart", 0.4)]
        self.device = device
        self.isolated = isolated
        self.load_timeout = load_timeout
        self.pipeline = None
        self._blended_voice_tensor = None
        self._request_queue = None
        self._response_queue = None
        self._worker_process = None
        self._request_ids = itertools.count(1)
        if self.isolated:
            self._start_worker()
        else:
            self._load_model()

    def _start_worker(self) -> None:
        """Start the isolated Kokoro worker process."""
        console.print(f"[yellow]Loading TTS worker (voice: {self.voice})[/yellow]")
        start = time.time()
        ctx = mp.get_context("spawn")
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()
        self._worker_process = ctx.Process(
            target=_tts_worker_main,
            args=(
                self._request_queue,
                self._response_queue,
                self.voice,
                self.speed,
                self.voice_blend,
                self.device,
            ),
            daemon=True,
        )
        self._worker_process.start()

        deadline = time.time() + self.load_timeout
        while True:
            if time.time() > deadline:
                self.close()
                raise RuntimeError(f"TTS worker loading timed out after {self.load_timeout}s")
            if not self._worker_process.is_alive():
                exitcode = self._worker_process.exitcode
                self.close()
                raise RuntimeError(f"TTS worker exited before ready (exit code {exitcode})")
            try:
                message = self._response_queue.get(timeout=0.1)
                break
            except queue.Empty:
                continue

        if message[0] == "error":
            self.close()
            raise RuntimeError(f"TTS worker failed to load:\n{message[2]}")
        if message[0] != "ready":
            self.close()
            raise RuntimeError(f"Unexpected TTS worker response: {message}")

        resolved_device = message[1]
        console.print(f"[green]TTS worker ready on {resolved_device.upper()} in {time.time() - start:.2f}s[/green]")

    def _load_model(self) -> None:
        """Load Kokoro TTS model with timeout."""
        console.print(f"[yellow]Loading TTS model (voice: {self.voice})[/yellow]")
        start = time.time()

        def do_load():
            device = resolve_torch_device(self.device)

            from kokoro import KPipeline

            # 'a' = American English, 'b' = British English
            lang_code = self.voice[0]  # 'a' or 'b'

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
            # Kokoro/PyTorch model construction is safer on the main thread on macOS.
            self.pipeline = do_load()

            # Create blended voice if configured
            if self.voice_blend:
                self._create_blended_voice()

            console.print(
                f"[green]TTS ready in {time.time() - start:.2f}s[/green]"
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
        self._blended_voice_tensor = _create_blended_voice_tensor(
            self.pipeline,
            self.voice_blend,
            self.VOICES,
        )

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
        if not self.isolated and self.pipeline is None:
            raise RuntimeError("Model not loaded")

        start = time.time()
        use_speed = speed if speed is not None else self.speed

        # Collect all audio chunks (convert tensors to numpy)
        audio_chunks = []
        for _, _, audio_chunk in self.synthesize_stream(text, speed=use_speed):
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
        if not self.isolated and self.pipeline is None:
            raise RuntimeError("Model not loaded")

        use_speed = speed if speed is not None else self.speed

        if self.isolated:
            yield from self._synthesize_stream_worker(text, use_speed)
            return

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
            audio_chunk = _audio_to_numpy(audio_chunk)

            total_audio_samples += len(audio_chunk)
            yield graphemes, phonemes, audio_chunk

        _log_stream_stats(start, first_chunk_time, chunk_count, total_audio_samples, self.sample_rate)

    def _synthesize_stream_worker(self, text: str, speed: Optional[float] = None):
        """Request streamed audio from the isolated Kokoro worker."""
        if not self._worker_process or not self._worker_process.is_alive():
            raise RuntimeError("TTS worker is not running")

        request_id = next(self._request_ids)
        self._request_queue.put(("synthesize", request_id, text, speed))

        start = time.time()
        first_chunk_time = None
        chunk_count = 0
        total_audio_samples = 0

        while True:
            try:
                message = self._response_queue.get(timeout=1.0)
            except queue.Empty:
                if not self._worker_process.is_alive():
                    raise RuntimeError("TTS worker exited unexpectedly")
                continue

            message_type = message[0]
            if message_type == "error":
                raise RuntimeError(f"TTS worker error:\n{message[2]}")
            if message_type == "done" and message[1] == request_id:
                break
            if message_type != "chunk" or message[1] != request_id:
                continue

            _, _, graphemes, phonemes, audio_chunk = message
            chunk_count += 1
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
            total_audio_samples += len(audio_chunk)
            yield graphemes, phonemes, audio_chunk

        _log_stream_stats(start, first_chunk_time, chunk_count, total_audio_samples, self.sample_rate)

    def warmup(self) -> None:
        """Warm up TTS to avoid cold-start latency on first real synthesis."""
        if not self.isolated and self.pipeline is None:
            return

        console.print("[dim]Warming up TTS...[/dim]")
        start = time.time()
        # Generate a short phrase to warm up the model
        for _ in self.synthesize_stream("Hi", speed=1.0):
            pass
        elapsed = time.time() - start
        console.print(f"[dim]TTS warm-up done in {elapsed:.2f}s[/dim]")

    def close(self) -> None:
        """Stop the isolated TTS worker if one is running."""
        if self._worker_process is None:
            return
        if self._worker_process.is_alive():
            try:
                self._request_queue.put(("close",))
                self._worker_process.join(timeout=2.0)
            except Exception:
                pass
        if self._worker_process.is_alive():
            self._worker_process.terminate()
            self._worker_process.join(timeout=2.0)
        self._worker_process = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


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
