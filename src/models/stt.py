"""
Speech-to-Text module using NVIDIA Parakeet TDT via MLX.
Supports batch transcription and streaming transcription during capture.
"""

import queue
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import numpy as np

from rich.console import Console
from rich.markup import escape

from config.settings import settings

console = Console()


def _to_mx_audio(audio: np.ndarray):
    """Convert numpy float32 audio to a normalized 1D mx.array."""
    import mlx.core as mx

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    peak = np.abs(audio).max() if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    return mx.array(audio.reshape(-1))


class SpeechToText:
    """Parakeet TDT speech recognition using MLX."""

    def __init__(
        self,
        model_name: str = "mlx-community/parakeet-tdt-0.6b-v3",
    ):
        self.model_name = model_name
        self.model = None
        # The streaming context swaps the encoder attention implementation on
        # enter/exit, so batch transcription must never overlap a stream
        self._model_lock = threading.Lock()
        self._load_model()

    def _load_model(self) -> None:
        """Load Parakeet model with timeout. Downloads if not cached."""
        console.print(f"[yellow]Loading STT model: {self.model_name}[/yellow]")
        start = time.time()

        def do_load():
            import mlx.core as mx
            from mlx.utils import tree_flatten
            from parakeet_mlx import from_pretrained

            model = from_pretrained(self.model_name)
            # Force weight evaluation here: lazy arrays stay bound to this
            # thread's MLX stream, and evaluating them later from another
            # thread fails with "There is no Stream(gpu, N) in current thread"
            mx.eval([value for _, value in tree_flatten(model.parameters())])
            return model

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(do_load)
                self.model = future.result(timeout=settings.model_load_timeout)

            console.print(
                f"[green]STT model ready in {time.time() - start:.2f}s[/green]"
            )
        except FuturesTimeoutError:
            raise RuntimeError(
                f"STT model loading timed out after {settings.model_load_timeout}s"
            )
        except ImportError as e:
            console.print(f"[red]Failed to import parakeet_mlx: {e}[/red]")
            console.print("[yellow]Run: pip install parakeet-mlx[/yellow]")
            raise

    def _is_hallucination(self, text: str) -> bool:
        """Detect common hallucination patterns like repetition loops."""
        words = text.lower().split()
        if len(words) < 5:
            return False

        # Check if same word repeated many times (>50% of text)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        max_repeat = max(word_counts.values())
        if max_repeat > len(words) * 0.5:
            console.print(f"[yellow]Hallucination detected (repetition)[/yellow]")
            return True

        return False

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as numpy array (float32, mono), already
                trimmed to speech boundaries by the pipeline VAD
            sample_rate: Sample rate in Hz (must be 16000, Parakeet's rate)

        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start = time.time()

        if len(audio) < sample_rate * 0.1:  # Less than 100ms of audio
            console.print("[dim]Audio too short, skipping[/dim]")
            return ""

        from parakeet_mlx.audio import get_logmel

        with self._model_lock:
            mel = get_logmel(_to_mx_audio(audio), self.model.preprocessor_config)
            result = self.model.generate(mel)[0]

        text = result.text.strip()
        elapsed = time.time() - start

        # Check for hallucination patterns
        if self._is_hallucination(text):
            console.print(f"[dim]STT ({elapsed:.2f}s): \\[rejected hallucination][/dim]")
            return ""

        console.print(f"[dim]STT ({elapsed:.2f}s): {escape(text)}[/dim]")

        return text

    def warmup(self) -> None:
        """Warm up the model to avoid cold-start latency on first real transcription."""
        if self.model is None:
            return

        console.print("[dim]Warming up STT...[/dim]")
        start = time.time()
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence

        from parakeet_mlx.audio import get_logmel

        with self._model_lock:
            mel = get_logmel(_to_mx_audio(dummy_audio), self.model.preprocessor_config)
            self.model.generate(mel)
        console.print(f"[dim]STT warm-up done in {time.time() - start:.2f}s[/dim]")


class StreamingTranscriber:
    """Feeds VAD-gated audio into a Parakeet stream while the user speaks.

    Runs a dedicated worker thread: encoding a batch of audio takes tens of
    milliseconds, which would stall the main loop's 32ms chunk cadence.
    By the time the utterance ends, the transcript is (nearly) ready, so
    speech-to-text drops off the response critical path.
    """

    _SENTINEL = object()

    def __init__(
        self,
        stt: SpeechToText,
        batch_seconds: float = 0.5,
        context_size: Tuple[int, int] = (256, 256),
        depth: int = 1,
        sample_rate: int = 16000,
        on_partial: Optional[Callable[[str], None]] = None,
    ):
        self.stt = stt
        self.batch_samples = int(batch_seconds * sample_rate)
        self.context_size = tuple(context_size)
        self.depth = depth
        self.on_partial = on_partial

        self._commands: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # Main-loop API (all non-blocking)

    def start(self, prebuffer: List[Tuple[int, np.ndarray]]) -> None:
        """Begin a streaming session, seeded with the capture pre-buffer."""
        self._commands.put(("start", list(prebuffer)))

    def add(self, seq: int, chunk: np.ndarray) -> None:
        """Feed one live capture chunk (deduplicated against the pre-buffer by seq)."""
        self._commands.put(("add", seq, chunk))

    def finalize(self) -> "Future[str]":
        """End the session; the returned future resolves to the final transcript."""
        future: Future = Future()
        self._commands.put(("finalize", future))
        return future

    def abort(self) -> None:
        """Discard the current session (e.g. input ignored or barge-in reset)."""
        self._commands.put(("abort",))

    def close(self) -> None:
        """Stop the worker thread."""
        self._commands.put(self._SENTINEL)
        self._worker.join(timeout=5.0)

    # Worker internals

    def _worker_loop(self) -> None:
        stream_ctx = None
        stream = None
        last_seq = -1
        pending: List[np.ndarray] = []
        pending_samples = 0
        last_partial = ""
        session_error: Optional[BaseException] = None

        def feed_pending(force: bool = False) -> None:
            nonlocal pending, pending_samples, last_partial
            if stream is None or not pending:
                return
            if not force and pending_samples < self.batch_samples:
                return
            batch = np.concatenate(pending)
            pending = []
            pending_samples = 0
            stream.add_audio(_to_mx_audio(batch))
            if self.on_partial is not None:
                text = stream.result.text.strip()
                if text and text != last_partial:
                    last_partial = text
                    self.on_partial(text)

        def close_stream() -> None:
            nonlocal stream_ctx, stream, pending, pending_samples, last_seq, last_partial, session_error
            # Clear state before exiting the context so a __exit__ that raises
            # can't leave stream_ctx set and make a later call double-release
            # the model lock.
            ctx = stream_ctx
            stream_ctx = None
            stream = None
            pending = []
            pending_samples = 0
            last_seq = -1
            last_partial = ""
            session_error = None
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                finally:
                    self.stt._model_lock.release()

        while True:
            command = self._commands.get()
            if command is self._SENTINEL:
                close_stream()
                return

            try:
                kind = command[0]
                if kind == "start":
                    close_stream()
                    self.stt._model_lock.acquire()
                    try:
                        stream_ctx = self.stt.model.transcribe_stream(
                            context_size=self.context_size, depth=self.depth
                        )
                        stream = stream_ctx.__enter__()
                    except BaseException:
                        self.stt._model_lock.release()
                        stream_ctx = None
                        stream = None
                        raise
                    for seq, chunk in command[1]:
                        if seq > last_seq:
                            last_seq = seq
                            pending.append(chunk)
                            pending_samples += len(chunk)
                    feed_pending()
                elif kind == "add":
                    _, seq, chunk = command
                    if stream is not None and seq > last_seq:
                        last_seq = seq
                        pending.append(chunk)
                        pending_samples += len(chunk)
                        feed_pending()
                elif kind == "finalize":
                    future = command[1]
                    if session_error is not None:
                        future.set_exception(session_error)
                    elif stream is None:
                        future.set_result("")
                    else:
                        try:
                            feed_pending(force=True)
                            text = stream.result.text.strip()
                            future.set_result(text)
                        except BaseException as exc:
                            future.set_exception(exc)
                    close_stream()
                elif kind == "abort":
                    close_stream()
            except BaseException as exc:
                # Remember the failure so finalize() surfaces it; a broken
                # session must not take down the worker thread
                session_error = exc
                console.print(
                    f"[red]Streaming STT error: {exc}[/red]\n"
                    f"[dim]{traceback.format_exc()}[/dim]"
                )


# Quick test
if __name__ == "__main__":
    import scipy.io.wavfile as wav

    stt = SpeechToText()

    # Test with a sample file if exists
    test_file = Path("test_audio.wav")
    if test_file.exists():
        rate, data = wav.read(test_file)
        audio = data.astype(np.float32)
        if data.dtype == np.int16:
            audio /= 32768.0
        text = stt.transcribe(audio, sample_rate=rate)
        console.print(f"[green]Transcription: {text}[/green]")
    else:
        console.print("[yellow]No test_audio.wav found. Create one to test.[/yellow]")
