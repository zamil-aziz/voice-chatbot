"""
Voice Pipeline Manager.
Orchestrates the full STT → LLM → TTS flow.
"""

import json
import re
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ..audio import AudioCapture, AudioPlayer, VoiceActivityDetector
from ..models import SpeechToText, LanguageModel, TextToSpeech, NotesRAG
from ..processing import TextPreprocessor, DynamicSpeedController
from config.settings import settings

console = Console()


class SentenceSegmenter:
    """Accumulates streamed model text into TTS-friendly spoken chunks."""

    SENTENCE_ENDINGS = ('.', '!', '?')
    FIRST_CLAUSE_TRIGGERS = (',', ';', ':', ' -')

    def __init__(self, first_min_chars: int = 18, first_min_words: int = 4):
        self.buffer = ""
        self.first_segment_emitted = False
        self.first_min_chars = first_min_chars
        self.first_min_words = first_min_words

    def add(self, text: str) -> list[str]:
        """Add streamed text and return zero or more complete spoken chunks."""
        self.buffer += text
        current = self.buffer.strip()
        if not current:
            return []

        should_emit = current.endswith(self.SENTENCE_ENDINGS)
        if not self.first_segment_emitted and not should_emit:
            word_count = len(re.findall(r"\b\w+\b", current))
            long_enough = len(current) >= self.first_min_chars and word_count >= self.first_min_words
            if long_enough and any(current.endswith(trigger) for trigger in self.FIRST_CLAUSE_TRIGGERS):
                should_emit = True

        if not should_emit:
            return []

        self.buffer = ""
        self.first_segment_emitted = True
        return [current]

    def flush(self) -> list[str]:
        """Return any remaining text as a final spoken chunk."""
        current = self.buffer.strip()
        self.buffer = ""
        if current:
            self.first_segment_emitted = True
            return [current]
        return []


class VoicePipeline:
    """
    Complete voice conversation pipeline.

    Handles:
    - Real-time audio capture with VAD
    - Speech-to-text transcription
    - LLM response generation
    - Text-to-speech synthesis
    """

    def __init__(
        self,
        stt: Optional[SpeechToText] = None,
        llm: Optional[LanguageModel] = None,
        tts: Optional[TextToSpeech] = None,
        rag: Optional[NotesRAG] = None,
    ):
        """
        Initialize the voice pipeline.

        Args:
            stt: Speech-to-text model (created if None)
            llm: Language model (created if None)
            tts: Text-to-speech model (created if None)
            rag: RAG system for personal notes (created if None)
        """
        console.print("[bold]Initializing Voice Pipeline[/bold]")
        console.print("=" * 50)

        # Initialize models (lazy loading)
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.rag = rag
        self._models_loaded = False

        # Audio components
        self.capture: Optional[AudioCapture] = None
        self.player: Optional[AudioPlayer] = None
        self.vad: Optional[VoiceActivityDetector] = None

        # Processing components for natural speech
        self.text_preprocessor = TextPreprocessor(settings.tts.text_processing)
        self.speed_controller = DynamicSpeedController(settings.tts.speed_control)

        # State
        self.is_running = False
        self._is_processing = False
        self._processing_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._active_threads: list = []

        # Reusable executor for background tasks (RAG, etc.)
        self._executor = ThreadPoolExecutor(max_workers=2)

    @property
    def is_processing(self) -> bool:
        """Thread-safe getter for is_processing flag."""
        with self._processing_lock:
            return self._is_processing

    @is_processing.setter
    def is_processing(self, value: bool) -> None:
        """Thread-safe setter for is_processing flag."""
        with self._processing_lock:
            self._is_processing = value

    def clear_conversation(self) -> None:
        """Clear conversation history (and the prompt cache built on it)."""
        if self.llm:
            self.llm.clear_history()

    def _load_models(self, skip_stt: bool = False) -> None:
        """Load all models in parallel if not already loaded.

        Args:
            skip_stt: If True, skip loading STT model (for text input mode)
        """
        if self._models_loaded:
            return

        console.print("\n[bold yellow]Loading AI models in parallel...[/bold yellow]")
        console.print("(This may take a moment on first run)\n")

        start_time = time.time()

        def load_stt():
            if self.stt is None:
                self.stt = SpeechToText()
            return "STT"

        def load_llm():
            if self.llm is None:
                self.llm = LanguageModel(
                    model_name=settings.llm.model_name,
                    max_tokens=settings.llm.max_tokens,
                    temperature=settings.llm.temperature,
                    top_p=settings.llm.top_p,
                    top_k=settings.llm.top_k,
                    min_p=settings.llm.min_p,
                    history_turns=settings.llm.history_turns,
                    enable_thinking=settings.llm.enable_thinking,
                    system_prompt=settings.llm.system_prompt,
                )
            return "LLM"

        def load_tts():
            if self.tts is None:
                # Convert voice blend config to tuple format if configured
                voice_blend = None
                if settings.tts.voice_blend:
                    voice_blend = [
                        (vb.voice, vb.weight)
                        for vb in settings.tts.voice_blend
                    ]
                self.tts = TextToSpeech(
                    voice=settings.tts.voice,
                    speed=settings.tts.speed,
                    voice_blend=voice_blend,
                    device=settings.tts.device,
                    isolated=settings.tts.isolated_process,
                    load_timeout=settings.model_load_timeout,
                )
            return "TTS"

        def load_rag():
            if self.rag is None:
                self.rag = NotesRAG()
            # Trigger lazy loading
            _ = self.rag.count()
            return "RAG"

        # Build list of models to load
        loaders = [load_llm, load_tts]
        if settings.rag.enabled:
            loaders.append(load_rag)
        if not skip_stt:
            loaders.append(load_stt)

        # Load models in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(loader) for loader in loaders]
            for future in as_completed(futures):
                model_name = future.result()
                console.print(f"[green]✓ {model_name} loaded[/green]")

        # Warm up models SEQUENTIALLY to avoid Metal GPU conflicts
        # (Multiple GPU operations in parallel cause command buffer crashes)
        console.print("\n[dim]Warming up models...[/dim]")
        if self.stt:
            self.stt.warmup()
        if self.llm:
            self.llm.warmup()
        if self.tts:
            self.tts.warmup()

        self._models_loaded = True
        total_time = time.time() - start_time
        console.print(f"\n[bold green]All models loaded in {total_time:.1f}s![/bold green]")

    def _init_audio(self) -> None:
        """Initialize audio components."""
        self.capture = AudioCapture(sample_rate=16000)
        self.player = AudioPlayer(sample_rate=24000)
        self.vad = VoiceActivityDetector(
            threshold=settings.vad.threshold,
            min_speech_duration_ms=settings.vad.min_speech_duration_ms,
            min_silence_duration_ms=settings.vad.min_silence_duration_ms,
            smoothing_window=settings.vad.smoothing_window,
        )

    def _log_turn(self, user_text: str, assistant_response: str, timing: Dict[str, float]) -> None:
        """Log a conversation turn to file if logging is enabled."""
        if not settings.logging.enabled or not settings.logging.log_conversations:
            return

        log_dir = Path(settings.logging.log_dir)
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"conversation_{datetime.now().strftime('%Y%m%d')}.jsonl"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_text,
            "assistant": assistant_response,
            "timing": timing,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _synthesize_sentence(
        self,
        sentence: str,
        on_first_audio: Optional[Callable[[], None]] = None,
    ) -> Dict[str, float]:
        """Synthesize and queue a single sentence for playback with enhancements."""
        # Step 1: Preprocess text for better prosody
        processed_sentence = self.text_preprocessor.process(sentence)
        if not processed_sentence:
            return {"tts_synthesis": 0.0, "audio_duration": 0.0, "chunks": 0}

        # Step 2: Calculate dynamic speed based on content
        speed = self.speed_controller.get_sentence_speed(
            processed_sentence,
            base_speed=settings.tts.speed
        )

        # Step 3: Synthesize with calculated speed
        start = time.time()
        chunks = 0
        samples = 0
        marked_first_audio = False
        for _, _, audio_chunk in self.tts.synthesize_stream(processed_sentence, speed=speed):
            if self._stop_event.is_set():
                break

            chunks += 1
            samples += len(audio_chunk)
            if self.player.queue_audio(audio_chunk, self.tts.sample_rate) and not marked_first_audio:
                marked_first_audio = True
                if on_first_audio:
                    on_first_audio()

        elapsed = time.time() - start
        audio_duration = samples / self.tts.sample_rate if self.tts and samples else 0.0
        return {
            "tts_synthesis": elapsed,
            "audio_duration": audio_duration,
            "chunks": chunks,
        }

    def _stream_response_to_speech(
        self,
        text: str,
        rag_context: Optional[list[str]] = None,
        stt_time: float = 0.0,
        rag_time: float = 0.0,
    ) -> tuple[str, Dict[str, float]]:
        """Stream LLM output into a background TTS worker and collect timings."""
        if not self.llm or not self.tts or not self.player:
            raise RuntimeError("Pipeline models and audio player must be loaded first")

        llm_start = time.time()
        segmenter = SentenceSegmenter()
        segment_queue: queue.Queue = queue.Queue()
        sentinel = object()
        spoken_segments: list[str] = []
        tts_errors: list[BaseException] = []
        metrics_lock = threading.Lock()
        metrics: Dict[str, float] = {
            "stt": stt_time,
            "rag": rag_time,
            "llm": 0.0,
            "llm_total": 0.0,
            "tts": 0.0,
            "tts_synthesis": 0.0,
            "tts_audio_duration": 0.0,
            "tts_chunks": 0.0,
            "playback_drain": 0.0,
            "first_audio": 0.0,
            "total": 0.0,
        }
        first_segment_time: Optional[float] = None
        first_tts_start: Optional[float] = None

        def mark_first_audio() -> None:
            with metrics_lock:
                if metrics["first_audio"] == 0.0:
                    metrics["first_audio"] = time.time() - llm_start

        def tts_worker() -> None:
            nonlocal first_tts_start
            while not self._stop_event.is_set():
                item = segment_queue.get()
                if item is sentinel:
                    segment_queue.task_done()
                    break

                sentence = item
                try:
                    if first_tts_start is None:
                        first_tts_start = time.time()
                    tts_stats = self._synthesize_sentence(sentence, on_first_audio=mark_first_audio)
                    with metrics_lock:
                        metrics["tts_synthesis"] += tts_stats["tts_synthesis"]
                        metrics["tts_audio_duration"] += tts_stats["audio_duration"]
                        metrics["tts_chunks"] += tts_stats["chunks"]
                except BaseException as exc:
                    tts_errors.append(exc)
                    break
                finally:
                    segment_queue.task_done()

        self.player.start_streaming()
        worker = threading.Thread(target=tts_worker, daemon=True)
        worker.start()

        def enqueue_spoken_segment(raw_segment: str) -> None:
            nonlocal first_segment_time
            segment = self.llm.clean_response_text(raw_segment)
            if not segment:
                return
            if first_segment_time is None:
                first_segment_time = time.time()
                metrics["llm"] = first_segment_time - llm_start
                console.print(f"[dim]LLM first chunk: {metrics['llm']:.2f}s[/dim]")
            spoken_segments.append(segment)
            segment_queue.put(segment)

        try:
            for token in self.llm.generate_stream(text, context=rag_context):
                if self._stop_event.is_set():
                    break
                if tts_errors:
                    # TTS failed: cancel generation now instead of discovering
                    # the error after the worker join
                    break

                for segment in segmenter.add(token):
                    enqueue_spoken_segment(segment)

            for segment in segmenter.flush():
                enqueue_spoken_segment(segment)
        finally:
            metrics["llm_total"] = time.time() - llm_start
            segment_queue.put(sentinel)
            worker.join(timeout=60.0)

            playback_start = time.time()
            self.player.stop_streaming()
            metrics["playback_drain"] = time.time() - playback_start

        if tts_errors:
            raise tts_errors[0]

        total_end = time.time()
        metrics["total"] = stt_time + (total_end - llm_start)
        if first_tts_start is not None:
            metrics["tts"] = total_end - first_tts_start
        metrics.update({f"llm_{k}": v for k, v in self.llm.last_stream_stats.items()})

        # Segments were cleaned as they were enqueued, so the joined text is
        # already the spoken response - no second cleaning pass needed
        response = " ".join(spoken_segments)
        self.llm.commit_turn(text, response)
        return response, metrics

    def process_text(self, text: str) -> None:
        """
        Process text input directly (skips STT).

        Args:
            text: User's text input
        """
        self.is_processing = True

        try:
            if not text.strip():
                return

            console.print(f"[bold white]You:[/bold white] {text}")

            # Retrieve relevant context from RAG
            rag_context = []
            rag_time = 0.0
            if self.rag:
                rag_start = time.time()
                rag_context = self.rag.search(text)
                rag_time = time.time() - rag_start

            # Stream LLM → TTS
            console.print("[cyan]Thinking...[/cyan]")
            response, timing = self._stream_response_to_speech(
                text,
                rag_context=rag_context,
                rag_time=rag_time,
            )
            console.print(f"[bold green]Assistant:[/bold green] {response}")

            # Log timing
            console.print(
                f"[dim]Timing: RAG={timing['rag']:.2f}s, "
                f"LLM first={timing['llm']:.2f}s, LLM total={timing['llm_total']:.2f}s, "
                f"TTS={timing['tts']:.2f}s, Total={timing['total']:.2f}s"
                + (f", First audio={timing['first_audio']:.2f}s" if timing["first_audio"] else "")
                + "[/dim]"
            )

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False

    def process_utterance(self, audio: np.ndarray) -> None:
        """
        Process a complete user utterance with streaming LLM→TTS.

        Args:
            audio: User's speech audio
        """
        self.is_processing = True

        try:
            # Step 1: Transcribe speech
            # Skip VAD trimming since pipeline VAD already detected speech boundaries
            console.print("\n[cyan]Transcribing...[/cyan]")
            start = time.time()
            text = self.stt.transcribe(audio)
            stt_time = time.time() - start

            if not text.strip():
                console.print("[dim]No speech detected[/dim]")
                return

            console.print(f"[bold white]You:[/bold white] {text}")

            # Start RAG search in background while we prepare LLM
            rag_future = None
            rag_start = None
            if self.rag:
                rag_start = time.time()
                rag_future = self._executor.submit(self.rag.search, text)

            # Step 2 & 3: Stream LLM → TTS (overlapped for lower latency)
            console.print("[cyan]Thinking...[/cyan]")

            # Get RAG results (should be ready by now or very soon)
            rag_context = []
            rag_time = 0.0
            if rag_future:
                try:
                    rag_context = rag_future.result(timeout=0.2)  # Max 200ms wait
                    if rag_start is not None:
                        rag_time = time.time() - rag_start
                except Exception:
                    pass  # Skip RAG if too slow

            response, timing = self._stream_response_to_speech(
                text,
                rag_context=rag_context,
                stt_time=stt_time,
                rag_time=rag_time,
            )
            console.print(f"[bold green]Assistant:[/bold green] {response}")

            # Log timing
            console.print(
                f"[dim]Timing: STT={stt_time:.2f}s, "
                f"RAG={timing['rag']:.2f}s, "
                f"LLM first={timing['llm']:.2f}s, LLM total={timing['llm_total']:.2f}s, "
                f"TTS={timing['tts']:.2f}s, Total={timing['total']:.2f}s"
                + (f", First audio={timing['first_audio']:.2f}s" if timing["first_audio"] else "")
                + "[/dim]"
            )

            # Log conversation turn if enabled
            self._log_turn(text, response, timing)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False

    def run(self) -> None:
        """
        Run the voice conversation loop.

        Press Ctrl+C to stop.
        """
        self._load_models()
        self._init_audio()

        console.print("\n" + "=" * 50)
        console.print("[bold green]Voice Assistant Ready![/bold green]")
        console.print("Speak into your microphone. Press Ctrl+C to quit.")
        console.print("=" * 50 + "\n")

        self.is_running = True
        self._stop_event.clear()

        try:
            self.capture.start()
            loop_start = time.monotonic()
            was_playing = False

            while not self._stop_event.is_set():
                # Get audio chunk
                chunk = self.capture.get_chunk(timeout=0.1)
                if chunk is None:
                    continue

                # Wall-clock timestamps so VAD durations stay correct even
                # when chunks are dropped under load.
                timestamp_ms = (time.monotonic() - loop_start) * 1000.0

                # Skip VAD while audio is playing: with speakers the
                # microphone re-hears the TTS output and would false-trigger.
                if self.player.is_playing:
                    was_playing = True
                    continue

                if was_playing:
                    # Playback just finished; drop any speech state and
                    # partial recording built up from TTS bleed.
                    was_playing = False
                    self.vad.reset()
                    if self.capture.recording:
                        self.capture.stop_recording()

                # Process VAD
                is_speech, speech_started, speech_ended = self.vad.process(
                    chunk, timestamp_ms
                )

                if speech_started:
                    self.capture.start_recording()

                if speech_ended:
                    # Always stop recording so audio can't accumulate
                    # unbounded across skipped turns.
                    audio = self.capture.stop_recording()
                    if self.is_processing:
                        console.print("[dim]Still responding, ignored input[/dim]")
                    elif len(audio) > 0:
                        # Clean up finished threads
                        self._active_threads = [t for t in self._active_threads if t.is_alive()]

                        # Process in background to keep capturing
                        thread = threading.Thread(
                            target=self.process_utterance,
                            args=(audio,),
                            daemon=True,
                        )
                        self._active_threads.append(thread)
                        thread.start()

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
        finally:
            self.stop()

    def run_text_mode(self) -> None:
        """
        Run in text input mode (type instead of speak).

        Press Ctrl+C to stop.
        """
        self._load_models(skip_stt=True)

        # Only need audio player for TTS output
        self.player = AudioPlayer(sample_rate=24000)

        console.print("\n" + "=" * 50)
        console.print("[bold green]Text Input Mode[/bold green]")
        console.print("Type your messages and press Enter. Press Ctrl+C to quit.")
        console.print("=" * 50 + "\n")

        self.is_running = True
        self._stop_event.clear()

        try:
            while not self._stop_event.is_set():
                try:
                    text = console.input("[bold white]You:[/bold white] ")
                    if text.strip():
                        self.process_text(text)
                        console.print()  # Add spacing between turns
                except EOFError:
                    break

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
        finally:
            self._stop_event.set()
            self.is_running = False
            if self.player:
                self.player.stop()
            console.print("[green]Text mode stopped.[/green]")

    def stop(self) -> None:
        """Stop the voice pipeline."""
        self._stop_event.set()
        self.is_running = False

        # Wait for active processing threads to finish (with timeout)
        for thread in self._active_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        self._active_threads.clear()

        # Shutdown background executor
        if self._executor:
            self._executor.shutdown(wait=False)

        if self.capture:
            self.capture.stop()
        if self.player:
            self.player.stop()
        if self.tts and hasattr(self.tts, "close"):
            self.tts.close()

        console.print("[green]Voice assistant stopped.[/green]")


# Quick test
if __name__ == "__main__":
    pipeline = VoicePipeline()
    pipeline.run()
