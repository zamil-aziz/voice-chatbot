"""
Voice Pipeline Manager.
Orchestrates the full STT → LLM → TTS flow.
"""

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ..audio import AudioCapture, AudioPlayer, VoiceActivityDetector, AudioPostProcessor
from ..models import SpeechToText, LanguageModel, TextToSpeech, NotesRAG
from ..processing import TextPreprocessor, DynamicSpeedController
from config.settings import settings

console = Console()


class VoicePipeline:
    """
    Complete voice conversation pipeline.

    Handles:
    - Real-time audio capture with VAD
    - Speech-to-text transcription
    - LLM response generation
    - Text-to-speech synthesis
    - Barge-in (interruption) handling
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
        self.post_processor = AudioPostProcessor(
            sample_rate=24000,
            config=settings.tts.post_processing
        )

        # State
        self.is_running = False
        self._is_processing = False
        self._processing_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._active_threads: list = []

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
                )
            return "TTS"

        def load_rag():
            if self.rag is None:
                self.rag = NotesRAG()
            # Trigger lazy loading
            _ = self.rag.count()
            return "RAG"

        # Build list of models to load
        loaders = [load_llm, load_tts, load_rag]
        if not skip_stt:
            loaders.append(load_stt)

        # Load models in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(loader) for loader in loaders]
            for future in as_completed(futures):
                model_name = future.result()
                console.print(f"[green]✓ {model_name} loaded[/green]")

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

    def _synthesize_sentence(self, sentence: str) -> None:
        """Synthesize and queue a single sentence for playback with enhancements."""
        # Step 1: Preprocess text for better prosody
        processed_sentence = self.text_preprocessor.process(sentence)

        # Step 2: Calculate dynamic speed based on content
        speed = self.speed_controller.get_sentence_speed(
            processed_sentence,
            base_speed=settings.tts.speed
        )

        # Step 3: Synthesize with calculated speed
        for _, _, audio_chunk in self.tts.synthesize_stream(processed_sentence, speed=speed):
            if self._stop_event.is_set():
                break

            # Step 4: Apply audio post-processing for naturalness
            if settings.tts.post_processing.enabled:
                audio_chunk = self.post_processor.process(audio_chunk)

            self.player.queue_audio(audio_chunk, self.tts.sample_rate)

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
            if self.rag:
                rag_context = self.rag.search(text, n_results=2)

            # Stream LLM → TTS
            console.print("[cyan]Thinking...[/cyan]")
            llm_start = time.time()
            tts_start = None
            first_audio_time = None

            # Start audio streaming
            self.player.start_streaming()

            # Buffer for accumulating LLM tokens until sentence boundary
            sentence_tokens: list = []
            response_tokens: list = []
            sentence_endings = ('.', '!', '?')

            for token in self.llm.generate_stream(text, context=rag_context):
                if self._stop_event.is_set():
                    break

                response_tokens.append(token)
                sentence_tokens.append(token)

                # Check for sentence boundary
                token_stripped = token.rstrip()
                if token_stripped and token_stripped[-1] in sentence_endings:
                    sentence = ''.join(sentence_tokens).strip()
                    if sentence:
                        if tts_start is None:
                            tts_start = time.time()
                            llm_time = tts_start - llm_start
                            console.print(f"[dim]LLM first sentence: {llm_time:.2f}s[/dim]")

                        self._synthesize_sentence(sentence)

                        if first_audio_time is None:
                            first_audio_time = time.time() - llm_start

                    sentence_tokens = []

            # Handle any remaining text
            remaining = ''.join(sentence_tokens).strip()
            if remaining:
                if tts_start is None:
                    tts_start = time.time()
                    llm_time = tts_start - llm_start
                self._synthesize_sentence(remaining)

            full_response = ''.join(response_tokens)

            self.player.stop_streaming()

            # Calculate timing
            total_end = time.time()
            if tts_start is None:
                tts_start = llm_start
                llm_time = 0
            tts_time = total_end - tts_start

            response = full_response.strip()
            console.print(f"[bold green]Assistant:[/bold green] {response}")

            # Log timing
            total_time = total_end - llm_start
            console.print(
                f"[dim]Timing: LLM={llm_time:.2f}s, TTS={tts_time:.2f}s, "
                f"Total={total_time:.2f}s"
                + (f", First audio={first_audio_time:.2f}s" if first_audio_time else "")
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
            console.print("\n[cyan]Transcribing...[/cyan]")
            start = time.time()
            text = self.stt.transcribe(audio)
            stt_time = time.time() - start

            if not text.strip():
                console.print("[dim]No speech detected[/dim]")
                return

            console.print(f"[bold white]You:[/bold white] {text}")

            # Retrieve relevant context from RAG
            rag_context = []
            if self.rag:
                rag_context = self.rag.search(text, n_results=2)

            # Step 2 & 3: Stream LLM → TTS (overlapped for lower latency)
            console.print("[cyan]Thinking...[/cyan]")
            llm_start = time.time()
            tts_start = None
            first_audio_time = None

            # Start audio streaming
            self.player.start_streaming()

            # Buffer for accumulating LLM tokens until sentence boundary
            # Using lists for efficient accumulation (avoids string concat overhead)
            sentence_tokens: list = []
            response_tokens: list = []
            sentence_endings = ('.', '!', '?')

            for token in self.llm.generate_stream(text, context=rag_context):
                if self._stop_event.is_set():
                    break

                response_tokens.append(token)
                sentence_tokens.append(token)

                # Check for sentence boundary (check token directly, not full buffer)
                token_stripped = token.rstrip()
                if token_stripped and token_stripped[-1] in sentence_endings:
                    sentence = ''.join(sentence_tokens).strip()
                    if sentence:
                        # First sentence: record time to first audio
                        if tts_start is None:
                            tts_start = time.time()
                            llm_time = tts_start - llm_start
                            console.print(f"[dim]LLM first sentence: {llm_time:.2f}s[/dim]")

                        # Synthesize this sentence immediately
                        self._synthesize_sentence(sentence)

                        if first_audio_time is None:
                            first_audio_time = time.time() - llm_start

                    sentence_tokens = []

            # Handle any remaining text (incomplete sentence)
            remaining = ''.join(sentence_tokens).strip()
            if remaining:
                if tts_start is None:
                    tts_start = time.time()
                    llm_time = tts_start - llm_start
                self._synthesize_sentence(remaining)

            full_response = ''.join(response_tokens)

            self.player.stop_streaming()

            # Calculate timing
            total_end = time.time()
            if tts_start is None:
                tts_start = llm_start
                llm_time = 0
            tts_time = total_end - tts_start

            response = full_response.strip()
            console.print(f"[bold green]Assistant:[/bold green] {response}")

            # Log timing
            total_time = stt_time + (total_end - llm_start)
            timing = {"stt": stt_time, "llm": llm_time, "tts": tts_time, "total": total_time}
            console.print(
                f"[dim]Timing: STT={stt_time:.2f}s, "
                f"LLM={llm_time:.2f}s, TTS={tts_time:.2f}s, "
                f"Total={total_time:.2f}s"
                + (f", First audio={first_audio_time:.2f}s" if first_audio_time else "")
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
            timestamp_ms = 0
            chunk_duration_ms = (self.capture.chunk_size / self.capture.sample_rate) * 1000

            while not self._stop_event.is_set():
                # Get audio chunk
                chunk = self.capture.get_chunk(timeout=0.1)
                if chunk is None:
                    continue

                timestamp_ms += chunk_duration_ms

                # Skip processing while audio is playing (barge-in disabled)
                # Note: Barge-in was causing false interruptions when using speakers
                # (microphone picking up TTS audio). Re-enable if using headphones.
                if self.player.is_playing:
                    continue

                # Process VAD
                is_speech, speech_started, speech_ended = self.vad.process(
                    chunk, timestamp_ms
                )

                if speech_started:
                    self.capture.start_recording()

                if speech_ended and not self.is_processing:
                    audio = self.capture.stop_recording()
                    if len(audio) > 0:
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

        if self.capture:
            self.capture.stop()
        if self.player:
            self.player.stop()

        console.print("[green]Voice assistant stopped.[/green]")


# Quick test
if __name__ == "__main__":
    pipeline = VoicePipeline()
    pipeline.run()
