"""
Voice Pipeline Manager.
Orchestrates the full STT → LLM → TTS flow.
"""

import json
import time
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ..audio import AudioCapture, AudioPlayer, VoiceActivityDetector
from ..models import SpeechToText, LanguageModel, TextToSpeech
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
    ):
        """
        Initialize the voice pipeline.

        Args:
            stt: Speech-to-text model (created if None)
            llm: Language model (created if None)
            tts: Text-to-speech model (created if None)
        """
        console.print("[bold]Initializing Voice Pipeline[/bold]")
        console.print("=" * 50)

        # Initialize models (lazy loading)
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self._models_loaded = False

        # Audio components
        self.capture: Optional[AudioCapture] = None
        self.player: Optional[AudioPlayer] = None
        self.vad: Optional[VoiceActivityDetector] = None

        # State
        self.is_running = False
        self.is_processing = False
        self._stop_event = threading.Event()

    def _load_models(self) -> None:
        """Load all models if not already loaded."""
        if self._models_loaded:
            return

        console.print("\n[bold yellow]Loading AI models...[/bold yellow]")
        console.print("(This may take a moment on first run)\n")

        if self.stt is None:
            self.stt = SpeechToText()

        if self.llm is None:
            self.llm = LanguageModel(
                model_name=settings.llm.model_name,
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature,
                system_prompt=settings.llm.system_prompt,
            )

        if self.tts is None:
            self.tts = TextToSpeech()

        self._models_loaded = True
        console.print("\n[bold green]All models loaded![/bold green]")

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
        """Synthesize and queue a single sentence for playback."""
        for _, _, audio_chunk in self.tts.synthesize_stream(sentence):
            if self._stop_event.is_set():
                break
            self.player.queue_audio(audio_chunk, self.tts.sample_rate)

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

            # Step 2 & 3: Stream LLM → TTS (overlapped for lower latency)
            console.print("[cyan]Thinking...[/cyan]")
            llm_start = time.time()
            tts_start = None
            first_audio_time = None

            # Start audio streaming
            self.player.start_streaming()

            # Buffer for accumulating LLM tokens until sentence boundary
            sentence_buffer = ""
            full_response = ""
            sentence_endings = ('.', '!', '?')

            for token in self.llm.generate_stream(text):
                if self._stop_event.is_set():
                    break

                full_response += token
                sentence_buffer += token

                # Check for sentence boundary
                # Look for sentence ending followed by space or end of token
                if any(sentence_buffer.rstrip().endswith(end) for end in sentence_endings):
                    sentence = sentence_buffer.strip()
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

                    sentence_buffer = ""

            # Handle any remaining text (incomplete sentence)
            if sentence_buffer.strip():
                if tts_start is None:
                    tts_start = time.time()
                    llm_time = tts_start - llm_start
                self._synthesize_sentence(sentence_buffer.strip())

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
                        # Process in background to keep capturing
                        threading.Thread(
                            target=self.process_utterance,
                            args=(audio,),
                            daemon=True,
                        ).start()

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the voice pipeline."""
        self._stop_event.set()
        self.is_running = False

        if self.capture:
            self.capture.stop()
        if self.player:
            self.player.stop()

        console.print("[green]Voice assistant stopped.[/green]")


# Quick test
if __name__ == "__main__":
    pipeline = VoicePipeline()
    pipeline.run()
