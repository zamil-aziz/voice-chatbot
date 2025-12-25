"""
Voice Pipeline Manager.
Orchestrates the full STT → LLM → TTS flow.
"""

import time
import threading
import numpy as np
from typing import Optional, Callable

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ..audio import AudioCapture, AudioPlayer, VoiceActivityDetector
from ..models import SpeechToText, LanguageModel, TextToSpeech

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
            self.llm = LanguageModel()

        if self.tts is None:
            self.tts = TextToSpeech()

        self._models_loaded = True
        console.print("\n[bold green]All models loaded![/bold green]")

    def _init_audio(self) -> None:
        """Initialize audio components."""
        self.capture = AudioCapture(sample_rate=16000)
        self.player = AudioPlayer(sample_rate=24000)
        self.vad = VoiceActivityDetector(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=700,  # Slightly longer for natural pauses
        )

    def process_utterance(self, audio: np.ndarray) -> None:
        """
        Process a complete user utterance.

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

            # Step 2: Generate response
            console.print("[cyan]Thinking...[/cyan]")
            start = time.time()
            response = self.llm.generate(text)
            llm_time = time.time() - start

            console.print(f"[bold green]Assistant:[/bold green] {response}")

            # Step 3: Synthesize speech
            console.print("[cyan]Speaking...[/cyan]")
            start = time.time()
            audio_out, sr = self.tts.synthesize(response)
            tts_time = time.time() - start

            # Step 4: Play audio (can be interrupted)
            self.player.play_async(audio_out, sr)

            # Log timing
            total_time = stt_time + llm_time + tts_time
            console.print(
                f"[dim]Timing: STT={stt_time:.2f}s, "
                f"LLM={llm_time:.2f}s, TTS={tts_time:.2f}s, "
                f"Total={total_time:.2f}s[/dim]"
            )

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
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

                # Check for barge-in (user interrupting)
                if self.player.is_playing:
                    prob = self.vad.get_speech_probability(chunk)
                    if prob > 0.6:  # User is speaking
                        console.print("[yellow]Interrupted![/yellow]")
                        self.player.stop()
                        self.vad.reset()
                        self.capture.clear_queue()
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
