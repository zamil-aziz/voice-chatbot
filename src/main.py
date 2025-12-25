#!/usr/bin/env python3
"""
Voice Chatbot - Local AI Voice Assistant
========================================

A fully local, privacy-respecting voice assistant running on Apple Silicon.

Components:
- STT: Whisper (via MLX)
- LLM: Llama 3.1 8B (via MLX)
- TTS: Kokoro
- VAD: Silero

Usage:
    python -m src.main              # Run voice assistant
    python -m src.main --test-stt   # Test speech-to-text
    python -m src.main --test-llm   # Test language model
    python -m src.main --test-tts   # Test text-to-speech
    python -m src.main --test-all   # Test all components
"""

import argparse
import sys
import time

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_stt():
    """Test speech-to-text component."""
    console.print(Panel("[bold]Testing Speech-to-Text (Whisper)[/bold]"))

    from .models import SpeechToText
    from .audio import AudioCapture

    stt = SpeechToText()

    console.print("\n[yellow]Recording 5 seconds of audio...[/yellow]")
    console.print("Speak now!\n")

    with AudioCapture() as capture:
        capture.start_recording()
        time.sleep(5)
        audio = capture.stop_recording()

    console.print(f"[green]Recorded {len(audio) / 16000:.2f}s of audio[/green]")
    console.print("[yellow]Transcribing...[/yellow]")

    text = stt.transcribe(audio)
    console.print(f"\n[bold green]Transcription:[/bold green] {text}")


def test_llm():
    """Test language model component."""
    console.print(Panel("[bold]Testing Language Model (Llama 3.1 8B)[/bold]"))

    from .models import LanguageModel

    llm = LanguageModel()

    prompts = [
        "Hello! Who are you?",
        "What's the weather like today?",
        "Tell me a short joke.",
    ]

    for prompt in prompts:
        console.print(f"\n[bold white]User:[/bold white] {prompt}")
        response = llm.generate(prompt)
        console.print(f"[bold green]Assistant:[/bold green] {response}")


def test_tts():
    """Test text-to-speech component."""
    console.print(Panel("[bold]Testing Text-to-Speech (Kokoro)[/bold]"))

    from .models import TextToSpeech
    from .audio import AudioPlayer

    tts = TextToSpeech(voice="af_bella")
    player = AudioPlayer(sample_rate=24000)

    texts = [
        "Hello! I'm your AI voice assistant.",
        "I can help you with many tasks.",
        "Just speak naturally, and I'll respond.",
    ]

    for text in texts:
        console.print(f"\n[yellow]Synthesizing:[/yellow] {text}")
        audio, sr = tts.synthesize(text)
        console.print(f"[green]Playing {len(audio) / sr:.2f}s of audio...[/green]")
        player.play(audio, sr)


def test_vad():
    """Test voice activity detection."""
    console.print(Panel("[bold]Testing Voice Activity Detection (Silero VAD)[/bold]"))

    from .audio import AudioCapture, VoiceActivityDetector
    import numpy as np

    vad = VoiceActivityDetector()

    console.print("\n[yellow]Listening for 10 seconds...[/yellow]")
    console.print("Speak to see VAD in action!\n")

    with AudioCapture() as capture:
        start = time.time()
        timestamp_ms = 0

        while time.time() - start < 10:
            chunk = capture.get_chunk(timeout=0.1)
            if chunk is None:
                continue

            timestamp_ms += 32  # ~32ms per chunk at 16kHz/512

            prob = vad.get_speech_probability(chunk)
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            status = "[green]SPEECH[/green]" if prob > 0.5 else "[dim]silence[/dim]"
            console.print(f"\r[{bar}] {prob:.2f} {status}  ", end="")

    console.print("\n\n[green]VAD test complete![/green]")


def test_all():
    """Test all components."""
    test_vad()
    console.print("\n" + "=" * 50 + "\n")
    test_stt()
    console.print("\n" + "=" * 50 + "\n")
    test_llm()
    console.print("\n" + "=" * 50 + "\n")
    test_tts()


def run_assistant():
    """Run the full voice assistant."""
    from .pipeline import VoicePipeline

    console.print(
        Panel.fit(
            "[bold blue]Voice Chatbot[/bold blue]\n"
            "[dim]Local AI Voice Assistant[/dim]",
            border_style="blue",
        )
    )

    pipeline = VoicePipeline()
    pipeline.run()


def main():
    parser = argparse.ArgumentParser(
        description="Voice Chatbot - Local AI Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--test-stt",
        action="store_true",
        help="Test speech-to-text component",
    )
    parser.add_argument(
        "--test-llm",
        action="store_true",
        help="Test language model component",
    )
    parser.add_argument(
        "--test-tts",
        action="store_true",
        help="Test text-to-speech component",
    )
    parser.add_argument(
        "--test-vad",
        action="store_true",
        help="Test voice activity detection",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all components",
    )

    args = parser.parse_args()

    try:
        if args.test_stt:
            test_stt()
        elif args.test_llm:
            test_llm()
        elif args.test_tts:
            test_tts()
        elif args.test_vad:
            test_vad()
        elif args.test_all:
            test_all()
        else:
            run_assistant()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
