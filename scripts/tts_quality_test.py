#!/usr/bin/env python3
"""
TTS Quality A/B Test Script

Generates the same phrases with different configurations for comparison.
This helps identify which settings produce the best audio quality.

Usage:
    python -m scripts.tts_quality_test

Output:
    Creates test_audio/ directory with WAV files for each configuration.
    Listen and compare to find the best settings.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import soundfile as sf
from rich.console import Console
from rich.table import Table

from src.models.tts import TextToSpeech
from src.processing.text_preprocessor import TextPreprocessor
from src.audio.post_processor import AudioPostProcessor
from config.settings import PostProcessingSettings

console = Console()

# Test phrases covering different emotional tones and content types
TEST_PHRASES = [
    # Curiosity / engagement
    "Oh, that's really interesting! Tell me more about it.",
    # Thinking / hesitation
    "Hmm... let me think about that for a second.",
    # Empathy / concern
    "I'm so sorry to hear that. Are you okay?",
    # Excitement / surprise
    "Wait, seriously?! That's amazing news!",
    # Factual / numbers (tests text preprocessing)
    "The total comes to fifty-three dollars and twenty-seven cents.",
]

# Test configurations: (name, voice, use_text_preprocessing, post_proc_settings)
CONFIGS = [
    (
        "01_bella_current",
        "af_bella",
        True,
        PostProcessingSettings(),
        "Current config (bella + full processing)"
    ),
    (
        "02_heart_full",
        "af_heart",
        True,
        PostProcessingSettings(),
        "Heart voice + full processing"
    ),
    (
        "03_heart_no_post",
        "af_heart",
        True,
        PostProcessingSettings(enabled=False),
        "Heart voice, no post-processing"
    ),
    (
        "04_heart_raw",
        "af_heart",
        False,
        PostProcessingSettings(enabled=False),
        "Heart voice, raw (no text or audio processing)"
    ),
    (
        "05_heart_dynamics_only",
        "af_heart",
        True,
        PostProcessingSettings(warmth_enabled=False),
        "Heart voice, dynamics only (no warmth)"
    ),
    (
        "06_heart_warmth_only",
        "af_heart",
        True,
        PostProcessingSettings(dynamics_enabled=False),
        "Heart voice, warmth only (no dynamics)"
    ),
    (
        "07_heart_light",
        "af_heart",
        True,
        PostProcessingSettings(compression_ratio=1.5, warmth_boost_db=1.0),
        "Heart voice, lighter processing (1.5 ratio, 1.0dB)"
    ),
    (
        "08_nicole_full",
        "af_nicole",
        True,
        PostProcessingSettings(),
        "Nicole voice (soft/calm) + full processing"
    ),
]


def main():
    output_dir = Path("test_audio")
    output_dir.mkdir(exist_ok=True)

    console.print("\n[bold cyan]TTS Quality A/B Test[/bold cyan]")
    console.print("=" * 60)
    console.print(f"Output directory: {output_dir.absolute()}")
    console.print(f"Configurations: {len(CONFIGS)}")
    console.print(f"Test phrases: {len(TEST_PHRASES)}")
    console.print(f"Total files: {len(CONFIGS) * len(TEST_PHRASES)}")
    console.print()

    # Create summary table
    table = Table(title="Test Configurations")
    table.add_column("#", style="cyan")
    table.add_column("Config Name", style="green")
    table.add_column("Description")

    for config_name, voice, use_text_proc, post_settings, description in CONFIGS:
        table.add_row(config_name[:2], config_name[3:], description)

    console.print(table)
    console.print()

    # Track which voices we need to load
    loaded_tts = {}
    preprocessor = TextPreprocessor()

    for config_name, voice, use_text_proc, post_settings, description in CONFIGS:
        console.print(f"\n[bold yellow]=== {config_name} ===[/bold yellow]")
        console.print(f"[dim]{description}[/dim]")

        # Load TTS model if not already loaded for this voice
        if voice not in loaded_tts:
            console.print(f"[dim]Loading voice: {voice}...[/dim]")
            loaded_tts[voice] = TextToSpeech(voice=voice)

        tts = loaded_tts[voice]
        post_processor = AudioPostProcessor(config=post_settings)

        for i, phrase in enumerate(TEST_PHRASES, 1):
            # Apply text preprocessing if enabled
            if use_text_proc:
                processed_text = preprocessor.process(phrase)
            else:
                processed_text = phrase

            # Synthesize
            audio, sr = tts.synthesize(processed_text)

            # Apply post-processing if enabled
            if post_settings.enabled:
                audio = post_processor.process(audio)

            # Save to file
            filename = output_dir / f"{config_name}_phrase{i}.wav"
            sf.write(str(filename), audio, sr)

            duration = len(audio) / sr
            console.print(f"  [green]Saved:[/green] {filename.name} ({duration:.2f}s)")

    # Print listening guide
    console.print("\n" + "=" * 60)
    console.print("[bold green]Done! Audio files saved to:[/bold green]", output_dir.absolute())
    console.print()
    console.print("[bold]Listening Guide:[/bold]")
    console.print("Compare files with the same phrase number across configs.")
    console.print("For example, compare all *_phrase1.wav files:")
    console.print()

    for i, phrase in enumerate(TEST_PHRASES, 1):
        console.print(f"  [cyan]Phrase {i}:[/cyan] {phrase[:50]}...")

    console.print()
    console.print("[bold]What to listen for:[/bold]")
    console.print("  - Naturalness: Does it sound like a real person?")
    console.print("  - Expression: Are emotions conveyed properly?")
    console.print("  - Clarity: Is the audio clear or muddy?")
    console.print("  - Pacing: Is the rhythm natural?")
    console.print()
    console.print("[yellow]Tip: Use a media player that can easily switch between files.[/yellow]")


if __name__ == "__main__":
    main()
