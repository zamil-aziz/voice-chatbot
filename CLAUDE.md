# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A fully local, privacy-respecting AI voice assistant for Apple Silicon Macs. All processing happens on-device using MLX-optimized models.

## Commands

```bash
# Run the voice assistant
python -m src.main

# Test individual components
python -m src.main --test-stt   # Speech-to-text (Whisper)
python -m src.main --test-llm   # Language model (Qwen)
python -m src.main --test-tts   # Text-to-speech (Kokoro)
python -m src.main --test-vad   # Voice activity detection (Silero)
python -m src.main --test-all   # All components
```

## Architecture

### Pipeline Flow
The core conversation loop (`VoicePipeline` in `src/pipeline/manager.py`) orchestrates:
1. **Audio Capture** → continuous microphone input at 16kHz
2. **VAD** → Silero VAD detects speech start/end (with configurable silence threshold)
3. **STT** → Whisper transcribes the recorded utterance
4. **LLM** → Qwen2.5 generates response (maintains conversation history, max 20 turns)
5. **TTS** → Kokoro synthesizes speech at 24kHz (with text preprocessing and dynamic speed)
6. **Post-Processing** → pitch variation, dynamics, and warmth for natural sound
7. **Playback** → async audio output with barge-in support (user can interrupt)

### Key Components

| Component | File | Model/Library |
|-----------|------|---------------|
| Speech-to-Text | `src/models/stt.py` | mlx-whisper (whisper-large-v3-turbo) |
| Language Model | `src/models/llm.py` | mlx-lm (Qwen2.5-7B-Instruct-4bit) |
| Text-to-Speech | `src/models/tts.py` | Kokoro (multiple voices) |
| Voice Activity | `src/audio/vad.py` | Silero VAD via PyTorch |
| Audio Capture | `src/audio/capture.py` | sounddevice (queue-based) |
| Audio Playback | `src/audio/playback.py` | sounddevice (async support) |
| Text Preprocessor | `src/processing/text_preprocessor.py` | Prosody enhancement for TTS |
| Speed Controller | `src/processing/speed_controller.py` | Emotion-aware speech pacing |
| Audio Post-Processor | `src/audio/post_processor.py` | Pitch variation, dynamics, warmth |

### Configuration
All settings in `config/settings.py` use Pydantic models. Key settings:
- Audio: 16kHz input, 24kHz TTS output
- VAD: 250ms min speech, 500ms silence to end turn
- LLM: 256 max tokens, 0.85 temperature, detailed persona system prompt ("Maya")
- TTS: `af_bella` default voice (Kokoro has American/British voices)
- Text Processing: interjection expansion, breathing pauses, emphasis markers
- Speed Control: slower for questions/empathy, faster for excitement
- Post-Processing: pitch micro-variation, dynamics compression, warmth boost

### Audio Flow Details
- `AudioCapture` uses a callback-based `sounddevice.InputStream` pushing to a queue
- `AudioPlayer.play_async()` enables non-blocking playback for barge-in detection
- VAD uses smoothed probability (10-sample history) to reduce false triggers

## Requirements
- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB RAM minimum
- ~15GB disk space for models (downloaded on first run)
- Python 3.10+
