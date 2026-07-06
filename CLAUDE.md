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
4. **RAG** → personal notes retrieved in the background (skipped if slower than 200ms)
5. **LLM** → Qwen generates a streamed response (keeps the last 6 conversation turns)
6. **TTS** → streamed LLM text is segmented into sentences and synthesized by Kokoro at 24kHz while generation continues (text preprocessing and dynamic speed applied per sentence)
7. **Playback** → ring-buffered streaming output; VAD is paused while audio plays to avoid the microphone re-hearing the TTS

### Key Components

| Component | File | Model/Library |
|-----------|------|---------------|
| Speech-to-Text | `src/models/stt.py` | mlx-whisper (whisper-large-v3-turbo) |
| Language Model | `src/models/llm.py` | mlx-lm (Qwen3-4B-Instruct-2507-4bit) |
| Text-to-Speech | `src/models/tts.py` | Kokoro (multiple voices, isolated subprocess) |
| Notes RAG | `src/models/rag.py` | sentence-transformers (all-MiniLM-L6-v2) |
| Voice Activity | `src/audio/vad.py` | Silero VAD (ONNX runtime by default) |
| Audio Capture | `src/audio/capture.py` | sounddevice (queue-based) |
| Audio Playback | `src/audio/playback.py` | sounddevice (streaming ring buffer) |
| Text Preprocessor | `src/processing/text_preprocessor.py` | TTS-safe text normalization |
| Speed Controller | `src/processing/speed_controller.py` | Emotion-aware speech pacing |

### Configuration
All settings in `config/settings.py` use Pydantic models. Key settings:
- Audio: 16kHz input, 24kHz TTS output
- VAD: 250ms min speech, 300ms silence to end turn
- LLM: 256 max tokens (safety net; persona keeps replies short), 0.7 temperature, "Maya" persona system prompt, 6 turns of history
- RAG: `data/notes.txt` (one note per line), min cosine similarity 0.30
- TTS: `af_heart` default voice (Kokoro has American/British voices)
- Text Processing: emoji stripping, symbol replacement, formatted-phone-number reading
- Speed Control: slower for questions/empathy, faster for excitement

### Audio Flow Details
- `AudioCapture` uses a callback-based `sounddevice.InputStream` pushing to a queue, with a 500ms pre-buffer so the start of an utterance is not lost while VAD confirms speech
- `AudioPlayer.start_streaming()`/`queue_audio()` provide gapless ring-buffered playback
- VAD uses smoothed probability (10-sample history) to reduce false triggers

## Requirements
- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB RAM minimum
- ~15GB disk space for models (downloaded on first run)
- Python 3.10+
