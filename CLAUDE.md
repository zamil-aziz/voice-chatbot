# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A fully local, privacy-respecting AI voice assistant for Apple Silicon Macs. All processing happens on-device using MLX-optimized models.

## Commands

```bash
# Run the voice assistant
python -m src.main

# Test individual components
python -m src.main --test-stt   # Speech-to-text (Parakeet)
python -m src.main --test-llm   # Language model (Qwen3.5)
python -m src.main --test-tts   # Text-to-speech (Kokoro)
python -m src.main --test-vad   # Voice activity detection (Silero)
python -m src.main --test-all   # All components
python -m src.main --text       # Text input mode (skips STT)
```

## Architecture

### Pipeline Flow
The core conversation loop (`VoicePipeline` in `src/pipeline/manager.py`) orchestrates:
1. **Audio Capture** → continuous microphone input at 16kHz; every chunk carries a sequence number shared by the live queue and the 500ms pre-buffer
2. **VAD** → Silero VAD detects speech start/end (smoothed probability for onset, raw probability for end-of-speech so turns end fast)
3. **Streaming STT** → while the user speaks, chunks are fed to a Parakeet stream on a worker thread (`StreamingTranscriber`); the transcript is (nearly) ready the moment the turn ends. Partial transcripts fire callbacks
4. **RAG prefetch** → partial transcripts trigger speculative note searches; at turn end the definitive search gets 200ms, falling back to the freshest prefetch
5. **LLM** → Qwen3.5 generates a streamed response. A conversation prompt cache holds the stable rendered-history prefix (discovered as the common prefix of consecutive prompts); each generation runs on a state-shared clone of it, so only the newest messages are processed each turn
6. **TTS** → streamed LLM text is segmented into sentences and synthesized in-process by Kokoro on MLX at 24kHz while generation continues (text preprocessing and dynamic speed applied per sentence)
7. **Playback** → ring-buffered streaming output with vectorized callbacks
8. **Barge-in** → the mic keeps listening during playback with a stricter VAD profile (no AEC, so only dominant speech triggers); user speech halts playback, cancels the turn via a cancel event, records the partial reply to history marked interrupted, and starts the new turn from the pre-buffer. `barge_in.enabled=false` restores half-duplex

### Key Components

| Component | File | Model/Library |
|-----------|------|---------------|
| Speech-to-Text | `src/models/stt.py` | parakeet-mlx (parakeet-tdt-0.6b-v3), batch + streaming |
| Language Model | `src/models/llm.py` | mlx-lm (Qwen3.5-4B-MLX-4bit), cross-turn prompt cache |
| Text-to-Speech | `src/models/tts.py` | mlx-audio Kokoro (Kokoro-82M-bf16), in-process |
| Notes RAG | `src/models/rag.py` | sentence-transformers (all-MiniLM-L6-v2) |
| Voice Activity | `src/audio/vad.py` | Silero VAD (ONNX runtime by default) |
| Audio Capture | `src/audio/capture.py` | sounddevice (queue-based, sequence-numbered chunks) |
| Audio Playback | `src/audio/playback.py` | sounddevice (streaming ring buffer) |
| Text Preprocessor | `src/processing/text_preprocessor.py` | TTS-safe text normalization |
| Speed Controller | `src/processing/speed_controller.py` | Emotion-aware speech pacing |

### Configuration
All settings in `config/settings.py` use Pydantic models. Key settings:
- Audio: 16kHz input, 24kHz TTS output
- VAD: 250ms min speech, 300ms silence to end turn, 4-chunk onset smoothing
- LLM: 256 max tokens (safety net; persona keeps replies short), 0.7 temperature, "Maya" persona system prompt, 6 turns of history (trimmed in batches so the prompt cache is rebuilt rarely)
- STT: 0.5s streaming batches, (256, 256) local attention context
- RAG: `data/notes.txt` (one note per line), min cosine similarity 0.30
- TTS: `af_heart` default voice (Kokoro has American/British voices)
- Barge-in: enabled by default; playback VAD threshold 0.75, 400ms min speech
- Text Processing: emoji stripping, symbol replacement, formatted-phone-number reading
- Speed Control: slower for questions/empathy, faster for excitement

### Audio Flow Details
- `AudioCapture` uses a callback-based `sounddevice.InputStream` pushing `(seq, chunk)` tuples to a queue, with a 500ms pre-buffer so the start of an utterance is not lost while VAD confirms speech; sequence numbers let the STT stream deduplicate the pre-buffer against live chunks
- `AudioPlayer.start_streaming()`/`queue_audio()` provide gapless ring-buffered playback; `is_playing` is true for the whole response (a briefly drained buffer between sentences does not reopen the mic)
- VAD onset uses smoothed probability (4-chunk history) to reject noise spikes; end-of-speech uses the raw probability so silence registers immediately

### MLX Threading Notes
- All three inference models (Parakeet, Qwen3.5, Kokoro) share the Metal device; kernels from concurrent threads serialize safely
- MLX lazy arrays are bound to the stream of the thread that created them: models loaded in a worker thread must have their weights force-evaluated there (see `SpeechToText._load_model`) or later use from other threads fails with a missing-stream error
- Parakeet's streaming context swaps the encoder attention implementation on enter/exit, so batch transcription and streaming are serialized by a lock

## Requirements
- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB RAM minimum
- ~5GB disk space for models (downloaded on first run)
- Python 3.10+
