# Voice Chatbot

A fully local, privacy-respecting AI voice assistant running on Apple Silicon (M1/M2/M3/M4).

## Features

- **100% Local**: All processing happens on your device
- **100% Free**: No API costs, no subscriptions
- **Privacy First**: Your conversations never leave your machine
- **Natural Voice**: 28 high-quality voices with Kokoro TTS
- **Fast**: Optimized for Apple Silicon with MLX
- **Streaming Pipeline**: transcription runs while you speak, and LLM generation and TTS synthesis overlap for lower latency
- **Interruptible**: barge in while the assistant is speaking to start a new turn
- **Personal Context**: RAG-powered retrieval from your notes
- **Apple Silicon Ready**: MLX acceleration for STT, LLM, and TTS in a single process

## Architecture

```
┌─────┐    ┌─────┐    ┌───────────────┐    ┌─────┐    ┌─────┐    ┌─────────┐
│ Mic │ -> │ VAD │ -> │ streaming STT │ -> │ LLM │ -> │ TTS │ -> │ Speaker │
└─────┘    └──┬──┘    └───────────────┘    └──┬──┘    └──┬──┘    └────┬────┘
              │     [transcribes while you speak]        │            │
              │                     [streaming: TTS starts per-sentence]
              └<─────────────── [barge-in: speech interrupts playback] ┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Speech-to-Text | Parakeet TDT 0.6B v3 (via MLX, streaming) |
| Language Model | Qwen3.5 4B 4-bit (via MLX, cross-turn prompt cache) |
| Text-to-Speech | Kokoro 82M on MLX (28 voices, in-process) |
| Voice Detection | Silero VAD (ONNX Runtime backend) |
| Context Retrieval | Sentence-Transformers (all-MiniLM-L6-v2) |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB RAM (minimum)
- Python 3.10+

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd voice-chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Voice Assistant
```bash
python -m src.main            # Voice mode (speak to the assistant)
python -m src.main --text     # Text input mode (type instead of speak; skips STT)
```

### Test Individual Components
```bash
python -m src.main --test-stt   # Test speech-to-text
python -m src.main --test-llm   # Test language model
python -m src.main --test-tts   # Test text-to-speech
python -m src.main --test-vad   # Test voice activity detection
python -m src.main --test-all   # Test all components
```

### Benchmark Latency
```bash
python -m scripts.benchmark_latency --stt
python -m scripts.benchmark_latency --llm
python -m scripts.benchmark_latency --tts
python -m scripts.benchmark_latency --all
```

## Project Structure

```
voice-chatbot/
├── src/
│   ├── audio/              # Audio capture and playback
│   │   ├── capture.py      # Microphone input (16kHz)
│   │   ├── playback.py     # Speaker output (24kHz)
│   │   ├── vad.py          # Voice activity detection
│   │   └── vad_singleton.py # Shared VAD model instance
│   ├── models/             # ML model wrappers
│   │   ├── stt.py          # Parakeet wrapper (batch + streaming)
│   │   ├── llm.py          # Qwen/LLM wrapper (prompt cache)
│   │   ├── tts.py          # Kokoro-on-MLX wrapper
│   │   └── rag.py          # RAG retrieval (sentence-transformers)
│   ├── processing/         # Text and speed processing
│   │   ├── text_preprocessor.py  # Prosody enhancement for TTS
│   │   └── speed_controller.py   # Emotion-aware speech pacing
│   ├── pipeline/           # Orchestration
│   │   └── manager.py      # Main pipeline controller
│   └── main.py             # Entry point
├── config/
│   └── settings.py         # Configuration
├── scripts/                # Evaluation and testing tools
│   ├── benchmark_latency.py     # STT/LLM/TTS latency benchmarks
│   ├── tts_quality_test.py      # TTS A/B quality comparison
│   └── eval_responses.py        # LLM response-quality evaluation
├── data/                   # Personal notes for RAG
├── requirements.txt
└── README.md
```

## Configuration

Edit `config/settings.py` to customize behavior. Key settings:

| Category | Setting | Default | Description |
|----------|---------|---------|-------------|
| Audio | `sample_rate` | 16000 | Capture rate (Hz) |
| VAD | `threshold` | 0.5 | Speech detection sensitivity |
| VAD | `min_silence_duration_ms` | 300 | Silence to end turn |
| STT | `model_name` | `mlx-community/parakeet-tdt-0.6b-v3` | Streaming speech recognition |
| LLM | `model_name` | `mlx-community/Qwen3.5-4B-MLX-4bit` | Default local chat model |
| LLM | `max_tokens` | 256 | Response length safety cap (persona keeps replies short) |
| LLM | `temperature` | 0.7 | Response creativity |
| LLM | `top_p` / `top_k` | `0.8` / `20` | Qwen3 non-thinking sampling defaults |
| TTS | `voice` | `af_heart` | Default voice |
| TTS | `speed` | 1.0 | Speech rate multiplier |
| Barge-in | `enabled` | `true` | Speak over the assistant to interrupt it |
| Barge-in | `playback_vad_threshold` | 0.75 | VAD strictness while audio plays (echo rejection) |
| Logging | `enabled` | `true` | Save conversation transcripts to `logs/` (stays on-device) |

## Available Voices

Kokoro provides 28 English voices with quality ratings:

**American Female (11 voices)**
| Voice | Grade | Description |
|-------|-------|-------------|
| `af_heart` | A | Highest quality, natural (default) |
| `af_bella` | A- | Warm, friendly |
| `af_nicole` | B- | Soft, calm |
| `af_sarah` | C+ | Clear, professional |
| `af_aoede` `af_kore` `af_nova` `af_alloy` `af_sky` `af_jessica` `af_river` | C-D | Additional options |

**American Male (9 voices)**
| Voice | Grade | Description |
|-------|-------|-------------|
| `am_fenrir` | C+ | Strong, clear |
| `am_michael` | C+ | Friendly, casual |
| `am_puck` | C+ | Energetic |
| `am_adam` | F+ | Deep, confident |
| `am_echo` `am_eric` `am_liam` `am_onyx` `am_santa` | C-D | Additional options |

**British Female (4 voices)**
| Voice | Grade | Description |
|-------|-------|-------------|
| `bf_emma` | B- | Elegant, refined |
| `bf_isabella` | C | Warm, articulate |
| `bf_alice` `bf_lily` | C-D | Additional options |

**British Male (4 voices)**
| Voice | Grade | Description |
|-------|-------|-------------|
| `bm_george` | C | Distinguished, clear |
| `bm_fable` | C | Storyteller |
| `bm_lewis` | D+ | Friendly |
| `bm_daniel` | D | Casual |

## Performance

| Metric | Typical Value |
|--------|---------------|
| First audio output | ~1-1.5 seconds after speech end |
| Memory usage | 6-9 GB RAM |
| Disk space (models) | ~5 GB |
| Models load time | ~10 seconds warm (first run downloads) |

## Troubleshooting

### "No module named 'mlx'"
```bash
pip install mlx mlx-lm parakeet-mlx mlx-audio
```

### Audio device issues
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Model download issues
Models are downloaded automatically on first run. Ensure you have ~5GB free disk space.

## License

MIT License - see [LICENSE](LICENSE) for details.
