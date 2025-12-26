# Voice Chatbot

A fully local, privacy-respecting AI voice assistant running on Apple Silicon (M1/M2/M3/M4).

## Features

- **100% Local**: All processing happens on your device
- **100% Free**: No API costs, no subscriptions
- **Privacy First**: Your conversations never leave your machine
- **Natural Voice**: 28 high-quality voices with Kokoro TTS
- **Fast**: Optimized for Apple Silicon with MLX
- **Streaming Pipeline**: TTS starts before LLM finishes for lower latency
- **Barge-in Support**: Interrupt the assistant while it's speaking
- **GPU Accelerated**: MPS (Metal) acceleration for TTS

## Architecture

```
┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────────┐
│ Mic │ -> │ VAD │ -> │ STT │ -> │ LLM │ -> │ TTS │ -> │ Speaker │
└─────┘    └─────┘    └─────┘    └──┬──┘    └──┬──┘    └─────────┘
                                    │          │
                              [streaming: TTS starts per-sentence]
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Speech-to-Text | Whisper Large-v3-turbo (via MLX) |
| Language Model | Qwen 2.5 7B 4-bit (via MLX) |
| Text-to-Speech | Kokoro (28 voices) |
| Voice Detection | Silero VAD |

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
python -m src.main
```

### Test Individual Components
```bash
python -m src.main --test-stt   # Test speech-to-text
python -m src.main --test-llm   # Test language model
python -m src.main --test-tts   # Test text-to-speech
python -m src.main --test-vad   # Test voice activity detection
python -m src.main --test-all   # Test all components
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
│   │   ├── stt.py          # Whisper wrapper
│   │   ├── llm.py          # Qwen/LLM wrapper
│   │   └── tts.py          # Kokoro wrapper
│   ├── pipeline/           # Orchestration
│   │   └── manager.py      # Main pipeline controller
│   └── main.py             # Entry point
├── config/
│   └── settings.py         # Configuration
├── requirements.txt
└── README.md
```

## Configuration

Edit `config/settings.py` to customize behavior. Key settings:

| Category | Setting | Default | Description |
|----------|---------|---------|-------------|
| Audio | `sample_rate` | 16000 | Capture rate (Hz) |
| VAD | `threshold` | 0.5 | Speech detection sensitivity |
| VAD | `min_silence_duration_ms` | 500 | Silence to end turn |
| LLM | `max_tokens` | 256 | Max response length |
| LLM | `temperature` | 0.7 | Response creativity |
| TTS | `voice` | `af_bella` | Default voice |
| TTS | `speed` | 1.0 | Speech rate multiplier |

## Available Voices

Kokoro provides 28 English voices with quality ratings:

**American Female (11 voices)**
| Voice | Grade | Description |
|-------|-------|-------------|
| `af_heart` | A | Highest quality, natural |
| `af_bella` | A- | Warm, friendly (default) |
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
| `bf_isabella` | C | Professional |
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
| First audio output | ~2-3 seconds after speaking |
| Memory usage | 8-12 GB RAM |
| Disk space (models) | ~15 GB |
| Models load time | ~30-60 seconds (first run downloads) |

## Troubleshooting

### "No module named 'mlx'"
```bash
pip install mlx mlx-lm mlx-whisper
```

### Audio device issues
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Model download issues
Models are downloaded automatically on first run. Ensure you have ~15GB free disk space.

## License

MIT License - see [LICENSE](LICENSE) for details.
