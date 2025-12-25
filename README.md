# Voice Chatbot

A fully local, privacy-respecting AI voice assistant running on Apple Silicon (M1/M2/M3/M4).

## Features

- **100% Local**: All processing happens on your device
- **100% Free**: No API costs, no subscriptions
- **Privacy First**: Your conversations never leave your machine
- **Natural Voice**: High-quality text-to-speech with Kokoro
- **Fast**: Optimized for Apple Silicon with MLX

## Tech Stack

| Component | Technology |
|-----------|------------|
| Speech-to-Text | Whisper Large-v3 (via MLX) |
| Language Model | Llama 3.1 8B 4-bit (via MLX) |
| Text-to-Speech | Kokoro |
| Voice Detection | Silero VAD |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB RAM (minimum)
- Python 3.10+

## Installation

```bash
# Clone or navigate to the project
cd ~/Personal/voice-chatbot

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
# Test speech-to-text
python -m src.main --test-stt

# Test language model
python -m src.main --test-llm

# Test text-to-speech
python -m src.main --test-tts

# Test voice activity detection
python -m src.main --test-vad

# Test all components
python -m src.main --test-all
```

## Project Structure

```
voice-chatbot/
├── src/
│   ├── audio/           # Audio capture and playback
│   │   ├── capture.py   # Microphone input
│   │   ├── playback.py  # Speaker output
│   │   └── vad.py       # Voice activity detection
│   ├── models/          # ML model wrappers
│   │   ├── stt.py       # Whisper wrapper
│   │   ├── llm.py       # Llama wrapper
│   │   └── tts.py       # Kokoro wrapper
│   ├── pipeline/        # Orchestration
│   │   └── manager.py   # Main pipeline controller
│   └── main.py          # Entry point
├── config/
│   └── settings.py      # Configuration
├── requirements.txt
└── README.md
```

## Configuration

Edit `config/settings.py` to customize:

- Audio settings (sample rate, chunk size)
- VAD sensitivity (speech detection threshold)
- LLM parameters (temperature, max tokens)
- TTS voice selection
- System prompt for the assistant

## Available Voices

Kokoro provides multiple high-quality voices:

**American English:**
- `af_bella` - Warm, friendly female
- `af_sarah` - Clear, professional female
- `am_adam` - Deep, confident male
- `am_michael` - Friendly, casual male

**British English:**
- `bf_emma` - Elegant, refined female
- `bm_george` - Distinguished, clear male

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

MIT License
