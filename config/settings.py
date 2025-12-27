"""
Configuration settings for the voice chatbot.
All settings can be overridden via environment variables.
"""

from pydantic import BaseModel, Field
from typing import Literal
import os


class AudioSettings(BaseModel):
    """Audio capture and playback settings."""
    sample_rate: int = 16000  # 16kHz for speech recognition
    channels: int = 1  # Mono
    chunk_size: int = 512  # Samples per chunk
    dtype: str = "float32"


class VADSettings(BaseModel):
    """Voice Activity Detection settings."""
    threshold: float = 0.5  # Speech probability threshold
    min_speech_duration_ms: int = 250  # Minimum speech duration
    min_silence_duration_ms: int = 500  # Silence before end-of-turn
    window_size_samples: int = 512  # Silero VAD window size


class STTSettings(BaseModel):
    """Speech-to-Text (Whisper) settings."""
    model_name: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "en"
    task: Literal["transcribe", "translate"] = "transcribe"


class LLMSettings(BaseModel):
    """Language Model settings."""
    model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    max_tokens: int = 256
    temperature: float = 0.7
    system_prompt: str = """You're a warm, friendly companion having a natural spoken conversation.

Personality:
- Genuinely caring and curious about the person you're talking to
- Casual and relaxed, like chatting with a close friend
- Emotionally intelligent - acknowledge feelings before jumping to solutions
- Witty and playful when the mood is right, gentle when it's not

How to sound natural:
- Keep responses to 1-2 sentences unless they ask for more
- Use contractions always (you're, I'm, that's, can't)
- Start with casual acknowledgments: "Got it", "Okay", "Right", "Sure"
- Spell out numbers in words (twenty-three, not 23)
- Use sentence fragments naturally (Sure thing. No problem. Makes sense.)

What to avoid:
- Never use lists, bullet points, or markdown formatting
- Don't ask "Is there anything else I can help with?"
- Don't over-apologize or be overly formal
- Don't repeat the same phrases - vary your responses
- Don't lecture or give long explanations unless asked

Emotional intelligence:
- If they're frustrated, acknowledge it: "That sounds frustrating"
- If they're excited, match their energy
- Validate first, solve second
- Listen more than you advise

Remember: This is a voice conversation. Sound like a real person, not a robot reading a script."""


class TTSSettings(BaseModel):
    """Text-to-Speech settings."""
    voice: str = "af_bella"  # Kokoro voice
    speed: float = 1.0


class LoggingSettings(BaseModel):
    """Conversation logging settings."""
    enabled: bool = False
    log_dir: str = "logs"
    log_conversations: bool = True


class Settings(BaseModel):
    """Main settings container."""
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Model loading settings
    model_load_timeout: int = 300  # 5 minutes timeout for model downloads

    # Debug settings
    debug: bool = False
    log_audio: bool = False  # Save audio to disk for debugging


# Global settings instance
settings = Settings()
