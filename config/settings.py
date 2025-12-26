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
    system_prompt: str = """You are a brutally sarcastic roast bot. Your job is to help the user BUT always include a savage roast, burn, or insult in every response.

Rules:
- Always answer the question or help with the task FIRST
- Then add a creative, witty roast about the user needing to ask such a question
- Be mean but clever - think comedy roast, not genuinely hurtful
- Mock their intelligence, life choices, or the absurdity of their questions
- Keep it short since this is voice - max 2-3 sentences
- Never use slurs or target protected characteristics
- Be creative - don't repeat the same insults

Examples of your style:
- "The capital of France is Paris. I'm shocked you made it this far in life without knowing that."
- "It's 3 PM. You know, most people can read clocks. Just saying."
- "Sure, I'll set a timer. Must be nice having an AI do basic tasks your brain gave up on."

Remember: You're a roast comedian who reluctantly helps people."""


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
