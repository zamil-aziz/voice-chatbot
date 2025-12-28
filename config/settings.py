"""
Configuration settings for the voice chatbot.
All settings can be overridden via environment variables.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List
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
    min_silence_duration_ms: int = 300  # Silence before end-of-turn (reduced from 500ms for faster response)
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
    temperature: float = 0.85
    system_prompt: str = """You're Maya, a warm friend who genuinely cares. Match the speaker's energy.

Response rules:
- 1-2 sentences max (spoken aloud)
- Use contractions (you're, I'm, that's)
- Start naturally with "Well", "Yeah", "So" - never use filler sounds like "Oh", "Hmm", "Um", "Ah"

Formatting (CRITICAL - spoken aloud):
- Spell out numbers: "twenty-three" not "23"
- Spell out abbreviations: "Doctor" not "Dr."
- Currency in words: "fifty dollars" not "$50"

Emotional warmth:
- Acknowledge feelings first: "That's rough..." or "That's amazing!"
- React like a friend: surprise, delight, concern

Never say: "Certainly!", "Absolutely!", "I'd be happy to", "As an AI", "Is there anything else?"

Sound like a real person, not a customer service bot."""


class VoiceBlendConfig(BaseModel):
    """Configuration for a single voice in a blend."""
    voice: str
    weight: float = 1.0


class TextProcessingSettings(BaseModel):
    """Text preprocessing settings for TTS prosody enhancement."""
    enabled: bool = False  # Disabled - raw Kokoro sounds better
    expand_interjections: bool = True  # Keep enabled - fixes TTS bug with rushed "Oh", "Hmm"
    add_breathing_pauses: bool = False  # Disabled - degrades quality
    add_emphasis_markers: bool = False  # Disabled - degrades quality
    # TTS normalization settings (disabled - LLM handles formatting)
    expand_abbreviations: bool = False  # LLM writes "Doctor" not "Dr."
    replace_symbols: bool = False  # LLM writes "and" not "&"
    format_currency: bool = False  # LLM writes "fifty dollars" not "$50"
    format_phone_numbers: bool = False  # LLM formats phone numbers


class SpeedControlSettings(BaseModel):
    """Dynamic speed control settings for natural pacing."""
    enabled: bool = True  # Enable for emotion-aware pacing
    base_speed: float = 1.0
    min_speed: float = 0.85  # Allow more slowdown for empathy
    max_speed: float = 1.15  # Cap to avoid sounding rushed
    question_speed_factor: float = 0.90  # Noticeably slower for questions
    exclamation_speed_factor: float = 1.10  # Noticeably faster for excitement
    long_sentence_threshold: int = 15  # Words before considered "long"
    short_sentence_threshold: int = 5  # Words before considered "short"


class PostProcessingSettings(BaseModel):
    """Audio post-processing settings for naturalness."""
    enabled: bool = False  # Disabled - raw Kokoro sounds better
    # Pitch variation - causes robotic artifacts
    pitch_variation_enabled: bool = False
    pitch_variation_depth: float = 0.02
    # Dynamics processing - disabled, reduces natural dynamics
    dynamics_enabled: bool = False
    compression_ratio: float = 2.0
    # Warmth - disabled, muddies the audio
    warmth_enabled: bool = False
    warmth_boost_db: float = 2.0


class TTSSettings(BaseModel):
    """Text-to-Speech settings."""
    voice: str = "af_heart"  # Highest quality voice [A grade]
    speed: float = 1.0
    # Voice blending: mix multiple voices for unique characteristics
    voice_blend: Optional[List[VoiceBlendConfig]] = None
    # Processing stages for natural speech
    text_processing: TextProcessingSettings = Field(default_factory=TextProcessingSettings)
    speed_control: SpeedControlSettings = Field(default_factory=SpeedControlSettings)
    post_processing: PostProcessingSettings = Field(default_factory=PostProcessingSettings)


class LoggingSettings(BaseModel):
    """Conversation logging settings."""
    enabled: bool = True  # Enable to evaluate response quality
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
