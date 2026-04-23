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
    device: Literal["auto", "cpu", "mps"] = "cpu"  # CPU is faster for tiny 512-sample chunks


class STTSettings(BaseModel):
    """Speech-to-Text (Whisper) settings."""
    model_name: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "en"
    task: Literal["transcribe", "translate"] = "transcribe"


class LLMSettings(BaseModel):
    """Language Model settings."""
    model_name: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    max_tokens: int = 96
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    history_turns: int = 4
    enable_thinking: bool = False
    system_prompt: str = """You're Maya, a warm, concise voice assistant.

Response rules:
- Keep answers to one or two natural spoken sentences unless the user asks for detail.
- Use contractions and everyday language.
- Match the user's mood without forcing the same opener every time.
- For emotions, acknowledge the feeling first, then offer one helpful thought or question.
- For facts, answer directly and briefly.

Never say: "Certainly", "Absolutely", "I'd be happy to", "As an AI", "Is there anything else?"
Do not use markdown, bullet points, code blocks, emojis, or stage directions."""


class VoiceBlendConfig(BaseModel):
    """Configuration for a single voice in a blend."""
    voice: str
    weight: float = 1.0


class TextProcessingSettings(BaseModel):
    """Text preprocessing settings for TTS prosody enhancement."""
    enabled: bool = True  # Enable deterministic TTS-safe text normalization
    expand_interjections: bool = True  # Keep enabled - fixes TTS bug with rushed "Oh", "Hmm"
    add_breathing_pauses: bool = False  # Disabled - degrades quality
    add_emphasis_markers: bool = False  # Disabled - degrades quality
    # TTS normalization settings (disabled - LLM handles formatting)
    expand_abbreviations: bool = False  # Avoid unsafe rewrites like "in." -> "inches"
    replace_symbols: bool = True
    format_currency: bool = False  # Disabled until comma-formatted amounts are safely normalized
    format_phone_numbers: bool = True


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
    device: Literal["auto", "cpu", "mps"] = "cpu"  # Avoid MLX/PyTorch GPU contention during streaming
    isolated_process: bool = True  # Avoid native sentencepiece/Kokoro shutdown crashes in main process
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
