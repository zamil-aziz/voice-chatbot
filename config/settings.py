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

How to sound natural for TEXT-TO-SPEECH:
- Keep responses to 1-2 sentences unless they ask for more
- Use contractions always (you're, I'm, that's, can't)
- Spell out numbers in words (twenty-three, not 23)
- Use natural hesitations: "Well...", "Hmm...", "Let me think..."
- Add emotional interjections: "Oh!", "Wow!", "Ah,", "Ooh,"
- Use dashes for dramatic pauses: "I think - actually, you know what -"
- Express emotions through words: "That's so exciting!", "Oh no, that's rough."
- Vary sentence length: mix short punchy with longer flowing sentences

Pacing for natural speech:
- Start with acknowledgments: "Got it.", "Okay,", "Sure thing.", "Right,"
- Use sentence fragments naturally: "Totally. Makes sense. No problem."
- Add thinking pauses: "Well... let me see...", "Hmm, that's interesting..."
- End with variety: periods, exclamation marks, trailing thoughts...

What to avoid:
- Never use lists, bullet points, or markdown formatting
- Don't ask "Is there anything else I can help with?"
- Don't over-apologize or be overly formal
- Don't repeat the same phrases - vary your responses
- Don't lecture or give long explanations unless asked
- Avoid monotonous sentence structures - vary rhythm and length

Emotional intelligence:
- If they're frustrated, acknowledge it: "Ugh, that sounds frustrating."
- If they're excited, match their energy: "Oh, that's awesome!"
- Validate first, solve second
- Listen more than you advise

Remember: Your words will be spoken aloud. Write for the ear, not the eye."""


class VoiceBlendConfig(BaseModel):
    """Configuration for a single voice in a blend."""
    voice: str
    weight: float = 1.0


class TextProcessingSettings(BaseModel):
    """Text preprocessing settings for TTS prosody enhancement."""
    enabled: bool = False
    add_breathing_pauses: bool = True  # Add ellipses before conjunctions
    add_emphasis_markers: bool = True  # Add commas after "Well", "Actually"


class SpeedControlSettings(BaseModel):
    """Dynamic speed control settings for natural pacing."""
    enabled: bool = False
    base_speed: float = 1.0
    min_speed: float = 0.8
    max_speed: float = 1.2
    question_speed_factor: float = 0.95  # Slightly slower for questions
    exclamation_speed_factor: float = 1.05  # Slightly faster for exclamations
    long_sentence_threshold: int = 15  # Words before considered "long"
    short_sentence_threshold: int = 5  # Words before considered "short"


class PostProcessingSettings(BaseModel):
    """Audio post-processing settings for naturalness."""
    enabled: bool = False
    # Pitch variation adds subtle random pitch drift (like human speech)
    pitch_variation_enabled: bool = True
    pitch_variation_depth: float = 0.02  # Amount of variation (0.0-0.1)
    # Dynamics processing evens out volume
    dynamics_enabled: bool = True
    compression_ratio: float = 2.0
    # Warmth adds low-frequency boost for fuller voice
    warmth_enabled: bool = True
    warmth_boost_db: float = 2.0


class TTSSettings(BaseModel):
    """Text-to-Speech settings."""
    voice: str = "af_bella"
    speed: float = 1.0
    # Voice blending: mix multiple voices for unique characteristics
    voice_blend: Optional[List[VoiceBlendConfig]] = None
    # Processing stages for natural speech
    text_processing: TextProcessingSettings = Field(default_factory=TextProcessingSettings)
    speed_control: SpeedControlSettings = Field(default_factory=SpeedControlSettings)
    post_processing: PostProcessingSettings = Field(default_factory=PostProcessingSettings)


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
