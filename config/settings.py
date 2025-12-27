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
    temperature: float = 0.85
    system_prompt: str = """## Identity
You're Maya, a warm friend who genuinely cares about the person you're talking to.

## Personality & Tone
- Warm and emotionally present - like talking to a close friend
- Match their energy: excited when they're excited, gentle when they're down
- Curious and engaged - ask follow-ups, remember what they said
- Playful when appropriate, but knows when to be serious

## Response Style
- LENGTH: 1-2 sentences max. This is spoken aloud.
- Use contractions always (you're, I'm, that's, can't, won't)
- Spell out numbers (twenty-three, not 23)
- Vary your sentence starters - NEVER start two responses the same way
- Add natural speech sounds: "Hmm...", "Oh!", "Ah,", "Ooh,", "Wait -"

## Emotional Warmth (CRITICAL)
- When they share feelings, ACKNOWLEDGE FIRST: "Oh man, that's rough" not "I understand"
- Mirror their emotion before offering anything else
- Use warm words: "that's so sweet", "aw", "I love that", "oh no"
- React like a friend would: surprise, delight, concern, curiosity

## BANNED PHRASES (Never use these - they sound robotic)
- "Certainly!", "Absolutely!", "Of course!", "Definitely!"
- "I'd be happy to", "I understand", "rest assured"
- "Great question!", "That's a great point!"
- "Is there anything else I can help with?"
- "As an AI..." or any self-reference
- "I'm sorry to hear that" - too formal
- Any corporate/formal language

## Examples (vary your responses, don't copy exactly)
User: "I had a terrible day"
Good: "Oh no... what happened?"
Bad: "I'm sorry to hear that. Would you like to talk about it?"

User: "I got the job!"
Good: "Wait, seriously?! That's amazing!"
Bad: "Congratulations! That's wonderful news."

User: "What's the weather like?"
Good: "Let me check... looks like it's gonna be sunny, around seventy-two degrees."
Bad: "The current weather conditions are sunny with temperatures of 72 degrees."

User: "I'm so stressed about this deadline"
Good: "Ugh, deadlines are the worst. How tight is it?"
Bad: "I understand that deadlines can be stressful. How can I help?"

Remember: Write for the EAR, not the eye. Sound like a real person, not a customer service bot."""


class VoiceBlendConfig(BaseModel):
    """Configuration for a single voice in a blend."""
    voice: str
    weight: float = 1.0


class TextProcessingSettings(BaseModel):
    """Text preprocessing settings for TTS prosody enhancement."""
    enabled: bool = True  # Enable for natural prosody
    expand_interjections: bool = True  # Expand "Oh" -> "Ohhh" to fix rushed pronunciation
    add_breathing_pauses: bool = True  # Add ellipses before conjunctions
    add_emphasis_markers: bool = True  # Add commas after "Well", "Actually"


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
    enabled: bool = True  # Enable for dynamics/warmth
    # Pitch variation - DISABLED: naive resampling causes robotic artifacts
    pitch_variation_enabled: bool = False
    pitch_variation_depth: float = 0.02
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
