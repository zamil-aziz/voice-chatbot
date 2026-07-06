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
    # Chunks averaged for onset detection; small keeps the smoothed signal
    # responsive (each chunk is 32ms). End-of-speech uses the raw probability
    # so a bigger window would not delay turn end, only speech start.
    smoothing_window: int = 4
    window_size_samples: int = 512  # Silero VAD window size
    device: Literal["auto", "cpu", "mps"] = "cpu"  # CPU is faster for tiny 512-sample chunks
    use_onnx: bool = True  # onnxruntime beats the JIT torch model on 512-sample chunks


class STTSettings(BaseModel):
    """Speech-to-Text (Parakeet TDT) settings."""
    model_name: str = "mlx-community/parakeet-tdt-0.6b-v3"
    # Streaming transcription: audio is fed to the encoder in batches of this
    # size while the user is still speaking, so the transcript is (nearly)
    # ready the moment the turn ends
    stream_batch_seconds: float = 0.5
    stream_context: tuple[int, int] = (256, 256)  # Local attention window (left, right)
    stream_depth: int = 1


class LLMSettings(BaseModel):
    """Language Model settings."""
    model_name: str = "mlx-community/Qwen3.5-4B-MLX-4bit"
    # Safety net, not a target: the persona keeps replies short, and a cap
    # this size means detailed answers no longer truncate mid-thought.
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    history_turns: int = 6
    enable_thinking: bool = False
    system_prompt: str = """You're Maya, a warm, concise voice assistant.

Response rules:
- Keep answers to one or two natural spoken sentences unless the user asks for detail.
- Use contractions and everyday language.
- Match the user's mood without forcing the same opener every time.
- For emotions, acknowledge the feeling first, then offer one helpful thought or question.
- For facts, answer directly and briefly.
- If you truly don't know something, say so in one short sentence.

Never say: "Certainly", "Absolutely", "I'd be happy to", "As an AI", "Is there anything else?"
Do not use markdown, bullet points, code blocks, emojis, or stage directions."""


class RAGSettings(BaseModel):
    """Personal-notes retrieval settings."""
    enabled: bool = True
    notes_file: str = "./data/notes.txt"
    embedder_model: str = "all-MiniLM-L6-v2"
    n_results: int = 2
    # MiniLM cosine scores below ~0.3 are mostly noise; anything retrieved
    # under this threshold would inject irrelevant notes into the reply.
    min_similarity: float = 0.30


class VoiceBlendConfig(BaseModel):
    """Configuration for a single voice in a blend."""
    voice: str
    weight: float = 1.0


class TextProcessingSettings(BaseModel):
    """Text preprocessing settings for TTS pronunciation."""
    enabled: bool = True  # Enable deterministic TTS-safe text normalization
    remove_fillers: bool = False  # Off: deleting "Oh"/"Hmm" flattens tone and breaks empathy pacing cues
    expand_abbreviations: bool = False  # Avoid unsafe rewrites like "in." -> "inches"
    replace_symbols: bool = True
    format_phone_numbers: bool = True  # Only touches explicitly formatted numbers like (502) 345-6789


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


class TTSSettings(BaseModel):
    """Text-to-Speech settings."""
    model_name: str = "mlx-community/Kokoro-82M-bf16"
    voice: str = "af_heart"  # Highest quality voice [A grade]
    speed: float = 1.0
    # Voice blending: mix multiple voices for unique characteristics
    voice_blend: Optional[List[VoiceBlendConfig]] = None
    # Processing stages for natural speech
    text_processing: TextProcessingSettings = Field(default_factory=TextProcessingSettings)
    speed_control: SpeedControlSettings = Field(default_factory=SpeedControlSettings)


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
    rag: RAGSettings = Field(default_factory=RAGSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Model loading settings
    model_load_timeout: int = 300  # 5 minutes timeout for model downloads

    # Debug settings
    debug: bool = False
    log_audio: bool = False  # Save audio to disk for debugging


# Global settings instance
settings = Settings()
