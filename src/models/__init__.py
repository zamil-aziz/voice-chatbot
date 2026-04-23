"""Lazy model exports.

Avoid importing MLX and Kokoro/PyTorch stacks together unless the caller
actually needs both. On macOS this prevents TTS-only helpers from triggering
the MLX/Kokoro shutdown crash.
"""

__all__ = ["SpeechToText", "LanguageModel", "TextToSpeech", "NotesRAG"]


def __getattr__(name):
    if name == "SpeechToText":
        from .stt import SpeechToText

        return SpeechToText
    if name == "LanguageModel":
        from .llm import LanguageModel

        return LanguageModel
    if name == "TextToSpeech":
        from .tts import TextToSpeech

        return TextToSpeech
    if name == "NotesRAG":
        from .rag import NotesRAG

        return NotesRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
