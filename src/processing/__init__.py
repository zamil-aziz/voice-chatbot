"""
Processing module for TTS enhancement.
Provides text preprocessing and speed control for natural speech.
"""

from .text_preprocessor import TextPreprocessor
from .speed_controller import DynamicSpeedController

__all__ = ["TextPreprocessor", "DynamicSpeedController"]
