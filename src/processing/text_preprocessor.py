"""
Text Preprocessor for TTS prosody enhancement.
Adds punctuation cues that Kokoro TTS uses for natural intonation and pacing.
"""

import re
from typing import Optional

from config.settings import TextProcessingSettings


class TextPreprocessor:
    """
    Preprocess text to improve TTS prosody.

    Kokoro TTS is sensitive to punctuation for pauses and intonation:
    - Periods (.) - Full stop with pitch drop
    - Commas (,) - Brief pause, slight pitch continuation
    - Ellipses (...) - Longer pause, uncertain intonation
    - Question marks (?) - Rising intonation
    - Exclamation marks (!) - Emphasis and energy
    - Dashes (-) - Dramatic pauses
    """

    def __init__(self, config: Optional[TextProcessingSettings] = None):
        self.config = config or TextProcessingSettings()

    def process(self, text: str) -> str:
        """
        Apply all enabled preprocessing steps.

        Args:
            text: Raw text to preprocess

        Returns:
            Preprocessed text with prosody cues
        """
        if not self.config.enabled:
            return text

        if self.config.add_breathing_pauses:
            text = self._add_breathing_pauses(text)

        if self.config.add_emphasis_markers:
            text = self._add_emphasis_markers(text)

        return text

    def _add_breathing_pauses(self, text: str) -> str:
        """
        Add ellipses for natural breathing pauses.

        Inserts subtle pauses before conjunctions in longer clauses,
        mimicking natural speech rhythm where speakers pause to breathe.

        Examples:
            "I went to the store and then I came home"
            -> "I went to the store... and then I came home"
        """
        # Add subtle pause before 'and', 'but', 'so' when preceded by 15+ chars
        # This creates natural breathing points in longer sentences
        patterns = [
            # Pause before conjunctions in long clauses
            (r'(\w{12,})\s+(and|but|so|or)\s+', r'\1... \2 '),
            # Pause after introductory clauses
            (r'^(You know|I mean|The thing is|Actually)\s+', r'\1... '),
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _add_emphasis_markers(self, text: str) -> str:
        """
        Add punctuation to emphasize key transition words.

        Adds commas after words like "Well", "Actually", "Honestly"
        which naturally have a pause after them in speech.

        Examples:
            "Well I think that's a good idea"
            -> "Well, I think that's a good idea"
        """
        # Words that naturally have a pause after them
        emphasis_words = [
            'Well', 'Actually', 'Honestly', 'Basically',
            'Look', 'See', 'Right', 'Okay', 'So',
            'Now', 'Anyway', 'Besides', 'Still',
        ]

        for word in emphasis_words:
            # Add comma after word if not already followed by punctuation
            pattern = rf'\b({word})\s+(?![,.\-!?])'
            replacement = r'\1, '
            text = re.sub(pattern, replacement, text)

        return text


# Quick test
if __name__ == "__main__":
    preprocessor = TextPreprocessor()

    test_sentences = [
        "Well I think that's a great idea.",
        "I went to the store and then I came home and made dinner.",
        "Actually I'm not sure about that.",
        "The thing is I really want to help you understand this concept.",
        "Okay so basically this is how it works.",
    ]

    print("Text Preprocessor Demo")
    print("=" * 60)

    for sentence in test_sentences:
        processed = preprocessor.process(sentence)
        print(f"\nOriginal:  {sentence}")
        print(f"Processed: {processed}")
