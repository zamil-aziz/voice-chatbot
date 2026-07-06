"""
Text Preprocessor for TTS prosody enhancement.
Adds punctuation cues that Kokoro TTS uses for natural intonation and pacing.
Also normalizes text for better TTS pronunciation (abbreviations, numbers, symbols).
"""

import re
from typing import Optional

from config.settings import TextProcessingSettings

# Emoji pattern for stripping emojis before TTS
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE
)


# Filler-word removal (opt-in). Only sentence-initial interjections are
# removed: dropping fillers mid-sentence can change meaning, and words like
# "like" carry meaning too often to strip safely at all.
FILLER_WORD_RE = re.compile(
    r"(^|[.!?]\s+)(?:Oh+|Hmm+|Aw+|Ah+|Uh+|Um+|Huh|Ooh+)\b[,.]?\s*",
    re.IGNORECASE,
)

# Abbreviation expansions for natural TTS pronunciation
ABBREVIATION_EXPANSIONS = [
    (r'\bDr\.', 'Doctor'),
    (r'\bMr\.', 'Mister'),
    (r'\bMrs\.', 'Missus'),
    (r'\bMs\.', 'Miz'),
    (r'\bSt\.', 'Street'),
    (r'\bAve\.', 'Avenue'),
    (r'\bBlvd\.', 'Boulevard'),
    (r'\bRd\.', 'Road'),
    (r'\bApt\.', 'Apartment'),
    (r'\bNo\.', 'Number'),
    (r'\be\.g\.', 'for example'),
    (r'\bi\.e\.', 'that is'),
    (r'\betc\.', 'etcetera'),
    (r'\bvs\.', 'versus'),
    (r'\bMin\.', 'Minimum'),
    (r'\bMax\.', 'Maximum'),
    (r'\bft\.', 'feet'),
    (r'\bin\.', 'inches'),
    (r'\blb\.', 'pounds'),
    (r'\boz\.', 'ounces'),
]

# Symbol replacements for natural speech
SYMBOL_REPLACEMENTS = [
    (r'&', ' and '),
    (r'(\d+)%', r'\1 percent'),  # "50%" -> "50 percent"
    (r'%', ' percent'),  # standalone %
    (r'\+', ' plus '),
    (r'@', ' at '),
    (r'#(\d+)', r'number \1'),  # "#5" -> "number 5"
]

# Phone numbers must be explicitly formatted (parentheses or separators);
# a bare 10-digit number is more often an ID, a year range, or a quantity.
PHONE_NUMBER_RE = re.compile(
    r"\(\d{3}\)\s?\d{3}[-.]\d{4}\b|\b\d{3}[-.]\d{3}[-.]\d{4}\b"
)


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
        # Strip emojis first - TTS will read them aloud otherwise
        text = EMOJI_PATTERN.sub('', text)

        if not self.config.enabled:
            return text

        if self.config.remove_fillers:
            text = self._remove_filler_words(text)

        if self.config.expand_abbreviations:
            text = self._expand_abbreviations(text)

        if self.config.replace_symbols:
            text = self._replace_symbols(text)

        if self.config.format_phone_numbers:
            text = self._format_phone_numbers(text)

        return text

    def _remove_filler_words(self, text: str) -> str:
        """
        Remove sentence-initial filler interjections.

        Examples:
            "Oh, okay." -> "okay."
            "Hmm, let me think." -> "let me think."
        """
        text = FILLER_WORD_RE.sub(r'\1', text)
        # Clean up any double spaces or leading spaces from removals
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'^\s+', '', text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand common abbreviations for natural TTS pronunciation.

        Examples:
            "Dr. Smith" -> "Doctor Smith"
            "123 Main St." -> "123 Main Street"
        """
        for pattern, replacement in ABBREVIATION_EXPANSIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _replace_symbols(self, text: str) -> str:
        """
        Replace symbols with their spoken equivalents.

        Examples:
            "Tom & Jerry" -> "Tom and Jerry"
            "50%" -> "50 percent"
        """
        for pattern, replacement in SYMBOL_REPLACEMENTS:
            text = re.sub(pattern, replacement, text)
        # Clean up any double spaces
        text = re.sub(r'  +', ' ', text)
        return text

    def _format_phone_numbers(self, text: str) -> str:
        """
        Format explicitly punctuated phone numbers for clear TTS pronunciation.

        Examples:
            "(502) 345-6789" -> "5 0 2, 3 4 5, 6 7 8 9"
            "502-345-6789" -> "5 0 2, 3 4 5, 6 7 8 9"
            "5023456789" is left alone (could be an ID or quantity)
        """
        def format_phone(match):
            digits = re.sub(r'\D', '', match.group(0))
            return f"{' '.join(digits[0:3])}, {' '.join(digits[3:6])}, {' '.join(digits[6:10])}"

        return PHONE_NUMBER_RE.sub(format_phone, text)


# Quick test
if __name__ == "__main__":
    preprocessor = TextPreprocessor()

    test_sentences = [
        # Symbol tests
        "That's $50 & change.",
        "You got 85% on the test!",
        "Email me at test@example.com",
        "Item #5 is on sale.",
        # Phone number tests (formatted numbers convert, bare digits don't)
        "My number is (502) 345-6789.",
        "Reach me at 502-345-6789.",
        "Order number 5023456789 shipped.",
        # Fillers survive by default (remove_fillers is off)
        "Hmm, let me think about that.",
        "Oh no, that sounds hard.",
    ]

    print("Text Preprocessor Demo")
    print("=" * 60)

    for sentence in test_sentences:
        processed = preprocessor.process(sentence)
        if sentence != processed:
            print(f"\nOriginal:  {sentence}")
            print(f"Processed: {processed}")
        else:
            print(f"\n[unchanged] {sentence}")
