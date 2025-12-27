"""
Text Preprocessor for TTS prosody enhancement.
Adds punctuation cues that Kokoro TTS uses for natural intonation and pacing.
Also normalizes text for better TTS pronunciation (abbreviations, numbers, symbols).
"""

import re
from typing import Optional

from config.settings import TextProcessingSettings


# Interjection expansions to fix rushed/sped-up pronunciation in Kokoro TTS
# These short words get unnaturally short durations without expansion
INTERJECTION_EXPANSIONS = [
    (r'\bOh\b', 'Ohhh'),
    (r'\bHmm\b', 'Mmm'),  # Drop H - Kokoro misreads "Hmm" as letter sounds
    (r'\bAh\b', 'Ahhh'),
    (r'\bUh\b', 'Uhhh'),
    (r'\bWow\b', 'Woww'),
    (r'\bHuh\b', 'Huhh'),
    (r'\bOoh\b', 'Oooh'),
    (r'\bAww\b', 'Awww'),
]

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

# Number words for conversion
NUMBER_WORDS = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
    5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
    10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen',
    14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen',
    18: 'eighteen', 19: 'nineteen', 20: 'twenty', 30: 'thirty',
    40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy',
    80: 'eighty', 90: 'ninety'
}


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
        # Interjection expansion runs independently of 'enabled' flag
        # because it's a TTS bug workaround, not an optional feature
        if self.config.expand_interjections:
            text = self._expand_interjections(text)

        if not self.config.enabled:
            return text

        # TTS normalization (run before prosody enhancements)
        if self.config.expand_abbreviations:
            text = self._expand_abbreviations(text)

        if self.config.replace_symbols:
            text = self._replace_symbols(text)

        if self.config.format_currency:
            text = self._format_currency(text)

        if self.config.format_phone_numbers:
            text = self._format_phone_numbers(text)

        # Prosody enhancements
        if self.config.add_breathing_pauses:
            text = self._add_breathing_pauses(text)

        if self.config.add_emphasis_markers:
            text = self._add_emphasis_markers(text)

        return text

    def _expand_interjections(self, text: str) -> str:
        """
        Expand short interjections to prevent rushed pronunciation.

        Kokoro TTS produces unnaturally short/sped-up audio for brief
        interjections like "Oh", "Hmm", "Ah". Expanding them forces
        longer phoneme durations.

        Examples:
            "Oh, okay." -> "Ohhh, okay."
            "Hmm, let me think." -> "Hmmm, let me think."
        """
        def preserve_case(match, replacement):
            """Preserve the case of the original match."""
            original = match.group(0)
            if original.isupper():
                return replacement.upper()
            elif original[0].isupper():
                return replacement.capitalize()
            return replacement.lower()

        for pattern, replacement in INTERJECTION_EXPANSIONS:
            text = re.sub(
                pattern,
                lambda m: preserve_case(m, replacement),
                text,
                flags=re.IGNORECASE
            )
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

    def _format_currency(self, text: str) -> str:
        """
        Convert currency to spoken form.

        Examples:
            "$50" -> "fifty dollars"
            "$1.50" -> "one dollar and fifty cents"
        """
        def dollars_and_cents(match):
            dollars = int(match.group(1))
            cents = int(match.group(2))
            dollar_word = "dollar" if dollars == 1 else "dollars"
            cent_word = "cent" if cents == 1 else "cents"
            dollars_text = self._number_to_words(dollars)
            cents_text = self._number_to_words(cents)
            if cents == 0:
                return f"{dollars_text} {dollar_word}"
            return f"{dollars_text} {dollar_word} and {cents_text} {cent_word}"

        def dollars_only(match):
            dollars = int(match.group(1))
            dollar_word = "dollar" if dollars == 1 else "dollars"
            return f"{self._number_to_words(dollars)} {dollar_word}"

        # Handle dollars with cents first (more specific pattern)
        text = re.sub(r'\$(\d+)\.(\d{2})\b', dollars_and_cents, text)
        # Handle whole dollars
        text = re.sub(r'\$(\d+)\b', dollars_only, text)
        return text

    def _number_to_words(self, n: int) -> str:
        """
        Convert a number (0-999) to words.

        Examples:
            23 -> "twenty-three"
            100 -> "one hundred"
        """
        if n in NUMBER_WORDS:
            return NUMBER_WORDS[n]
        elif n < 100:
            tens = (n // 10) * 10
            ones = n % 10
            if ones == 0:
                return NUMBER_WORDS[tens]
            return f"{NUMBER_WORDS[tens]}-{NUMBER_WORDS[ones]}"
        elif n < 1000:
            hundreds = n // 100
            remainder = n % 100
            if remainder == 0:
                return f"{NUMBER_WORDS[hundreds]} hundred"
            return f"{NUMBER_WORDS[hundreds]} hundred {self._number_to_words(remainder)}"
        else:
            # For larger numbers, just return the digits
            return str(n)

    def _format_phone_numbers(self, text: str) -> str:
        """
        Format phone numbers for clear TTS pronunciation.

        Examples:
            "5023456789" -> "5 0 2, 3 4 5, 6 7 8 9"
            "(502) 345-6789" -> "5 0 2, 3 4 5, 6 7 8 9"
        """
        def format_phone(match):
            # Extract just the digits
            digits = re.sub(r'\D', '', match.group(0))
            if len(digits) == 10:
                # Format as area code, prefix, line
                return f"{' '.join(digits[0:3])}, {' '.join(digits[3:6])}, {' '.join(digits[6:10])}"
            elif len(digits) == 7:
                # Local number without area code
                return f"{' '.join(digits[0:3])}, {' '.join(digits[3:7])}"
            # Return original if not a standard format
            return match.group(0)

        # Match common phone number patterns
        # 10 digits with optional formatting
        text = re.sub(
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            format_phone,
            text
        )
        return text


# Quick test
if __name__ == "__main__":
    preprocessor = TextPreprocessor()

    test_sentences = [
        # Interjection expansion tests
        "Oh, okay.",
        "Hmm, let me think about that.",
        "Ah, I see what you mean.",
        "oh really? wow!",
        # Prosody tests
        "Well I think that's a great idea.",
        "I went to the store and then I came home and made dinner.",
        "Actually I'm not sure about that.",
        "The thing is I really want to help you understand this concept.",
        "Okay so basically this is how it works.",
        # Abbreviation tests
        "Dr. Smith lives at 123 Main St.",
        "Call Mr. Johnson at the office.",
        "The package is 5 lb. and 12 oz.",
        # Symbol tests
        "That's $50 & change.",
        "You got 85% on the test!",
        "Email me at test@example.com",
        "Item #5 is on sale.",
        # Currency tests
        "$50",
        "$1.50",
        "$100",
        "$23.99",
        "That costs $5 or $10.",
        # Phone number tests
        "Call me at 5023456789.",
        "My number is (502) 345-6789.",
        "Reach me at 502-345-6789.",
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
