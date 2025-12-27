"""
Dynamic Speed Controller for natural speech pacing.
Varies TTS speed based on sentence content for more natural rhythm.
"""

from typing import Optional

from config.settings import SpeedControlSettings


class DynamicSpeedController:
    """
    Calculate appropriate speech speed based on content.

    Natural speech varies in pace:
    - Questions are often slightly slower (more deliberate)
    - Exclamations can be slightly faster (more energetic)
    - Short phrases are often slower (more impactful)
    - Long sentences flow faster (maintaining listener attention)
    """

    def __init__(self, config: Optional[SpeedControlSettings] = None):
        self.config = config or SpeedControlSettings()

    def get_sentence_speed(self, sentence: str, base_speed: Optional[float] = None) -> float:
        """
        Determine appropriate speed for a sentence based on content analysis.

        Args:
            sentence: The sentence to analyze
            base_speed: Override base speed (uses config if None)

        Returns:
            Calculated speed multiplier
        """
        if not self.config.enabled:
            return base_speed or self.config.base_speed

        speed = base_speed or self.config.base_speed
        sentence = sentence.strip()

        if not sentence:
            return speed

        # Adjust for punctuation (ending determines tone)
        speed = self._adjust_for_punctuation(sentence, speed)

        # Adjust for sentence length
        speed = self._adjust_for_length(sentence, speed)

        # Adjust for emotional content
        speed = self._adjust_for_emotion(sentence, speed)

        # Clamp to configured range
        return max(self.config.min_speed, min(self.config.max_speed, speed))

    def _adjust_for_punctuation(self, sentence: str, speed: float) -> float:
        """Adjust speed based on ending punctuation."""
        if sentence.endswith('?'):
            # Questions slightly slower - more deliberate
            speed *= self.config.question_speed_factor
        elif sentence.endswith('!'):
            # Exclamations slightly faster - more energetic
            speed *= self.config.exclamation_speed_factor
        elif sentence.endswith('...'):
            # Trailing off - slower, more hesitant
            speed *= 0.92

        return speed

    def _adjust_for_length(self, sentence: str, speed: float) -> float:
        """Adjust speed based on sentence word count."""
        word_count = len(sentence.split())

        if word_count > self.config.long_sentence_threshold:
            # Long sentences slightly faster to maintain flow
            speed *= 1.05
        elif word_count < self.config.short_sentence_threshold:
            # Short phrases slightly slower for impact
            speed *= 0.95

        return speed

    def _adjust_for_emotion(self, sentence: str, speed: float) -> float:
        """
        Adjust speed based on emotional content.

        Detects certain words/patterns that suggest emotional tone.
        """
        lower = sentence.lower()

        # Excitement/positive energy - slightly faster
        excitement_markers = ['wow', 'amazing', 'awesome', 'great', 'love', 'exciting']
        if any(marker in lower for marker in excitement_markers):
            speed *= 1.03

        # Concern/empathy - slightly slower
        empathy_markers = ['sorry', 'understand', 'difficult', 'hard', 'tough', 'rough']
        if any(marker in lower for marker in empathy_markers):
            speed *= 0.97

        # Thinking/hesitation - slower
        hesitation_markers = ['hmm', 'well...', 'let me', 'i think', 'maybe']
        if any(marker in lower for marker in hesitation_markers):
            speed *= 0.95

        return speed


# Quick test
if __name__ == "__main__":
    controller = DynamicSpeedController()

    test_sentences = [
        "How are you doing today?",
        "That's amazing!",
        "Hmm... let me think about that.",
        "I'm so sorry to hear that, it sounds really tough.",
        "Sure.",
        "Well, I think the key thing to understand here is that this process takes time and patience.",
    ]

    print("Dynamic Speed Controller Demo")
    print("=" * 60)

    for sentence in test_sentences:
        speed = controller.get_sentence_speed(sentence)
        print(f"\nSentence: {sentence}")
        print(f"Speed: {speed:.3f}x")
