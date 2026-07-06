import unittest

from config.settings import TextProcessingSettings
from src.models.llm import LanguageModel
from src.models.tts import _create_blended_voice_tensor
from src.pipeline.manager import SentenceSegmenter
from src.processing.text_preprocessor import TextPreprocessor


class SentenceSegmenterTests(unittest.TestCase):
    def test_waits_for_more_than_tiny_first_clause(self):
        segmenter = SentenceSegmenter()

        self.assertEqual(segmenter.add("Well,"), [])
        self.assertEqual(segmenter.add(" I can help with that,"), ["Well, I can help with that,"])

    def test_emits_sentence_endings(self):
        segmenter = SentenceSegmenter()

        self.assertEqual(segmenter.add("That sounds rough."), ["That sounds rough."])
        self.assertEqual(segmenter.flush(), [])

    def test_flushes_remaining_text(self):
        segmenter = SentenceSegmenter()

        self.assertEqual(segmenter.add("Let me check"), [])
        self.assertEqual(segmenter.flush(), ["Let me check"])


class ResponseCleanupTests(unittest.TestCase):
    def test_removes_think_blocks_and_markdown(self):
        text = "<think>private chain</think> **Sure.** - Try `rice`."

        self.assertEqual(LanguageModel.clean_response_text(text), "Sure. Try rice.")

    def test_drops_unclosed_think_block(self):
        self.assertEqual(LanguageModel.clean_response_text("<think>private chain"), "")
        self.assertEqual(
            LanguageModel.clean_response_text("Visible answer. <think>private chain"),
            "Visible answer.",
        )

    def test_removes_links_but_keeps_text(self):
        text = "Read [the guide](https://example.com) when you have time."

        self.assertEqual(
            LanguageModel.clean_response_text(text),
            "Read the guide when you have time.",
        )


class PromptCachePrefixTests(unittest.TestCase):
    def test_common_prefix_lengths(self):
        self.assertEqual(LanguageModel._common_prefix_len([], []), 0)
        self.assertEqual(LanguageModel._common_prefix_len([1, 2, 3], [1, 2, 3]), 3)
        self.assertEqual(LanguageModel._common_prefix_len([1, 2, 3], [1, 2, 4, 5]), 2)
        self.assertEqual(LanguageModel._common_prefix_len([1, 2], [1, 2, 3]), 2)
        self.assertEqual(LanguageModel._common_prefix_len([9, 2], [1, 2, 3]), 0)


class CommitTurnTests(unittest.TestCase):
    @staticmethod
    def _bare_llm():
        llm = object.__new__(LanguageModel)
        llm.history_turns = 6
        llm.conversation_history = []
        return llm

    def test_commit_turn_appends_user_and_assistant(self):
        llm = self._bare_llm()

        llm.commit_turn("Hi", "Hello there.")

        self.assertEqual(
            llm.conversation_history,
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there."},
            ],
        )

    def test_interrupted_turn_is_marked(self):
        llm = self._bare_llm()

        llm.commit_turn("Count to thirty", "One, two, three,", interrupted=True)

        self.assertEqual(
            llm.conversation_history[-1]["content"],
            "One, two, three, [interrupted by the user]",
        )

    def test_commit_turn_trims_history_in_batches(self):
        llm = self._bare_llm()
        llm.history_turns = 1

        # Slack lets history exceed the cap so the prompt cache is rebuilt
        # rarely; the trim fires once the slack is exhausted
        for i in range(LanguageModel.HISTORY_TRIM_SLACK_TURNS + 1):
            llm.commit_turn(f"user {i}", f"reply {i}")
        self.assertEqual(
            len(llm.conversation_history),
            (LanguageModel.HISTORY_TRIM_SLACK_TURNS + 1) * 2,
        )

        llm.commit_turn("last user", "last reply")

        self.assertEqual(
            llm.conversation_history,
            [
                {"role": "user", "content": "last user"},
                {"role": "assistant", "content": "last reply"},
            ],
        )


class VoiceBlendTests(unittest.TestCase):
    def test_skips_invalid_blend_voices_and_uses_valid_pair(self):
        class FakePipeline:
            def load_voice(self, voice_name):
                if voice_name == "bad_voice":
                    raise RuntimeError("missing")
                return {"af_bella": 10.0, "af_heart": 20.0}[voice_name]

        blended = _create_blended_voice_tensor(
            FakePipeline(),
            [("bad_voice", 0.9), ("af_bella", 0.6), ("af_heart", 0.4)],
            {"bad_voice": "Bad", "af_bella": "Bella", "af_heart": "Heart"},
        )

        self.assertEqual(blended, 14.0)

    def test_invalid_blend_falls_back_to_default_voice(self):
        class FakePipeline:
            def load_voice(self, voice_name):
                raise RuntimeError("missing")

        blended = _create_blended_voice_tensor(
            FakePipeline(),
            [("missing_one", 0.6), ("missing_two", 0.4)],
            {"missing_one": "Missing one", "missing_two": "Missing two"},
        )

        self.assertIsNone(blended)


class HistoryTrimTests(unittest.TestCase):
    def test_zero_history_turns_clears_history(self):
        llm = object.__new__(LanguageModel)
        llm.history_turns = 0
        llm.conversation_history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]

        LanguageModel._trim_history(llm)

        self.assertEqual(llm.conversation_history, [])


class DefaultSettingsTests(unittest.TestCase):
    def test_abbreviation_normalization_is_off_by_default(self):
        settings = TextProcessingSettings()

        self.assertFalse(settings.expand_abbreviations)
        self.assertEqual(TextPreprocessor(settings).process("I'm in."), "I'm in.")

    def test_fillers_survive_by_default(self):
        settings = TextProcessingSettings()

        self.assertFalse(settings.remove_fillers)
        self.assertEqual(
            TextPreprocessor(settings).process("Hmm, let me think."),
            "Hmm, let me think.",
        )
        self.assertEqual(
            TextPreprocessor(settings).process("Oh no, that sounds hard."),
            "Oh no, that sounds hard.",
        )

    def test_disabled_preprocessor_only_strips_emojis(self):
        settings = TextProcessingSettings(enabled=False, remove_fillers=True)

        self.assertEqual(
            TextPreprocessor(settings).process("Oh, okay then."),
            "Oh, okay then.",
        )


class FillerRemovalTests(unittest.TestCase):
    def test_removes_sentence_initial_fillers_when_enabled(self):
        settings = TextProcessingSettings(remove_fillers=True)
        preprocessor = TextPreprocessor(settings)

        self.assertEqual(preprocessor.process("Oh, okay."), "okay.")
        self.assertEqual(
            preprocessor.process("That's fine. Um, mostly."),
            "That's fine. mostly.",
        )

    def test_keeps_meaningful_mid_sentence_words(self):
        settings = TextProcessingSettings(remove_fillers=True)
        preprocessor = TextPreprocessor(settings)

        self.assertEqual(
            preprocessor.process("Try things like, resting more."),
            "Try things like, resting more.",
        )


class PhoneNumberTests(unittest.TestCase):
    def test_formats_punctuated_phone_numbers(self):
        preprocessor = TextPreprocessor(TextProcessingSettings())

        self.assertEqual(
            preprocessor.process("Call (502) 345-6789."),
            "Call 5 0 2, 3 4 5, 6 7 8 9.",
        )
        self.assertEqual(
            preprocessor.process("Call 502-345-6789."),
            "Call 5 0 2, 3 4 5, 6 7 8 9.",
        )

    def test_leaves_bare_ten_digit_numbers_alone(self):
        preprocessor = TextPreprocessor(TextProcessingSettings())

        self.assertEqual(
            preprocessor.process("Order 5023456789 shipped."),
            "Order 5023456789 shipped.",
        )


class RagPromptInjectionTests(unittest.TestCase):
    def test_no_context_returns_original_message(self):
        self.assertEqual(
            LanguageModel._build_user_content("Hi there", None),
            "Hi there",
        )

    def test_context_wraps_but_preserves_message(self):
        content = LanguageModel._build_user_content("Hi there", ["Dina likes tea."])

        self.assertIn("Dina likes tea.", content)
        self.assertIn("Hi there", content)
        self.assertIn("ignore them silently", content)


if __name__ == "__main__":
    unittest.main()
