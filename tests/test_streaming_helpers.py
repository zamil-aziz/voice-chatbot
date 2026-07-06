import unittest

from config.settings import TextProcessingSettings
from src.models.llm import LanguageModel
from src.models.tts import _create_blended_voice_tensor
from src.pipeline.manager import SentenceSegmenter, ThinkBlockStreamFilter
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


class ThinkBlockStreamFilterTests(unittest.TestCase):
    def test_suppresses_open_think_block_until_close_arrives(self):
        stream_filter = ThinkBlockStreamFilter()

        self.assertEqual(stream_filter.add("<think>I should answer briefly."), "")
        self.assertEqual(stream_filter.add("</think>Sure."), "Sure.")
        self.assertEqual(stream_filter.flush(), "")

    def test_handles_tags_split_across_chunks(self):
        stream_filter = ThinkBlockStreamFilter()

        self.assertEqual(stream_filter.add("<thi"), "")
        self.assertEqual(stream_filter.add("nk>private"), "")
        self.assertEqual(stream_filter.add("</thi"), "")
        self.assertEqual(stream_filter.add("nk>Public."), "Public.")

    def test_preserves_non_think_angle_text(self):
        stream_filter = ThinkBlockStreamFilter()

        self.assertEqual(stream_filter.add("Use < 5 items."), "Use < 5 items.")


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
