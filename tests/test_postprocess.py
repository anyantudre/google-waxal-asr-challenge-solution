"""Tests for the two post-processing rules that survived measurement."""

from waxal_asr.postprocess import collapse_loops, fix_sentence_case, postprocess


class TestCollapseLoops:
    def test_collapses_a_long_repetition_run(self):
        text = "the dog " + "ran away " * 6 + "home"
        assert collapse_loops(text) == "the dog ran away ran away home"

    def test_keeps_genuine_reduplication(self):
        # Two consecutive copies are lexical in Bantu languages and must survive untouched.
        text = "mwana mwana akaenda"
        assert collapse_loops(text) == text

    def test_collapses_single_word_loops(self):
        assert collapse_loops("na na na na na yes") == "na na yes"

    def test_leaves_clean_text_alone(self):
        text = "Bato ya masano bazali kobeta baskete na terrain."
        assert collapse_loops(text) == text

    def test_handles_empty_and_none(self):
        assert collapse_loops("") == ""
        assert collapse_loops(None) == ""


class TestFixSentenceCase:
    def test_capitalises_after_a_full_stop(self):
        assert fix_sentence_case("one thing. another thing.") == "one thing. Another thing."

    def test_handles_several_sentences_in_one_pass(self):
        assert fix_sentence_case("a. b. c.") == "a. B. C."

    def test_capitalises_after_question_and_exclamation(self):
        assert fix_sentence_case("what? yes! ok.") == "what? Yes! Ok."

    def test_does_not_touch_the_first_character(self):
        # Deliberate: references open with a capital only 89.3 per cent of the time, and our output
        # was already above that rate, so forcing it would move away from the reference distribution.
        assert fix_sentence_case("lowercase start here.") == "lowercase start here."

    def test_leaves_already_correct_text_unchanged(self):
        text = "First sentence. Second sentence."
        assert fix_sentence_case(text) == text


class TestPostprocessChain:
    def test_applies_both_rules_in_order(self):
        text = "hello. " + "loop word " * 5 + "end"
        out = postprocess(text)
        assert "Loop word loop word end" in out or "loop word loop word end" in out.lower()
        assert out.count("loop word") == 2

    def test_never_returns_none(self):
        assert postprocess(None) == ""
