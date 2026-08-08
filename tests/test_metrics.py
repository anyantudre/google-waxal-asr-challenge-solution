"""Tests for the competition metric, in particular its asymmetry.

The property under test is the one that shaped the entire solution: WER ignores casing and
punctuation, CER does not. If a refactor ever breaks this, every offline comparison silently stops
predicting the leaderboard, which is exactly what happened to us for two phases.
"""

import pytest

from waxal_asr.metrics import score, score_dual


class TestScoreDualAsymmetry:
    def test_casing_changes_cer_but_not_wer(self):
        refs = ["Bato ya masano bazali kobeta baskete."]
        cased = ["Bato ya masano bazali kobeta baskete."]
        uncased = ["bato ya masano bazali kobeta baskete."]

        good = score_dual(refs, cased)
        bad = score_dual(refs, uncased)

        assert good.wer == bad.wer, "WER must ignore casing"
        assert bad.cer > good.cer, "CER must penalise the wrong casing"
        assert good.score > bad.score

    def test_punctuation_changes_cer_but_not_wer(self):
        refs = ["Ndinoda kuenda kumusha, ndichadzoka mangwana."]
        with_punct = ["Ndinoda kuenda kumusha, ndichadzoka mangwana."]
        without = ["Ndinoda kuenda kumusha ndichadzoka mangwana"]

        assert score_dual(refs, with_punct).wer == score_dual(refs, without).wer
        assert score_dual(refs, without).cer > score_dual(refs, with_punct).cer

    def test_a_perfect_transcript_scores_one(self):
        refs = ["Awa tozali komona bato mingi."]
        assert score_dual(refs, list(refs)).score == pytest.approx(1.0)

    def test_word_errors_hurt_more_than_character_errors(self):
        # A wrong word costs a full word error plus its characters; a single wrong character costs
        # only the character. This is why our best-in-field CER did not translate into rank one.
        refs = ["mwana akaenda kumusha nhasi"]
        one_char = ["mwana akaenda kumusha nhesi"]
        one_word = ["mwana akaenda kumusha zvakanaka"]
        assert score_dual(refs, one_word).score < score_dual(refs, one_char).score

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            score_dual(["a", "b"], ["a"])


class TestScore:
    def test_identical_text_scores_one(self):
        assert score(["hello world"], ["hello world"]).score == pytest.approx(1.0)

    def test_empty_reference_does_not_raise(self):
        result = score([""], ["something"])
        assert result.score <= 1.0
