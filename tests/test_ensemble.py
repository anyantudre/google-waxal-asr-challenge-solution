"""Tests for the character-level ROVER vote.

These pin the behaviours the solution actually depends on: the anchor wins ties, a majority can
overrule it, an empty arm abstains rather than voting for silence, and the output is never empty.
"""

from waxal_asr.ensemble import _combine

WEIGHTS = [1.0] * 4


def combine(hyps, wc=2.0, skeleton="anchor"):
    return _combine(hyps, [1.0] * len(hyps), wc, skeleton)


class TestAnchorBehaviour:
    def test_unanimous_agreement_returns_that_text(self):
        assert combine(["the cat sat"] * 3) == "the cat sat"

    def test_anchor_survives_a_lone_dissenter(self):
        # One member disagreeing carries weight 1.0, below the threshold of 2.0.
        out = combine(["the cat sat", "the dog sat", "the cat sat"])
        assert out == "the cat sat"

    def test_members_overrule_the_anchor_when_they_agree(self):
        # Two members agreeing reach the threshold, so the anchor is overruled.
        out = combine(["the dog sat", "the cat sat", "the cat sat"])
        assert "cat" in out

    def test_anchor_identity_changes_the_result(self):
        # Anchor choice was worth 0.0015 WER on the leaderboard, so it must be observable here.
        a = combine(["alpha text here", "beta text here", "gamma text here"])
        b = combine(["beta text here", "alpha text here", "gamma text here"])
        assert a != b


class TestSafetyGuards:
    def test_an_empty_arm_abstains_rather_than_voting_for_silence(self):
        out = combine(["the cat sat", "", ""])
        assert out == "the cat sat", "empty hypotheses must be treated as missing data"

    def test_output_is_never_empty_when_the_anchor_has_text(self):
        assert combine(["something", "", ""]).strip() != ""

    def test_an_empty_anchor_falls_back_without_crashing(self):
        assert combine(["", "text one", "text two"]) is not None

    def test_single_arm_is_returned_unchanged(self):
        assert combine(["only one arm"]) == "only one arm"


class TestThreshold:
    def test_a_higher_threshold_protects_the_anchor(self):
        hyps = ["the dog sat", "the cat sat", "the cat sat"]
        assert "cat" in combine(hyps, wc=2.0)
        assert combine(hyps, wc=5.0) == "the dog sat"
