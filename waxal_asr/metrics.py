"""The competition metric, computed the way the leaderboard computes it.

    leaderboard score = 1 - 0.5 * (WER + CER)      higher is better

The two halves do NOT use the same text:

* WER is computed on normalised text. Casing and punctuation are removed before comparison.
* CER is computed on the raw string. Casing and punctuation are scored.

That asymmetry was established experimentally, not assumed. Two submissions identical except for the
capitalisation of 50 rows returned a WER identical to nine decimal places (0.382307342 both times)
while CER moved from 0.109473 to 0.109366. If both halves normalised, CER could not have changed; if
neither did, WER could not have stayed identical.

The consequence shaped the whole solution: a lowercase, punctuation-free model is capped on CER no
matter how good its words are. Rebuilding the CTC vocabulary to include capitals and punctuation was
worth 0.0171 on the leaderboard, the largest single modelling gain we measured.

`score_dual` is the function to use for any decision. `score` is kept for the normalised-only view,
which is still the right comparison when contrasting two arms that share an output convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

from waxal_asr.normalize import normalize_text


@dataclass
class ScoreResult:
    """WER, CER and the leaderboard score, on whichever text the caller supplied."""

    wer: float
    cer: float
    score: float

    def __str__(self) -> str:
        return f"score={self.score:.4f}  (WER={self.wer:.4f}  CER={self.cer:.4f})"


def _leaderboard(wer: float, cer: float) -> float:
    return 1.0 - 0.5 * (wer + cer)


def score(references: list[str], hypotheses: list[str]) -> ScoreResult:
    """Score a set of pairs without changing the text, for like-for-like arm comparisons."""
    if len(references) != len(hypotheses):
        raise ValueError(f"length mismatch: {len(references)} references, {len(hypotheses)} hypotheses")
    # jiwer raises on an empty reference, so substitute a single space. This affects only clips whose
    # reference is genuinely empty, of which the test set has none.
    refs = [r if r.strip() else " " for r in references]
    wer = jiwer.wer(refs, hypotheses)
    cer = jiwer.cer(refs, hypotheses)
    return ScoreResult(wer=wer, cer=cer, score=_leaderboard(wer, cer))


def score_dual(raw_references: list[str], hypotheses: list[str]) -> ScoreResult:
    """Score exactly as the leaderboard does: WER on normalised text, CER on raw text.

    Pass the untouched reference strings and the untouched model output. Normalisation of the WER
    half happens here so that a caller cannot accidentally normalise the CER half as well, which is
    the mistake that made our offline numbers disagree with the leaderboard for two phases.
    """
    if len(raw_references) != len(hypotheses):
        raise ValueError(
            f"length mismatch: {len(raw_references)} references, {len(hypotheses)} hypotheses"
        )
    refs_norm = [normalize_text(r) or " " for r in raw_references]
    hyps_norm = [normalize_text(h) for h in hypotheses]
    refs_raw = [r if r.strip() else " " for r in raw_references]

    wer = jiwer.wer(refs_norm, hyps_norm)
    cer = jiwer.cer(refs_raw, hypotheses)
    return ScoreResult(wer=wer, cer=cer, score=_leaderboard(wer, cer))
