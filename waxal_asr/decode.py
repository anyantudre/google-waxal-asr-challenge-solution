"""CTC decoding, including the blank-penalty correction that produced our largest single gain.

Greedy CTC takes the arg max at every frame. When the blank symbol marginally outranks the best
character, that character is dropped silently, and often the whole word with it. Measured against
matched references our hypotheses carried only 97.9 per cent of the reference word count: 1.30 words
per second against a reference rate of 1.41.

Every missing word is a deletion. Deletions cost WER heavily and CER barely, which was exactly our
error signature: the best CER in the competition alongside the worst WER of the leading entries.

Subtracting a constant from the blank logit before the arg max shifts that tie-break towards
emitting. The constant is chosen so the output word rate matches the reference rate, not by tuning on
the leaderboard. A penalty of 1.5 brought a representative arm to 1.417 words per second.

The re-decoded arms are worse on their own (0.743 against 0.746 for the same checkpoint decoded
normally) because some of the recovered words are wrong. They are valuable as ensemble members: the
vote keeps the recovered words the other arms confirm and discards the rest.
"""

from __future__ import annotations

from typing import Iterable

REFERENCE_WORDS_PER_SECOND = 1.41  # measured on matched lin/sna references


def apply_blank_penalty(logits, blank_id: int, penalty: float):
    """Return a copy of `logits` with `penalty` subtracted from the blank column.

    logits: torch.Tensor of shape (time, vocab) for one utterance.
    """
    out = logits.clone()
    out[:, blank_id] -= penalty
    return out


def greedy_decode(logits, processor, blank_id: int, penalty: float = 0.0) -> str:
    """Decode one utterance's logits to text, optionally penalising the blank symbol."""
    if penalty:
        logits = apply_blank_penalty(logits, blank_id, penalty)
    ids = logits.argmax(-1).unsqueeze(0)
    return processor.batch_decode(ids)[0].strip()


def words_per_second(transcripts: Iterable[str], durations: Iterable[float]) -> float:
    """Mean words per second across a set of transcripts, used to choose the penalty.

    Compare the result against REFERENCE_WORDS_PER_SECOND: a value below it means the decoder is
    still dropping words, and a value above it means it has started inventing them.
    """
    pairs = [(t, d) for t, d in zip(transcripts, durations) if d and d > 0]
    if not pairs:
        return 0.0
    return sum(len(t.split()) / d for t, d in pairs) / len(pairs)
