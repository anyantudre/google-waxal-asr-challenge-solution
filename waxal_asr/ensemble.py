"""Reference-free character-level ROVER over several arms' transcripts.

The first hypothesis is the anchor: its text survives every slot unless the other arms dissent with
combined weight at least `wc`. Anchor identity therefore decides the output far more than any single
member does, and swapping it was worth 0.0015 WER on the leaderboard.

Two properties of the vote explain why the ensemble works at all here:

* Deletions are unrecoverable but insertions are filtered. A member that omits a word gives the vote
  nothing to find, whereas a member that emits a spurious word is simply outvoted. This is why the
  deliberately over-generating blank-penalty arms improve the ensemble by 0.0104 despite scoring
  0.743 on their own, well below the 0.764 of the ensemble they belong to.
* Members must be both strong and decorrelated. Adding arms from weak checkpoints cost 0.0003, and
  adding arms from the same lineage as the anchor cost 0.0014, while four arms from four different
  strong checkpoints gained 0.0040.

Empty hypotheses are treated as missing data rather than as a prediction of silence. Without that,
one silent arm drags the vote towards the empty string and the clip is lost.
"""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein


def _mbr_skeleton(hyps):
    """Index of the medoid hypothesis (smallest total char edit distance to the others).
    Reference-free: it only asks which candidate is most central, never which is correct."""
    best, bi = None, 0
    for i, h in enumerate(hyps):
        d = sum(Levenshtein.distance(h, g) for j, g in enumerate(hyps) if j != i)
        if best is None or d < best:
            best, bi = d, i
    return bi


def _align_to(skel, other):
    """Map `other` onto the skeleton: per-slot character (''=deletion) + text inserted before each slot."""
    out = [None] * len(skel)
    ins = [""] * (len(skel) + 1)
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(skel, other):
        if tag == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = other[j1 + k]
        elif tag == "replace":
            n, m = i2 - i1, j2 - j1
            for k in range(n):
                out[i1 + k] = other[j1 + k] if k < m else ""
            if m > n:
                out[i2 - 1] = other[j1 + n - 1: j2]
        elif tag == "delete":
            for k in range(i1, i2):
                out[k] = ""
        elif tag == "insert":
            ins[i1] += other[j1:j2]
    return out, ins


def _rover(hyps, anchor, weights, wc, skeleton="mbr"):
    """Character-level confusion-network vote over `hyps`, anchored on hyps[anchor].

    skeleton='mbr', vote onto the most central hypothesis. Best on holdout (-0.0090) but the
                        skeleton is non-champion on ~70% of clips, so segmentation/length come from
                        a weaker arm and the downside is not bounded if the test set shifts.
    skeleton='anchor', vote onto the champion's own string: strictly safer, ~85% of the gain
                        (holdout -0.0076). The fallback when the leaderboard is ambiguous.
    """
    skel = hyps[anchor] if skeleton == "anchor" else hyps[_mbr_skeleton(hyps)]
    if not skel:
        return hyps[anchor]
    cols, inss = [], []
    for h in hyps:
        c, i = _align_to(skel, h)
        cols.append(c)
        inss.append(i)

    out, n = [], len(skel)
    for pos in range(n + 1):
        tally = {}
        for a in range(len(hyps)):
            tally[inss[a][pos]] = tally.get(inss[a][pos], 0.0) + weights[a]
        cand = max(tally.items(), key=lambda kv: (kv[1], kv[0] != ""))
        anchor_ins = inss[anchor][pos]
        out.append(cand[0] if cand[0] != anchor_ins and tally.get(cand[0], 0) >= wc else anchor_ins)
        if pos == n:
            break
        tally = {}
        for a in range(len(hyps)):
            tally[cols[a][pos]] = tally.get(cols[a][pos], 0.0) + weights[a]
        anchor_c = cols[anchor][pos]
        best = max(tally.items(), key=lambda kv: kv[1])
        out.append(best[0] if best[0] != anchor_c and best[1] >= wc else anchor_c)
    return " ".join("".join(out).split())


def _combine(hyps, weights, wc, skeleton="mbr"):
    """ROVER with two safety guards, both justified independently of any score:
    an empty hypothesis is missing data (abstain), and we never emit an empty transcription."""
    anchor_text = hyps[0]
    keep = [i for i, h in enumerate(hyps) if h.strip()]
    if 0 not in keep or len(keep) < 2:
        return anchor_text
    out = _rover([hyps[i] for i in keep], keep.index(0), [weights[i] for i in keep], wc, skeleton)
    return out if out.strip() else anchor_text
