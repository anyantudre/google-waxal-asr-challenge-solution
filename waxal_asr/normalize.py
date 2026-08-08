"""Text normalization policy.

CRITICAL RULE: this policy is calibrated ONLY on the train/validation splits.
The Phase-1 test transcriptions are publicly leaked but OFF-LIMITS for any tuning
(including normalization). See CHALLENGE_DESCRIPTION.md \u00a75.

The same policy must be applied to (a) training targets, (b) local-eval references,
and (c) model predictions before scoring/submission, so it lives in one place.
Every option is config-driven (cfg.normalize) so we can ablate it as an experiment.

HARD RULES (verified against orthography sources + Train.csv, 2026-07-02, see
research/Q4_text_norm.md). These prevent real WER/CER inflation:
  * NEVER strip the apostrophe (U+0027), it is LEXICAL in all three languages:
    Luganda ng'=\u014b and n' connective (~2.7/row), Shona n'anga, Lingala elisions n'a/d'/l'.
  * NEVER strip the hyphen (-), lexical reduplication (mwana-mwasi, yana-siya).
  * NEVER use NFKC or accent-stripping, Lingala carries real French accents (\u00e9/\u00e8/\u00f4\u2026),
    Luganda carries \u014b/\u1d51; NFKC would delete/mangle them and inflate CER. Use NFC only.
  * None of the 3 languages mark tone in orthography, so lowercase/strip_punct/NFC/collapse_ws
    are all SAFE. The `_PUNCT` set below is deliberately conservative and OMITS ' and -.
Best tunable win (train/val-only): unify Luganda's velar-nasal spelling, see
`unify_luganda_velar_nasal` below (data mixes \u014b, ng', and modifier \u1d51 inconsistently).
"""
from __future__ import annotations
import re
import unicodedata

# Conservative punctuation set: MUST NOT contain the apostrophe (') or hyphen (-),
# which are lexical in lin/sna/lug (see HARD RULES above).
_PUNCT = re.compile(r"[.,!?;:\"()\[\]{}\u00ab\u00bb\u2026\u201c\u201d\u201e\u201f\u2022]+")
_WS = re.compile(r"\s+")

# Luganda velar nasal \u014b is written inconsistently: literal \u014b (U+014B), the keyboard
# form ng', and the modifier \u1d51 (U+1D51). Map all to one canonical form so targets,
# refs, and predictions match. Direction (ng' vs \u014b) is an ablation, pick on train/val.
_LUG_ENG_VARIANTS = {"\u014b": "ng'", "\u014a": "ng'", "\u1d51": "ng'"}


def normalize_text(
    text: str,
    lowercase: bool = True,
    nfc: bool = True,
    strip_punct: bool = True,
    collapse_ws: bool = True,
) -> str:
    if text is None:
        return ""
    if nfc:
        text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = _PUNCT.sub(" ", text)
    if collapse_ws:
        text = _WS.sub(" ", text).strip()
    return text


def unify_luganda_velar_nasal(text: str, canonical: str = "ng'") -> str:
    """Map Luganda's inconsistent velar-nasal spellings (\u014b / ng' / \u1d51) to one form.

    Luganda-only, opt-in pre-step (NOT one of the four cfg.normalize flags). Apply
    identically to targets, local-eval refs, and predictions. `canonical` is an
    ablation to tune on train/val: "ng'" (default, the majority keyboard form) or "\u014b".
    """
    if canonical == "ng'":
        for variant, repl in _LUG_ENG_VARIANTS.items():
            text = text.replace(variant, repl)
    elif canonical == "\u014b":
        text = text.replace("ng'", "\u014b").replace("\u014a", "\u014b").replace("\u1d51", "\u014b")
    else:
        raise ValueError("canonical must be \"ng'\" or \"\u014b\"")
    return text


def make_normalizer(cfg) -> "callable":
    """Return a single-arg normalizer bound to the config's normalize options.

    `cfg.normalize.unify_luganda` (null | "ng'" | "\u014b", default null) opt-in enables H3
    velar-nasal unification, applied AFTER the base steps (so lowercasing has already run).
    Applied to targets, refs, and predictions identically. Only lug text contains these
    variants, so it is a no-op for lin/sna.
    """
    n = cfg.normalize
    canon = n.get("unify_luganda") if hasattr(n, "get") else getattr(n, "unify_luganda", None)

    def _norm(t):
        t = normalize_text(
            t, lowercase=n.lowercase, nfc=n.nfc, strip_punct=n.strip_punct, collapse_ws=n.collapse_ws
        )
        if canon:
            t = unify_luganda_velar_nasal(t, canonical=canon)
        return t

    return _norm
