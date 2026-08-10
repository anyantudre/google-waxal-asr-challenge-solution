"""CTC character-vocabulary builder.

CTC models (XLS-R, wav2vec2, w2v-BERT) have NO tokenizer on the hub, we build a
character-level vocab from the normalized train+val transcripts. Character-level is
ideal here: the challenge scores 0.5*CER, and a char vocab lets the model emit any
orthography directly.

CRITICAL: a character the vocab lacks can never be emitted and maps to [UNK],
inflating WER/CER. So we FORCE-INCLUDE the lexical characters of lin/sna/lug,
the apostrophe, hyphen, ŋ, and Lingala French accents, so no required character
can be absent from the vocabulary regardless of what the transcripts contain.
"""
from __future__ import annotations

import json
from pathlib import Path

# Chars that are LEXICAL in lin/sna/lug and must be emittable even if rare in a split.
REQUIRED_CHARS = ["'", "-", "ŋ", "é", "è", "ê", "ô", "î", "â", "à", "ï", "ç", "ñ", "ë", "ü"]


def build_ctc_vocab(texts, save_dir: str | Path, extra_required: list[str] | None = None) -> dict:
    """Build a char vocab from an iterable of NORMALIZED transcripts and save vocab.json.

    Returns the vocab dict. `save_dir` gets vocab.json (load it into a Wav2Vec2CTCTokenizer).
    Text MUST already be normalized with the same policy used for training targets.
    """
    chars = set()
    for t in texts:
        chars.update(t)
    chars.discard(" ")  # space -> word delimiter token below, not a literal char

    for c in REQUIRED_CHARS + (extra_required or []):
        chars.add(c)

    vocab = {c: i for i, c in enumerate(sorted(chars))}
    vocab["|"] = len(vocab)          # word delimiter (space)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)      # CTC blank

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)  # ensure_ascii=False keeps ŋ/accents as single codepoints
    return vocab


def build_ctc_tokenizer(vocab_dir: str | Path):
    """Assemble a Wav2Vec2CTCTokenizer from a saved vocab.json."""
    from transformers import Wav2Vec2CTCTokenizer
    return Wav2Vec2CTCTokenizer(
        str(Path(vocab_dir) / "vocab.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
