"""Model registry: the plug'n'play seam.

A model adapter hides ALL architecture specifics behind one interface
(waxal_asr.models.base.ASRAdapter) so that train.py / infer.py never branch on the model. To add a
new architecture, write one adapter and register it here, nothing else changes.

    cfg.model.type -> adapter class

Registered types:
    "ctc"      -> wav2vec2 / XLS-R (char vocab built from corpus)   [ctc.py]
    "seq2seq"  -> Whisper / any AutoModelForSpeechSeq2Seq            [seq2seq.py]
    "w2v_bert" -> facebook/w2v-bert-2.0 (input_features frontend)    [w2v_bert.py]
"""
from __future__ import annotations

from waxal_asr.models.base import ASRAdapter

_REGISTRY: dict[str, type[ASRAdapter]] = {}


def register(name: str):
    def deco(cls: type[ASRAdapter]):
        _REGISTRY[name] = cls
        return cls
    return deco


def build_model(cfg) -> ASRAdapter:
    """Instantiate the adapter named by cfg.model.type."""
    # imports here so registration happens and optional deps stay lazy
    from waxal_asr.models import ctc, seq2seq, w2v_bert  # noqa: F401
    key = cfg.model.type
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model.type={key!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](cfg)
