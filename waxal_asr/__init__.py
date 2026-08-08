"""WAXAL ASR: architecture-agnostic ASR pipeline for the Google WAXAL challenge.

Everything is config-driven and model-agnostic: a run is fully described by a YAML
config (see `configs/`), and any HuggingFace ASR checkpoint (Whisper/seq2seq or
wav2vec2, MMS/CTC, later Gemma-audio) plugs in through the `waxal.models` registry.

Public API kept intentionally small:
    from waxal_asr.config  import load_config, set_seed
    from waxal_asr.data    import load_splits
    from waxal_asr.metrics import score            # 0.5*WER + 0.5*CER
    from waxal_asr.models  import build_model       # config -> model adapter
"""
__version__ = "0.1.0"
LANGUAGES = ("lin", "sna", "lug")  # Lingala, Shona, Luganda
