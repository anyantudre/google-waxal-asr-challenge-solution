"""Inference with Sunbird-51, the one third-party model in the ensemble, used zero-shot.

Three things about this checkpoint need handling, none of them optional:

1. Its `tokenizer_config.json` stores `extra_special_tokens` as a list where transformers expects a
   mapping, so a plain load raises AttributeError. Passing an explicit empty mapping overrides it.
2. Its `generation_config.lang_to_id` is a list rather than a mapping, which breaks generation.
   It is repaired after loading.
3. The revision is pinned. The model card states the weights will be replaced before the final
   release, so an unpinned load would silently change the submission.

Long audio is decoded in overlapping windows. The test set averages 20.2 seconds and 3.3 per cent of
clips run past 30 seconds, which is Whisper's entire receptive field: a plain feature-extractor call
truncates anything longer and raises no error, so those clips would quietly lose their endings.

Lingala and Shona are genuine Whisper language tokens in this checkpoint, so forcing them per clip is
safe. That is not true of every language it supports, several of which occupy repurposed slots.
"""

from __future__ import annotations

import json
from pathlib import Path

from waxal_asr.audio import load_clip
from waxal_asr.config import SUNBIRD_51, SUNBIRD_51_REVISION

WINDOW_SECONDS = 28.0  # two seconds of headroom below Whisper's 30 second window
OVERLAP_SECONDS = 2.0
BEAMS = 8  # beam 8 was worth 0.0224 over greedy for this model
LANG_TOKEN = {"lin": "lingala", "sna": "shona"}


def _load():
    import torch
    from transformers import (
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperTokenizerFast,
    )

    try:
        tokenizer = WhisperTokenizerFast.from_pretrained(SUNBIRD_51, extra_special_tokens={})
    except Exception:
        tokenizer = WhisperTokenizerFast.from_pretrained(SUNBIRD_51)
    extractor = WhisperFeatureExtractor.from_pretrained(SUNBIRD_51)
    model = WhisperForConditionalGeneration.from_pretrained(
        SUNBIRD_51, revision=SUNBIRD_51_REVISION, dtype=torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    generation = model.generation_config
    if isinstance(getattr(generation, "lang_to_id", None), list):
        generation.lang_to_id = {t: tokenizer.convert_tokens_to_ids(t) for t in generation.lang_to_id}
    generation.forced_decoder_ids = None
    return tokenizer, extractor, model, device


def transcribe_sunbird(
    clip_ids: list[str],
    audio_dir: Path,
    lang_map: dict | None = None,
    sample_rate: int = 16000,
) -> dict[str, str]:
    """Transcribe every clip, forcing the language when a language map is supplied."""
    import torch

    tokenizer, extractor, model, device = _load()
    window = int(WINDOW_SECONDS * sample_rate)
    stride = int((WINDOW_SECONDS - OVERLAP_SECONDS) * sample_rate)

    predictions: dict[str, str] = {}
    for index, clip_id in enumerate(clip_ids):
        waveform = load_clip(clip_id, audio_dir, sample_rate)
        chunks = (
            [waveform]
            if len(waveform) <= window
            else [waveform[s : s + window] for s in range(0, len(waveform), stride) if s < len(waveform)]
        )
        kwargs = {"max_new_tokens": 200, "num_beams": BEAMS}
        if lang_map:
            entry = lang_map.get(clip_id) or {}
            language = LANG_TOKEN.get(entry.get("lang"))
            if language and entry.get("conf", 0) >= 0.5:
                kwargs["language"] = language

        pieces = []
        for chunk in chunks:
            features = extractor(chunk, sampling_rate=sample_rate, return_tensors="pt")
            features = features.input_features.to(device, torch.float16)
            with torch.no_grad():
                generated = model.generate(features, **kwargs)
            pieces.append(tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip())
        predictions[clip_id] = " ".join(p for p in pieces if p)
        if index % 50 == 0:
            print(f"[sunbird] {index + 1}/{len(clip_ids)}", flush=True)
    return predictions


def load_lang_map(path: Path) -> dict:
    """Read the per-clip language map written by waxal_asr.lid."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("per_clip", data)
