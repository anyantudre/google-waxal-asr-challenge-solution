"""Open-set spoken language identification over the test audio.

    python -m waxal_asr.lid --audio-dir data/raw/test_audio --out data/interim/lid.json

The output routes the third-party Sunbird arm, which needs a language token per clip, and it is what
established the composition of the corrected Phase 2 test set. Both uses want the same thing: an
honest per-clip label with a confidence attached.

Open set, deliberately. A closed-set decision restricted to the competition languages cannot report
that a clip is something else, so it converts every out-of-set clip into a confident wrong answer.
Running the full label space instead let us see that 34 clips were neighbouring-language confusions
rather than coverage gaps, and that the corrected test set contains no Luganda at all. See
docs/SOLUTION.md for the measurement.

The model is ``waxal_asr.config.LID_MODEL``. Clips whose confidence falls below the routing
threshold keep their label here and are simply not routed downstream, which is the safe default: an
unrouted clip is transcribed by a multilingual model, whereas a misrouted one is transcribed by a
model that has never seen its language.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from waxal_asr.audio import load_clip
from waxal_asr.config import INTERIM_DATA_DIR, LID_MODEL, RAW_DATA_DIR

SAMPLE_RATE = 16000


def identify(clip_ids: list[str], audio_dir: Path, model_id: str = LID_MODEL) -> dict[str, dict]:
    """Label every clip with its most likely language and the model's confidence.

    Args:
        clip_ids: identifiers to label.
        audio_dir: directory holding one audio file per identifier.
        model_id: an MMS language identification checkpoint.

    Returns:
        Mapping of clip identifier to ``{"lang": iso code, "conf": probability}``. The code is
        whatever the model returned, so it may fall outside the competition languages.
    """
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    labels: dict[str, dict] = {}
    for index, clip_id in enumerate(clip_ids):
        waveform = load_clip(clip_id, audio_dir, SAMPLE_RATE)
        inputs = extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probabilities = model(**inputs).logits.softmax(-1)[0]
        best = int(probabilities.argmax())
        labels[clip_id] = {
            "lang": model.config.id2label[best],
            "conf": round(float(probabilities[best]), 6),
        }
        if index % 50 == 0:
            print(f"[lid] {index + 1}/{len(clip_ids)}", flush=True)
    return labels


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio-dir", type=Path, default=RAW_DATA_DIR / "test_audio")
    parser.add_argument("--test-csv", type=Path, default=RAW_DATA_DIR / "Test.csv")
    parser.add_argument("--out", type=Path, default=INTERIM_DATA_DIR / "lid.json")
    parser.add_argument("--model", default=LID_MODEL)
    args = parser.parse_args()

    from waxal_asr.modeling.predict import read_test_ids

    if not args.audio_dir.exists():
        raise SystemExit(f"{args.audio_dir} not found. See the README for the expected layout.")
    ids = read_test_ids(args.test_csv)
    print(f"[lid] {len(ids)} clips, model {args.model}")

    labels = identify(ids, args.audio_dir, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"model": args.model, "per_clip": labels}, ensure_ascii=False),
        encoding="utf-8",
    )

    counts: dict[str, int] = {}
    for entry in labels.values():
        counts[entry["lang"]] = counts.get(entry["lang"], 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print(f"[lid] wrote {args.out}")
    print("[lid] " + ", ".join(f"{lang} {n}" for lang, n in top))


if __name__ == "__main__":
    main()
