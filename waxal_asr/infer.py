"""Inference -> Zindi submission.

Same code path for Phase 1 and Phase 2. Language routing:
  * Phase 1: language is known (Test.csv ID prefix e.g. `lug_...`).
  * Phase 2: NO metadata -> either one multilingual model (design A) or LID routing
    (design B, waxal_asr.lid). `route_language` centralizes this switch.

Writes a SampleSubmission-format CSV (columns: ID,Target).
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from waxal_asr.config import DATA_DIR, RESULTS_DIR
from waxal_asr.models import build_model
from waxal_asr.normalize import make_normalizer


def route_language(clip_id: str | None, cfg) -> str | None:
    """Phase 1: parse language from the ID prefix. Phase 2: return None (handled by
    a multilingual model) or route per language from a language map, see waxal_asr.lid."""
    if cfg.infer.get("language_from_id", True) and clip_id and "_" in clip_id:
        lang = clip_id.split("_")[0]
        if lang in set(cfg.data.languages):     # ignore non-language prefixes, Phase-2 ids may contain '_'
            return lang
    return None


_EMPTY_FALLBACK = "a"  # last resort if Train.csv is unavailable


def _fallback_words(cfg) -> dict:
    """Per-language most-common training word (+ '_default' global). A valid, in-domain, NON-EMPTY
    fallback for clips the model transcribes as empty (CTC greedy blank-collapse). Zindi rejects
    blank Targets ("Missing entries for IDs ...") so every row must carry some text."""
    from collections import Counter
    try:
        from waxal_asr.data import load_zindi_csv, TEXT_COLUMN, ID_COLUMN
        norm = make_normalizer(cfg)
        df = load_zindi_csv("Train.csv")
        per, allc = {}, Counter()
        for cid, txt in zip(df[ID_COLUMN], df[TEXT_COLUMN]):
            words = norm(txt).split()
            per.setdefault(str(cid).split("_")[0], Counter()).update(words)
            allc.update(words)
        fb = {lang: c.most_common(1)[0][0] for lang, c in per.items() if c}
        fb["_default"] = allc.most_common(1)[0][0] if allc else _EMPTY_FALLBACK
        return fb
    except Exception:
        return {"_default": _EMPTY_FALLBACK}


def run(cfg, test_csv: str = "Test.csv", out_csv: str | None = None, batch_size: int = 16):
    adapter = build_model(cfg).load()
    adapter.model.eval()
    import torch
    if torch.cuda.is_available():
        adapter.model.to("cuda")
    normalizer = make_normalizer(cfg)

    test_path = Path(test_csv)
    if not test_path.exists():                       # bare name -> resolve under data/
        test_path = DATA_DIR / test_csv
    if out_csv is None:
        out_csv = str(RESULTS_DIR / "submissions" / f"{Path(cfg.train.get('output_dir') or 'submission').name}.csv")

    test = pd.read_csv(test_path)
    # Phase-2's file is unseen. Tolerate a differently-cased id column and fail LOUD (not a bare KeyError)
    # if none is found. astype(str): numeric ids would otherwise break route_language's `"_" in clip_id`.
    id_col = next((c for c in ("ID", "id", "Id", "audio_id", "filename") if c in test.columns), None)
    if id_col is None:
        raise KeyError(f"no id column in {test_path.name}; columns = {list(test.columns)}")
    ids = test[id_col].astype(str).tolist()

    # Phase-2 (audio_dir) loads each clip from disk ON DEMAND per batch, so peak RAM is one batch, 
    # the whole (possibly-large) test set is never materialized. Phase-1 (HF, bounded ~4k) caches.
    fetch = _audio_fetcher(ids, cfg)

    # Group by language (ID prefix) so a per-language LM / route can apply; positional writes keep ID order.
    import inspect
    from collections import defaultdict
    accepts_lang = "lang" in inspect.signature(adapter.transcribe).parameters
    groups: dict = defaultdict(list)
    for i, cid in enumerate(ids):
        groups[route_language(cid, cfg)].append(i)
    preds: list = [None] * len(ids)
    for lang, idxs in groups.items():
        for s in range(0, len(idxs), batch_size):
            ci = idxs[s : s + batch_size]
            chunk = fetch([ids[j] for j in ci])
            out = adapter.transcribe(chunk, lang=lang) if accepts_lang else adapter.transcribe(chunk)
            for j, p in zip(ci, out):
                preds[j] = p

    targets = [normalizer(p) for p in preds]
    # Zindi rejects blank Targets. CTC greedy decode emits an empty string on some hard clips
    # (concentrated in the weakest language) -> fill with a valid per-language fallback word.
    blanks = [k for k, t in enumerate(targets) if not t.strip()]
    if blanks:
        fb = _fallback_words(cfg)
        for k in blanks:
            targets[k] = fb.get(route_language(ids[k], cfg), fb["_default"])
        print(f"[infer] filled {len(blanks)} empty prediction(s) with fallback words")

    sub = pd.DataFrame({"ID": ids, "Target": targets})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"[infer] wrote {len(sub)} rows -> {out_csv}")
    return sub


def _load_one_from_dir(clip_id, audio_dir, sr):
    """Load a single <id>.* clip from an audio dump: mono, resampled to sr."""
    import glob
    import soundfile as sf
    from pathlib import Path
    # glob.escape: an id could contain a glob metachar ([, ?, *). sorted(): deterministic pick if an
    # id somehow has >1 file (e.g. both .wav and .mp3) instead of glob's arbitrary order.
    matches = sorted(glob.glob(glob.escape(str(Path(audio_dir) / str(clip_id))) + ".*"))
    if not matches:
        raise FileNotFoundError(f"no audio file for id {clip_id!r} in {audio_dir}")
    wav, s = sf.read(matches[0], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if s != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=s, target_sr=sr)
    return wav


def _audio_fetcher(ids, cfg):
    """Return a callable: batch_ids -> [np.ndarray], loading audio in ID order.

    Phase-2 (cfg.infer.audio_dir set): loads each <id>.* from disk ON DEMAND, so peak memory is one
    batch, the whole test set is never materialized (Phase-2 may be far larger than Phase-1's 4253).
    Phase-1 (HF `test` splits, bounded ~4k): loads once and serves from an in-memory cache.
    """
    audio_dir = cfg.infer.get("audio_dir")
    if audio_dir:
        sr = cfg.data.sample_rate
        return lambda batch_ids: [_load_one_from_dir(i, audio_dir, sr) for i in batch_ids]
    cache = dict(zip(ids, _load_audio_for_ids(ids, cfg)))
    return lambda batch_ids: [cache[i] for i in batch_ids]


def _load_audio_for_ids(ids, cfg):
    """Return audio arrays aligned with `ids`.

    Phase 2 (audio-only dump): set cfg.infer.audio_dir -> loads <id>.* from disk.
    Phase 1: streams the *_asr `test` splits from HF and collects the requested ids.
    """
    sr = cfg.data.sample_rate
    audio_dir = cfg.infer.get("audio_dir")
    if audio_dir:
        return [_load_one_from_dir(i, audio_dir, sr) for i in ids]

    # Phase-1 HF path: stream each language's test split, collect the requested ids by `id`
    from datasets import load_dataset, Audio
    want = set(ids)
    id2arr: dict = {}
    for lang in cfg.data.languages:
        # test AUDIO only: select_columns drops `transcription`, so the leaked Phase-1 test labels
        # are never even loaded into the process (auditable no-leak guarantee). Streaming pulls only
        # the `test` shards. Inherits the HF_HUB_DOWNLOAD_TIMEOUT set by slurm_train.sh.
        ds = load_dataset(cfg.data.hf_dataset, cfg.data.configs[lang], split="test",
                          streaming=True).select_columns(["id", "audio"]).cast_column(
                          "audio", Audio(sampling_rate=sr))
        for ex in ds:
            if ex["id"] in want:
                id2arr[ex["id"]] = ex["audio"]["array"]
        if len(id2arr) >= len(want):
            break
    missing = want - set(id2arr)
    if missing:
        raise KeyError(f"{len(missing)} test ids not found in HF test splits (e.g. {list(missing)[:3]})")
    return [id2arr[i] for i in ids]
