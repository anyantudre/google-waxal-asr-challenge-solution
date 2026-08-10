"""Data loading + splits.

Audio lives on HuggingFace `google/WaxalNLP` (configs lin_asr / sna_asr / lug_asr);
Zindi ships only text+IDs (Train.csv/Test.csv) that map 1:1 to HF rows by `id`.
We load audio from HF and align it with the Zindi split.

Local evaluation:
  * `validation`, the split WAXAL already marks in Train.csv (`original_split`).
  * `speaker_disjoint` holdout, carved from train on unseen speakers, which mimics
    Phase 2 ("new speakers, no metadata") far better than the Phase-1 test set.

Schema CONFIRMED (datasets-server API + card + README, 2026-07-02):
each ASR config (lin_asr/sna_asr/lug_asr) has 6 columns
  [id, speaker_id, transcription, language, gender, audio]
and 4 splits [train, validation, test, unlabeled]. `speaker_id` is present -> speaker-disjoint
splits work. `audio` is MP3-encoded -> decode + resample to 16 kHz. The `unlabeled` split has
transcription="" (use for SSL / pseudo-labelling, never for supervised targets). This module
never loads the `test` split: it is the Phase-1 test set, which Phase 1 forbade training on.
Phase 2 permits it and it IS used, as disclosed in README.md and docs/SOLUTION.md, but through
the separate corpus-construction path, not through these splits.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from waxal_asr.config import DATA_DIR, RESULTS_DIR

# Confirmed dataset schema, the single place to adjust if the dataset changes
AUDIO_COLUMN = "audio"
SPEAKER_COLUMN = "speaker_id"
TEXT_COLUMN = "transcription"
ID_COLUMN = "id"
GENDER_COLUMN = "gender"         # only demographic column (Male/Female); no age


def load_zindi_csv(name: str = "Train.csv") -> pd.DataFrame:
    """Read a Zindi CSV from data/ (backslash-escaped quotes -> escapechar='\\')."""
    return pd.read_csv(DATA_DIR / name, escapechar="\\")


def load_hf_audio(language_config: str, split: str = "train", streaming: bool = False):
    """Load one HF audio config (e.g. 'lug_asr'). Import kept local so the package
    imports without `datasets` installed (e.g. for pure-metrics use)."""
    from datasets import Audio, load_dataset
    ds = load_dataset("google/WaxalNLP", language_config, split=split, streaming=streaming)
    ds = ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=16000))
    return ds


def split_speaker_disjoint(df: pd.DataFrame, holdout_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole speakers (not random rows) so the eval set has speakers the
    model never trained on, the key Phase-2 generalization signal.

    HARD-FAILS if speaker_id is missing/incomplete. A silent random-row fallback would
    leak speakers train<->eval and quietly destroy our only Phase-2 proxy, every gate
    would then optimize an over-optimistic metric (panel-flagged top single-point-of-failure).
    """
    if SPEAKER_COLUMN not in df.columns or df[SPEAKER_COLUMN].isna().any():
        missing = "column absent" if SPEAKER_COLUMN not in df.columns else f"{df[SPEAKER_COLUMN].isna().sum()} null values"
        raise ValueError(
            f"speaker-disjoint split requires a complete '{SPEAKER_COLUMN}' ({missing}). "
            "Fix the HF<->Zindi id join before building the holdout, do NOT fall back to a "
            "random-row split (would leak speakers and invalidate the Phase-2 proxy)."
        )
    speakers = df[SPEAKER_COLUMN].unique()
    held_speakers = set(pd.Series(speakers).sample(frac=holdout_frac, random_state=seed).tolist())
    held = df[df[SPEAKER_COLUMN].isin(held_speakers)]
    train = df[~df[SPEAKER_COLUMN].isin(held_speakers)]
    assert set(train[SPEAKER_COLUMN]).isdisjoint(held_speakers), "train/held speakers overlap"
    return train, held


def _load_external(spec: dict, sr: int, cap=None):
    """Load one external ASR corpus, mapped to WAXAL's TRAIN schema [audio, transcription].

    spec keys: EITHER hf (+config?, split?) for an HF corpus, OR hf+data_files=<parquet glob> to fetch
    only specific shards of a huge corpus (Open Bible: 730MB/shard, streaming stalls), OR local=<manifest.tsv>
    for a local corpus (OpenSLR after download+extract). Plus text_col?('transcription'), audio_col?('audio'),
    max_rows? (down-weight cap). External corpora are speaker-disjoint from the WAXAL holdout by
    construction (different sources). `cap` (from cfg.data.limit) forces a small sample for smoke runs.
    """
    import itertools

    from datasets import Audio, Dataset, load_dataset

    audio_col, text_col = spec.get("audio_col", AUDIO_COLUMN), spec.get("text_col", TEXT_COLUMN)
    caps = [int(x) for x in (spec.get("max_rows"), cap) if x]
    n = min(caps) if caps else None                    # smoke `cap` must cap EVERY source, even a max_rows one

    local = spec.get("local")
    if local:                                          # local manifest (TSV/CSV): audio-path col + text col
        # QUOTE_NONE: transcripts legitimately contain " and ', and at pandas' default quoting one
        # stray quote swallows every following line until the next, merging thousands of rows into a
        # single label: which surfaced as "Labels' sequence length 21522 cannot exceed 448 tokens".
        import csv as _csv
        m = pd.read_csv(local, sep=spec.get("sep", "\t"), quoting=_csv.QUOTE_NONE,
                        on_bad_lines="skip", dtype=str, keep_default_na=False)
        if n and len(m) > int(n):
            m = m.head(int(n))
        cols = {AUDIO_COLUMN: m[audio_col].astype(str).tolist(),
                TEXT_COLUMN: m[text_col].astype(str).tolist()}
        keep = [AUDIO_COLUMN, TEXT_COLUMN]
        # Preserve `lang` when the manifest carries it. Whisper labels must begin with a LANGUAGE token,
        # and a multilingual fine-tune needs it PER EXAMPLE: dropping this column forced every language
        # to train under one token while inference forced a different one, a silent train/test mismatch.
        if "lang" in m.columns:
            cols["lang"] = m["lang"].astype(str).tolist()
            keep.append("lang")
        ds = Dataset.from_dict(cols)
        ds = ds.cast_column(AUDIO_COLUMN, Audio(sampling_rate=sr)).select_columns(keep)
        if cap:
            # SMOKE ONLY: data._load builds the WAXAL smoke split with Dataset.from_list, whose audio column
            # is a DECODED struct, not the Audio feature. concatenate_datasets refuses to align the two
            # ("key audio ... has unexpected type"), so materialise to the same decoded form here.
            # The full path leaves this untouched: there both sides are Audio<->Audio and already align.
            ds = Dataset.from_list(list(ds))
        return ds

    hf, config, split = spec["hf"], spec.get("config"), spec.get("split", "train")
    data_files = spec.get("data_files")                # explicit parquet shard glob for HUGE corpora
    if n and not data_files:                            # streamed sample: smoke, or a cap on a SMALL corpus
        # cast the STREAM to Audio, then from_list -> a decoded-audio struct, EXACTLY matching how
        # data._load builds the WAXAL smoke split (do NOT re-cast to the Audio feature here, or the
        # two won't align under concatenate_datasets). The full-parquet path below yields Audio<->Audio.
        stream = load_dataset(hf, config, split=split, streaming=True).cast_column(
            audio_col, Audio(sampling_rate=sr))
        rows = [{AUDIO_COLUMN: r[audio_col], TEXT_COLUMN: str(r[text_col])}
                for r in itertools.islice(stream, int(n))]
        return Dataset.from_list(rows).select_columns([AUDIO_COLUMN, TEXT_COLUMN])

    # Full corpus OR a bounded set of parquet shards (data_files): both are CACHED + memory-mapped, so a
    # resumed job reuses the download. Streaming a 730MB-shard corpus (e.g. Open Bible) instead re-pulls
    # whole shards every job and stalls; data_files fetches only the listed shards, then max_rows caps rows.
    if data_files:
        ds = load_dataset(hf, data_files={split: data_files}, split=split).select_columns([audio_col, text_col])
        if n and len(ds) > int(n):
            ds = ds.select(range(int(n)))
    else:
        ds = load_dataset(hf, config, split=split).select_columns([audio_col, text_col])
    ren = {}
    if audio_col != AUDIO_COLUMN:
        ren[audio_col] = AUDIO_COLUMN
    if text_col != TEXT_COLUMN:
        ren[text_col] = TEXT_COLUMN
    ds = (ds.rename_columns(ren) if ren else ds).cast_column(AUDIO_COLUMN, Audio(sampling_rate=sr))
    return ds.select_columns([AUDIO_COLUMN, TEXT_COLUMN])


def holdout_manifest_path(seed: int) -> Path:
    return RESULTS_DIR / f"holdout_speakers_seed{seed}.json"


def canonical_held_speakers(cfg) -> set | None:
    """Held speaker_ids from the CANONICAL holdout manifest (built once over ALL languages),
    or None if no matching manifest exists.

    One canonical held set shared across arms is what makes a per-language arm's holdout a strict
    SUBSET of the joint champion's holdout: so the two are comparable offline. Re-carving per arm
    (the old path) picks different speakers for lin-only vs joint, which silently leaked and made
    one early arm's holdout a mirage. Ignored if the manifest's holdout_frac disagrees
    with cfg (seed is already pinned in the filename)."""
    path = holdout_manifest_path(cfg.seed)
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    if abs(float(m.get("holdout_frac", -1)) - float(cfg.data.holdout_frac)) > 1e-9:
        return None
    return set(m["held_speakers"])


def load_splits(cfg):
    """Return {'train', 'val', 'holdout'} as HF Datasets with columns [audio, transcription,
    language, speaker_id, gender]. The model adapter owns feature extraction, so this stays
    architecture-agnostic.

    - Concatenates cfg.data.languages (each via its *_asr config) for the train + validation splits.
    - Carves a SPEAKER-DISJOINT holdout from train (group on speaker_id), our Phase-2 proxy.
    - cfg.data.limit (int or null): per-language cap via streaming, for fast LOCAL/smoke runs;
      null = full download. This function never loads the `test` split (see the module docstring).
    """
    import itertools

    import pandas as pd
    from datasets import Audio, Dataset, concatenate_datasets, load_dataset

    hf, configs, sr = cfg.data.hf_dataset, cfg.data.configs, cfg.data.sample_rate
    limit = cfg.data.get("limit")

    def _load(split):                                  # split is "train"/"validation", NEVER "test"
        parts = []
        for lang in cfg.data.languages:                # lang is the bare code: lin / sna / lug
            name = configs[lang]
            if limit:                                  # fast local/smoke: stream one split, cap rows
                stream = load_dataset(hf, name, split=split, streaming=True).cast_column(
                    AUDIO_COLUMN, Audio(sampling_rate=sr))
                parts.append(Dataset.from_list(list(itertools.islice(stream, int(limit)))))
            else:
                # Full run: fetch ONLY this split's parquet shards via explicit data_files, so the
                # builder never prepares/downloads the huge `unlabeled`/`test` splits. That side-download
                # (non-streaming load_dataset prepares ALL splits) hung the job, and pulling `test` would
                # be a leak risk. Repo layout: data/ASR/<code>/<code>-<split>-*.parquet (bare code folder).
                ds = load_dataset(
                    hf, data_files={split: f"data/ASR/{lang}/{lang}-{split}-*.parquet"}, split=split
                ).cast_column(AUDIO_COLUMN, Audio(sampling_rate=sr))
                parts.append(ds)
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    train_full = _load("train")
    val = _load("validation")

    # drop clips outside [min, max] duration, prevents OOM on the long tail (lug 20% >30s, up to 44s)
    # and keeps Whisper (30s cap) from training on truncated audio. HF caches the filtered set across jobs.
    max_s = cfg.data.get("max_duration_s")
    if max_s:
        lo, hi = int(cfg.data.get("min_duration_s", 0.0) * sr), int(max_s * sr)
        n0 = len(train_full)
        train_full = train_full.filter(lambda ex: lo <= len(ex[AUDIO_COLUMN]["array"]) <= hi)
        val = val.filter(lambda ex: lo <= len(ex[AUDIO_COLUMN]["array"]) <= hi)
        print(f"[data] duration filter [{cfg.data.get('min_duration_s', 0)}-{max_s}s]: train {n0} -> {len(train_full)}")

    # OPTIONAL label-noise gate (off unless configured). WAXAL is in-the-wild and a measurable slice of it is
    # MISLABELLED: on validation, 6.5% of clips score CER>=0.5 against gold and 1.1% score >=1.0, and the worst
    # cases are refs that belong to a DIFFERENT clip entirely (verified by inspection, 2026-07-26; concentrated
    # in Lingala). The organisers confirmed the Phase-2 test set will be CLEAN, so training on those pairs
    # teaches noise we will never be scored on.
    # Deliberately MODEL-INDEPENDENT: a transcript that is implausibly long/short for its audio duration is
    # misaligned regardless of any model's opinion. Using model disagreement instead would also delete
    # hard-but-correct clips wherever our model is simply weak: the opposite of what we want.
    # TRAIN ONLY: val/holdout keep every clip, so the eval metric stays comparable with all earlier numbers
    # (and is never flattered by dropping clips our own model happens to fail).
    cps_lo, cps_hi = cfg.data.get("min_chars_per_sec"), cfg.data.get("max_chars_per_sec")
    if cps_lo or cps_hi:
        lo_c, hi_c = float(cps_lo or 0.0), float(cps_hi or 1e9)

        def _plausible(ex):
            dur = len(ex[AUDIO_COLUMN]["array"]) / sr
            if dur <= 0:
                return False
            return lo_c <= len((ex[TEXT_COLUMN] or "").strip()) / dur <= hi_c

        n0 = len(train_full)
        train_full = train_full.filter(_plausible)
        print(f"[data] label-noise gate [{lo_c}-{hi_c} chars/s, TRAIN only]: {n0} -> {len(train_full)} "
              f"({n0 - len(train_full)} dropped)")

    # speaker-disjoint holdout: HARD-FAIL if speaker_id is missing/incomplete (Phase-2 proxy integrity)
    speakers = train_full[SPEAKER_COLUMN]
    if any(s is None or s == "" for s in speakers):
        raise ValueError(f"'{SPEAKER_COLUMN}' has null/empty values, cannot build a speaker-disjoint holdout.")
    uniq = sorted(set(speakers))
    canonical = canonical_held_speakers(cfg)
    if canonical is None:
        # No manifest -> single-arm carve (smoke / back-compat). WARNING: a per-language arm carves a
        # DIFFERENT speaker set than the joint champion under this path, so their holdouts are NOT
        # offline-comparable. Provide results/holdout_speakers_seed<seed>.json (the file read by
        # canonical_held_speakers above) for a canonical, cross-arm-comparable holdout.
        held = set(pd.Series(uniq).sample(frac=cfg.data.holdout_frac, random_state=cfg.seed).tolist())
        print("[data] WARN: no holdout manifest, single-arm carve (not cross-arm comparable).")
    else:
        # This arm holds out exactly the canonical speakers present in its languages -> a strict
        # subset of the joint holdout, so arms are matched.
        held = canonical & set(uniq)
        print(f"[data] canonical holdout manifest: {len(held)}/{len(uniq)} arm speakers held")
    train = train_full.filter(lambda s: s not in held, input_columns=[SPEAKER_COLUMN])
    holdout = train_full.filter(lambda s: s in held, input_columns=[SPEAKER_COLUMN])
    print(f"[data] {len(uniq)} train speakers -> {len(held)} held; "
          f"train={len(train)} val={len(val)} holdout={len(holdout)} rows")

    # Per-language oversampling: TRAIN ONLY (holdout/val stay untouched, so the Phase-2 proxy still
    # measures the real language mix). Motivation: lin is 42.5% of train rows but ~80% of holdout error
    # mass, so the sampling rate is mismatched to where the difficulty actually is. Runs BEFORE the
    # external block, which drops every column but [audio, transcription] (we need `id` for the language).
    oversample = cfg.data.get("oversample")
    if oversample:
        extra = []
        for lang, k in dict(oversample).items():
            reps = int(k) - 1                       # k=2 -> one extra copy (each lin clip seen 2x/epoch)
            if reps <= 0:
                continue
            sub = train.filter(lambda i, L=lang: str(i).startswith(f"{L}_"), input_columns=[ID_COLUMN])
            if not len(sub):
                raise ValueError(f"oversample: no train rows with id prefix '{lang}_', check the language key")
            extra += [sub] * reps
        if extra:
            n0 = len(train)
            train = concatenate_datasets([train, *extra])
            print(f"[data] +oversample {dict(oversample)}: train {n0} -> {len(train)} rows")

    # External auxiliary data (Stage 3): added to TRAIN ONLY so the speaker-disjoint holdout + WAXAL
    # val stay pure (Phase-2 proxy integrity). Reduce train to [audio, transcription] first so schemas
    # match across corpora. Each source is down-weightable via its `max_rows`.
    external = cfg.data.get("external")
    if external:
        train = train.select_columns([AUDIO_COLUMN, TEXT_COLUMN])
        ext = [_load_external(dict(e), sr, limit) for e in external]
        train = concatenate_datasets([train, *ext])
        print(f"[data] +external: {sum(len(e) for e in ext)} rows from {len(external)} source(s) -> train={len(train)}")

    return {"train": train, "val": val, "holdout": holdout}
