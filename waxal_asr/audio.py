"""Audio utilities + augmentation for Phase-2 robustness (speaker/recording generalization).

Config-driven (cfg.augment), OFF by default, applied to the TRAIN split ONLY, never to
eval/test. Uses audiomentations (MIT). We augment the raw 16 kHz mono array before feature
extraction, so it works for every adapter (waveform-in).

NOTE: augmentation runs on the fly in the data collator: the dataset map caches the raw
waveform once (augmentation is not part of the cache fingerprint, so the map never busts),
and the collator augments and re-extracts features per batch, giving fresh randomness every
epoch. MP3 caveat: WAXAL audio is ALREADY MP3, so Mp3Compression models
bitrate/transcode variation at LOW probability, not a naive double-encode.
Open noise/RIR corpora: MUSAN (OpenSLR SLR17, CC-BY), RIR (OpenSLR SLR28, Apache).
"""
from __future__ import annotations

import numpy as np


def ensure_mono_16k(wav: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """Downmix to mono and resample to target_sr (librosa imported lazily)."""
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def build_augmenter(cfg):
    """Return an audiomentations.Compose from cfg.augment, or None if disabled."""
    a = cfg.augment
    if not a.get("enabled"):
        return None
    from audiomentations import (
        AddBackgroundNoise,
        AddGaussianSNR,
        AirAbsorption,
        ApplyImpulseResponse,
        Compose,
        Gain,
        Mp3Compression,
        PitchShift,
        TimeStretch,
    )
    transforms = []

    def _path(p):
        """Expand ~ and $VARS so configs stay machine-independent and carry no absolute paths."""
        import os
        return os.path.expanduser(os.path.expandvars(str(p)))

    # order = room -> channel -> codec
    if a.get("rir_dir"):
        transforms.append(ApplyImpulseResponse(ir_path=_path(a.rir_dir), p=a.get("p_rir", 0.3)))
    if a.get("noise_dir"):
        transforms.append(AddBackgroundNoise(
            sounds_path=_path(a.noise_dir), min_snr_db=a.get("noise_min_snr", 5.0),
            max_snr_db=a.get("noise_max_snr", 25.0), p=a.get("p_noise", 0.5)))
    transforms += [
        AddGaussianSNR(min_snr_db=10.0, max_snr_db=40.0, p=a.get("p_gaussian", 0.2)),
        Gain(min_gain_db=-8.0, max_gain_db=8.0, p=a.get("p_gain", 0.4)),
        PitchShift(min_semitones=-2, max_semitones=2, p=a.get("p_pitch", 0.25)),
        TimeStretch(min_rate=0.9, max_rate=1.1, leave_length_unchanged=True, p=a.get("p_speed", 0.25)),
        AirAbsorption(p=a.get("p_air", 0.15)),
    ]
    # Mp3Compression needs the optional `fast_mp3_augment` backend (audiomentations>=0.40). WAXAL audio is
    # already MP3 so this is low-value: include it only if requested AND the backend is importable, else
    # skip so a missing optional dep never breaks training on any of the 3 environments.
    if a.get("p_mp3", 0) > 0:
        try:
            import fast_mp3_augment  # noqa: F401
            transforms.append(Mp3Compression(min_bitrate=24, max_bitrate=64, p=a.p_mp3))
        except Exception as e:
            print(f"[audio] MP3 augmentation unavailable ({type(e).__name__}), skipping (WAXAL is already MP3)")
    return Compose(transforms)


def maybe_augment(augmenter, array) -> np.ndarray:
    """Apply the augmenter to a mono 16 kHz array if provided; else return it unchanged."""
    arr = np.asarray(array, dtype=np.float32)
    if augmenter is None:
        return arr
    return augmenter(samples=arr, sample_rate=16000)


# Int16 waveform caching
# The on-the-fly aug path caches the RAW waveform in the Map. Stored as float32 that is ~1.7MB/row, which
# capped `map_in_memory` runs at ~45k rows on the 90G node (77k rows -> OUT_OF_MEMORY, measured 2026-07-25).
# 16-bit PCM is the native precision of this corpus, so int16 is lossless here and HALVES the footprint,
# roughly doubling how much data a run can hold in RAM.

def to_int16(wav: np.ndarray) -> np.ndarray:
    """float32 [-1,1] -> int16, for compact caching."""
    return np.clip(np.asarray(wav, dtype=np.float32) * 32767.0, -32768, 32767).astype(np.int16)


def from_int16(wav) -> np.ndarray:
    """Cached waveform -> float32 [-1,1]. Accepts already-float input unchanged (eval paths)."""
    arr = np.asarray(wav)
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32767.0
    return arr.astype(np.float32, copy=False)


def load_clip(clip_id, audio_dir, sample_rate: int = 16000):
    """Load one `<clip_id>.<ext>` file from a directory as mono audio at `sample_rate`.

    The extension is discovered rather than assumed: the competition shipped WAV for the corrected
    test set and MP3 earlier, and the corpora we build are FLAC.
    """
    import glob
    from pathlib import Path

    import soundfile as sf

    # glob.escape guards ids containing a glob metacharacter; sorted() makes the choice
    # deterministic if an id somehow has more than one file.
    matches = sorted(glob.glob(glob.escape(str(Path(audio_dir) / str(clip_id))) + ".*"))
    if not matches:
        raise FileNotFoundError(f"no audio file for id {clip_id!r} in {audio_dir}")
    wave, rate = sf.read(matches[0], dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=-1)
    if rate != sample_rate:
        import librosa

        wave = librosa.resample(wave, orig_sr=rate, target_sr=sample_rate)
    return wave
