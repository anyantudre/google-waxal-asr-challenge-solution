"""Project paths, model registry and YAML configuration loading.

Paths follow the cookiecutter-data-science layout and can be overridden with environment variables
so the same code runs unchanged on a laptop and on a batch node.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

PROJ_ROOT = Path(os.environ.get("WAXAL_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = Path(os.environ.get("WAXAL_MODELS_DIR", PROJ_ROOT / "models"))
REPORTS_DIR = PROJ_ROOT / "reports"
CONFIGS_DIR = PROJ_ROOT / "configs"

# Training-side outputs. RESULTS_DIR holds the speaker-disjoint holdout manifest and any evaluation
# artefacts; RUNS_DIR collects one TensorBoard subdirectory per run, so several arms can be compared
# with a single `tensorboard --logdir runs`. Both are created on demand.
RESULTS_DIR = PROJ_ROOT / "results"
RUNS_DIR = PROJ_ROOT / "runs"

# The nine published checkpoints, and where their weights live on the Hugging Face Hub. Setting
# WAXAL_MODELS_DIR makes the code load from disk instead, which is what the batch jobs do.
#
# Repository names follow waxal-<architecture>-<languages>-<variant>, so the languages a checkpoint
# covers are readable from its name: `lin` and `sna` are single-language specialists, `linsna` and
# `linsnalug` are multilingual. The short keys on the left are the internal arm names used by the
# configs, the recipes and the cache filenames.
HF_NAMESPACE = os.environ.get("WAXAL_HF_NAMESPACE", "anyantudre")

ENSEMBLE_ARMS = {
    "s43": f"{HF_NAMESPACE}/waxal-w2vbert-linsna-seed43",
    "s44": f"{HF_NAMESPACE}/waxal-w2vbert-linsna-seed44",
    "soup5": f"{HF_NAMESPACE}/waxal-w2vbert-linsna-soup5",
    "p1raw": f"{HF_NAMESPACE}/waxal-w2vbert-linsnalug-raw",
    "linspec_r": f"{HF_NAMESPACE}/waxal-w2vbert-lin-specialist",
    "snaspec_r": f"{HF_NAMESPACE}/waxal-w2vbert-sna-specialist",
    "p1av": f"{HF_NAMESPACE}/waxal-w2vbert-linsna-afrivoicemix",
    "distil": f"{HF_NAMESPACE}/waxal-w2vbert-linsna-distilled",
    "turbo_linsna_r": f"{HF_NAMESPACE}/waxal-whisper-turbo-linsna",
    "s46": f"{HF_NAMESPACE}/waxal-w2vbert-linsna-seed46",
    "soup6": f"{HF_NAMESPACE}/waxal-w2vbert-linsna-soup6",
}

# Third-party models used without fine-tuning. Sunbird-51 is pinned because its card states the
# weights will be replaced; an unpinned load would silently change the submission.
SUNBIRD_51 = "Sunbird/asr-whisper-51-african-languages"
SUNBIRD_51_REVISION = "5d4f0038"
LID_MODEL = "facebook/mms-lid-4017"

# Blank penalties used to build the ensemble members, chosen to bracket the reference word rate.
BLANK_PENALTIES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)

LANGUAGES = ("lin", "sna")


def resolve_model(name: str) -> str:
    """Return a local directory if one exists under MODELS_DIR, otherwise the Hub id.

    This is what lets the same command run offline on a machine that has already downloaded the
    weights, and online on a fresh one.
    """
    local = MODELS_DIR / name
    if (local / "config.json").exists():
        return str(local)
    if name in ENSEMBLE_ARMS:
        return ENSEMBLE_ARMS[name]
    return name


def save_config(cfg, output_dir: str | os.PathLike) -> None:
    """Write the fully resolved config beside the weights it produced.

    Every checkpoint directory therefore carries the exact settings that trained it, including any
    command line overrides, which is what makes a run reconstructable months later.
    """
    from omegaconf import OmegaConf

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(output_dir) / "config.yaml")


def set_seed(seed: int = 42) -> None:
    """Seed every source of randomness that affects training or decoding."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | Path, overrides: list[str] | None = None):
    """Load a YAML config, merge it over configs/base.yaml, and apply dotted CLI overrides.

    An override looks like `train.lr=1e-5` or `data.languages=[lin,sna]`.
    """
    import yaml

    def _merge(base: dict, extra: dict) -> dict:
        out = dict(base)
        for k, v in extra.items():
            out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    path = Path(path)
    cfg: dict = {}
    base = CONFIGS_DIR / "base.yaml"
    if base.exists() and path.resolve() != base.resolve():
        cfg = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
    cfg = _merge(cfg, yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    for item in overrides or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return _Namespace(cfg)


class _Namespace(dict):
    """Dictionary with attribute access, so configs read as `cfg.train.lr`."""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return _Namespace(value) if isinstance(value, dict) else value

    def get(self, item, default=None):
        value = super().get(item, default)
        return _Namespace(value) if isinstance(value, dict) else value
