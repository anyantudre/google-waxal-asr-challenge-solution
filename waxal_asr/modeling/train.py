"""Training entry point: one arm from one config.

    python -m waxal_asr.modeling.train --config configs/w2vbert_s43.yaml --name s43
    python -m waxal_asr.modeling.train --config configs/w2vbert_p1av.yaml --name p1av train.lr=1e-5

Any trailing ``key=value`` arguments are configuration overrides applied on top of the file, so a
one-off variation needs no new config. ``--name`` decides where the weights land, under
``models/<name>``; it defaults to the config's own stem.

Training resumes automatically from the newest checkpoint in the output directory, which is what
makes a long run survive a scheduler wall clock: submit the same command again and it continues.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from waxal_asr.config import MODELS_DIR, load_config


def main() -> None:
    """Command line entry point."""
    # The HuggingFace Trainer wraps the model in DataParallel when it sees more than one GPU without
    # a torchrun launch, and that breaks variable-length CTC training: each device pads its own
    # sub-batch to a different length, so the gathered tensors do not line up. Pin to one device
    # before torch initialises CUDA. A real multi-GPU launch sets WORLD_SIZE, and is left alone.
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None and not (
        os.environ.get("WORLD_SIZE") or os.environ.get("LOCAL_RANK")
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="path to a YAML under configs/")
    parser.add_argument("--name", default=None, help="output name; defaults to the config stem")
    args, overrides = parser.parse_known_args()

    if not Path(args.config).exists():
        raise SystemExit(f"{args.config} not found")

    from waxal_asr import train as trainer

    cfg = load_config(args.config, overrides)
    name = args.name or Path(args.config).stem
    cfg.train.output_dir = str(MODELS_DIR / name)
    print(f"[train] {args.config} -> {cfg.train.output_dir}")
    if overrides:
        print(f"[train] overrides: {' '.join(overrides)}")
    trainer.run(cfg)


if __name__ == "__main__":
    main()
