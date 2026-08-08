"""Generic, architecture-agnostic training loop.

Wraps HuggingFace Trainer/Seq2SeqTrainer so we inherit robust checkpointing and RESUME
(essential for the remote GPU host's 4h wall-clock: re-submit -> resume from the latest checkpoint).
Nothing here is model-specific; the adapter (waxal.models) supplies the model, collator,
Trainer class, TrainingArguments class, eval-decode + logits hooks. See scripts/train.py.
"""
from __future__ import annotations
from pathlib import Path

from waxal_asr.config import set_seed, save_config, RUNS_DIR
from waxal_asr.models import build_model
from waxal_asr.metrics import score
from waxal_asr.normalize import make_normalizer
from waxal_asr.data import TEXT_COLUMN, load_splits


def _has_checkpoint(output_dir: str) -> bool:
    return any(Path(output_dir).glob("checkpoint-*"))


def run(cfg):
    set_seed(cfg.seed)
    output_dir = cfg.train.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir)

    adapter = build_model(cfg)
    if not adapter.trainable:
        raise RuntimeError(
            f"model.type={cfg.model.type!r} is inference-only, use scripts/infer.py, not train."
        )

    splits = load_splits(cfg)

    # CTC-family bases have no hub tokenizer -> build a char vocab from normalized train+val text
    if getattr(adapter, "needs_vocab", False) and not cfg.model.get("vocab_dir"):
        from waxal_asr.vocab import build_ctc_vocab
        norm = make_normalizer(cfg)
        texts = [norm(t) for t in splits["train"][TEXT_COLUMN]]
        texts += [norm(t) for t in splits["val"][TEXT_COLUMN]]
        build_ctc_vocab(texts, output_dir)
        cfg.model.vocab_dir = output_dir

    # adapters that own their training loop (Gemma: TRL SFT + LoRA) handle everything themselves
    if getattr(adapter, "custom_train", False):
        adapter.load()
        adapter.fit(cfg, splits)
        (Path(output_dir) / ".done").touch()
        print("[train] done (custom) ->", output_dir)
        return

    adapter.load()

    # augment the TRAIN split only (off unless cfg.augment.enabled); never eval. On-the-fly: preprocess
    # caches the RAW waveform (stable Map, no cache-bust) and the collator augments+extracts per batch,
    # so _collate_aug must OUTLIVE the _augment reset below (data_collator() is built later).
    from waxal_asr.audio import build_augmenter
    aug = build_augmenter(cfg)
    adapter._collate_aug = aug              # used by the collator for the raw-waveform (train) path
    adapter._augment = aug                  # tells preprocess to cache RAW waveform for the train split
    train_ds = adapter.preprocess(splits["train"])
    adapter._augment = None                 # eval preprocess bakes features (no aug)

    # cap the in-training eval set (Whisper generation over the full val set would blow the 4h budget).
    # The REAL gate uses the full speaker-disjoint holdout via scripts/evaluate.py; this is monitoring only.
    val_split = splits["val"]
    n_eval = cfg.train.get("eval_max_samples")
    if n_eval and len(val_split) > n_eval:
        val_split = val_split.shuffle(seed=cfg.seed).select(range(n_eval))
    eval_ds = adapter.preprocess(val_split)
    normalizer = make_normalizer(cfg)

    def compute_metrics(pred):
        # pred.predictions are already ids: seq2seq via generate; CTC via preprocess_logits_for_metrics
        label_ids = [[t for t in row if t != -100] for row in pred.label_ids]
        hyps = adapter.decode_preds(pred.predictions)
        refs = adapter.decode_refs(label_ids)
        r = score([normalizer(x) for x in refs], [normalizer(x) for x in hyps])
        return {"wer": r.wer, "cer": r.cer, "score": r.score}

    # HF requires save_steps to be a round multiple of eval_steps when load_best_model_at_end=True.
    # Align save UP to the nearest multiple of eval so checkpoints coincide with evals (this is why the
    # smoke test: which sets eval_steps==save_steps, never caught the base config's 300 vs 400 mismatch).
    import math
    eval_steps = int(cfg.train.eval_steps)
    save_steps = int(cfg.train.save_steps)
    if save_steps % eval_steps != 0:
        aligned = math.ceil(save_steps / eval_steps) * eval_steps
        print(f"[train] save_steps {save_steps} -> {aligned} (must be a multiple of eval_steps={eval_steps} for load_best_model_at_end)")
        save_steps = aligned

    # optionally push a RESUMABLE checkpoint to the HF hub during training (survives a killed
    # preempted session): hub_strategy="checkpoint" writes a 'last-checkpoint/' to the repo each save.
    # Enabled only when cfg.train.hub_model_id is set (unset on local/server -> no behaviour change).
    import os as _os
    hub_args = {}
    if cfg.train.get("hub_model_id"):
        hub_args = dict(push_to_hub=True, hub_model_id=cfg.train.hub_model_id, hub_strategy="checkpoint",
                        hub_private_repo=True,
                        hub_token=_os.environ.get("HF_TOKEN") or _os.environ.get("WAXAL_HF_TOKEN"))

    targs = adapter.training_args_class()(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=float(cfg.train.lr),
        warmup_ratio=cfg.train.warmup_ratio,
        num_train_epochs=cfg.train.num_epochs,
        max_steps=cfg.train.max_steps,
        fp16=cfg.train.get("fp16", False),
        bf16=cfg.train.get("bf16", False),          # prefer bf16 on the RTX 5090 (Whisper-safe)
        auto_find_batch_size=cfg.train.get("auto_find_batch_size", False),  # OOM backstop
        dataloader_num_workers=cfg.train.get("dataloader_num_workers", 0),  # parallelize on-the-fly collator aug/FE
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=False,                    # lower 0.5*WER+0.5*CER is better
        logging_steps=25,
        logging_first_step=True,
        report_to=["tensorboard"],
        logging_dir=str(RUNS_DIR / Path(output_dir).name),   # runs/<name> -> compare all via --logdir runs
        run_name=Path(output_dir).name,
        seed=cfg.seed,
        **hub_args,
        **adapter.training_extra_args(),
    )

    trainer = adapter.trainer_class()(
        model=adapter.model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=adapter.data_collator(),
        compute_metrics=compute_metrics,
        processing_class=adapter.processor,
        preprocess_logits_for_metrics=adapter.preprocess_logits_for_metrics(),
    )

    # resume priority: HF-pushed 'last-checkpoint/' (cross-session) -> local checkpoint-* -> fresh
    resume = False
    if cfg.train.get("resume", True):
        lc = Path(output_dir) / "last-checkpoint"
        if (lc / "trainer_state.json").exists():
            resume = str(lc)
        elif _has_checkpoint(output_dir):
            resume = True
    print(f"[train] {'RESUMING from ' + str(resume) if resume else 'starting fresh'} in {output_dir}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output_dir)  # final best model + processor

    # final metrics of the best model -> final_metrics.json (comparable summary; monitor.py reads it)
    try:
        import json
        m = trainer.evaluate()
        summary = {k.replace("eval_", ""): v for k, v in m.items() if k.startswith("eval_")}
        (Path(output_dir) / "final_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[train] final:", {k: round(v, 4) for k, v in summary.items() if isinstance(v, (int, float))})
    except Exception as e:  # never fail the run over a metrics dump
        print("[train] final eval skipped:", e)

    (Path(output_dir) / ".done").touch()
    print("[train] done ->", output_dir)
