"""The single interface every architecture implements.

train.py and infer.py depend ONLY on this, that's what makes models swappable by a
one-line config change. An adapter owns: the HF model + processor, how audio+text
become model inputs (preprocessing / collator), how the Trainer is configured, and how
to decode to text.

Two axes differ across architectures and are exposed as hooks so the generic training
loop stays model-agnostic:
  * trainer_class() / training_args_class(), CTC uses Trainer/TrainingArguments;
    seq2seq (Whisper) needs Seq2SeqTrainer/Seq2SeqTrainingArguments (predict_with_generate).
  * preprocess_logits_for_metrics(), CTC must argmax logits on-GPU before
    metric accumulation (else eval OOMs on big logits); seq2seq returns generated ids.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ASRAdapter(ABC):
    #: inference-only adapters (e.g. Omnilingual/fairseq2) set this False so train.py refuses to train them
    trainable: bool = True
    #: adapters that own their whole training loop (e.g. Gemma via TRL SFT + LoRA) set this True
    #: and implement fit(cfg, splits); train.run delegates to it instead of the generic HF-Trainer path
    custom_train: bool = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> "ASRAdapter":
        """Load processor + model from cfg.model.id. Return self."""

    @abstractmethod
    def preprocess(self, dataset):
        """Map a raw {audio, transcription} HF dataset to model-ready tensors/features."""

    @abstractmethod
    def data_collator(self):
        """Return a collator that pads a batch appropriately for this architecture."""

    @abstractmethod
    def transcribe(self, batch_audio) -> list[str]:
        """Run inference on a list of audio arrays -> list of raw transcripts."""

    # Evaluation decoding. The default is a sequence-to-sequence token decode; CTC overrides it.
    def decode_preds(self, pred_ids) -> list[str]:
        return self.processor.batch_decode(pred_ids, skip_special_tokens=True)

    def decode_refs(self, label_ids) -> list[str]:
        return self.processor.batch_decode(label_ids, skip_special_tokens=True)

    # Training hooks. The defaults suit CTC; the sequence-to-sequence adapter overrides them.
    def trainer_class(self):
        from transformers import Trainer
        return Trainer

    def training_args_class(self):
        from transformers import TrainingArguments
        return TrainingArguments

    def training_extra_args(self) -> dict:
        """Extra kwargs for the *_TrainingArguments (e.g. predict_with_generate, group_by_length)."""
        return {}

    def preprocess_logits_for_metrics(self):
        """Return a fn(logits, labels)->ids to shrink eval memory, or None (seq2seq)."""
        return None
