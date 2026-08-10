"""w2v-BERT 2.0 adapter (CTC): facebook/w2v-bert-2.0 (MIT), our best-Shona base.

Differs from wav2vec2/XLS-R and so needs its own adapter (not the generic CTC one):
  * uses `input_features` (mel), NOT `input_values` (waveform);
  * processor = Wav2Vec2BertProcessor wrapping a SeamlessM4TFeatureExtractor;
  * freeze via freeze_base_model() (there is no freeze_feature_encoder here);
  * add_adapter downsamples 2x -> guard against too-short clips (ctc_zero_infinity).
Shares the char-vocab build, decode hooks, and KenLM path with CTCAdapter.
"""
from __future__ import annotations

from dataclasses import dataclass

from waxal_asr.data import TEXT_COLUMN
from waxal_asr.models import register
from waxal_asr.models.ctc import CTCAdapter
from waxal_asr.normalize import make_normalizer


@register("w2v_bert")
class W2VBertAdapter(CTCAdapter):
    needs_vocab = True

    def load(self):
        from transformers import (
            SeamlessM4TFeatureExtractor,
            Wav2Vec2BertForCTC,
            Wav2Vec2BertProcessor,
        )

        from waxal_asr.vocab import build_ctc_tokenizer

        vocab_dir = self.cfg.model.get("vocab_dir") or self.cfg.train.output_dir
        tokenizer = build_ctc_tokenizer(vocab_dir)
        fe = SeamlessM4TFeatureExtractor.from_pretrained(self.cfg.model.id)
        self.processor = Wav2Vec2BertProcessor(feature_extractor=fe, tokenizer=tokenizer)
        self.model = Wav2Vec2BertForCTC.from_pretrained(
            self.cfg.model.id,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            add_adapter=self.cfg.model.get("add_adapter", True),
            # layerdrop MUST be passed IN, not set afterwards. Wav2Vec2BertAdapter.__init__ CAPTURES
            # `self.layerdrop = config.layerdrop` at construction and its forward uses that copy, so a
            # post-hoc `config.layerdrop = 0.0` reaches the encoder (which re-reads config live) but
            # NOT the adapter: which kept the hub's 0.1. With add_adapter=true and num_adapter_layers=1,
            # that dropped the ONLY adapter layer on ~10% of training forward passes. That layer does the
            # stride-2 downsample, yet Wav2Vec2BertForCTC always computes input_lengths as the DOWNSAMPLED
            # count -> ctc_loss saw full-rate frames but only the first T/2 of them, i.e. the model was
            # trained to emit the WHOLE transcript from the FIRST HALF of the audio. ~27% of optimizer
            # steps at grad_accum=3. Verified against transformers 4.57.6 (adapter :559 captures, :583 uses).
            layerdrop=0.0,
            ignore_mismatched_sizes=True,
        )
        if self.cfg.model.get("freeze_base"):
            self.model.freeze_base_model()
        self.model.config.layerdrop = 0.0                      # encoder reads this live
        _adapter = getattr(self.model.wav2vec2_bert, "adapter", None)
        assert _adapter is None or _adapter.layerdrop == 0.0, (
            f"adapter layerdrop is {_adapter.layerdrop}, not 0.0, the CTC length bookkeeping does not "
            "follow a dropped adapter layer; training would be corrupted on those batches."
        )
        if self.cfg.model.get("gradient_checkpointing"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.cfg.train.get("output_dir"):                       # None at eval/infer with the base config
            self.processor.save_pretrained(self.cfg.train.output_dir)
        return self

    def preprocess(self, dataset):
        proc, norm = self.processor, make_normalizer(self.cfg)
        aug = getattr(self, "_augment", None)   # set by train.run for the TRAIN split of an aug arm only

        if aug is None:
            # eval, or any non-aug arm: bake the (expensive) mel features into the cache. UNCHANGED.
            def _prep(batch):
                feats = proc(batch["audio"]["array"], sampling_rate=16000).input_features[0]
                batch["input_features"] = feats
                batch["input_length"] = len(feats)
                batch["labels"] = proc(text=norm(batch[TEXT_COLUMN])).input_ids
                return batch
            return dataset.map(_prep, remove_columns=dataset.column_names,
                               num_proc=self.cfg.data.get("map_num_proc"),
                               keep_in_memory=self.cfg.data.get("map_in_memory", False))

        # aug arm (train split): cache the RAW waveform only (aug not in the fingerprint -> the expensive
        # mel Map runs ONCE and never busts), and augment + mel-extract fresh per batch in the collator
        # (per-epoch aug). input_length (raw samples) is a valid group_by_length proxy (monotonic in mel len).
        def _prep_raw(batch):
            from waxal_asr.audio import to_int16
            wav = to_int16(batch["audio"]["array"])   # int16 numpy: half the RAM of float32, lossless at 16k
                                                      # (NOT list() -> 8x RAM -> map OOM)
            batch["waveform"] = wav
            batch["input_length"] = len(wav)
            batch["labels"] = proc(text=norm(batch[TEXT_COLUMN])).input_ids
            return batch
        return dataset.map(_prep_raw, remove_columns=dataset.column_names,
                           num_proc=self.cfg.data.get("map_num_proc"),
                           keep_in_memory=self.cfg.data.get("map_in_memory", False))

    def data_collator(self):
        return _W2VBertCollator(self.processor, getattr(self, "_collate_aug", None))


@dataclass
class _W2VBertCollator:
    processor: object
    augmenter: object = None   # on-the-fly aug for the raw-waveform (train) path; None on the eval path

    def __call__(self, features):
        if "waveform" in features[0]:            # aug train path: augment + mel-extract per batch
            from waxal_asr.audio import from_int16, maybe_augment
            input_features = [
                {"input_features": self.processor(maybe_augment(self.augmenter, from_int16(f["waveform"])),
                                                  sampling_rate=16000).input_features[0]}
                for f in features
            ]
        else:                                    # eval / no-aug: mel features already extracted in .map
            input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")
        batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        return batch
