"""Seq2seq adapter: Whisper / any AutoModelForSpeechSeq2Seq.

Encoder-decoder ASR: log-mel in, token sequence out, cross-entropy loss, beam/greedy
generation. Whisper specifics handled here (verified against current transformers):
  * language/task set on generation_config; legacy forced_decoder_ids nulled (deprecated).
  * Luganda is NOT in Whisper's 99 languages -> use a proxy token (cfg.model.language,
    default 'sw' Swahili) for the joint lin/sna/lug run.
  * collator strips the prepended decoder-start token from labels (else labels mis-shift).
  * uses Seq2SeqTrainer + Seq2SeqTrainingArguments (predict_with_generate).
"""
from __future__ import annotations

from dataclasses import dataclass

from waxal_asr.data import TEXT_COLUMN
from waxal_asr.models import register
from waxal_asr.models.base import ASRAdapter
from waxal_asr.normalize import make_normalizer


@register("seq2seq")
class Seq2SeqAdapter(ASRAdapter):
    def load(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        lang = self.cfg.model.get("language")          # e.g. 'sw' proxy; None -> multilingual auto
        task = self.cfg.model.get("task", "transcribe")
        # only pass language/task for Whisper-style processors (keeps other seq2seq models safe)
        proc_kwargs = {"language": lang, "task": task} if lang else {}
        # Sunbird/asr-whisper-51-african-languages stores `extra_special_tokens` as a LIST in its
        # tokenizer_config.json, but transformers calls .keys() on it -> "AttributeError: 'list' object
        # has no attribute 'keys'". Passing an explicit dict overrides the broken field. Harmless for
        # every other checkpoint, so it is unconditional (with a fallback for older transformers that
        # do not accept the kwarg).
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.cfg.model.id, extra_special_tokens={}, **proc_kwargs)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(self.cfg.model.id, **proc_kwargs)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.cfg.model.id, revision=self.cfg.model.get("revision"))  # 4.x loads fp32 by default
        # modern language/task API: forced_decoder_ids is deprecated and must be nulled
        if lang:
            self.model.generation_config.language = lang
            self.model.generation_config.task = task
        self.model.generation_config.forced_decoder_ids = None
        self.model.config.use_cache = False            # mutually exclusive w/ grad checkpointing
        if self.cfg.model.get("gradient_checkpointing"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return self

    # Sunbird's 51-language Whisper does NOT add new tokens: it REMAPS unused Whisper language slots and
    # retrains the embedding there. So `<|ach|>` does not exist (it resolves to <|endoftext|>), and the
    # only way to address Acholi is the slot Whisper calls Sundanese. Verified against the loaded
    # tokenizer: ach->50357, lug->50332, nyn->50322, sog/xog->50310, luy->50330, and lin/sna use their
    # GENUINE Whisper tokens (50353/50324). Passing the wrong name here trains the wrong language head.
    SUNBIRD_LANG_NAME = {
        "ach": "sundanese", "lug": "sindhi", "nyn": "sinhala", "sog": "basque",
        "xog": "basque", "luy": "belarusian", "lin": "lingala", "sna": "shona",
        # mas = WAXAL's Masaaba (myx). Sunbird slot 50329 = stock <|ka|>; verified live 2026-08-02.
        "mas": "georgian",
    }

    def _labels_for(self, proc, text, row):
        """Tokenize with the LANGUAGE token this row actually needs.

        Whisper labels are [<|startoftranscript|>, <LANG>, <|transcribe|>, <|notimestamps|>, ...]. The
        default call emits NO language token at all, and one global cfg.model.language would train every
        language under a single token while inference forces a different one, silent, and it would look
        like a merely mediocre model rather than a broken one.
        """
        if self.cfg.model.get("per_example_language"):
            lang = row.get("lang") or str(row.get("id", "")).split("_")[0]
            name = self.SUNBIRD_LANG_NAME.get(lang)
            if name:
                # tokenizer prefix state is global, so this REQUIRES a serial map (map_num_proc: null);
                # with workers the state would race across processes.
                proc.tokenizer.set_prefix_tokens(language=name, task="transcribe",
                                                 predict_timestamps=False)
        return proc(text=text).input_ids

    def preprocess(self, dataset):
        proc, norm = self.processor, make_normalizer(self.cfg)
        aug = getattr(self, "_augment", None)   # set by train.run for the train split only
        if self.cfg.model.get("per_example_language") and self.cfg.data.get("map_num_proc"):
            raise ValueError("per_example_language needs map_num_proc: null, the tokenizer's prefix "
                             "state is global and would race across map workers.")

        if aug is None:
            # eval / non-aug arm: bake the mel features. UNCHANGED behaviour.
            def _prep(batch):
                batch["input_features"] = proc(batch["audio"]["array"], sampling_rate=16000).input_features[0]
                batch["labels"] = self._labels_for(proc, norm(batch[TEXT_COLUMN]), batch)
                return batch
            return dataset.map(_prep, remove_columns=dataset.column_names,
                               num_proc=self.cfg.data.get("map_num_proc"),
                               keep_in_memory=self.cfg.data.get("map_in_memory", False))

        # aug arm (train split): cache the RAW waveform only, augment + mel-extract fresh per batch in the
        # collator. Two reasons this is REQUIRED for Whisper, not just nice-to-have:
        #  1. size, Whisper zero-pads every clip to 30s, so a baked mel is a FIXED ~1.5MB/row (128x3000)
        #     regardless of clip length; 41k rows = ~64G of cache. The raw waveform is ~21G instead.
        #  2. correctness, baking applies ONE fixed augmentation per clip for the whole run (the c3 bug);
        #     on-the-fly gives fresh per-epoch aug, which is the entire point of multi-condition training.
        def _prep_raw(batch):
            from waxal_asr.audio import to_int16
            wav = to_int16(batch["audio"]["array"])    # int16: half the RAM of float32, lossless at 16k
            batch["waveform"] = wav
            batch["labels"] = self._labels_for(proc, norm(batch[TEXT_COLUMN]), batch)
            return batch
        return dataset.map(_prep_raw, remove_columns=dataset.column_names,
                           num_proc=self.cfg.data.get("map_num_proc"),
                           keep_in_memory=self.cfg.data.get("map_in_memory", False))

    def data_collator(self):
        return _Seq2SeqCollator(self.processor, self.model.config.decoder_start_token_id,
                                getattr(self, "_collate_aug", None))

    def trainer_class(self):
        from transformers import Seq2SeqTrainer
        return Seq2SeqTrainer

    def training_args_class(self):
        from transformers import Seq2SeqTrainingArguments
        return Seq2SeqTrainingArguments

    def training_extra_args(self):
        args = {"predict_with_generate": True, "generation_num_beams": self.cfg.decode.num_beams}
        if getattr(self, "_collate_aug", None) is not None:
            args["remove_unused_columns"] = False   # keep `waveform` for the on-the-fly collator
        return args

    def transcribe(self, batch_audio):
        import torch

        proc, model = self.processor, self.model
        model.config.use_cache = True
        feats = proc(batch_audio, sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
        with torch.no_grad():
            gen = model.generate(feats, num_beams=self.cfg.decode.num_beams,
                                 max_new_tokens=self.cfg.decode.get("max_new_tokens", 225))
        return proc.batch_decode(gen, skip_special_tokens=True)


@dataclass
class _Seq2SeqCollator:
    processor: object
    decoder_start_token_id: int
    augmenter: object = None                 # set for aug arms -> augment + mel-extract per batch

    def __call__(self, features):
        if "waveform" in features[0]:        # on-the-fly path: fresh aug + mel every epoch
            from waxal_asr.audio import from_int16, maybe_augment
            wavs = [maybe_augment(self.augmenter, from_int16(f["waveform"])) for f in features]
            batch = self.processor.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt")
        else:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # cut the BOS/decoder-start token the tokenizer prepends: the model adds it back
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
