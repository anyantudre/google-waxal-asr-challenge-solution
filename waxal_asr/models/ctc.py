"""CTC adapter: wav2vec2 / XLS-R / any AutoModelForCTC.

CTC bases (XLS-R, wav2vec2) have NO tokenizer on the hub, so we build a char vocab
from the normalized train+val transcripts (waxal_asr.vocab), train.run does this before
load() for any adapter with `needs_vocab=True`. Char-level directly optimizes CER and
lets the model emit ', -, ŋ, and Lingala accents (which the vocab force-includes).

Inference is greedy by default. Set `cfg.decode.ctc_beam_width > 1` for pyctcdecode beam search,
and `cfg.decode.lm` to a KenLM .arpa/.bin (a single path, or a {lin/sna/lug: path} map for
per-language LMs) for LM-boosted decoding. KenLM is Linux/WSL2-only; the path falls back to
greedy if pyctcdecode/kenlm can't be imported or the LM can't be built.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from waxal_asr.data import TEXT_COLUMN
from waxal_asr.models import register
from waxal_asr.models.base import ASRAdapter
from waxal_asr.normalize import make_normalizer


@register("ctc")
class CTCAdapter(ASRAdapter):
    needs_vocab = True  # build a char vocab from the corpus before load()

    def load(self):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor

        from waxal_asr.vocab import build_ctc_tokenizer

        vocab_dir = self.cfg.model.get("vocab_dir") or self.cfg.train.output_dir
        tokenizer = build_ctc_tokenizer(vocab_dir)
        fe = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=True,  # attn mask REQUIRED for XLS-R
        )
        self.processor = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tokenizer)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.cfg.model.id,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,     # CTC blank
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,                  # avoid inf loss on too-short clips
            ignore_mismatched_sizes=True,            # lm_head is re-init to our vocab
        )
        self.model.freeze_feature_encoder()          # canonical wav2vec2 recipe (freezes the CNN)
        self.model.config.layerdrop = 0.0            # determinism (seed-everything rule)
        if self.cfg.model.get("gradient_checkpointing"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.cfg.train.get("output_dir"):                       # None at eval/infer with the base config
            self.processor.save_pretrained(self.cfg.train.output_dir)  # resume/infer reload same vocab
        return self

    def preprocess(self, dataset):
        proc, norm = self.processor, make_normalizer(self.cfg)
        aug = getattr(self, "_augment", None)   # set by train.run for the TRAIN split of an aug arm only

        if aug is None:
            # eval, or any non-aug arm: bake features into the cache (one-time, stable). UNCHANGED.
            def _prep(batch):
                batch["input_values"] = proc(batch["audio"]["array"], sampling_rate=16000).input_values[0]
                batch["input_length"] = len(batch["input_values"])   # enables group_by_length
                batch["labels"] = proc(text=norm(batch[TEXT_COLUMN])).input_ids
                return batch
            return dataset.map(_prep, remove_columns=dataset.column_names,
                               num_proc=self.cfg.data.get("map_num_proc"),
                               keep_in_memory=self.cfg.data.get("map_in_memory", False))

        # aug arm (train split): cache the RAW waveform only, aug is NOT in the fingerprint, so the Map
        # caches ONCE and never busts (fixes the ~40G re-Map churn), and the collator augments + extracts
        # features fresh per batch => per-epoch augmentation. input_length (raw samples) is a valid
        # group_by_length proxy (monotonic in the feature length).
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
        return _CTCCollator(self.processor, getattr(self, "_collate_aug", None))

    def decode_preds(self, pred_ids):
        # CTC greedy: default group_tokens=True collapses repeats+blanks
        return self.processor.batch_decode(pred_ids)

    def decode_refs(self, label_ids):
        # references are plain char sequences: do NOT CTC-collapse them
        return self.processor.batch_decode(label_ids, group_tokens=False)

    def training_extra_args(self):
        if bool(self.cfg.get("augment") and self.cfg.augment.get("enabled")):
            # On-the-fly aug caches raw 'waveform' + 'input_length' (no model-signature columns). Keep them
            # (remove_unused_columns=False), or the Trainer strips both -> the collator can't find 'waveform'
            # AND the length sampler can't find 'input_length'. length_column_name then feeds group_by_length.
            return {"group_by_length": True, "length_column_name": "input_length",
                    "remove_unused_columns": False}
        return {"group_by_length": True}   # baked path: unchanged (sampler infers length from input_values)

    def preprocess_logits_for_metrics(self):
        # argmax on-GPU so eval doesn't accumulate (batch x time x vocab) logits -> OOM
        def _fn(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)
        return _fn

    def transcribe(self, batch_audio, lang=None):
        import torch

        proc, model = self.processor, self.model
        inputs = proc(batch_audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        dec = self._beam_decoder(lang)
        if dec is None:                                     # greedy, the default, works everywhere
            return proc.batch_decode(logits.argmax(dim=-1))
        decoder, beam_width = dec
        logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        return [decoder.decode(logp[i], beam_width=beam_width).strip() for i in range(logp.shape[0])]

    def _pyctc_labels(self):
        """Vocab tokens in index order, mapped to pyctcdecode conventions:
        the CTC blank (pad) -> '' and the word-delimiter ('|') -> ' '."""
        tok = self.processor.tokenizer
        toks = [t for t, _ in sorted(tok.get_vocab().items(), key=lambda kv: kv[1])]
        return ["" if t == tok.pad_token else " " if t == tok.word_delimiter_token else t for t in toks]

    def _beam_decoder(self, lang=None):
        """Cached (pyctcdecode decoder, beam_width) for `lang`, or None to signal greedy.
        Returns None whenever beam/LM isn't configured, OR pyctcdecode/kenlm can't be imported, OR the
        LM can't be built: so CTC always degrades to greedy (keeps Windows/local runs working)."""
        dcfg = self.cfg.get("decode") or {}
        lm = dcfg.get("lm")
        lm_path = None if lm is None else (lm if isinstance(lm, str) else (lm.get(lang) if lang else None))
        beam_width = int(dcfg.get("ctc_beam_width", 1) or 1)
        if not lm_path and beam_width <= 1:
            return None                                     # nothing configured -> greedy
        cache = self.__dict__.setdefault("_dec_cache", {})
        key = str(lm_path) if lm_path else f"__beam{beam_width}"
        if key not in cache:
            cache[key] = self._build_decoder(lm_path, beam_width if beam_width > 1 else 100, dcfg)
        return cache[key]

    @staticmethod
    def _lm_unigrams(lm_path):
        """Word list for pyctcdecode, read from `<lm_stem>.unigrams.txt` next to the LM binary.

        Without it pyctcdecode logs "No known unigrams provided, decoding results might be a lot
        worse" and builds NO char trie, so it cannot prune implausible partial words and skips the
        unk_score_offset branch entirely. It only auto-extracts unigrams from .arpa, never from a
        .bin, which is why a KenLM binary needs the sidecar word list.
        """
        if not lm_path:
            return None
        uni = Path(str(lm_path)).with_suffix(".unigrams.txt")
        if not uni.exists():
            print(f"[ctc] WARN: {uni} missing, pyctcdecode will decode WITHOUT a char trie "
                  f"(degraded word handling). Provide the word list alongside the LM.")
            return None
        return [w for w in uni.read_text(encoding="utf-8").splitlines() if w]

    def _build_decoder(self, lm_path, beam_width, dcfg):
        try:
            from pyctcdecode import build_ctcdecoder
            unigrams = self._lm_unigrams(lm_path)
            decoder = build_ctcdecoder(
                self._pyctc_labels(),
                kenlm_model_path=str(lm_path) if lm_path else None,
                unigrams=unigrams,
                alpha=float(dcfg.get("lm_alpha", 0.5)),
                beta=float(dcfg.get("lm_beta", 1.5)),
            )
            print(f"[ctc] beam decode ON (beam_width={beam_width}, lm={lm_path or 'none'}, "
                  f"unigrams={len(unigrams) if unigrams else 0})")
            return (decoder, beam_width)
        except Exception as e:  # pyctcdecode/kenlm missing, or unreadable LM -> greedy
            if not getattr(self, "_warned_beam", False):
                print(f"[ctc] beam/LM decode unavailable ({type(e).__name__}: {e}) -> greedy fallback")
                self._warned_beam = True
            return None


@dataclass
class _CTCCollator:
    processor: object
    augmenter: object = None   # on-the-fly aug for the raw-waveform (train) path; None on the eval path

    def __call__(self, features):
        if "waveform" in features[0]:            # aug train path: augment + feature-extract per batch
            from waxal_asr.audio import from_int16, maybe_augment
            input_features = [
                {"input_values": self.processor(maybe_augment(self.augmenter, from_int16(f["waveform"])),
                                                sampling_rate=16000).input_values[0]}
                for f in features
            ]
        else:                                    # eval / no-aug: features already extracted in .map
            input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")
        # -100 = ignore index for CTC loss on padded label positions
        batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        return batch
