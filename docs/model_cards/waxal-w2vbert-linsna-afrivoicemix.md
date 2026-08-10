---
language:
- ln
- sn
license: apache-2.0
tags:
- automatic-speech-recognition
- ctc
- waxal
- lingala
- shona
base_model: facebook/w2v-bert-2.0
---

# waxal-w2vbert-linsna-afrivoicemix

This model was developed for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) on Zindi.

`waxal-w2vbert-linsna-afrivoicemix` continues the `waxal-w2vbert-linsnalug-raw` checkpoint, a
fine-tune of [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0), on a single mixed
corpus: WAXAL Lingala and Shona gold together with both AfriVoice corpora, present in every batch for
the whole run. It is the deliberate counter-experiment to the two-stage specialists, which see exactly
the same external audio but sequenced, AfriVoice first and WAXAL gold last. Because the ordering is the
only intended difference, this arm is what turned a plausible curriculum heuristic into a measured
result, and the result was negative for mixing: 0.2905 on the holdout for mixing throughout, against
0.2785 for the sequenced Lingala specialist and 0.1616 for the sequenced Shona specialist. It is kept
in the submission's core ensembles as a decorrelated member, not as a strong solo model. It also
inherits the parent
checkpoint's raw character vocabulary unchanged, capitals and the punctuation CER scores included.

## Training data

| source | rows | hours | licence |
|---|---|---|---|
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona train and validation | 33,002 | about 179 | CC-BY-4.0 |
| [KasuleTrevor/Lingala_100hrs](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs), AfriVoice Lingala | 22,131 | 104.5 | CC-BY-4.0 |
| [realtime-speech/shona1](https://huggingface.co/datasets/realtime-speech/shona1), AfriVoice Shona | 15,923 | 91.0 | CC-BY-4.0 |

All three sources are mixed into one training set and there is no stage boundary. Luganda is dropped
from this arm because the corrected Phase 2 test set was measured to contain none. Rows and hours above
are per-source totals from the corpus build; the speaker-disjoint holdout and the validation rows are
removed before training, and the duration and characters-per-second filters drop outliers.

## Training configuration

Settings:

| setting | value |
|---|---|
| starting weights | `waxal-w2vbert-linsnalug-raw` |
| seed | 42 |
| epochs | 2 |
| learning rate | 2.0e-5 |
| warmup ratio | 0.1 |
| batch | 12 per device, 3 gradient-accumulation steps (effective 36) |
| precision | bf16, gradient checkpointing on |
| vocabulary | inherited from the parent checkpoint via `vocab_dir`, not rebuilt, so the CTC token order is preserved |
| target normalization | NFC and whitespace collapse only: no lowercasing, no punctuation stripping |
| augmentation | on, applied in the collator |
| hardware | 1x NVIDIA RTX 5090 (32 GB VRAM), 32 CPU cores, 92 GB RAM |

Augmentation is applied on the fly in the data collator, so each epoch sees a fresh draw of the same
clip:

- MUSAN noise (OpenSLR 17, `noise/` subset only, not speech), p = 0.5, SNR 5 to 20 dB
- simulated room impulse responses (OpenSLR 28), p = 0.3
- Gaussian noise p = 0.1, random gain p = 0.3
- pitch and speed perturbation disabled (p = 0.0)

Targets are raw for the same reason as every other arm here: CER is computed on the raw string and WER
is computed on lowercased but otherwise unmodified text, so punctuation errors are charged twice and a
model trained on stripped targets is capped on both metrics.

## Evaluation

| split | error, 0.5 * (WER + CER) |
|---|---|
| holdout, both languages, the checkpoint published here | 0.2599 |
| holdout, both languages, the original checkpoint | 0.2905 |

| leaderboard | score | CER | WER |
|---|---|---|---|
| public, this checkpoint alone | 0.726089 | 0.127544 | 0.420278 |

The holdout is a speaker-disjoint carve of the WAXAL **training** data. It is **not** the competition
test set, and it shares recording conditions and speakers' language variety with the training corpus,
so it flatters any arm tuned on that corpus and gives no independent read on generalisation. Lower is
better; the competition score is `1 - 0.5*(WER + CER)`.

**This repository holds a retrained checkpoint, not the weights used in the submission.** The
original was lost to a disk cleanup, so the arm was retrained from the same config at the same seed.
It is a genuinely different model: it disagrees with the original on 766 of the 892 test clips, 85.9
per cent. The 26-member ensemble built with it nonetheless changed on only 50 of 892 rows and scored
0.764774 against 0.764915 for the original, a difference of 0.000141 against a standard error of
about 0.0013.

Read the two holdout figures with care. The retrained checkpoint's 0.2599 looks far better than the
original's 0.2905, yet the ensemble containing it scored fractionally worse on the leaderboard. The
two holdout numbers may not share a carve, and in any case a member's solo quality is a poor
predictor of what it contributes to a vote.

This is a negative result and is reported as one. The intended benefit of mixing was that WAXAL text
would anchor the output conventions while AfriVoice supplied in-domain acoustics, avoiding the
punctuation drift the sequenced specialists show at the end of their AfriVoice stage, which their
refresh epoch then has to undo. That benefit
did not appear in the score: the sequenced specialists finished ahead on both languages. The rule taken
from the comparison, external acoustics first and in-style gold last, was then applied to every later
arm, including the Whisper one. The refresh is not free money either: on an arm supplemented with
already-punctuated FLEURS text it changed nothing (0.2809 before, 0.2812 after), which is consistent
with transcription convention rather than acoustics being the thing the refresh repairs.

## Intended use and limitations

Intended use is automatic speech recognition for Lingala and Shona photo-description speech:
spontaneous descriptions of images, on consumer recording equipment, averaging 20.2 seconds per clip.
It is not a general-purpose model for either language and has not been evaluated outside this domain.
The competition targets Lingala, Shona and Luganda; the corrected Phase 2 test set was measured to
contain only Lingala and Shona, and Luganda is dropped from this arm's corpus, so it should not be
given Luganda audio.

By solo holdout error this is the weakest of the w2v-BERT arms in this solution, and that is the point
of it. It exists as a controlled comparison first and as an ensemble member second: its training
distribution differs from every other arm, so its errors are correspondingly decorrelated, and source
diversity was measured to be worth more than parameter diversity (four members from four checkpoints
gained 0.0040, six members from two checkpoints gained 0.0006). The submission's 26-member
character-level ROVER core scored 0.764915, and the scored final submission, recipe p2n_mbr, selects
per clip across seven such ensembles and scored 0.766791 public and 0.772552 private. Solo strength
is not the only criterion for membership: the blank-penalty
re-decodes over-generate words and lose score in isolation (0.743 against 0.746 for the same
checkpoint) while contributing +0.0104 in total, because a character vote can filter a spurious word
and can never recover a missing one. The limit of that argument was measured as well: genuinely weak
checkpoints did not help, and a 21-member vote including them scored 0.764152 against 0.764759 for a
25-member vote of strong sources.

## How to load

```python
import torch
import soundfile as sf
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC

MODEL_ID = "anyantudre/waxal-w2vbert-linsna-afrivoicemix"   # or a local checkpoint directory

processor = Wav2Vec2BertProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2BertForCTC.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()

wav, sr = sf.read("clip.wav")          # mono, resample to 16 kHz before this call
inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

print(processor.batch_decode(logits.argmax(-1))[0])
```

When transcribing a batch rather than a single clip, pass the attention mask through to the model.
Clips in a batch have different lengths and the shorter ones are zero-padded; without the mask,
self-attention treats that padding as real audio and the encoder output changes for every frame, not
just the padded tail. Omitting it altered roughly half of all transcripts in a side-by-side check.
