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

# waxal-w2vbert-linsna-seed46

This model was developed for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) on Zindi.

`waxal-w2vbert-linsna-seed46` is a Lingala and Shona CTC model, and the most decorrelated member of
the family. It is a fresh fine-tune of
[facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0), a 600M parameter encoder, at
seed 46, with its own raw character vocabulary built from unnormalised transcripts, capitals and the
punctuation CER scores included.

It shares its corpus and every hyperparameter with the seed-43 arm. Only the seed differs. That is
the point: a fresh initialisation is the cheapest way to obtain a model whose errors are unlike the
rest of the family, and error diversity is what a character-level vote converts into accuracy.

## How different it actually is

Measured over the 892 clip evaluation set, comparing greedy transcripts clip by clip:

| against | identical transcripts |
|---|---|
| `waxal-w2vbert-linsna-seed43` | 110 of 892, 12.3 per cent |
| `waxal-w2vbert-linsna-seed44` | 77 of 892, 8.6 per cent |
| `waxal-w2vbert-linsna-distilled` | 103 of 892, 11.5 per cent |

Agreement below 13 per cent with every existing arm, while scoring comparably on its own, is the
profile a vote wants from a new member.

## Training data

| source | rows | hours | licence |
|---|---|---|---|
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona train and validation | 33,002 | about 179 | CC-BY-4.0 |
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona **test** split | 3,615 (1,866 lin, 1,749 sna) | 19.2 | CC-BY-4.0 |

30,987 rows after the speaker-disjoint holdout and the validation rows are removed.

**Disclosure.** The WAXAL Phase 1 test split is used as training data here. Phase 2 permits this and
the organisers confirmed it publicly. It is gold reference text released by the organisers, not
pseudo-labels, and it is not the Phase 2 test set.

## Training configuration

Settings:

| setting | value |
|---|---|
| starting weights | `facebook/w2v-bert-2.0` |
| seed | 46 |
| vocabulary | built fresh from unnormalised transcripts, 97 tokens |
| epochs | 8 |
| learning rate | 1.0e-4 |
| warmup ratio | 0.1 |
| batch | 12 per device, 3 gradient-accumulation steps (effective 36) |
| precision | bf16, gradient checkpointing on |
| augmentation | on, applied in the collator |

Trained with `add_adapter: true` and no inherited vocabulary, so this checkpoint's CTC token order
differs from the lineage rooted at `waxal-w2vbert-linsnalug-raw` and it is **not** weight-average
compatible with it. It combines by voting, not by averaging. Target normalization is NFC and
whitespace collapse only: no lowercasing, no punctuation stripping, because CER is computed on the
raw string and WER does not strip punctuation. The feature cache was held in RAM rather than on
disk, which changes nothing about the result. Hardware: 1x NVIDIA RTX 5090 (32 GB VRAM), 32 CPU
cores, 92 GB RAM.

Augmentation is applied on the fly in the data collator, so each epoch sees a fresh draw of the same
clip:

- MUSAN noise (OpenSLR 17, `noise/` subset only, not speech), p = 0.5, SNR 5 to 20 dB
- simulated room impulse responses (OpenSLR 28), p = 0.3
- Gaussian noise p = 0.1, random gain p = 0.3
- pitch and speed perturbation disabled (p = 0.0)

## Evaluation

| split | error, 0.5 * (WER + CER) |
|---|---|
| holdout, Lingala and Shona combined | **0.2588**, the lowest of any arm in this family |

The holdout is a speaker-disjoint carve of the WAXAL **training** data. It is **not** the competition
test set, and it shares recording conditions and language variety with the training corpus, so it
flatters any arm tuned on that corpus and gives no independent read on generalisation. Lower is
better; the competition score is `1 - 0.5*(WER + CER)`.

No solo public-leaderboard score was recorded for this checkpoint.

Read the holdout figure with care. It is the best in the family, yet substituting this arm for the
weakest member of a 26-member ensemble moved that ensemble from 0.766580 to 0.766477, which is
slightly worse and well inside measurement noise. A member's solo quality is a poor predictor of
what it contributes to a vote, and this checkpoint is a clean example of the gap.

## Intended use and limitations

Intended use is automatic speech recognition for Lingala and Shona photo-description speech:
spontaneous descriptions of images, on consumer recording equipment, averaging 20.2 seconds per clip.
It has not seen Luganda; for Luganda, start from `waxal-w2vbert-linsnalug-raw`, the only published
checkpoint in this family that did.

This checkpoint is one component of an ensemble. It contributes to one of the five ensembles that
the best submission votes over, and it is published so that result can be rebuilt in full. On its
own it is an ordinary member of the family rather than the strongest one.

## How to load

```python
import torch
import soundfile as sf
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC

MODEL_ID = "anyantudre/waxal-w2vbert-linsna-seed46"   # or a local checkpoint directory

processor = Wav2Vec2BertProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2BertForCTC.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()

wav, sr = sf.read("clip.wav")   # mono, resample to 16 kHz before this call
inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

print(processor.batch_decode(logits.argmax(-1))[0])
```

When transcribing a batch rather than a single clip, pass the attention mask through to the model.
Clips in a batch have different lengths and the shorter ones are zero-padded; without the mask,
self-attention treats that padding as real audio and the encoder output changes for every frame, not
just the padded tail. Omitting it altered roughly half of all transcripts in a side-by-side check.
