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
- model-soup
base_model: facebook/w2v-bert-2.0
---

# waxal-w2vbert-linsna-soup5

This model was developed for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) on Zindi.

`waxal-w2vbert-linsna-soup5` is not a trained checkpoint. It is the **uniform weight average of five
checkpoints** from the `waxal-w2vbert-linsnalug-raw` lineage, produced by averaging their parameter
tensors elementwise. No gradient step was taken to create it and no data was seen. It covers Lingala
and Shona.

Averaging is only meaningful because all five members share one vocabulary and one CTC token order.
Every member either is `waxal-w2vbert-linsnalug-raw` or continues from it with `vocab_dir` pointing
back at it, so position `k` of the CTC head means the same character in all five. Averaging
checkpoints whose vocabularies were built independently, for instance the seed-43 and seed-44 arms,
would produce a meaningless model; those arms combine by voting instead.

The result is the strongest w2v-BERT arm apart from the distilled one, and it is the checkpoint that
`waxal-w2vbert-linsna-distilled` continues from.

## The five members

| member | continues | epochs and learning rate |
|---|---|---|
| `p1raw` (`waxal-w2vbert-linsnalug-raw`) | `facebook/w2v-bert-2.0` | 12 at 1.0e-4 |
| `linspec_r` (`waxal-w2vbert-lin-specialist`) | `linspec`, its AfriVoice Lingala stage 1 | 1 at 1.0e-5 |
| `snaspec_r` (`waxal-w2vbert-sna-specialist`) | `snaspec`, its AfriVoice Shona stage 1 | 1 at 1.0e-5 |
| `linspec_p2` | `linspec_r` | 2 at 1.5e-5, then 1 at 1.0e-5 |
| `snaspec_p2` | `snaspec_r` | 2 at 1.5e-5, then 1 at 1.0e-5 |

The two `_p2` members are pseudo-label rounds over the WAXAL unlabeled pools, refreshed on WAXAL
gold. They are published only as ingredients of this average, not as standalone repositories.
Self-training on those unlabeled pools was flat in both languages when measured directly, and
`snaspec_p2` was one of the checkpoints whose blank-penalty members took the vote from 0.764476 down
to 0.764152. Neither is strong enough to vote on its own account.

Averaging is the way to use a checkpoint that is not good enough to vote. A weight average commits to
one set of parameters and therefore one transcript, so a mediocre ingredient is diluted; a vote gives
every member a say in every character slot, so a mediocre member injects its errors directly.

## Training data

No data was used to produce this checkpoint. Transitively, through its five members, it derives from:

| source | rows | hours | licence |
|---|---|---|---|
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona train and validation | 33,002 | about 179 | CC-BY-4.0 |
| [KasuleTrevor/Lingala_100hrs](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs), AfriVoice Lingala | 22,131 | 104.5 | CC-BY-4.0 |
| [realtime-speech/shona1](https://huggingface.co/datasets/realtime-speech/shona1), AfriVoice Shona | 15,923 | 91.0 | CC-BY-4.0 |
| Pseudo-labels over [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) **unlabeled** Lingala | 22,986, top 60 per cent kept | not measured | derived |
| Pseudo-labels over [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) **unlabeled** Shona | 22,396, top 60 per cent kept | not measured | derived |

The pseudo-labels are our own models' output over WAXAL's unlabeled pools. They are **not** test
audio. Self-training on those pools was flat in both languages when measured directly, which is why
the `_p2` checkpoints survive only as soup ingredients.

## How it was produced

Uniform average, weight 0.2 each, over every floating-point parameter tensor. Buffers and the
processor are taken from `waxal-w2vbert-linsnalug-raw` unchanged. The operation is CPU only, takes
seconds, and is exactly reproducible from the five published or derived members.

Hardware is irrelevant to reproducing this checkpoint; the members were trained on 1x NVIDIA RTX 5090
(32 GB VRAM), 32 CPU cores, 92 GB RAM.

## Evaluation

| split | error, 0.5 * (WER + CER) |
|---|---|
| holdout, Shona | 0.1610 |

| leaderboard | score | CER | WER |
|---|---|---|---|
| public, this checkpoint alone | 0.746054 | 0.114864 | 0.393027 |

The holdout is a speaker-disjoint carve of the WAXAL **training** data. It is **not** the competition
test set, and it shares recording conditions and language variety with the training corpus, so it
flatters any arm tuned on that corpus and gives no independent read on generalisation. Lower is
better; the competition score is
`1 - 0.5*(WER + CER)`.

For comparison on the same leaderboard: `waxal-w2vbert-linsnalug-raw` alone scored 0.742142 and the
routed specialist pair scored 0.744835, so the average beats every one of its own ingredients. Only
`waxal-w2vbert-linsna-distilled`, which continues from this checkpoint, scored higher as a single
model, at 0.746787.

## Intended use and limitations

Intended use is automatic speech recognition for Lingala and Shona photo-description speech:
spontaneous descriptions of images, on consumer recording equipment, averaging 20.2 seconds per clip.
It has not effectively seen Luganda: `waxal-w2vbert-linsnalug-raw` contributes Luganda exposure at
weight 0.2 while the other four members were tuned on Lingala and Shona only, so Luganda behaviour
should be assumed degraded relative to that checkpoint and is untested either way.

An honest caveat about the holdout figure above: it is Shona only, because that is the only holdout
figure recorded for this checkpoint. Three of the five members are Shona-tuned or Shona-inclusive,
so the average is expected to be stronger on Shona than on Lingala, and the single figure should not
be read as an overall error.

This model is one component of the ensembles behind the submission. The 26-member character-level
ROVER core scored 0.764915, well above this checkpoint's solo 0.746054, and the scored final
submission, recipe p2n_mbr, selects per clip across seven such ensembles and scored 0.766791 public
and 0.772552 private. This checkpoint appears in each core ensemble four times: once greedily
decoded and three times re-decoded at blank penalties 1.0, 1.5 and 2.0.

## How to load

```python
import torch
import soundfile as sf
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC

MODEL_ID = "anyantudre/waxal-w2vbert-linsna-soup5"   # or a local checkpoint directory

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
