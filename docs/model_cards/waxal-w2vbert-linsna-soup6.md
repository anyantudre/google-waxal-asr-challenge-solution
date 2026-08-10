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

# waxal-w2vbert-linsna-soup6

This model was developed for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) on Zindi.

`waxal-w2vbert-linsna-soup6` is not a trained checkpoint. It is the **uniform weight average of six
checkpoints**, produced by averaging their parameter tensors elementwise. No gradient step was taken
to create it and no data was seen. It covers Lingala and Shona.

It extends the five-way average `waxal-w2vbert-linsna-soup5` with one further member: the distilled
arm, which continues from that soup and therefore still shares its vocabulary and CTC token order.
Averaging a checkpoint with its own descendant is unusual, and the question it answers is whether
the distillation step produced something worth folding back in.

## The six members

| member | continues |
|---|---|
| `waxal-w2vbert-linsnalug-raw` | `facebook/w2v-bert-2.0` |
| `waxal-w2vbert-lin-specialist` | its AfriVoice Lingala stage 1 |
| `waxal-w2vbert-sna-specialist` | its AfriVoice Shona stage 1 |
| `linspec_p2` | the Lingala specialist, pseudo-label round |
| `snaspec_p2` | the Shona specialist, pseudo-label round |
| `waxal-w2vbert-linsna-distilled` | `waxal-w2vbert-linsna-soup5` |

Averaging is only meaningful because all six share one vocabulary and one CTC token order. Every
member either is the root checkpoint or continues from it with the vocabulary pinned, so position
`k` of the CTC head means the same character in all six. Averaging checkpoints whose vocabularies
were built independently, for instance the fresh-seed arms, would produce a model that loads without
error and transcribes nonsense; the build script therefore checks parameter names and shapes and
refuses to proceed on a mismatch.

The two `_p2` members are pseudo-label rounds over the WAXAL unlabeled pools. They are archived
rather than released as standalone repositories, because neither is strong enough to vote on its own
account.

## Training data

No data was used to produce this checkpoint. Transitively, through its six members, it derives from:

| source | rows | hours | licence |
|---|---|---|---|
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona train and validation | 33,002 | about 179 | CC-BY-4.0 |
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona test split | 3,615 | 19.2 | CC-BY-4.0 |
| [KasuleTrevor/Lingala_100hrs](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs), AfriVoice Lingala | 22,131 | 104.5 | CC-BY-4.0 |
| [realtime-speech/shona1](https://huggingface.co/datasets/realtime-speech/shona1), AfriVoice Shona | 15,923 | 91.0 | CC-BY-4.0 |
| Pseudo-labels over [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) **unlabeled** Lingala and Shona | 22,986 and 22,396, top 60 per cent kept | not measured | derived |
| Ensemble transcripts for the 892 test clips, via the distilled member | 892 | 5.0 | derived |

**Disclosure.** One of the six members, the distilled arm, was trained on the 892 competition test
clips using an ensemble's own transcripts as targets. That is self-training on unlabeled test audio,
which the organisers permit subject to disclosure, and no reference transcript for any test clip was
used. This average therefore inherits that property. The distilled member's own card states the
caveats in full.

## How it was produced

Uniform average, weight one sixth each, over every floating-point parameter tensor. Buffers and the
processor are taken from the root checkpoint unchanged. The operation is CPU only, takes minutes,
and is exactly reproducible from the six members.

## Evaluation

No solo public-leaderboard score was recorded for this checkpoint, and no holdout evaluation was run
on it.

What was measured is its effect as an ensemble member. Substituting it for the five-way soup in
every slot of a 26-member ensemble moved that ensemble from 0.766580 to 0.766226, slightly worse and
well inside measurement noise. As a member it is genuinely distinct from the five-way soup: their
greedy transcripts agree on only 592 of 892 clips.

The honest summary is that folding the distilled arm back into the average changed the model
substantially and changed the result by nothing measurable.

## Intended use and limitations

Intended use is automatic speech recognition for Lingala and Shona photo-description speech:
spontaneous descriptions of images, on consumer recording equipment, averaging 20.2 seconds per clip.
Luganda behaviour should be assumed degraded: only one of the six members ever saw Luganda, at weight
one sixth, and it is untested here.

Because one member was adapted to one specific set of 892 recordings, this average carries a share of
that adaptation. For general Lingala or Shona use, prefer `waxal-w2vbert-linsna-soup5`, which does
not, or `waxal-w2vbert-linsna-seed44`.

This checkpoint contributes to one of the seven ensembles that the scored submission, recipe
p2n_mbr (public 0.766791, private 0.772552), selects across, and it is published so that result can
be rebuilt in full.

## How to load

```python
import torch
import soundfile as sf
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC

MODEL_ID = "anyantudre/waxal-w2vbert-linsna-soup6"   # or a local checkpoint directory

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
