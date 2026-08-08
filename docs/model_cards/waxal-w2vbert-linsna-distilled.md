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
- knowledge-distillation
base_model: facebook/w2v-bert-2.0
---

# waxal-w2vbert-linsna-distilled

This model was developed for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) on Zindi.

`waxal-w2vbert-linsna-distilled` is a Lingala and Shona CTC model and the **strongest single
checkpoint** in this solution, at 0.746787 on the public leaderboard and 0.2746 on the holdout. It
continues from `waxal-w2vbert-linsna-soup5` and reuses that checkpoint's vocabulary and CTC token
order.

It is trained by ensemble distillation: the 26-member ROVER ensemble scores 0.764915 while its best
single member scores about 0.746, and this arm recovers part of that gap in a single
checkpoint by learning from the ensemble's own transcripts.

## Disclosure, read this first

**This model is trained on the 892 Phase 2 test clips.** The audio is the competition test audio and
the targets are our own ensemble's transcripts of it, mixed with WAXAL gold. This is self-training on
unlabeled test audio, which the organisers explicitly permit subject to disclosure, and it is
disclosed here and throughout the documentation of the solution repository that accompanies these
models.

Three points bound what that means:

- **No reference transcript for any test clip was used.** The targets are model output, not labels.
- **No reference transcript for any test clip exists publicly.** The 892 clips were fingerprinted
  against every transcribed AfriVoice clip, 16,815 rows of `realtime-speech/shona1` and 23,540 rows
  of `KasuleTrevor/Lingala_100hrs`, in two passes: an exact one (duration within 80 ms, two second
  waveform correlation) and a tolerant one (duration within 2 seconds, excerpt slid across the first
  8 seconds). The ASR Africa Shona benchmark was scanned as well. There were no matches in any pass.
  This check is independent of the training procedure and establishes that no public label could have
  leaked in even accidentally.
- **The gain is not free skill.** The model can only learn what the ensemble already produced,
  including its errors. Anchoring this checkpoint in the final vote rather than letting it vote as a
  member scored 0.763419 against 0.764915, which is the measured cost of trusting it too much.

Anyone reusing this checkpoint outside the competition should understand that it has been adapted to
one specific set of 892 recordings. For general Lingala or Shona use, prefer
`waxal-w2vbert-linsna-soup5`, which it continues from, or `waxal-w2vbert-linsna-seed44`.

## Training data

| source | rows | hours | licence |
|---|---|---|---|
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona **test** split, gold | 3,615 (1,866 lin, 1,749 sna) | 19.2 | CC-BY-4.0 |
| Our ensemble's transcripts for the 892 Phase 2 test clips | 892 | 5.0 | derived |

The WAXAL gold rows are not optional decoration. Training on the ensemble's transcripts alone would
let the model drift onto its own errors, since nothing would hold it to the reference transcription
conventions. The gold rows anchor casing, punctuation and spelling to what the metric actually scores.

Transitively, through `waxal-w2vbert-linsna-soup5`, this checkpoint also derives from WAXAL train and
validation, [KasuleTrevor/Lingala_100hrs](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs)
(CC-BY-4.0), [realtime-speech/shona1](https://huggingface.co/datasets/realtime-speech/shona1)
(CC-BY-4.0), and pseudo-labels over the WAXAL unlabeled pools. See that checkpoint's card.

## Training configuration

Settings:

| setting | value |
|---|---|
| starting weights | `waxal-w2vbert-linsna-soup5` |
| vocabulary | inherited via `vocab_dir`, CTC token order preserved |
| epochs | 3 |
| learning rate | 8.0e-6 |
| warmup ratio | 0.1 |
| batch | 12 per device, 3 gradient-accumulation steps (effective 36) |
| precision | bf16, gradient checkpointing on |
| augmentation | on, applied in the collator |

The learning rate is an order of magnitude below the one used to train an arm from the base model.
It is refining a converged checkpoint over 4,507 rows, and a larger rate would overwrite what the
soup already knows. Target normalization is NFC and whitespace collapse only: no lowercasing, no
punctuation stripping, because CER is computed on the raw string and WER does not strip punctuation.
Hardware: 1x NVIDIA RTX 5090 (32 GB VRAM), 32 CPU cores, 92 GB RAM.

Because the vocabulary is inherited rather than rebuilt, this checkpoint remains weight-average
compatible with the whole `waxal-w2vbert-linsnalug-raw` lineage.

## Evaluation

| split | error, 0.5 * (WER + CER) |
|---|---|
| holdout, Lingala and Shona combined | **0.2746**, the best of any single arm |

| leaderboard | score | CER | WER |
|---|---|---|---|
| public, this checkpoint alone | **0.746787** | 0.113117 | 0.393308 |

The holdout is a speaker-disjoint carve of the WAXAL **training** data. It is **not** the competition
test set, and it shares recording conditions and language variety with the training corpus, so it
flatters any arm tuned on that corpus and gives no independent read on generalisation. Lower is
better; the competition score is
`1 - 0.5*(WER + CER)`.

The holdout result is the more informative of the two here. The holdout contains no test audio, so
the improvement on it cannot be an artefact of having seen the test recordings: the model genuinely
learned something transferable from the ensemble's output, rather than only memorising 892 clips.

Both figures still sit below the ensemble's 0.764915. Distillation narrowed the gap between one model
and the vote; it did not close it.

## Intended use and limitations

Intended use is automatic speech recognition for Lingala and Shona photo-description speech:
spontaneous descriptions of images, on consumer recording equipment, averaging 20.2 seconds per clip.
It has not seen Luganda; for Luganda, start from `waxal-w2vbert-linsnalug-raw`, the only published
checkpoint in this family that did. The competition targets Lingala, Shona and Luganda, and the
corrected Phase 2 test set was measured to contain only the first two.

Beyond the adaptation caveat above, the ordinary ensemble caveat applies: this model is one component
of the final submission, which combines 26 character-level ROVER members and scored 0.764915. Adding
this arm to the 25-member vote moved it from 0.764759 to 0.764915, a difference of 0.000156. On a
public split of roughly 268 clips the standard error on a paired difference is about 0.0013, so that
gain is **not** distinguishable from noise. It was included in the final pick on the reasoning that a
strong, differently trained member should not hurt, not because the measurement showed it helped.

## How to load

```python
import torch
import soundfile as sf
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC

MODEL_ID = "anyantudre/waxal-w2vbert-linsna-distilled"   # or a local checkpoint directory

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
