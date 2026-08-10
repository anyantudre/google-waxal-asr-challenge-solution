---
language:
- ln
- sn
- lg
license: apache-2.0
tags:
- automatic-speech-recognition
- ctc
- waxal
- lingala
- shona
- luganda
base_model: facebook/w2v-bert-2.0
---

# waxal-w2vbert-linsnalug-raw

This model was developed for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) on Zindi.

`waxal-w2vbert-linsnalug-raw` is the root checkpoint of this solution and the one that carries its
largest single modelling gain. It fine-tunes
[facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0), a 600M parameter encoder, with
a CTC head over a **raw** character vocabulary: 97 tokens covering the letters, the 26 capitals, and
the punctuation `! , . : ; ?`. Capitals and punctuation are therefore ordinary emittable output
symbols rather than something stripped during normalisation. Switching from the earlier 81-token
stripped vocabulary to this one was worth **+0.0171** on the public leaderboard, more than any other
single change made during the competition.

It covers all three competition languages, and its frontmatter language list is `[ln, sn, lg]` for
that reason. Every other w2v-BERT checkpoint in this family either continues from it or shares its
token order, which is what makes weight averaging possible across the lineage.

## Why the raw vocabulary matters

The competition metric is `1 - 0.5 * (WER + CER)`, and the two halves are computed on different text.
Two controlled experiments established this:

1. Two submissions identical except for the capitalisation of 50 rows returned WER identical to nine
   decimal places (0.382307342 both times) while CER moved from 0.109473 to 0.109366. WER lowercases;
   CER does not.
2. Two submissions identical except for 87 added commas returned different WER (0.361457 against
   0.362198). WER does not strip punctuation either.

So CER is computed on the raw string and WER on lowercased but otherwise unmodified text. A model
that cannot emit a capital or a full stop is capped on CER no matter how good its words are, and
punctuation errors are charged twice, once to each metric.

## Training data

| source | rows | hours | licence |
|---|---|---|---|
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona train and validation | 33,002 | about 179 | CC-BY-4.0 |
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Luganda train and validation | not measured | not measured | CC-BY-4.0 |

The WAXAL train and validation splits are recorded jointly, with no per-language breakdown, so no
Lingala-only, Shona-only or Luganda-only row count is asserted here. The speaker-disjoint holdout and
the validation rows are removed before training.

## Training configuration

Settings:

| setting | value |
|---|---|
| starting weights | `facebook/w2v-bert-2.0` |
| languages | Lingala, Shona, Luganda |
| vocabulary | built fresh from unnormalised transcripts, 97 tokens |
| epochs | 12 |
| learning rate | 1.0e-4 |
| warmup ratio | 0.1 |
| batch | 12 per device, 3 gradient-accumulation steps (effective 36) |
| precision | bf16, gradient checkpointing on |
| augmentation | on, applied in the collator |

Trained with `add_adapter: true` and no `vocab_dir`, since this is the checkpoint that defines the
vocabulary rather than inheriting one. Target normalization is NFC and whitespace collapse only: no
lowercasing and no punctuation stripping, for the reason given above. Hardware: 1x NVIDIA RTX 5090
(32 GB VRAM), 32 CPU cores, 92 GB RAM.

Augmentation is applied on the fly in the data collator, so each epoch sees a fresh draw of the same
clip:

- MUSAN noise (OpenSLR 17, `noise/` subset only, not speech), p = 0.5, SNR 5 to 20 dB
- simulated room impulse responses (OpenSLR 28), p = 0.3
- Gaussian noise p = 0.1, random gain p = 0.3
- pitch and speed perturbation disabled (p = 0.0)

## Evaluation

| split | error, 0.5 * (WER + CER) |
|---|---|
| holdout, Lingala | 0.3131 |

| leaderboard | score | CER | WER |
|---|---|---|---|
| public, this checkpoint alone | 0.742142 | 0.115951 | 0.399766 |

The holdout is a speaker-disjoint carve of the WAXAL **training** data. It is **not** the competition
test set, and it shares recording conditions and language variety with the training corpus, so it
flatters any arm tuned on that corpus and gives no independent read on generalisation. Lower is
better; the competition score is `1 - 0.5*(WER + CER)`.

For context on the leaderboard row: the previous best configuration, identical except for its
stripped 81-token vocabulary, scored 0.724984 (CER 0.130719, WER 0.419312). The difference between
those two rows is the raw vocabulary and nothing else.

## Intended use and limitations

Intended use is automatic speech recognition for photo-description speech: spontaneous descriptions
of images, on consumer recording equipment, averaging 20.2 seconds per clip. This checkpoint is the
generalist of the family and is the one to start from for Luganda, since it is the only published
checkpoint that saw Luganda audio. Its Luganda performance was never measured, because the corrected
Phase 2 test set contains no Luganda: open-set language identification over all 892 clips found
roughly half Lingala and half Shona, and the single clip one model labelled Luganda was inspected and
is Lingala.

For Lingala or Shona alone, the specialists `waxal-w2vbert-lin-specialist` and
`waxal-w2vbert-sna-specialist` are stronger, and both continue from this checkpoint.

This model is one component of the ensembles behind the submission. The 26-member character-level
ROVER core scored 0.764915, well above this checkpoint's solo 0.742142, and the scored final
submission, recipe p2n_mbr, selects per clip across seven such ensembles and scored 0.766791 public
and 0.772552 private. Solo strength is not the only criterion for membership in those core
ensembles. Some members are deliberately weak alone, notably the blank-penalty
re-decodes, which over-generate words and lose score in isolation (0.743 against 0.746 for the same
checkpoint) while contributing +0.0104 in total, because a character vote can filter a spurious word
and can never recover a missing one.

## How to load

```python
import torch
import soundfile as sf
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC

MODEL_ID = "anyantudre/waxal-w2vbert-linsnalug-raw"   # or a local checkpoint directory

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
