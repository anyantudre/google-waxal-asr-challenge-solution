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

# waxal-w2vbert-linsna-seed43

This model was developed for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) on Zindi.

`waxal-w2vbert-linsna-seed43` is a Lingala and Shona CTC model, and the **anchor of the final
ensemble**. It is a fresh fine-tune of
[facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0), a 600M parameter encoder, at
seed 43, with its own raw character vocabulary built from unnormalised transcripts, capitals and the
punctuation CER scores included.

The word "fresh" is the point. This checkpoint does **not** continue from
`waxal-w2vbert-linsnalug-raw`; it starts again from the public base with a different seed, so its
errors are decorrelated from that entire lineage. Fresh seeds were the most valuable ensemble members
measured in this project, worth about +0.0032 each, and were cheaper and more effective than adding
new architectures.

Its role as anchor matters more than its solo score. In character-level ROVER the anchor supplies the
skeleton, and its text survives every slot unless the members outvote it, so anchor choice moves the
result more than any single member does. Using the stronger `distil` arm as the anchor instead scored
0.763419, below the 0.764915 obtained with this checkpoint anchoring and `distil` voting as a member.

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
| seed | 43 |
| vocabulary | built fresh from unnormalised transcripts, 97 tokens |
| epochs | 8 |
| learning rate | 1.0e-4 |
| warmup ratio | 0.1 |
| batch | 12 per device, 3 gradient-accumulation steps (effective 36) |
| precision | bf16, gradient checkpointing on |
| augmentation | on, applied in the collator |

Trained with `add_adapter: true` and no `vocab_dir`. Because the vocabulary is rebuilt from these
transcripts, this checkpoint's CTC token order differs from the `waxal-w2vbert-linsnalug-raw` lineage
and it is **not** weight-average compatible with it. It combines by voting, not by averaging. Target
normalization is NFC and whitespace collapse only: no lowercasing, no punctuation stripping, because
CER is computed on the raw string and WER does not strip punctuation. Hardware: 1x NVIDIA RTX 5090
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
| holdout, Lingala and Shona combined | 0.2862 |

The holdout is a speaker-disjoint carve of the WAXAL **training** data. It is **not** the competition
test set, and it shares recording conditions and language variety with the training corpus, so it
flatters any arm tuned on that corpus and gives no independent read on generalisation. Lower is
better; the competition score is `1 - 0.5*(WER + CER)`.

No solo public-leaderboard score was recorded for this checkpoint. Holdout and leaderboard did
not always agree in this project, so the holdout figure should be read as a training diagnostic, not
as a ranking.

## Intended use and limitations

Intended use is automatic speech recognition for Lingala and Shona photo-description speech:
spontaneous descriptions of images, on consumer recording equipment, averaging 20.2 seconds per clip.
It has not seen Luganda; for Luganda, start from `waxal-w2vbert-linsnalug-raw`, the only published
checkpoint in this family that did. The competition targets Lingala, Shona and Luganda, and the
corrected Phase 2 test set was measured to contain only the first two.

This model is one component of an ensemble; the final submission combines 26 character-level ROVER
members and scored 0.764915. Solo strength is not the only selection criterion. Some members are
deliberately weak alone, notably the blank-penalty re-decodes, which over-generate words and lose
score in isolation (0.743 against 0.746 for the same checkpoint) while contributing +0.0104 in total,
because a character vote can filter a spurious word and can never recover a missing one. The limit of
that argument was also measured: genuinely weak checkpoints did not help, and a 21-member vote
including them scored 0.764152 against 0.764759 for a 25-member vote of strong sources.

## How to load

```python
import torch
import soundfile as sf
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC

MODEL_ID = "anyantudre/waxal-w2vbert-linsna-seed43"   # or a local checkpoint directory

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
