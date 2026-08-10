---
language:
- sn
license: apache-2.0
tags:
- automatic-speech-recognition
- ctc
- waxal
- shona
base_model: facebook/w2v-bert-2.0
---

# waxal-w2vbert-sna-specialist

This model was developed for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) on Zindi.

`waxal-w2vbert-sna-specialist` is a single-language Shona CTC model, and its frontmatter language list
is `[sn]` for that reason. It continues the `waxal-w2vbert-linsnalug-raw` checkpoint, itself a
fine-tune of [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0), and inherits that
checkpoint's raw character vocabulary unchanged, capitals and the punctuation CER scores included, so
the CTC head keeps its token order and stays averageable with its siblings. Training follows the same
two-stage curriculum as the Lingala specialist: AfriVoice Shona first, then a refresh on WAXAL Shona
gold alone. In the final system this model is the Shona branch of a language-identification-routed pair
and one of the ensemble members, and it produced the Shona pseudo-labels used in a later experiment.

## Training data

| stage | source | rows | hours | licence |
|---|---|---|---|---|
| 1 | [realtime-speech/shona1](https://huggingface.co/datasets/realtime-speech/shona1), AfriVoice Shona | 15,923 | 91.0 | CC-BY-4.0 |
| 1 and 2 | [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Shona train and validation | not measured separately | not measured separately | CC-BY-4.0 |

The WAXAL Lingala and Shona train and validation splits are recorded jointly, as 33,002 rows and
about 179 hours, with no per-language breakdown, so no Shona-only row count or duration is asserted
here. The speaker-disjoint holdout and the validation rows are removed before training. Stage 1 sees
both sources in every batch; stage 2 sees WAXAL Shona only.

## Training configuration

Stage 1 and stage 2 settings:

| setting | stage 1 | stage 2 |
|---|---|---|
| starting weights | `waxal-w2vbert-linsnalug-raw` | stage 1 |
| corpus | WAXAL Shona plus AfriVoice Shona | WAXAL Shona only |
| epochs | 3 | 1 |
| learning rate | 3.0e-5 | 1.0e-5 |
| warmup ratio | 0.1 | 0.1 |
| batch | 12 per device, 3 gradient-accumulation steps (effective 36) | 12 per device, 3 gradient-accumulation steps (effective 36) |
| precision | bf16, gradient checkpointing on | bf16, gradient checkpointing on |
| augmentation | on, applied in the collator | on, applied in the collator |

Both stages run with `add_adapter: true`, seed 42, and `vocab_dir` pointing at the parent checkpoint,
so no vocabulary is rebuilt and the CTC token order is preserved. Target normalization is NFC and
whitespace collapse only: no lowercasing, no punctuation stripping, because CER is computed on the raw
string and WER does not strip punctuation. Hardware: 1x NVIDIA RTX 5090 (32 GB VRAM), 32 CPU cores,
92 GB RAM.

Augmentation is applied on the fly in the data collator, so each epoch sees a fresh draw of the same
clip:

- MUSAN noise (OpenSLR 17, `noise/` subset only, not speech), p = 0.5, SNR 5 to 20 dB
- simulated room impulse responses (OpenSLR 28), p = 0.3
- Gaussian noise p = 0.1, random gain p = 0.3
- pitch and speed perturbation disabled (p = 0.0)

**Why the order, and why the refresh.** The curriculum was measured rather than assumed. Mixing the
same data throughout scored 0.2905 on the holdout; sequencing external audio first and in-style WAXAL
gold last scored 0.2785, both figures measured on the Lingala arm. The mechanism is transcription
convention, not acoustics: AfriVoice transcripts carry less punctuation than WAXAL's, so a model
trained on them drifts away from the reference conventions and the refresh epoch pulls it back.
Where the external corpus is already punctuated the refresh buys nothing, which was also measured on
a FLEURS-supplemented arm (0.2809 before the refresh, 0.2812 after).

## Evaluation

| split | error, 0.5 * (WER + CER) |
|---|---|
| holdout, Shona | 0.1616 |

The holdout is a speaker-disjoint carve of the WAXAL **training** data. It is **not** the competition
test set, and it shares recording conditions and speakers' language variety with the training corpus,
so it flatters any arm tuned on that corpus and gives no independent read on generalisation. Lower is
better; the competition score is `1 - 0.5*(WER + CER)`.

No solo public-leaderboard score was recorded for this checkpoint on its own. Routed with its Lingala
counterpart by open-set language identification, the specialist pair scored 0.744835 on the public
leaderboard (CER 0.115141, WER 0.395189), above the general `waxal-w2vbert-linsnalug-raw` arm at
0.742142. Holdout and leaderboard did not always agree in this project, so the holdout figure above
should be read as a training diagnostic, not as a ranking.

Shona is consistently easier than Lingala for this model family: 0.1616 here against 0.2785 for the
Lingala specialist on the same kind of holdout. A five-way weight average that includes this checkpoint
reached 0.1610 on Shona. Self-training on the WAXAL unlabeled Shona pool, using this model's own
pseudo-labels (22,396 rows, top 60 per cent kept), was flat in both languages and was dropped.

## Intended use and limitations

Intended use is automatic speech recognition for Shona photo-description speech: spontaneous
descriptions of images, on consumer recording equipment, averaging 20.2 seconds per clip. It is a
specialist and should be given Shona audio only. Its behaviour on Lingala, on Luganda, or on any other
language is untested and expected to be worse than that of a model trained for those languages. The
competition targets Lingala, Shona and Luganda; the corrected Phase 2 test set was measured to contain
only Lingala and Shona, which is why this pair of specialists covers those two.

In the final system the routing decision comes from open-set language identification rather than from
the clip identifiers, which carry no language information: `facebook/mms-lid-4017` called 423 of the
892 test clips Shona at confidence 0.980 and 437 Lingala at confidence 0.990, leaving 32
low-confidence clips, one of which it labelled Luganda. The 34 clips that either identification
model (the other being `facebook/mms-lid-256`) declined to call Lingala
or Shona were inspected individually: clips labelled Chichewa, Ndau or Venda are transcribed as Shona
by every arm, so they are neighbouring-language confusions rather than coverage gaps.

This model is one component of the ensembles behind the submission. The 26-member character-level
ROVER core scored 0.764915, and the scored final submission, recipe p2n_mbr, selects per clip across
seven such ensembles and scored 0.766791 public and 0.772552 private. Solo strength is not the only
criterion for membership in those core ensembles. Some members are deliberately weak alone,
notably the blank-penalty re-decodes, which over-generate words and lose score in isolation (0.743
against 0.746 for the same checkpoint) while contributing +0.0104 in total, because a character vote
can filter a spurious word and can never recover a missing one. The limit of that argument was also
measured: genuinely weak checkpoints did not help, and a 21-member vote including them scored 0.764152
against 0.764759 for a 25-member vote of strong sources.

## How to load

```python
import torch
import soundfile as sf
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC

MODEL_ID = "anyantudre/waxal-w2vbert-sna-specialist"   # or a local checkpoint directory

processor = Wav2Vec2BertProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2BertForCTC.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()

wav, sr = sf.read("shona_clip.wav")   # mono, resample to 16 kHz before this call
inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

print(processor.batch_decode(logits.argmax(-1))[0])
```

When transcribing a batch rather than a single clip, pass the attention mask through to the model.
Clips in a batch have different lengths and the shorter ones are zero-padded; without the mask,
self-attention treats that padding as real audio and the encoder output changes for every frame, not
just the padded tail. Omitting it altered roughly half of all transcripts in a side-by-side check.
