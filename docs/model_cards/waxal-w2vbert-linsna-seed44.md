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

# waxal-w2vbert-linsna-seed44

This model was developed for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) on Zindi.

`waxal-w2vbert-linsna-seed44` is a Lingala and Shona CTC model, and the most broadly trained arm in
the family. It is a fresh fine-tune of
[facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0), a 600M parameter encoder, at
seed 44, with its own raw character vocabulary built from unnormalised transcripts.

It is deliberately decorrelated from the rest of the ensemble on **two axes at once**: a fresh
initialisation, like `waxal-w2vbert-linsna-seed43`, and a different corpus, since it adds four public
external sources that no other arm sees together. That combination is why it earns a place alongside
its seed-43 sibling rather than duplicating it. Source diversity was the more valuable of the two
axes: four members drawn from four different checkpoints gained 0.0040, while six members drawn from
only two checkpoints gained 0.0006.

## Training data

| source | rows | hours | licence |
|---|---|---|---|
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona train and validation | 33,002 | about 179 | CC-BY-4.0 |
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona **test** split | 3,615 (1,866 lin, 1,749 sna) | 19.2 | CC-BY-4.0 |
| [google/fleurs](https://huggingface.co/datasets/google/fleurs), config `ln_cd` | 2,847 | 15.0 | CC-BY-4.0 |
| [google/fleurs](https://huggingface.co/datasets/google/fleurs), config `sn_zw` | 3,704 | 15.0 | CC-BY-4.0 |
| [shunyalabs/lingala-speech-dataset](https://huggingface.co/datasets/shunyalabs/lingala-speech-dataset) | 4,341 | 21.5 | see the dataset card |
| [asr-africa/ASRAfricaDataEfficiencyBenchmark](https://huggingface.co/datasets/asr-africa/ASRAfricaDataEfficiencyBenchmark), config `Shona` | 1,055 | 6.0 | see the dataset card |

42,934 rows in total after the speaker-disjoint holdout and the validation rows are removed.

FLEURS is read using its `raw_transcription` field, which preserves casing and punctuation. The
lowercased `transcription` field would cap CER and would defeat the raw vocabulary.

**Disclosure.** The WAXAL Phase 1 test split is used as training data here. Phase 2 permits this and
the organisers confirmed it publicly. It is gold reference text released by the organisers, not
pseudo-labels, and it is not the Phase 2 test set.

## Training configuration

Settings:

| setting | value |
|---|---|
| starting weights | `facebook/w2v-bert-2.0` |
| seed | 44 |
| vocabulary | built fresh from unnormalised transcripts, 97 tokens |
| epochs | 8 |
| learning rate | 1.0e-4 |
| warmup ratio | 0.1 |
| batch | 12 per device, 3 gradient-accumulation steps (effective 36) |
| precision | bf16, gradient checkpointing on |
| augmentation | on, applied in the collator |

Trained with `add_adapter: true` and no `vocab_dir`, so this checkpoint's CTC token order differs
from the `waxal-w2vbert-linsnalug-raw` lineage and it is **not** weight-average compatible with it.
It combines by voting, not by averaging. Target normalization is NFC and whitespace collapse only.
Hardware: 1x NVIDIA RTX 5090 (32 GB VRAM), 32 CPU cores, 92 GB RAM.

Augmentation is applied on the fly in the data collator, so each epoch sees a fresh draw of the same
clip:

- MUSAN noise (OpenSLR 17, `noise/` subset only, not speech), p = 0.5, SNR 5 to 20 dB
- simulated room impulse responses (OpenSLR 28), p = 0.3
- Gaussian noise p = 0.1, random gain p = 0.3
- pitch and speed perturbation disabled (p = 0.0)

## Evaluation

| split | error, 0.5 * (WER + CER) |
|---|---|
| holdout, Lingala and Shona combined | 0.2809 |

The holdout is a speaker-disjoint carve of the WAXAL **training** data. It is **not** the competition
test set, and it shares recording conditions and language variety with the training corpus, so it
flatters any arm tuned on that corpus and gives no independent read on generalisation. Lower is
better; the competition score is `1 - 0.5*(WER + CER)`.

No solo public-leaderboard score was recorded for this checkpoint.

**A negative result worth recording.** A variant, `s44r`, added a final low-learning-rate refresh
epoch over WAXAL alone, the same two-stage curriculum that helps the per-language specialists a great
deal (0.2905 mixed against 0.2785 sequenced). Here it did nothing: 0.2809 before, 0.2812 after. The
reason is that the refresh exists to repair transcription conventions, not acoustics. AfriVoice
transcripts carry much less punctuation than WAXAL's, so a specialist trained on them drifts away
from the reference conventions and needs the refresh to return to them. FLEURS
`raw_transcription`, Shunya and ASR Africa are already punctuated, so there was nothing to repair.
The refresh is worth its cost only when the external corpus disagrees with the target conventions.

## Intended use and limitations

Intended use is automatic speech recognition for Lingala and Shona photo-description speech:
spontaneous descriptions of images, on consumer recording equipment, averaging 20.2 seconds per clip.
Of the published checkpoints this is the one that saw the widest range of recording conditions, so it
is the reasonable default for Lingala or Shona audio that is not photo-description speech, though
that use is untested. It has not seen Luganda; for Luganda, start from
`waxal-w2vbert-linsnalug-raw`.

This model is one component of the ensembles behind the submission. The 26-member character-level
ROVER core scored 0.764915, and the scored final submission, recipe p2n_mbr, selects per clip across
seven such ensembles and scored 0.766791 public and 0.772552 private. Solo strength is not the only
criterion for membership in those core ensembles. Some members are
deliberately weak alone, notably the blank-penalty re-decodes, which over-generate words and lose
score in isolation (0.743 against 0.746 for the same checkpoint) while contributing +0.0104 in total,
because a character vote can filter a spurious word and can never recover a missing one.

## How to load

```python
import torch
import soundfile as sf
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC

MODEL_ID = "anyantudre/waxal-w2vbert-linsna-seed44"   # or a local checkpoint directory

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
