---
language:
- ln
- sn
license: apache-2.0
tags:
- automatic-speech-recognition
- whisper
- seq2seq
- waxal
- lingala
- shona
base_model: openai/whisper-large-v3-turbo
---

# waxal-whisper-turbo-linsna

This model was developed for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) on Zindi.

`waxal-whisper-turbo-linsna` is a full fine-tune, not a LoRA, of
[openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) on WAXAL Lingala
and Shona, followed by one low-learning-rate refresh epoch. It is the only sequence-to-sequence model
we trained for this solution: every other arm we trained is a w2v-BERT CTC model, and the one
third-party Whisper-family member in the final system is used zero-shot rather than fine-tuned. That
difference is the whole reason this arm exists. An encoder-decoder model with a subword vocabulary and
a language model built into its decoder fails differently from a character CTC model, which is what a
vote needs. Whisper-turbo was chosen over the other candidates because it has genuine Whisper language
tokens for Lingala (`<|ln|>`) and Shona (`<|sn|>`), with no remapped or proxy language slots. Targets
are kept raw, with casing and punctuation intact, for the same reason as the CTC arms: CER is computed
on the raw string and WER does not strip punctuation.

## Training data

| source | rows | hours | licence |
|---|---|---|---|
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona train and validation | 33,002 | about 179 | CC-BY-4.0 |
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona test split, gold | 3,615 (1,866 lin, 1,749 sna) | 19.2 | CC-BY-4.0 |

Both stages train on this same corpus; the second stage differs only in learning rate and length. Use
of the WAXAL Phase 1 test split for training is permitted in Phase 2 and the organisers confirmed this
publicly. The AfriVoice corpora, [KasuleTrevor/Lingala_100hrs](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs)
and [realtime-speech/shona1](https://huggingface.co/datasets/realtime-speech/shona1), are deliberately
excluded here and are used by the per-language CTC specialists instead. Clips are filtered to 1 to 30
seconds, 30 seconds being Whisper's input window, and to 1 to 25 characters per second; the
speaker-disjoint holdout and the validation rows are removed before training.

## Training configuration

Stage 1 and stage 2 settings:

| setting | stage 1 | stage 2 |
|---|---|---|
| starting weights | `openai/whisper-large-v3-turbo` | stage 1 |
| epochs | 3 | 1 |
| learning rate | 1.0e-5 | 5.0e-6 |
| warmup ratio | 0.1 | 0.1 |
| batch | 4 per device, 8 gradient-accumulation steps (effective 32) | 4 per device, 8 gradient-accumulation steps (effective 32) |
| precision | bf16, gradient checkpointing on | bf16, gradient checkpointing on |
| seed | 42 | 42 |

| setting | value |
|---|---|
| vocabulary | Whisper's own subword tokenizer, unchanged; no character vocabulary is built |
| language conditioning | per-example language token, `task=transcribe` |
| target normalization | NFC and whitespace collapse only: no lowercasing, no punctuation stripping |
| augmentation | disabled |
| evaluation during training | every 400 steps, capped at 300 generated samples |
| hardware | 1x NVIDIA RTX 5090 (32 GB VRAM), 32 CPU cores, 92 GB RAM |

Two settings differ deliberately from the CTC arms. The learning rate is an order of magnitude lower,
because this base is already a competent multilingual model and the aim is to move it into the WAXAL
recording domain, not to teach it the languages. And multi-condition augmentation (MUSAN noise, room
impulse response reverberation) is switched off, even though it helps the CTC arms, because what is
being bought from this base is its generalisation, which noise injection risks fighting.

Per-example language conditioning is load-bearing. With one global language token, every example would
train under a single token while inference forces a different one, a silent train and test mismatch
that reads as a mediocre model rather than as a broken one.

The `decode` block in the training configuration (`num_beams: 1`, `max_new_tokens: 200`) is also the
decoding of record for the submitted ensemble member. See the decoding section below.

## Decoding of record, and the 30 second window

The submitted ensemble member was generated greedily (`num_beams: 1`) at `max_new_tokens: 200`, with
no forced language token and **no windowing**: clips are passed to the feature extractor as they
are, so Whisper's fixed 30 second receptive field truncates the 3.3 per cent of test clips that run
longer (the audio averages 20.2 seconds and ranges from 1.01 to 35.2). That is what
`waxal_asr/modeling/whisper.py` in the solution repository does, and rebuilding the submission
requires decoding this arm exactly that way.

For standalone reuse, windowing is worth having: a plain feature-extractor call truncates long audio
and raises no error, so long clips quietly lose their endings. The third-party Sunbird arm in the
same system decodes in overlapping windows of 28 seconds, 2 seconds of headroom below the limit,
with 2 seconds of overlap (`waxal_asr/modeling/sunbird.py`), and that implementation applies
unchanged to this checkpoint. The two arms differ on this deliberately: the submitted member of this
arm predates the windowing path, and the configuration that produced it is preserved as the
configuration of record. The CTC arms are convolutional and streaming over time, so they have no
equivalent limit and no windowing.

## Evaluation

| split | error, 0.5 * (WER + CER) |
|---|---|
| holdout, both languages | 0.2772 |

The holdout is a speaker-disjoint carve of the WAXAL **training** data. It is **not** the competition
test set, and it shares recording conditions and speakers' language variety with the training corpus,
so it flatters any arm tuned on that corpus and gives no independent read on generalisation. Lower is
better; the competition score is `1 - 0.5*(WER + CER)`.

This arm is the clearest illustration in the project of why holdout numbers were not trusted for
ranking. On holdout, 0.2772 is a strong figure, better than the sequenced CTC specialists' 0.2785 on
Lingala. On the public leaderboard the same fine-tune scored 0.7331, below the general
`waxal-w2vbert-linsnalug-raw` CTC arm at 0.742142, below the routed CTC specialists at 0.744835, and
below a third-party 51-language Whisper model used zero-shot at 0.737168. Architectural diversity was
attempted three times, here with Whisper-turbo, with XLS-R which was the weakest arm trained, and with
a fine-tune of that third-party 51-language Whisper model which lost to its own zero-shot output. None
of the three paid off as a solo model. Fresh random seeds on the strongest architecture were worth more,
at roughly +0.0032 per ensemble member.

## Intended use and limitations

Intended use is automatic speech recognition for Lingala and Shona photo-description speech:
spontaneous descriptions of images, on consumer recording equipment, averaging 20.2 seconds per clip.
The competition targets Lingala, Shona and Luganda; the corrected Phase 2 test set was measured to
contain only Lingala and Shona, which is why this arm was trained on those two. Whisper's own Luganda
support is untouched by this fine-tune but was not evaluated.

The submitted member was generated without a forced language token: the model inferred the language
per clip, matching the pipeline in the solution repository. For standalone reuse, supplying the
token helps, and in the final system the language labels exist anyway because the third-party
Sunbird arm is routed by open-set language identification: `facebook/mms-lid-4017` called 437 of
the 892 test clips Lingala at confidence 0.990 and 423 Shona at confidence 0.980, leaving 32
low-confidence clips, one of which it labelled Luganda.

This model is a weak component by solo leaderboard score, and that is not by itself disqualifying.
The submission's 26-member character-level ROVER core scored 0.764915, and the scored final
submission, recipe p2n_mbr, selects per clip across seven such ensembles and scored 0.766791 public
and 0.772552 private. Several members are deliberately weak
alone: the blank-penalty re-decodes of the CTC arms over-generate words and lose score in isolation
(0.743 against 0.746 for the same checkpoint) while contributing +0.0104 in total, because a character
vote can filter a spurious word and can never recover a missing one. What such a vote needs is members
of comparable strength with uncorrelated errors, and a Whisper-family member is uncorrelated with the
CTC family by construction.

Two decoding notes are specific to this family, and both are standalone-reuse advice rather than
the submission recipe. Beam search helps: beam width 8 was measured at +0.0224 over greedy for
Whisper models in this system (the Sunbird arm uses it; the submitted member of this arm was
greedy, and rebuilding the submission must keep it greedy). And encoder-decoder models loop:
collapsing repeated runs in the output was worth
+0.0069 when introduced. Reduplication is lexical in both languages, so loop collapse is applied after
generation rather than suppressed with an n-gram constraint during it.

## How to load

```python
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_ID = "anyantudre/waxal-whisper-turbo-linsna"   # or a local checkpoint directory

processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()

wav, sr = sf.read("clip.wav")          # mono, resample to 16 kHz before this call
inputs = processor(wav, sampling_rate=16000, return_tensors="pt")

# The settings of record, which reproduce the submitted ensemble member: greedy, no language
# forcing, and clips past 30 seconds truncated by the feature extractor. For standalone reuse,
# beam search (num_beams=8, +0.0224 measured), a language token from identification
# (language="ln" or "sn", task="transcribe"), and windowing long clips all help, but each one
# changes the output away from the submitted member.
ids = model.generate(
    inputs.input_features,
    num_beams=1,
    max_new_tokens=200,
)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```
