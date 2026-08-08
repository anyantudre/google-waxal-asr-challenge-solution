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

The `decode` block in the training configuration (`num_beams: 1`, `max_new_tokens: 200`) is a training-time setting
and is not the inference recipe. See below.

## Long-form windowing

This is the one operational difference between this arm and every CTC arm in the system, and it is not
optional. The Phase 2 test audio averages 20.2 seconds and ranges from 1.01 to 35.2 seconds, so 3.3 per
cent of clips run past 30 seconds, which is Whisper's entire receptive field. A plain feature-extractor
call truncates anything longer and raises no error, so those clips would quietly lose their endings.
Long audio is therefore decoded in overlapping windows of 28 seconds, 2 seconds of headroom below the
limit, with 2 seconds of overlap; the same windowing implementation is used by every Whisper-family
arm in the system. The CTC arms are convolutional and streaming over time, so they have
no equivalent limit and no windowing.

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

The model must be told which language to transcribe. In the final system that decision comes from
open-set language identification, not from the clip identifiers, which carry no language information:
`facebook/mms-lid-4017` called 437 of the 892 test clips Lingala at confidence 0.990 and 423 Shona at
confidence 0.980, leaving 32 low-confidence clips and one labelled Luganda.

This model is a weak component by solo leaderboard score, and that is not by itself disqualifying. The
final submission combines 26 character-level ROVER members, several of which are deliberately weak
alone: the blank-penalty re-decodes of the CTC arms over-generate words and lose score in isolation
(0.743 against 0.746 for the same checkpoint) while contributing +0.0104 in total, because a character
vote can filter a spurious word and can never recover a missing one. What such a vote needs is members
of comparable strength with uncorrelated errors, and a Whisper-family member is uncorrelated with the
CTC family by construction.

Two decoding notes are specific to this family. Beam search matters: beam width 8 was measured at
+0.0224 over greedy for Whisper models, so the training config's `num_beams: 1` must not be carried
into inference. And encoder-decoder models loop: collapsing repeated runs in the output was worth
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
# clips longer than 30 s must be split into overlapping windows before this call
inputs = processor(wav, sampling_rate=16000, return_tensors="pt")

ids = model.generate(
    inputs.input_features,
    language="ln",          # "ln" for Lingala, "sn" for Shona; supply from language identification
    task="transcribe",
    num_beams=8,
    max_new_tokens=200,
)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```
