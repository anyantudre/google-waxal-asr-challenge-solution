# Solution: Google WAXAL ASR Challenge

Username: anyantudre

Automatic speech recognition for **Lingala**, **Shona** and **Luganda**, built on the WAXAL corpus.
**Final result: 2nd place** of the competition, private leaderboard **0.772552** (CER 0.110664,
WER 0.344233), scored on the recipe `p2n_mbr`, whose public score was 0.766791 (CER 0.108130,
WER 0.358288).

The competition covers all three languages. Open-set language identification with two independent
models over the whole of the corrected Phase 2 test set found it to be roughly half Lingala and half
Shona, containing no Luganda. The ensemble is therefore tuned for Lingala and Shona, while the
training code and the root checkpoint keep full Luganda support. The measurement is in
[Languages](#languages).

Everything here runs from published weights. `make submission` rebuilds the best result, and the
ensemble that defines it is data rather than code, in
[configs/ensembles.yaml](configs/ensembles.yaml). [docs/SOLUTION.md](docs/SOLUTION.md) is the
reference record: every number quoted anywhere in this repository appears there with its provenance,
including how far each one sits from measurement noise.

## Summary

The competition metric is `1 - 0.5 * (WER + CER)`, and the two halves are computed on different
text: **WER on lowercased but otherwise unmodified text, CER on the raw string.** That single fact
drove most of the solution. A lowercase, punctuation-free model is capped on CER regardless of how
good its words are, and punctuation errors are charged twice, once to each metric. So the character
vocabulary of every CTC model here includes capitals and punctuation. That change alone was worth
0.0171 on the leaderboard.

The pipeline in one paragraph. The test set is 892 clips, roughly half Lingala and half Shona. Nine
acoustic models are fine-tuned from `facebook/w2v-bert-2.0` and `openai/whisper-large-v3-turbo` on
WAXAL plus five public external corpora, using a two-stage curriculum: external audio first, then a
final low-learning-rate pass over WAXAL alone. Two further checkpoints are weight averages over the
strongest lineage. Each model is decoded several times with different blank penalties, which
corrects a systematic word-dropping bias in greedy CTC decoding. The resulting transcript sets,
which include one third-party zero-shot model, feed seven character-level ROVER ensembles, six
anchored on the seed-43 arm and one on seed 46, plus a meta vote over five of them; the submission
keeps, for every clip, the candidate transcript most central by character edit distance to the 26
members of the corrected ensemble, then applies loop collapse and sentence-case repair.

Three findings did most of the work:

1. **Raw vocabulary (+0.0171).** CER is scored on raw text, so the model must be able to emit
   capitals and punctuation as ordinary output symbols.
2. **Blank-penalty decoding (+0.0104).** Greedy CTC drops a character whenever the blank symbol
   marginally outranks the best character, and often loses the whole word. Our output carried only
   97.9 per cent of the reference word count, 1.30 words per second against 1.41. Penalising the
   blank restores those words. The re-decoded models are *worse alone* but much stronger as ensemble
   members, because a vote can filter a spurious word but cannot recover a missing one.
3. **Ensemble composition.** Members must be strong and mutually decorrelated. Four arms from four
   different checkpoints gained 0.0040, while six arms from two checkpoints gained 0.0006, and
   adding arms from weak checkpoints lost 0.0003 (0.764476 down to 0.764152, with four weak members
   added and nothing else changed).

A full account of what was tried, including the measured negative results, is in
[docs/SOLUTION.md](docs/SOLUTION.md). That document is also the reference record for every number
quoted anywhere in this repository.

## Repository layout

```
configs/              Training configuration, one YAML per arm
  ensembles.yaml      Ensemble recipes: which arms vote, at which blank penalty
data/                 Not in the repository; every subdirectory is created on demand
  raw/                Competition data, placed here by you: test audio and Test.csv
  external/           Public corpora fetched by `make data`
  interim/            Per-arm transcripts and the language map
  processed/          The final submission CSV
docs/                 Solution write-up, inference guide, conventions
  SOLUTION.md         The detailed account, and every published number with its provenance
  dataset_card.md     The training corpus: every subset's derivation from its upstream source
  model_cards/        Cards for the eleven checkpoints, kept here so the repository is
                      self-contained; the same cards are published on the Hub repositories
models/               Downloaded or trained weights, also created on demand
tests/                Test suite, runs without a GPU or any downloads
waxal_asr/            The package
  config.py           Paths, the model registry, YAML loading
  data.py             Training data loading and the speaker-disjoint holdout
  audio.py            Loading, resampling, augmentation
  normalize.py        Text normalisation policy
  vocab.py            Character vocabulary built from the training transcripts
  metrics.py          The competition metric, including its asymmetry
  decode.py           CTC decoding and the blank penalty
  ensemble.py         Character-level ROVER
  postprocess.py      Loop collapse and sentence case
  lid.py              Open-set language identification
  analysis.py         Data insights report
  train.py            Training loop, architecture agnostic
  infer.py            Inference used during training and evaluation
  models/             Model adapters (CTC, sequence to sequence)
  modeling/
    train.py          Training entry point
    predict.py        Inference entry point
    ctc.py            Batched CTC transcription
    whisper.py        The fine-tuned Whisper arm: greedy, truncates past 30 seconds
    sunbird.py        The third-party arm, with long-form windowing
```

# Setup

1. **Python 3.11 or 3.12.** The submitted result was produced with Python 3.12.13. The pinned
   numpy needs at least 3.11, and the pinned audiomentations does not install on 3.13.

2. **Install the dependencies.**

   ```bash
   python -m venv .venv && source .venv/bin/activate
   make requirements
   ```

   On Windows the activation command is `.venv\Scripts\activate`.

   For an NVIDIA RTX 5090 or any other Blackwell card, torch must be at least 2.7 built against
   CUDA 12.8. Older builds do not support compute capability 12.0 and will fail at the first
   forward pass.

   ```bash
   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
   ```

3. **No system packages are required.** Audio decoding uses `soundfile` and `librosa`, both of which
   install from PyPI. FFmpeg is not needed.

4. **Expected layout before inference.** Place the competition data as follows:

   ```
   data/raw/
     Test.csv            Column: ID
     test_audio/         One file per ID, for example ID_AAOODF.wav
   ```

   The audio may be WAV, MP3 or FLAC at any sample rate. The loader discovers the extension and
   resamples to 16 kHz.

5. **Model weights** download automatically from the Hugging Face Hub on first use. To fetch them
   ahead of time, or to work offline:

   ```bash
   huggingface-cli download anyantudre/waxal-w2vbert-linsna-seed43 --local-dir models/s43
   export WAXAL_MODELS_DIR=models
   ```

   Any arm present under `WAXAL_MODELS_DIR` is loaded from disk; the rest come from the Hub.

# Hardware

Training and inference ran on a single machine:

- **GPU:** 1x NVIDIA RTX 5090, 32 GB VRAM
- **CPU:** 32 cores
- **RAM:** 92 GB
- **Scheduler:** SLURM, 4 hour wall-clock limit per job, every run resumable from its last checkpoint

Ensembling and post-processing are CPU-only and were run on a laptop. No cloud or paid compute was
used at any point.

| Step | Time |
|---|---|
| Training one CTC arm (8 epochs, roughly 31,000 clips) | 8 to 10 hours |
| Training the Whisper-turbo arm | 12 hours |
| Language identification over 892 clips | 25 minutes |
| Inference, one arm over 892 clips | 20 minutes |
| Blank-penalty re-decode, per checkpoint | 90 seconds (logits are cached, then decoded per penalty) |
| Ensembling, selection and post-processing | seconds |
| Training-corpus download, `make data` | about an hour (14 GB), training only |
| **Full submission (`p2n_mbr`) from cached arm outputs** | **under 5 minutes** |
| **Full submission (`p2n_mbr`) from scratch, 12 checkpoints** | **about 7 hours, plus `make lid`** |

Total training was roughly 80 GPU hours. The competition score can be reproduced without any of it,
using the published weights.

# Run training

Training is not needed to reproduce the submission, since all weights are published. These are the
steps that produced them.

1. **Build the corpora.** The public sources sum to about 451 hours of audio; no single arm trains
   on all of them.

   ```bash
   make data
   ```

   This downloads the packaged training corpus `anyantudre/waxal-linsna` (about 14 GB: 16 kHz mono
   FLAC, already converted, with a manifest per corpus) into `data/external`. The WaxalNLP train and
   validation splits are not included; training reads them directly from `google/WaxalNLP`. How each
   subset was derived from its upstream source, including the pseudo-label recipe, is documented in
   [docs/dataset_card.md](docs/dataset_card.md). Sources, licences and row counts are in
   [Data and licences](#data-and-licences) below and in [docs/SOLUTION.md](docs/SOLUTION.md). All
   are public. Network access is required.

2. **Train one arm.**

   ```bash
   make train ARM=s43
   ```

   `ARM` is the config stem after `w2vbert_`: `s43`, `s44`, `p1raw`, `linspec`, `linspec_r`,
   `snaspec`, `snaspec_r`, `p1av`. The Whisper arm's configs carry no such prefix, so train it by
   calling the module directly with `configs/turbo_linsna.yaml` and then `configs/turbo_linsna_r.yaml`.
   The distilled arm has no config here because it trains on the ensemble's own transcripts of the
   test clips, which only exist once the ensemble has been run; its weights are published, and its
   full configuration is in its model card. The seed-46 arm reruns the seed-43 configuration at
   seed 46, and the two soup checkpoints are uniform weight averages, both documented in their
   cards. Weights are written to `models/<arm>/` and are roughly 2.3 GB per CTC arm, 3.2 GB for
   the Whisper arm. Seeds are fixed per arm in the config. Training is repeatable in configuration
   but not bitwise: GPU kernel selection makes retrained weights differ (see the republished p1av
   arm in [docs/SOLUTION.md](docs/SOLUTION.md)), which is why the published weights are the
   reproduction path.

3. **Two-stage arms.** The per-language specialists train in two passes, external audio then a WAXAL
   refresh. The order matters and was measured: mixing the corpora throughout scored 0.2905 on the
   holdout, while training in this order scored 0.2785.

   ```bash
   make train ARM=linspec      # stage 1, AfriVoice Lingala
   make train ARM=linspec_r    # stage 2, WAXAL refresh
   ```

# Run inference

**Quick verification**, one arm, about 2 GB of weights and 20 minutes:

```bash
make predict
```

**Reproduce the submitted result**, recipe `p2n_mbr`, a per-clip selection across seven ensembles:

```bash
make lid
make submission
```

`make lid` must run first: it writes the language map that routes the third-party Sunbird arm per
clip. Both write `data/processed/submission.csv` with columns `ID` and `Target`.

Ensembles are defined in [configs/ensembles.yaml](configs/ensembles.yaml), not in code, so a
different combination needs no Python change. Each recipe is named after the submission file that
was sent to Zindi, so a recipe maps to exactly one leaderboard row:

```bash
make recipes
```

| recipe | members | public score |
|---|---|---|
| `p2n_mbr` | selection | **0.766791**, the best result, run by `make submission` |
| `p2n_meta` | 5 ensembles | 0.766683, a vote over five complete ensembles |
| `p2n_ens_weighted` | 26 | 0.766580, the two weakest members at half weight |
| `p2n_ens_masked` | 26 | 0.766563 |
| `p2n_ens_s46swap` | 26 | 0.766477, seed 46 in place of the weakest arm |
| `p2n_ens_soup6` | 26 | 0.766226, the six-way soup in place of the five-way |
| `p2n_ens_s46anchor` | 26 | 0.766083, seed 46 as the anchor, a `p2n_mbr` candidate |
| `p2n_ens_wide` | 38 | 0.765960, penalties widened to 0.5 and 2.5 |
| `p2n_ens_distil` | 26 | 0.764915, an earlier decoding path, kept as a fixed reference |
| `p2n_ens_bp25` | 25 | 0.764759, without the distilled arm |
| `p2n_ens_bp10` | 17 | 0.764476 |
| `p2n_distil_nl_f` | 1 | 0.746787, the strongest single checkpoint, run by `make predict` |

A recipe lists models, other recipes, or a selection. `p2n_meta` votes over five complete
ensembles; `p2n_mbr` goes one step further and selects, per clip, the transcript most central to
the 26 members by character edit distance. Voting reduces bias, the second-level vote reduces
variance, and the selection optimises expected character error directly. The selection was measured
twice with different candidate pools, scoring 0.766791 and 0.766773.

`make submission` runs `p2n_mbr`. To run another recipe, call the module directly:

```bash
python -m waxal_asr.modeling.predict --recipe p2n_ens_bp25
```

What happens during `make submission`:

1. Each checkpoint is loaded once and every clip is transcribed. Logits are decoded again at each
   blank penalty in the recipe, which is cheap because the forward pass is not repeated.
2. Each arm's transcripts are cached in `data/interim/<arm>_bp<penalty>.json`. Re-running skips any
   arm already cached, so an interrupted run resumes rather than restarting. Delete the cache to
   force a fresh decode.
3. Each of the seven candidate ensembles is combined by character-level ROVER with the vote
   threshold at 2.0, six anchored on the seed-43 arm and one on seed 46; `p2n_meta` additionally
   votes over five of them. The seed-43 anchor is deliberately not the strongest arm: anchoring on
   the strongest scored 0.763419 against 0.764915. The threshold and the skeleton are recorded in
   the recipe file rather than left to the vote function.
4. For every clip, the candidate transcript with the smallest mean normalised character edit
   distance to the 26 members of `p2n_ens_masked` is kept.
5. Loop collapse and sentence-case repair are applied to every member and again to every voted text.
6. The result is validated before it is written: row count, identifier set, and no empty cell.

Interim files total roughly 20 MB. The final CSV is about 300 KB.

A general guide to running inference on new audio, including single clips and other languages, is in
[docs/INFERENCE.md](docs/INFERENCE.md).

**Troubleshooting.**

- *403 from the Hugging Face Hub.* Check the repository name and your network; the model
  repositories are public. If the Hub is unreachable, download the weights manually and set
  `WAXAL_MODELS_DIR`.
- *CUDA out of memory.* Lower the batch size: `--batch-size 4`. Inference fits in 8 GB at batch 4,
  measured on an RTX 4070; the default of 8 does not fit there. The batch size does not change the
  transcripts of any recipe except the legacy unmasked members of `p2n_ens_distil`.
- *No audio file found for ID.* The loader expects `data/raw/test_audio/<ID>.<ext>`. Check that the
  identifiers in `Test.csv` match the filenames.
- *An empty cell in the submission.* One test clip is 1.01 seconds long and some models return
  nothing for it. `predict.py` fails loudly rather than writing an invalid file; the ensemble path
  fills such cells from the anchor.
- *No GPU.* Everything runs on CPU, roughly 20 times slower. The test suite needs no GPU at all.

# Languages

The competition covers Lingala, Shona and Luganda. Two independent open-set language identification
models, [facebook/mms-lid-4017](https://huggingface.co/facebook/mms-lid-4017) and
[facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256), were run over all 892 corrected
Phase 2 test clips. Both put the set at roughly half Lingala and half Shona. The 34 clips that
either model declined to call Lingala or Shona were inspected individually and are all
neighbouring-language confusions rather than coverage gaps. **The single clip labelled Luganda,
`ID_REPXZM`, is Lingala**, and all eight arms agree on it. There is no Luganda in this test set and
therefore no Luganda coverage gap in the submission. The per-model counts, their confidences and the
inspection are in [docs/SOLUTION.md](docs/SOLUTION.md#test-set-composition).

Luganda support is nonetheless retained where it costs nothing. `waxal-w2vbert-linsnalug-raw`, the
root checkpoint of the whole family, was trained on all three languages and is published as such;
nothing in the package hardcodes a language, and retargeting is described in
[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

# Models

Eleven checkpoints are published, named `waxal-<architecture>-<languages>-<variant>` so the languages
a checkpoint covers are readable from its name. One third-party model is used zero-shot.

| checkpoint | languages | role | card |
|---|---|---|---|
| `waxal-w2vbert-linsnalug-raw` | ln, sn, lg | root of the family, the raw-vocabulary gain | [card](docs/model_cards/waxal-w2vbert-linsnalug-raw.md) |
| `waxal-w2vbert-linsna-seed43` | ln, sn | ensemble anchor | [card](docs/model_cards/waxal-w2vbert-linsna-seed43.md) |
| `waxal-w2vbert-linsna-seed44` | ln, sn | widest corpus, decorrelated on two axes | [card](docs/model_cards/waxal-w2vbert-linsna-seed44.md) |
| `waxal-w2vbert-lin-specialist` | ln | Lingala branch of the routed pair | [card](docs/model_cards/waxal-w2vbert-lin-specialist.md) |
| `waxal-w2vbert-sna-specialist` | sn | Shona branch of the routed pair | [card](docs/model_cards/waxal-w2vbert-sna-specialist.md) |
| `waxal-w2vbert-linsna-afrivoicemix` | ln, sn | the mixed-curriculum counter-experiment | [card](docs/model_cards/waxal-w2vbert-linsna-afrivoicemix.md) |
| `waxal-w2vbert-linsna-soup5` | ln, sn | five-way weight average | [card](docs/model_cards/waxal-w2vbert-linsna-soup5.md) |
| `waxal-w2vbert-linsna-distilled` | ln, sn | strongest single checkpoint, see its disclosure | [card](docs/model_cards/waxal-w2vbert-linsna-distilled.md) |
| `waxal-w2vbert-linsna-seed46` | ln, sn | fresh seed, the most decorrelated arm | [card](docs/model_cards/waxal-w2vbert-linsna-seed46.md) |
| `waxal-w2vbert-linsna-soup6` | ln, sn | six-way weight average | [card](docs/model_cards/waxal-w2vbert-linsna-soup6.md) |
| `waxal-whisper-turbo-linsna` | ln, sn | architectural diversity, the one seq2seq fine-tune | [card](docs/model_cards/waxal-whisper-turbo-linsna.md) |

# Testing

```bash
make test
```

90 tests covering the metric asymmetry, the ensemble vote and its safety guards, both
post-processing rules, the submission validator, the recipe file's contents, and the formatting
rules this repository enforces. They use synthetic data, so they need no GPU, no model weights and
no network access, and run in a few seconds.

# Data and licences

Every source is publicly available. Licences are as published by each dataset. The derived
training corpus that `make data` downloads is published as
[anyantudre/waxal-linsna](https://huggingface.co/datasets/anyantudre/waxal-linsna); its derivation
from the sources below is documented in [docs/dataset_card.md](docs/dataset_card.md).

| source | rows | hours | licence |
|---|---|---|---|
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona train and validation | 33,002 | about 179 | CC-BY-4.0 |
| [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona test split | 3,615 | 19.2 | CC-BY-4.0 |
| [KasuleTrevor/Lingala_100hrs](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs), AfriVoice Lingala | 22,131 | 104.5 | CC-BY-4.0 |
| [realtime-speech/shona1](https://huggingface.co/datasets/realtime-speech/shona1), AfriVoice Shona | 15,923 | 91.0 | CC-BY-4.0 |
| [shunyalabs/lingala-speech-dataset](https://huggingface.co/datasets/shunyalabs/lingala-speech-dataset) | 4,341 | 21.5 | see the dataset card |
| [google/fleurs](https://huggingface.co/datasets/google/fleurs), config `ln_cd` | 2,847 | 15.0 | CC-BY-4.0 |
| [google/fleurs](https://huggingface.co/datasets/google/fleurs), config `sn_zw` | 3,704 | 15.0 | CC-BY-4.0 |
| [asr-africa/ASRAfricaDataEfficiencyBenchmark](https://huggingface.co/datasets/asr-africa/ASRAfricaDataEfficiencyBenchmark), config `Shona` | 1,055 | 6.0 | see the dataset card |

Models and augmentation sources:

| source | licence |
|---|---|
| [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) | MIT |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | Apache-2.0 |
| [facebook/mms-lid-4017](https://huggingface.co/facebook/mms-lid-4017) and [facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256) | CC-BY-NC-4.0 |
| [Sunbird/asr-whisper-51-african-languages](https://huggingface.co/Sunbird/asr-whisper-51-african-languages) | see the model card, pinned to revision `5d4f0038` |
| MUSAN noise (OpenSLR 17) | CC-BY-4.0 |
| Simulated room impulse responses (OpenSLR 28) | Apache-2.0 |

FLEURS is read using its `raw_transcription` field, which preserves casing and punctuation. The
lowercased `transcription` field would cap CER.

Sunbird 51 is pinned by revision because its card states the weights will be replaced; an unpinned
load would silently change the submission.

# Disclosures

- **The WAXAL Phase-1 test split was used for training.** Phase 2 permits this, and the organisers
  confirmed it publicly. It contributes 3,615 clips and 19.2 hours of gold reference text.
- **Pseudo-labels were used**, generated by our own models. Two kinds: over WAXAL's *unlabeled*
  pools, and over the 892 Phase 2 test clips.
- **One checkpoint, `waxal-w2vbert-linsna-distilled`, is trained on the 892 Phase-2 test clips**,
  using our own ensemble's transcripts as targets and mixed with WAXAL gold. This is self-training
  on unlabeled test audio, which the organisers explicitly permit subject to disclosure. **No
  reference transcript for any test clip was used, and none exists publicly.** That checkpoint is
  the strongest single arm (0.746787) and votes as one of 26 members. An otherwise identical
  ensemble without it scores 0.000156 lower, which is well inside measurement noise. Its card
  states the caveats in full.
- **Blank-penalty and token-penalty decoding are decoding parameters, not adaptation.** They change
  how logits are turned into text and involve no training and no test-set statistics.
- **Provenance was verified.** The test clips were fingerprinted against every transcribed AfriVoice
  clip, roughly 40,000 in total, with both an exact and a tolerant matcher. There were no matches,
  so no test clip has a public transcription. This check is independent of the training procedure.

# A note on measurement

The public leaderboard covers roughly 30 per cent of the test set, about 268 clips. For a paired
comparison between two similar systems the standard error on the score difference is about 0.0013,
so **differences below roughly 0.0027 cannot be distinguished from noise**. The last-day candidate
submissions span 0.7643 to 0.7668, and most neighbouring pairs in that range are inside the band.
The two final picks were chosen on the principle of maximum averaging, the selection recipe
`p2n_mbr` first because it also optimises expected character error directly, rather than on ranking
within the noise, and any claim in this repository that rests on a difference smaller than the band
is labelled as such where it appears.
