# INFERENCE - reproducing any recorded submission from released weights

This document answers "how do I run it". It gives the exact commands, arguments, file paths,
schemas, runtimes, failure modes and the pre-submission checklist.

It deliberately does not argue for the design. Why blank-penalty decoding exists, why language
identification is open set, how ensemble members were selected, and what was tried and rejected are
all in [SOLUTION.md](SOLUTION.md), which is the single source of truth for facts, measurements and
reasoning. Where a number appears below it is quoted from there.

Nothing here requires training. Every published checkpoint is on the Hugging Face Hub and downloads
on first use. The path is deterministic: the CTC arms decode greedily, the Whisper-based arm uses
fixed-width beam search with sampling off, and the vote is a pure function of its members, so
re-running a recipe on the same weights and the same audio reproduces the same CSV. Seeds are fixed
per arm (42, 43, 44); they matter for training, not for this path.

## 1. The recorded submissions

Four recipes are recorded, each named after the submission file that was sent to Zindi, so a recipe
maps to exactly one leaderboard row.

| recipe | members | public score | CER | WER |
|---|---|---|---|---|
| **`p2n_ens_distil`** | 26 | **0.764915** | 0.108961 | 0.361209 |
| `p2n_ens_bp25` | 25 | 0.764759 | 0.109025 | 0.361457 |
| `p2n_ens_bp10` | 17 | 0.764476 | 0.108578 | 0.362471 |
| `p2n_distil_nl_f` | 1 | 0.746787 | 0.113117 | 0.393308 |

Each one is reproduced by a single command, which differs only in the recipe name:

```
python -m waxal_asr.modeling.predict --recipe p2n_ens_distil
python -m waxal_asr.modeling.predict --recipe p2n_ens_bp25
python -m waxal_asr.modeling.predict --recipe p2n_ens_bp10
python -m waxal_asr.modeling.predict --recipe p2n_distil_nl_f
```

**`p2n_ens_masked` at 0.766563 is the best result, and it is what `make submission` runs.** It is
the 26-member vote with a correct forward pass throughout: the attention mask is passed for every
member, and the Whisper arm is decoded as a generator rather than through the CTC path.

`p2n_ens_distil` is the same 26 members with two differences, both recorded in the recipe: its
blank-penalty members omit the attention mask, and it predates the Whisper decoding path. It scores
0.764915 and is kept because reproducing a known result exactly is a useful check on an
installation. `p2n_ens_bp25` drops the distilled arm; `p2n_ens_bp10` is a 17-member configuration.
`p2n_distil_nl_f` is a single arm, the strongest single checkpoint, and the cheapest way to verify
the pipeline without downloading nine sets of weights; `make predict` runs it.

One caveat applies to every recipe. The `p1av` checkpoint published here is a retrain rather than
the original weights, so any ensemble containing it differs slightly from the corresponding
historical file. For `p2n_ens_distil` the difference is 0.000141, about a tenth of the standard
error. It is described under
[The republished p1av arm](SOLUTION.md#the-republished-p1av-arm).

### Where recipes live

Recipes are data, not code. They are defined in [`configs/ensembles.yaml`](../configs/ensembles.yaml)
and read at run time by `waxal_asr/modeling/predict.py`, so **adding a recipe needs no Python
change**: append an entry under `recipes:` and pass its name to `--recipe`.

To see what is available, without loading any weights:

```
python -m waxal_asr.modeling.predict --list        # or: make recipes
```

```
recipes in configs/ensembles.yaml:
  p2n_ens_distil     26 member(s)  public 0.764915  26 members, the blank-penalty core plus the distilled arm
  p2n_ens_bp25       25 member(s)  public 0.764759  25 members, the blank-penalty core
  p2n_ens_bp10       17 member(s)  public 0.764476  17 members, the configuration that established source diversity
  p2n_distil_nl_f     1 member(s)  public 0.746787  the distilled arm on its own
```

A recipe is a list of members. Each member is an `arm` key plus an optional `blank_penalty`, and the
**first member is the anchor**: its text survives every slot unless the others outvote it, so member
order is significant. A penalty of 0, or none at all, means ordinary greedy decoding.

```yaml
  my_recipe:
    description: two arms, one of them re-decoded
    members:
      - {arm: s43}                     # anchor, greedy
      - {arm: s43, blank_penalty: 1.5}
      - {arm: soup5}
```

The `defaults` block at the top of the file supplies `vote_threshold: 2.0` and `skeleton: anchor`
for every recipe. Both differ from the underlying vote function's own defaults, which is why they are
recorded in the file rather than left implicit. A recipe may override either key inline.

`tests/test_submission_pipeline.py` asserts that the recipe file parses, that the three submitted
recipes are still present, that `p2n_ens_distil` still has 26 members anchored on `s43`, and that
every arm named in a recipe matches the registry in `waxal_asr/config.py`.

## 2. Setup

### Package versions

The submitted result was produced with Python 3.12.13, torch 2.11.0+cu128, transformers 4.57.6,
datasets 3.6.0, soundfile 0.14.0, librosa 0.11.0 and numpy 2.4.6. These pins are in
[`requirements.txt`](../requirements.txt).

Install torch first, from the CUDA 12.8 index. The GPU used is Blackwell (sm_120) and the default
PyPI wheel does not carry that architecture:

```
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt        # or: make requirements
```

No system packages are needed. Audio decoding goes through `soundfile` and `librosa`; FFmpeg is not
required.

### Hardware

One NVIDIA RTX 5090 (32 GB VRAM), 32 CPU cores, 92 GB RAM. Inference is single-GPU throughout, and
the vote and post-processing are CPU only. Everything runs on CPU if no GPU is present, roughly
twenty times slower.

### Environment variables

```
export HF_TOKEN=<hugging face token>                    # model downloads
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WAXAL_MODELS_DIR=models                          # optional, load weights from disk
```

`WAXAL_MODELS_DIR` is the offline switch. `waxal_asr.config.resolve_model` returns
`<WAXAL_MODELS_DIR>/<arm>` when that directory contains a `config.json`, and the Hub id otherwise, so
arms can be mixed between disk and Hub with no flag changes. `WAXAL_ROOT` relocates the whole data
tree the same way.

### Input layout

`predict.py` defaults to the layout below, so with the data in place the command in Section 1 takes
no path arguments:

```
data/raw/
  Test.csv            one column, ID
  test_audio/         one file per ID, for example ID_AAOODF.wav
```

The corrected Phase 2 test set is 892 clips of 48 kHz WAV, mean duration 20.2 seconds, range 1.01 to
35.2 seconds, of which 3.3 per cent run past 30 seconds. Nothing needs converting first. Audio
loading is centralised in `waxal_asr.audio.load_clip`, which every arm uses and which

- resolves `<audio_dir>/<id>.*` by glob, escaping the id and sorting matches so a duplicated id with
  two extensions resolves deterministically,
- reads with `soundfile` as float32,
- downmixes to mono by averaging channels,
- resamples with `librosa` to 16 kHz.

The extension is discovered rather than assumed: the competition shipped WAV for the corrected test
set and MP3 earlier, and the corpora built for training are FLAC. The source files are read as
shipped and never rewritten.

Only the `ID` column of `Test.csv` is read, so a placeholder `Target` column (as in the sample
submission) is ignored. **Output row order follows `Test.csv`**, not the directory listing.

If the CSV is unavailable it can be rebuilt from the audio directory:

```
python - <<'PY'
import csv, glob, os
ids = sorted(os.path.splitext(os.path.basename(f))[0]
             for f in glob.glob("data/raw/test_audio/*"))
with open("data/raw/Test.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["ID"]); w.writerows([[i] for i in ids])
print(f"[ids] {len(ids)} -> data/raw/Test.csv")
PY
```

### Weights

Nine checkpoints are published under the `anyantudre` namespace, plus one third-party model used
zero-shot. The registry is `ENSEMBLE_ARMS` in `waxal_asr/config.py`, and the same mapping is repeated
in the `arms:` block of `configs/ensembles.yaml` for readers who do not want to read Python; a test
keeps the two from drifting apart.

| arm key | repository | role |
|---|---|---|
| `s43` | `anyantudre/waxal-w2vbert-linsna-seed43` | anchors every ensemble here |
| `s44` | `anyantudre/waxal-w2vbert-linsna-seed44` | widest corpus |
| `soup5` | `anyantudre/waxal-w2vbert-linsna-soup5` | five-way weight average |
| `p1raw` | `anyantudre/waxal-w2vbert-linsnalug-raw` | root of the family |
| `linspec_r` | `anyantudre/waxal-w2vbert-lin-specialist` | Lingala specialist |
| `snaspec_r` | `anyantudre/waxal-w2vbert-sna-specialist` | Shona specialist |
| `p1av` | `anyantudre/waxal-w2vbert-linsna-afrivoicemix` | mixed-curriculum counter-experiment |
| `distil` | `anyantudre/waxal-w2vbert-linsna-distilled` | strongest single checkpoint |
| `turbo_linsna_r` | `anyantudre/waxal-whisper-turbo-linsna` | Whisper large-v3-turbo |
| `sunbird51` | `Sunbird/asr-whisper-51-african-languages` | third party, zero-shot, pinned |

A CTC arm is roughly 2.3 GB and the Whisper arm roughly 3.2 GB. To fetch one ahead of time:

```
huggingface-cli download anyantudre/waxal-w2vbert-linsna-seed43 --local-dir models/s43
export WAXAL_MODELS_DIR=models
```

The eight w2v-BERT arms share a 97-token raw character vocabulary that includes the 26 capitals and
`! , . : ; ?`.

## 3. What the command does

`waxal_asr/modeling/predict.py` is the only entry point. Given a recipe it runs these stages in
order.

| stage | module | what it does |
|---|---|---|
| 1 | `waxal_asr/modeling/ctc.py` | one forward pass per CTC arm, batched, over every clip |
| 2 | `waxal_asr/decode.py` | greedy arg max, optionally after subtracting the blank penalty |
| 3 | `waxal_asr/modeling/sunbird.py` | the third-party zero-shot arm, windowed, beam 8 |
| 4 | `waxal_asr/postprocess.py` | loop collapse then sentence case, on every member |
| 5 | `waxal_asr/ensemble.py` | character-level ROVER over the members |
| 6 | `waxal_asr/postprocess.py` | the same two rules again, on the voted text |
| 7 | `waxal_asr/modeling/predict.py` | validate, then write `data/processed/submission.csv` |

Stages 4 and 6 run the identical function; both rules are idempotent, so the second application is a
safety net rather than a second edit.

### Command line

```
python -m waxal_asr.modeling.predict \
    --recipe p2n_ens_distil \
    --audio-dir data/raw/test_audio \
    --test-csv data/raw/Test.csv \
    --out data/processed/submission.csv \
    --recipes-file configs/ensembles.yaml
```

Every flag above is at its default, so the short form in Section 1 is equivalent. `--list` prints the
recipe table and exits. `--recipe` defaults to `p2n_distil_nl_f`, the cheap single-arm recipe, so an
argument-free invocation cannot accidentally start a nine-checkpoint download.

The console output identifies the recipe and the anchor before any weights load:

```
[predict] 892 clips, recipe p2n_ens_distil
[predict] recipe with 26 member(s), anchor is s43
[ctc] 8/892
```

### The per-arm cache

Every `(arm, penalty)` pair is written to `data/interim/<arm>_bp<penalty>.json` as a flat mapping of
clip id to post-processed transcript, with the decimal point removed from the penalty:

```
data/interim/s43_bp0.json     penalty 0, greedy
data/interim/s43_bp1.json     penalty 1.0
data/interim/s43_bp15.json    penalty 1.5
data/interim/s43_bp2.json     penalty 2.0
```

This is the resumability mechanism and it matters operationally in three ways. An interrupted run
resumes instead of restarting. A second recipe that shares an arm reuses it, which is why running all
four recipes costs barely more than running `p2n_ens_distil` alone. And **a stale cache is silently
authoritative**: if weights or audio change, delete the affected JSON files or the old transcripts
will be voted on. `[predict] s43_bp15: reusing cache` on the console is the signal that no forward
pass happened.

Interim files total roughly 20 MB for the 26-member recipe. The final CSV is about 300 KB.

### Blank-penalty members

A member with `blank_penalty: 1.5` subtracts 1.5 from the CTC blank logit before the arg max, which
restores characters, and often whole words, that greedy decoding drops. It is a decoding parameter:
no model is updated and no test transcription is used. The measurement behind it, and the reason
these members are worth having even though the re-decoded arms score worse on their own, are in
[SOLUTION.md](SOLUTION.md#blank-penalty-decoding).

Operationally there are two things to know. The forward pass is shared, so decoding an arm at four
penalties costs one pass plus four cheap arg maxes, which is what makes 26 members affordable. And
the penalty is chosen by matching the output word rate to the reference rate of 1.41 words per
second, not by tuning against the leaderboard; `waxal_asr.decode.words_per_second` computes the
figure to compare, and `REFERENCE_WORDS_PER_SECOND` in the same module is the target. A penalty of
1.5 brought a representative arm to 1.417 words per second. The penalties used in the recorded
recipes are 1.0, 1.5 and 2.0.

## 4. The Sunbird-51 arm

This is the one third-party model in the ensemble, used zero-shot, and it is the only arm with
operational quirks that a reader loading it directly will hit. `waxal_asr/modeling/sunbird.py`
handles all of them; this section exists so the failures are recognisable if the model is loaded any
other way.

**Pinned revision.** The model card states the weights will be replaced. `waxal_asr/config.py` pins
`SUNBIRD_51_REVISION = "5d4f0038"` and passes it to `from_pretrained`. Reproduction requires that
revision; `main` is not a stable target.

**Tokenizer, fast only.** The repository ships `tokenizer.json` but no `vocab.json` or `merges.txt`,
so the slow `WhisperTokenizer` receives `vocab_file=None` and raises
`TypeError: expected str, bytes or os.PathLike object, not NoneType`. The module loads
`WhisperTokenizerFast` explicitly rather than going through `AutoProcessor` or `pipeline`.

**Tokenizer, `extra_special_tokens`.** The repository's `tokenizer_config.json` stores
`extra_special_tokens` as a list, while transformers calls `.keys()` on it. That is the real source
of the `AttributeError: 'list' object has no attribute 'keys'` this checkpoint is known for. Passing
an explicit empty mapping overrides the broken field before it is read:

```python
tokenizer = WhisperTokenizerFast.from_pretrained(SUNBIRD_51, extra_special_tokens={})
```

**Generation config, `lang_to_id`.** The same repository ships `generation_config.lang_to_id` as a
list where transformers expects a mapping, so any `generate()` call touching language selection
fails. It is rebuilt after loading and the stale forced ids are cleared:

```python
generation = model.generation_config
if isinstance(getattr(generation, "lang_to_id", None), list):
    generation.lang_to_id = {t: tokenizer.convert_tokens_to_ids(t) for t in generation.lang_to_id}
generation.forced_decoder_ids = None
```

**Long-form windowing.** Whisper's receptive field is 30 seconds and a plain feature-extractor call
truncates anything longer without raising, so the tail of a long clip would simply vanish. The module
decodes any clip up to `WINDOW_SECONDS = 28.0` in one pass, and otherwise cuts it into 28 second
windows with `OVERLAP_SECONDS = 2.0` of overlap and joins the pieces. Decoding is `num_beams = 8`,
`max_new_tokens = 200`, fp16.

**Optional language forcing.** `transcribe_sunbird` accepts a `lang_map` argument: a mapping of clip
id to `{"lang": ..., "conf": ...}`. When supplied, a clip whose language is `lin` or `sna` with
confidence at least 0.5 has that Whisper language token forced, and every other clip falls back to
the model's own detection. Only Lingala and Shona are wired up, because they are genuine Whisper
language tokens in this checkpoint; several of the 51 languages it supports occupy repurposed slots
and would need raw token ids instead. `waxal_asr.modeling.sunbird.load_lang_map` reads such a mapping
from JSON, accepting either a bare mapping or a document with a `per_clip` key.

**`predict.py` supplies the map.** The submitted member was language routed, so running the arm
unrouted would produce a different member and therefore a different ensemble. `predict.py` reads
`data/interim/lid.json` and fails with an explicit message if it is absent, rather than silently
falling back to the model's own detection. Any recipe containing `sunbird51` therefore needs
`make lid` to have run first; `p2n_distil_nl_f` does not, because it has a single CTC member.

## 5. Language identification

Open-set language identification is what established the composition of the test set, and it is
reported in [SOLUTION.md](SOLUTION.md#test-set-composition): `facebook/mms-lid-4017` returns 437
Lingala (49.0 per cent, confidence 0.990) and 423 Shona (47.4 per cent, confidence 0.980), with 32
low-confidence clips and 1 Luganda; `facebook/mms-lid-256` returns 440 and 440 (49.3 per cent each,
confidence 0.996) with 12 low confidence. Both agree the set is half Lingala and half Shona.

That measurement is both an input to the design and a step in the recipes. **The three ensemble
recipes need a language map**, because each contains the routed `sunbird51` arm;
`p2n_distil_nl_f` does not. Produce it with:

```bash
make lid
# or: python -m waxal_asr.lid --audio-dir data/raw/test_audio --out data/interim/lid.json
```

`waxal_asr/lid.py` runs the open set, so a clip that is neither Lingala nor Shona is reported as
what it is rather than forced into the nearer of the two. `waxal_asr/config.py` records
`LID_MODEL = "facebook/mms-lid-4017"` as the checkpoint of record.

The map format that `load_lang_map` accepts is:

```json
{
  "per_clip": {
    "ID_AAOODF": {"lang": "lin", "conf": 0.9989},
    "ID_AAPBIE": {"lang": "sna", "conf": 0.9961}
  }
}
```

`lang` is an ISO-639-3 code and `conf` is the model's confidence. A low-confidence prediction should
be treated as unknown rather than as a language: `mms-lid-4017` spreads probability across 4017
classes, so its tail predictions sit far below the 0.99 the two main classes reach.

## 6. Post-processing

Two rules, in this order, both in `waxal_asr/postprocess.py` and both applied automatically by
`predict.py` to every member and again to the voted text. Neither needs invoking by hand.

**`collapse_loops`** repairs an n-gram repeated to the token limit, which seq2seq decoding produces
on hard audio and which scores near zero on the affected row. Only a run of at least three
consecutive repeats of the same 1-, 2-, 3- or 4-word gram is collapsed, and only the repeats beyond
the second are dropped, so a genuinely doubled phrase survives. The conservatism is deliberate;
reduplication is lexical in these languages, which is also why `no_repeat_ngram_size` is never used
at decode time.

**`fix_sentence_case`** capitalises the first letter after a sentence terminator (`[.!?]` followed by
whitespace), repeating until the string stops changing because one pass misses `". a. b"`. It affects
CER only.

Both rules are idempotent, so running them again on a finished CSV is a check rather than a change.
Every other style rule that was considered was measured and rejected; the list and the measurements
are in [SOLUTION.md](SOLUTION.md#what-did-not-work).

## 7. Validation before submitting

`predict.py` validates before it writes, and exits non-zero rather than producing an invalid file.
Three gates, all in `validate()`:

1. **Row count** matches the test list.
2. **Identifier sets are identical.** Checked as set equality, not as a count, because a substituted
   id is exactly what a misaligned join produces and a count check cannot see it. The error names how
   many are missing and how many are unexpected.
3. **No empty transcript.** Zindi rejects a submission containing an empty cell outright, and one
   clip in this test set is short enough to provoke one.

The header is written as exactly `ID,Target` and rows follow the order of `Test.csv`.

The ensemble adds two guards of its own, upstream of those gates: a member that emits nothing for a
clip abstains from that clip rather than voting for the empty string, and if the vote still produces
an empty string the anchor's text is returned.

To re-check a CSV that came from somewhere else, or after any manual edit:

```
python - <<'PY'
import csv
sub  = list(csv.reader(open("data/processed/submission.csv", encoding="utf-8")))
test = list(csv.DictReader(open("data/raw/Test.csv", encoding="utf-8")))
ids  = [r["ID"] for r in test]
assert sub[0] == ["ID", "Target"], sub[0]
assert len(sub) - 1 == len(ids), len(sub) - 1
assert all(r[1].strip() for r in sub[1:]), "empty target"
assert {r[0] for r in sub[1:]} == set(ids), "id mismatch"
print("ok:", len(sub) - 1, "rows, 0 blank, ids match")
PY
```

Two provenance facts hold for every recipe here and are worth confirming before any submission is
claimed as reproducible. No language model, lexicon correction or test-time adaptation is present in
the path. And no model in the path was trained on reference transcripts for the 892 test clips: the
clips were fingerprinted against every transcribed AfriVoice clip and against the ASR Africa Shona
benchmark with no matches in any pass, and the one arm trained on test audio (`distil`) used our own
ensemble's transcripts as targets. Both are documented under
[Disclosures](SOLUTION.md#disclosures).

## 8. Known gaps

Recorded so a reader does not lose time discovering them.

- **`predict.py` exposes no batch-size flag.** The CTC batch size is the `batch_size = 8` default of
  `waxal_asr.modeling.ctc.transcribe_ctc`. Lowering it for a smaller card means editing that default
  rather than passing an argument.

## 9. Runtime

Approximate wall clock on one RTX 5090 (32 GB), 32 CPU cores, 92 GB RAM, over the 892-clip test set.

| step | scope | approximate wall clock |
|---|---|---|
| CTC forward pass | per arm | 20 minutes |
| Blank-penalty re-decode | per checkpoint, logits already computed | 90 seconds |
| Sunbird-51 arm | 892 clips, beam 8, windowed | comparable to one CTC arm |
| Vote and post-processing | any recipe | seconds |
| **`p2n_ens_distil` from cached arm outputs** | 26 members | **under 5 minutes** |
| **`p2n_ens_distil` from scratch** | 10 checkpoints | **about 6 hours** |

The dominant cost is the forward pass multiplied by the number of distinct checkpoints, which is ten
for the 26-member recipe and one for `p2n_distil_nl_f`. Everything after the GPU work is CPU-bound
and negligible. The penalty sweep is cheap only because the logits are decoded repeatedly rather than
recomputed; a naive implementation would pay a full forward pass per penalty.

For reference, training is not on this path at all, but the arms took 8 to 10 hours each for a CTC
arm and 12 hours for the Whisper arm, roughly 80 GPU hours in total.

## 10. Failure modes

| symptom | cause and fix |
|---|---|
| `403` from the Hugging Face Hub | the model repositories are gated until the competition closes. Request access, or download the weights manually and set `WAXAL_MODELS_DIR`. |
| `unknown recipe '<name>'` | the name is not in `configs/ensembles.yaml`. Run `--list` for the four in Section 1; earlier drafts of this repository used different recipe names. |
| `no audio file for id 'ID_XXXX' in ...` | the loader expects `data/raw/test_audio/<ID>.<ext>`. Check that the ids in `Test.csv` match the filenames, including case. |
| `expected a CSV with an ID column` | `--test-csv` points at a file whose header has no `ID` column. |
| `N empty transcript(s), which Zindi rejects` | a single-arm recipe returned nothing for a very short clip. Use an ensemble recipe, where the anchor fills the cell, or inspect that clip. |
| CUDA out of memory | lower `batch_size` in `waxal_asr/modeling/ctc.py`; see [Known gaps](#8-known-gaps). Inference fits in 8 GB at batch 4. |
| output looks stale after changing weights | a per-arm cache was reused. Delete the relevant `data/interim/<arm>_bp<penalty>.json` and re-run. |
| `AttributeError: 'list' object has no attribute 'keys'` | the Sunbird checkpoint loaded outside `waxal_asr/modeling/sunbird.py`. See Section 4. |
| the score differs from 0.764915 by 0.000141 | expected. The published `p1av` is a retrain; `p2n_ens_distil` from published weights returns 0.764774. |

## 11. Summary

```
# one arm, verifies the setup end to end, about 2 GB of weights
python -m waxal_asr.modeling.predict --recipe p2n_distil_nl_f

# the best recorded submission, 26 members, public 0.764915
python -m waxal_asr.modeling.predict --recipe p2n_ens_distil

# the other two recorded ensembles
python -m waxal_asr.modeling.predict --recipe p2n_ens_bp25
python -m waxal_asr.modeling.predict --recipe p2n_ens_bp10

# all four write data/processed/submission.csv, validated before writing
```

`make predict` is the first of these and `make submission` the second.
