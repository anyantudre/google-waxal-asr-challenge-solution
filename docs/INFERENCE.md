# INFERENCE - reproducing any recorded submission from released weights

This document answers "how do I run it". It gives the exact commands, arguments, file paths,
schemas, runtimes, failure modes and the pre-submission checklist.

It deliberately does not argue for the design. Why blank-penalty decoding exists, why language
identification is open set, how ensemble members were selected, and what was tried and rejected are
all in [SOLUTION.md](SOLUTION.md), which is the single source of truth for facts, measurements and
reasoning. Where a number appears below it is quoted from there.

Nothing here requires training. Every published checkpoint is on the Hugging Face Hub and downloads
on first use. The path is deterministic: the CTC arms decode greedily (optionally with a blank
penalty), the fine-tuned Whisper arm generates greedily, the third-party Sunbird arm uses beam
search with sampling off, and the vote and the selection are pure functions of their members, so
re-running a recipe on the same weights and the same audio reproduces the same CSV. Seeds are fixed
per arm (42, 43, 44, 46); they matter for training, not for this path.

## 1. The recorded submissions

Twelve recipes are recorded, each named after the submission file that was sent to Zindi, so a
recipe maps to exactly one leaderboard row. The ones a reader is most likely to want:

| recipe | shape | public score | CER | WER |
|---|---|---|---|---|
| **`p2n_mbr`** | selection over 7 ensembles | **0.766791** | 0.108130 | 0.358288 |
| `p2n_meta` | vote over 5 ensembles | 0.766683 | 0.108386 | 0.358249 |
| `p2n_ens_masked` | 26 members | 0.766563 | 0.108364 | 0.358509 |
| `p2n_ens_distil` | 26 members | 0.764915 | 0.108961 | 0.361209 |
| `p2n_ens_bp25` | 25 members | 0.764759 | 0.109025 | 0.361457 |
| `p2n_distil_nl_f` | 1 member | 0.746787 | 0.113117 | 0.393308 |

Each one is reproduced by a single command, which differs only in the recipe name:

```
python -m waxal_asr.modeling.predict --recipe p2n_mbr
python -m waxal_asr.modeling.predict --recipe p2n_ens_masked
python -m waxal_asr.modeling.predict --recipe p2n_ens_distil
python -m waxal_asr.modeling.predict --recipe p2n_distil_nl_f
```

**`p2n_mbr` is the scored final submission** (`cand_mbr.csv`, public 0.766791, private 0.772552,
second place), **and it is what `make submission` runs.** It builds seven complete ensembles, six
anchored on the seed-43 arm and one on seed 46, plus a meta vote over five of them, and then keeps,
for every clip, the candidate transcript with the smallest mean normalised character edit distance
to the 26 members of `p2n_ens_masked`. Every member is decoded with a correct forward pass: the
attention mask is passed throughout, and the Whisper arm is decoded as a generator rather than
through the CTC path.

`p2n_ens_distil` was the second final pick: the same 26 members as `p2n_ens_masked` with two
differences, both recorded in the recipe. Its blank-penalty members omit the attention mask, and it
predates the Whisper decoding path. It scores 0.764915 and is kept because reproducing a known
historical result exactly is a useful check on an installation. `p2n_ens_bp25` drops the distilled
arm; `p2n_ens_bp10` is a 17-member configuration. `p2n_distil_nl_f` is a single arm, the strongest
single checkpoint, and the cheapest way to verify the pipeline without downloading twelve sets of
weights; `make predict` runs it.

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
  p2n_ens_masked     26 member(s)  public 0.766563  26 members, the corrected forward pass
  p2n_ens_weighted   26 member(s)  public 0.76658  26 members, the two weakest at half weight
  p2n_ens_soup6      26 member(s)  public 0.766226  26 members, the six-way soup in place of the five-way
  p2n_ens_s46swap    26 member(s)  public 0.766477  26 members, seed 46 in place of the weakest arm
  p2n_ens_wide       38 member(s)  public 0.76596  38 members, the penalty bracket widened to 0.5 and 2.5
  p2n_meta            5 ensembles  public 0.766683  a vote over five complete ensembles
  p2n_ens_s46anchor  26 member(s)  public 0.766083  26 members, seed 46 as the anchor
  p2n_mbr             7 candidates  public 0.766791  per-clip selection across seven ensembles, judged by the 26 members
  p2n_distil_nl_f     1 member(s)  public 0.746787  the distilled arm on its own
```

A recipe takes one of three shapes, and defines exactly one of these keys:

- **`members`**: a list of arms that vote. Each member is an `arm` key plus an optional
  `blank_penalty`, and the **first member is the anchor**: its text survives every slot unless the
  others outvote it, so member order is significant. A penalty of 0, or none at all, means ordinary
  greedy decoding. An optional `weight` scales a member's vote.
- **`ensembles`**: a list of other recipe names. Each named recipe is built in full and the results
  vote as members, first one anchoring. `p2n_meta` is this shape.
- **`select`** plus **`judged_by`**: a list of candidate recipe names and one electorate recipe.
  Every candidate is built in full, and for each clip the candidate transcript with the smallest
  mean normalised character edit distance to the electorate's members is kept; ties keep the
  earliest candidate, so order matters here too. `p2n_mbr`, the scored submission, is this shape.

```yaml
  my_recipe:
    description: two arms, one of them re-decoded
    members:
      - {arm: s43}                     # anchor, greedy
      - {arm: s43, blank_penalty: 1.5}
      - {arm: soup5}
```

The `defaults` block at the top of the file supplies `vote_threshold: 2.0` and `skeleton: anchor`
for every recipe. The skeleton differs from the vote function's own default of `mbr`, and the
threshold has no default in the vote function at all, which is why both are recorded in the file
rather than left implicit. A recipe may override either key inline.

`tests/test_submission_pipeline.py` asserts that the recipe file parses, that the submitted recipes
are still present, that `p2n_mbr` still names its seven candidates in order and is judged by
`p2n_ens_masked`, that `p2n_ens_distil` still has 26 members anchored on `s43`, and that every arm
named in a recipe matches the registry in `waxal_asr/config.py`.

## 2. Setup

### Package versions

The submitted result was produced with Python 3.12.13, torch 2.11.0+cu128, transformers 4.57.6,
datasets 3.6.0, soundfile 0.14.0, librosa 0.11.0 and numpy 2.4.6. These pins are in
[`requirements.txt`](../requirements.txt), recorded from the training environment's `pip freeze`.
Python 3.11 or 3.12 is required: numpy 2.4.6 needs at least 3.11, and the pinned audiomentations
does not install on 3.13.

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

Eleven checkpoints are published under the `anyantudre` namespace, plus one third-party model used
zero-shot. The registry is `ENSEMBLE_ARMS` in `waxal_asr/config.py`, and the same mapping is repeated
in the `arms:` block of `configs/ensembles.yaml` for readers who do not want to read Python; a test
keeps the two from drifting apart.

| arm key | repository | role |
|---|---|---|
| `s43` | `anyantudre/waxal-w2vbert-linsna-seed43` | anchors six of the seven ensembles |
| `s44` | `anyantudre/waxal-w2vbert-linsna-seed44` | widest corpus |
| `soup5` | `anyantudre/waxal-w2vbert-linsna-soup5` | five-way weight average |
| `soup6` | `anyantudre/waxal-w2vbert-linsna-soup6` | six-way weight average |
| `p1raw` | `anyantudre/waxal-w2vbert-linsnalug-raw` | root of the family |
| `linspec_r` | `anyantudre/waxal-w2vbert-lin-specialist` | Lingala specialist |
| `snaspec_r` | `anyantudre/waxal-w2vbert-sna-specialist` | Shona specialist |
| `p1av` | `anyantudre/waxal-w2vbert-linsna-afrivoicemix` | mixed-curriculum counter-experiment |
| `distil` | `anyantudre/waxal-w2vbert-linsna-distilled` | strongest single checkpoint |
| `s46` | `anyantudre/waxal-w2vbert-linsna-seed46` | most decorrelated arm, anchors the seventh |
| `turbo_linsna_r` | `anyantudre/waxal-whisper-turbo-linsna` | Whisper large-v3-turbo, greedy, no windowing |
| `sunbird51` | `Sunbird/asr-whisper-51-african-languages` | third party, zero-shot, pinned, beam 8, windowed |

A CTC arm is roughly 2.3 GB and the Whisper arm roughly 3.2 GB. To fetch one ahead of time:

```
huggingface-cli download anyantudre/waxal-w2vbert-linsna-seed43 --local-dir models/s43
export WAXAL_MODELS_DIR=models
```

The ten w2v-BERT arms share a 97-token raw character vocabulary that includes the 26 capitals and
`! , . : ; ?`.

## 3. What the command does

`waxal_asr/modeling/predict.py` is the only entry point. Given a recipe it runs these stages in
order.

| stage | module | what it does |
|---|---|---|
| 1 | `waxal_asr/modeling/ctc.py` | one forward pass per CTC arm, batched, over every clip |
| 2 | `waxal_asr/decode.py` | greedy arg max, optionally after subtracting the blank penalty |
| 3 | `waxal_asr/modeling/whisper.py` | the fine-tuned Whisper arm: generates greedily, truncates past 30 s |
| 4 | `waxal_asr/modeling/sunbird.py` | the third-party zero-shot arm, windowed, beam 8, language routed |
| 5 | `waxal_asr/postprocess.py` | loop collapse then sentence case, on every member |
| 6 | `waxal_asr/ensemble.py` | character-level ROVER over the members of each ensemble |
| 7 | `waxal_asr/postprocess.py` | the same two rules again, on the voted text |
| 8 | `waxal_asr/modeling/predict.py` | for `p2n_mbr`: per-clip selection across the finished ensembles |
| 9 | `waxal_asr/modeling/predict.py` | validate, then write `data/processed/submission.csv` |

Stages 5 and 7 run the identical function; both rules are idempotent, so the second application is a
safety net rather than a second edit. For the scored recipe, stages 6 and 7 repeat for each of the
seven candidate ensembles (and once more for the meta vote inside `p2n_meta`), all drawing on the
same per-arm cache, before stage 8 selects per clip.

### Error handling and logging

Console output is prefixed by stage: `[predict]` for recipe assembly, `[ctc]`, `[whisper]` and
`[sunbird]` for per-arm progress counters (`[ctc] 8/892`), and `[lid]` for language identification.
`[predict] <arm>_bp<penalty>: reusing cache` means no forward pass ran for that member; see the
cache section below. Every failure path exits non-zero with a named cause rather than writing a
partial file: a missing recipe, a missing language map, a malformed test CSV, and each of the three
validation gates in Section 7 all abort the run. Training logs its per-step metrics to
`runs/<name>` (TensorBoard) and writes `final_metrics.json` next to the weights under the model
output directory; nothing in the inference path needs either.

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

Members decoded without the attention mask (the legacy blank-penalty members of `p2n_ens_distil`)
cache with a `_nomask` suffix, for example `s43_bp15_nomask.json`: a different forward pass is a
different member, so it must be a different cache file.

This is the resumability mechanism and it matters operationally in three ways. An interrupted run
resumes instead of restarting. A second recipe that shares an arm reuses it, which is why running
every recorded recipe costs barely more than running the largest one alone. And **a stale cache is
silently authoritative**: if weights or audio change, delete the affected JSON files, including any
`_nomask` variants, or the old transcripts will be voted on. `[predict] s43_bp15: reusing cache` on
the console is the signal that no forward pass happened.

Interim files total roughly 30 MB for the scored recipe. The final CSV is about 300 KB.

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

That measurement is both an input to the design and a step in the recipes. **Every recipe containing
the routed `sunbird51` arm needs a language map**, which is every recorded recipe except
`p2n_distil_nl_f`: the scored `p2n_mbr` needs it for all seven of its candidate ensembles and for
its electorate. Produce it with:

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

## 8. Running on new audio

The test-set path above generalises to any audio directory: point the two path flags at your data.
`--test-csv` needs a CSV with an `ID` column (the rebuild snippet in Section 2 generates one from a
directory), and `--audio-dir` a directory with one file per identifier.

```
python -m waxal_asr.modeling.predict --recipe p2n_ens_masked \
    --audio-dir /path/to/audio --test-csv /path/to/ids.csv --out my_transcripts.csv
```

Two caveats. Any recipe containing `sunbird51` needs a matching language map first: run
`python -m waxal_asr.lid --audio-dir /path/to/audio --out data/interim/lid.json`. And the per-arm
cache keys on arm and penalty only, not on the audio, so **delete `data/interim/*.json` when
switching audio directories** or the old transcripts will be reused. For a single model on new
audio, `p2n_distil_nl_f` is the lightest recipe; for Luganda, the only published checkpoint that has
seen it is `waxal-w2vbert-linsnalug-raw` (arm `p1raw`).

## 9. Known gaps

Recorded so a reader does not lose time discovering them.

- **The legacy unmasked members are batch-composition-dependent.** Without the attention mask,
  zero-padding leaks into self-attention, so those members' transcripts depend on how clips are
  grouped into batches. Rebuilding `p2n_ens_distil` byte-for-byte therefore requires the default
  `--batch-size 8` and the clip order of `Test.csv`. The masked members, which is everything the
  scored `p2n_mbr` uses, do not have this property: any batch size gives the same transcripts.

## 10. Runtime

Approximate wall clock on one RTX 5090 (32 GB), 32 CPU cores, 92 GB RAM, over the 892-clip test set.

| step | scope | approximate wall clock |
|---|---|---|
| `make lid` | 892 clips, two passes of a 1B LID model | 25 minutes |
| CTC forward pass | per arm | 20 minutes |
| Blank-penalty re-decode | per checkpoint, logits already computed | 90 seconds |
| Fine-tuned Whisper arm | 892 clips, greedy | comparable to one CTC arm |
| Sunbird-51 arm | 892 clips, beam 8, windowed | comparable to one CTC arm |
| Votes, selection and post-processing | any recipe | seconds |
| **`p2n_mbr` from cached arm outputs** | 7 ensembles plus the electorate | **under 5 minutes** |
| **`p2n_mbr` from scratch** | 12 distinct checkpoints | **about 7 hours, plus `make lid`** |
| `p2n_ens_distil` from scratch | 10 distinct checkpoints | about 6 hours |

The dominant cost is the forward pass multiplied by the number of distinct checkpoints: twelve for
the scored recipe (the ten of `p2n_ens_distil` plus `s46` and `soup6`), one for `p2n_distil_nl_f`.
Everything after the GPU work is CPU-bound and negligible. The penalty sweep is cheap only because
the logits are decoded repeatedly rather than recomputed; a naive implementation would pay a full
forward pass per penalty. On an 8 GB card at `--batch-size 4`, budget roughly double the GPU time.

For reference, training is not on this path at all, but the arms took 8 to 10 hours each for a CTC
arm and 12 hours for the Whisper arm, roughly 80 GPU hours in total. Building the training corpus
(`make data`, about 14 GB in roughly 54,000 files) is a download of an hour or so on a fast
connection, and is likewise not needed for inference.

## 11. Failure modes

| symptom | cause and fix |
|---|---|
| `403` from the Hugging Face Hub | check the repository name and your network; the model repositories are public. If the Hub is unreachable, download the weights manually and set `WAXAL_MODELS_DIR`. |
| `unknown recipe '<name>'` | the name is not in `configs/ensembles.yaml`. Run `--list` for the twelve recorded recipes; earlier drafts of this repository used different recipe names. |
| `no audio file for id 'ID_XXXX' in ...` | the loader expects `data/raw/test_audio/<ID>.<ext>`. Check that the ids in `Test.csv` match the filenames, including case. |
| `expected a CSV with an ID column` | `--test-csv` points at a file whose header has no `ID` column. |
| `N empty transcript(s), which Zindi rejects` | a single-arm recipe returned nothing for a very short clip. Use an ensemble recipe, where the anchor fills the cell, or inspect that clip. |
| CUDA out of memory | pass `--batch-size 4`, which fits in 8 GB (measured on an 8 GB RTX 4070; the default 8 does not fit there). The masked members' transcripts do not depend on the batch size; see [Known gaps](#9-known-gaps) for the one legacy exception. |
| output looks stale after changing weights | a per-arm cache was reused. Delete the relevant `data/interim/<arm>_bp<penalty>.json`, including `_nomask` variants, and re-run. |
| `AttributeError: 'list' object has no attribute 'keys'` | the Sunbird checkpoint loaded outside `waxal_asr/modeling/sunbird.py`. See Section 4. |
| `p2n_ens_distil` differs from 0.764915 by 0.000141 | expected. The published `p1av` is a retrain; that recipe from published weights returns 0.764774. The same arm shifts the other recipes comparably. |

## 12. Summary

```
# one arm, verifies the setup end to end, about 2 GB of weights
python -m waxal_asr.modeling.predict --recipe p2n_distil_nl_f

# the language map, required by every other recipe
python -m waxal_asr.lid --audio-dir data/raw/test_audio --out data/interim/lid.json

# the scored final submission, public 0.766791, private 0.772552
python -m waxal_asr.modeling.predict --recipe p2n_mbr

# the historical 26-member reference, public 0.764915
python -m waxal_asr.modeling.predict --recipe p2n_ens_distil

# all recipes write data/processed/submission.csv, validated before writing
```

`make predict` runs the first of these, `make lid` the second, and `make submission` the third,
which is the scored recipe `p2n_mbr`.
