---
license: cc-by-4.0
language:
- ln
- sn
task_categories:
- automatic-speech-recognition
pretty_name: WAXAL Phase-2 Lingala/Shona training corpus (derived)
tags:
- lingala
- shona
- ctc
- speech
- waxal
---

# WAXAL Phase-2 Lingala / Shona training corpus (derived)

This corpus was built for the [Google WAXAL ASR Challenge](https://zindi.world/competitions/google-waxal-asr-challenge) on Zindi.

This repository is the exact training corpus behind our Google WAXAL ASR Challenge (Phase 2)
submission: the TSV manifests plus the derived 16 kHz mono FLAC audio that our training configs read.
It exists so that the whole recipe can be rebuilt and audited from one place. The manifests below,
together with the `google/WaxalNLP` train and validation splits read directly from upstream, are the
training data of every fine-tuned arm we submitted with one exception, stated plainly: the
distilled arm also trained on the 892 Phase 2 test clips, using our own ensemble's transcripts as
targets. That is self-training on unlabeled test audio, which the organisers permit subject to
disclosure, and no reference transcript for any test clip was used. Those derived targets are not
redistributed here. One further ensemble member is a third-party model used zero-shot, which we
did not train at all.

It is a **derived redistribution of publicly licensed corpora**. We did not record any audio. Each
subset is a resampled, re-encoded and gated copy of an upstream public dataset, with the upstream
transcription preserved verbatim apart from the whitespace/quote cleaning described below. Upstream
links are listed for every source, and each source remains subject to its own upstream terms.

The task is Lingala (`ln` / `lin`) and Shona (`sn` / `sna`) transcription of long, spontaneous
photo-description recordings. The Phase-2 test set is 892 clips, mean 20.2 s, and open-set language
ID puts it at roughly 50/50 Lingala/Shona.

## Sources

Row counts and hours are corpus build statistics from our own build logs.

| subset in this repo | upstream source | rows | hours | licence |
|---|---|---|---|---|
| `data/afrivoice_lin/` | [`KasuleTrevor/Lingala_100hrs`](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs) (AfriVoice Lingala mirror) | 22,131 | 104.5 | CC-BY-4.0 |
| `data/afrivoice_sna/` | [`realtime-speech/shona1`](https://huggingface.co/datasets/realtime-speech/shona1) (AfriVoice Shona mirror) | 15,923 | 91.0 | CC-BY-4.0 |
| `data/ext_shunya_lin/` | [`shunyalabs/lingala-speech-dataset`](https://huggingface.co/datasets/shunyalabs/lingala-speech-dataset) | 4,341 | 21.5 | upstream terms (see link) |
| `data/ext_fleurs_lin/` | [`google/fleurs`](https://huggingface.co/datasets/google/fleurs), config `ln_cd` | 2,847 | 15.0 | upstream terms (see link) |
| `data/ext_fleurs_sna/` | [`google/fleurs`](https://huggingface.co/datasets/google/fleurs), config `sn_zw` | 3,704 | 15.0 | upstream terms (see link) |
| `data/ext_asrafrica_sna/` | [`asr-africa/ASRAfricaDataEfficiencyBenchmark`](https://huggingface.co/datasets/asr-africa/ASRAfricaDataEfficiencyBenchmark), config `Shona` | 1,055 | 6.0 | upstream terms (see link) |
| `data/waxal_test/` | [`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP), lin+sna **test** split | 3,615 (1,866 lin / 1,749 sna) | 19.2 | upstream terms (see link) |

The `cc-by-4.0` tag in the front matter covers the manifests we wrote and the AfriVoice-derived
audio, which is CC-BY-4.0 upstream. It is not a relicensing of the other subsets: FLEURS, Shunya
Labs, ASR-Africa and WaxalNLP material stays under the terms published in the linked repositories,
and anyone reusing this repository should read those terms per subset.

### Not redistributed here

The `google/WaxalNLP` lin+sna **train/validation** splits (33,002 rows, ~179 hours) are the official
challenge data and were the backbone of every fine-tuned arm, but they are read directly from the upstream
repository at training time (`data.languages: [lin, sna]` in our configs) and are not copied into
this repository. Get them from `google/WaxalNLP`.

The 892 Phase-2 test clips are **not** in this repository in any form.

## Layout and manifest schema

Each subset is a directory holding one or more TSV manifests and an `audio/` directory of FLAC files.
The `data/` prefix is kept because the `audio` column of every manifest is a path relative to the
repository root, and mirroring the build-time layout means those paths resolve without rewriting:

```
data/afrivoice_lin/      manifest_lin.tsv         audio/av_lin_<split>_<index>.flac
data/afrivoice_sna/      manifest_sna.tsv         audio/av_sna_<split>_<index>.flac
data/ext_shunya_lin/     manifest_lin.tsv         audio/av_lin_<split>_<index>.flac
data/ext_fleurs_lin/     manifest_lin.tsv         audio/av_lin_<split>_<index>.flac
data/ext_fleurs_sna/     manifest_sna.tsv         audio/av_sna_<split>_<index>.flac
data/ext_asrafrica_sna/  manifest_sna.tsv         audio/av_sna_<split>_<index>.flac
data/waxal_test/         manifest_test.tsv        audio/wt_<lang>_<index>.flac
                         manifest_test_lin.tsv    (language-filtered copies of manifest_test.tsv,
                         manifest_test_sna.tsv     read by the per-language specialist configs)
```

Four corpora hold more audio files than the Hub allows in one directory, which is capped at 10,000.
Their audio is therefore split across numbered shard directories of at most 9,000 files each, and
the `audio` column of the corresponding manifest names the sharded path:

```
data/afrivoice_lin/audio/000/av_lin_train_000000.flac
data/afrivoice_lin/audio/001/av_lin_train_009000.flac
```

| corpus | audio files | shard directories |
|---|---|---|
| `afrivoice_lin` | 22,131 | 3 |
| `afrivoice_sna` | 15,923 | 2 |

The remaining corpora are small enough to sit in a single `audio/` directory. Either way the manifest
is the authority: read the `audio` column and resolve it against the snapshot root, and the sharding
never has to be thought about.

Manifests are tab-separated with a header row and four columns:

```
audio	transcription	lang	duration
```

| column | meaning |
|---|---|
| `audio` | path to the FLAC file, relative to the repository root, e.g. `data/afrivoice_sna/audio/av_sna_train_000001.flac` |
| `transcription` | the transcript as delivered upstream, with its original casing and punctuation, never lowercased or stripped |
| `lang` | `lin` or `sna` |
| `duration` | clip length in seconds, two decimal places |

The pseudo-label manifests described below carry one extra column, `conf`: the labelling model's
mean maximum softmax probability over non-blank CTC frames for that clip.

Transcripts keep their upstream text with one cleaning step: tab, newline, carriage-return and
double-quote characters are replaced by a space and the result is stripped. This is a parser
requirement, not a normalisation choice: the manifests are read with a TSV reader, and a single
stray quote character had previously merged thousands of rows into one label. No lowercasing, no
punctuation removal, no Unicode folding beyond what the upstream text already carried. Raw cased and
punctuated text is deliberate: the challenge metric scores character error rate on raw text.

## Audio normalisation

Every FLAC file in this repository was produced the same way:

- downmixed to **mono** by averaging channels;
- resampled to **16 kHz** with `librosa.resample` when the upstream sampling rate differed;
- written as **FLAC** (`soundfile`, float32 input), which is roughly half the size of the equivalent
  WAV and lossless;
- **duration gate 1-40 s**;
- **characters-per-second gate 1-25** (transcript length divided by duration), a label-noise filter
  that catches truncated references and misaligned rows.

Two subsets deviate from those gates, and the deviation is in the builder source:

- `waxal_test/` uses the 1-40 s duration gate and requires a transcript of at least 3 characters, but
  applies no characters-per-second gate; these rows are organiser-published gold and were not
  filtered for label noise.
- `pseudo_lin/` and `pseudo_sna/` use a **1-30 s** duration gate, the 1-25 characters-per-second
  gate, a minimum hypothesis length of 5 characters, and a minimum CTC confidence of 0.5 before a
  clip is written at all.

## How each subset was built

All builders are in `scripts/` in the research repository that produced this corpus, not in the
released solution package, which ships the inference and training pipeline rather than the corpus
builders. The commands are recorded here so the derivation can be audited and repeated. They stream the upstream datasets rather
than materialising them, and they append to the manifest and skip already-written filenames, so an
interrupted build is resumed by re-running the same command.

**`scripts/build_afrivoice.py`** builds the six external subsets. It iterates every split the hub
reports for the chosen config, picks the text column (`--text-col`, otherwise the first of
`raw_transcription`, `transcription`, `transcript`, `text`, `sentence` that exists), applies the
gates above, writes FLAC and appends the manifest row. `--max-hours` caps the ingested audio.

```
python scripts/build_afrivoice.py --lang lin --repo KasuleTrevor/Lingala_100hrs
python scripts/build_afrivoice.py --lang sna --repo realtime-speech/shona1
python scripts/build_afrivoice.py --lang sna --repo asr-africa/ASRAfricaDataEfficiencyBenchmark \
    --config Shona --out data/ext_asrafrica_sna --max-hours 40
python scripts/build_afrivoice.py --lang lin --repo shunyalabs/lingala-speech-dataset \
    --text-col transcript --out data/ext_shunya_lin --max-hours 90
python scripts/build_afrivoice.py --lang lin --repo google/fleurs --config ln_cd \
    --text-col raw_transcription --out data/ext_fleurs_lin --max-hours 15
python scripts/build_afrivoice.py --lang sna --repo google/fleurs --config sn_zw \
    --text-col raw_transcription --out data/ext_fleurs_sna --max-hours 15
```

Two column choices matter. FLEURS ships both `transcription` (lowercased, unpunctuated) and
`raw_transcription` (cased, punctuated); we take the raw field because character error rate is scored
on raw text. The Shunya Lingala repository names its text column `transcript`, which none of the
fallbacks would have found, so it is passed explicitly. `scripts/slurm_extbuild.sh` is the batch
wrapper that ran the last four builds together.

**`scripts/build_waxal_test.py`** builds `waxal_test/` from the `google/WaxalNLP` lin and sna test
parquet shards. Those transcriptions are public and the organisers confirmed for Phase 2 that they
may be trained on; they were not used in Phase 1. They are the most test-style gold text available,
so they are used in the final in-style refresh stage of every arm.

```
python scripts/build_waxal_test.py --langs lin,sna --out data/waxal_test
```

**`scripts/pseudo_p2.py`** builds the pseudo-label subsets, and **`scripts/filter_pseudo.py`**
selects the kept fraction. Details in the next section.

**`scripts/build_corpus.py`** and **`scripts/build_waxal_langs.py`** are earlier multi-language
builders from the first phase of the work (they also cover Luganda and other WaxalNLP languages).
They did not contribute any subset shipped here; they are named because they appear in the solution
repository and a reviewer will find them.

## Pseudo-labels, and why they are not here

Two pseudo-label subsets were built during the work and are **not shipped in this repository**.
They are machine transcripts, not human labels; they add 11.8 GB; and self-training on them was a
measured negative result, so nothing in the solution depends on having them. They are fully
regenerable from the commands below, which is why the recipe is documented here in full.

They were produced only on the WaxalNLP **unlabeled** pools for Lingala and Shona. The labelling
models were our own per-language specialists, `w2vbert_linspec_r` for Lingala and `w2vbert_snaspec_r`
for Shona, both of them `facebook/w2v-bert-2.0` CTC fine-tunes with the raw cased-and-punctuated
character vocabulary. `scripts/pseudo_p2.py` streams the unlabeled parquet shards, resamples to 16 kHz, runs a
single batched CTC forward pass in bfloat16, greedy-decodes, and computes a per-clip confidence as the
mean maximum softmax probability over the frames whose argmax is not the blank token. A clip is
written only if its hypothesis is at least 5 characters, its confidence is at least 0.5, its duration
is 1-30 s and its characters-per-second ratio is 1-25. The manifest is appended and already-processed
clip ids are skipped, so labelling survives a wall-clock limit.

```
python scripts/pseudo_p2.py --ckpt experiments/w2vbert_linspec_r_s42 --langs lin --out data/pseudo_lin
python scripts/pseudo_p2.py --ckpt experiments/w2vbert_snaspec_r_s42 --langs sna --out data/pseudo_sna
python scripts/filter_pseudo.py --in data/pseudo_lin/manifest_pseudo.tsv \
    --out data/pseudo_lin/manifest_keep.tsv --keep-frac 0.6
python scripts/filter_pseudo.py --in data/pseudo_sna/manifest_pseudo.tsv \
    --out data/pseudo_sna/manifest_keep.tsv --keep-frac 0.6
```

`filter_pseudo.py` sorts by confidence **within each language** and keeps the top fraction, 60 per cent here,
giving the 22,986 Lingala and 22,396 Shona rows in the table. Cutting per language rather than
globally stops the easier language from crowding out the harder one. Both manifests are produced by
the commands above, so a different threshold can be applied without re-labelling, but neither they
nor their audio are redistributed in this repository.

Negative result, stated plainly: **this self-training round did not help.** On our speaker-disjoint
holdout the Lingala specialist went from 0.2785 to 0.2800 error and the Shona specialist moved
0.1616 to 0.1598 and back to 0.1616, which is flat. The labelled portion of the same collection was already
saturated. The recipe is documented because the models trained on these labels became useful
*ensemble* members, not because the pseudo-labels improved a single model.

## What is not in this repository

The 892 Phase-2 test clips are not here in any form. Neither are the two pseudo-label subsets,
which were produced on the WaxalNLP **unlabeled** pools only and are regenerable from the recipe
below. The `waxal_test/` subset is the **Phase-1** WaxalNLP test
split, whose transcriptions are published by the organisers and whose use in Phase 2 was
organiser-confirmed; it is not the Phase-2 blind set.

## Loading

The repository is a file tree of manifests and audio, not a `datasets` loading script. Read a
manifest and resolve the paths:

```python
import pandas as pd
from huggingface_hub import snapshot_download

root = snapshot_download("anyantudre/waxal-linsna", repo_type="dataset",
                         allow_patterns=["data/afrivoice_sna/*"])
df = pd.read_csv(f"{root}/data/afrivoice_sna/manifest_sna.tsv", sep="\t",
                 quoting=3)              # quoting=3 (QUOTE_NONE): transcripts are raw text
print(df.columns.tolist(), len(df))      # ['audio', 'transcription', 'lang', 'duration']

import soundfile as sf
wav, sr = sf.read(f"{root}/{df.audio[0]}")   # the audio column resolves against the snapshot root
```

`allow_patterns` matters: fetch only the subsets you need. The audio is FLAC at 16 kHz mono, so
`soundfile.read` returns a float array ready for a feature extractor with no resampling.

## Attribution

If you use this corpus, cite the upstream datasets, not this repository:

- [`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP)
- [`KasuleTrevor/Lingala_100hrs`](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs), AfriVoice Lingala
- [`realtime-speech/shona1`](https://huggingface.co/datasets/realtime-speech/shona1), AfriVoice Shona
- [`shunyalabs/lingala-speech-dataset`](https://huggingface.co/datasets/shunyalabs/lingala-speech-dataset)
- [`google/fleurs`](https://huggingface.co/datasets/google/fleurs)
- [`asr-africa/ASRAfricaDataEfficiencyBenchmark`](https://huggingface.co/datasets/asr-africa/ASRAfricaDataEfficiencyBenchmark)

The AfriVoice recordings were collected by Digital Umuganda. If you are a rights holder for any
subset and want it removed from this mirror, open a discussion on the repository and it will be taken
down.

## Build environment

The corpora were built and the models trained on a single machine: 1x NVIDIA RTX 5090 (32 GB VRAM),
32 CPU cores, 92 GB RAM, with a batch scheduler imposing a 4-hour wall clock per job, which is why
every builder is resumable. Software: python 3.12.13, torch 2.11.0+cu128, transformers 4.57.6,
datasets 3.6.0, soundfile 0.14.0, librosa 0.11.0, numpy 2.4.6.
