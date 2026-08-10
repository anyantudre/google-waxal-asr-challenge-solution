# Solution details

This document is the detailed account behind the README: what was built, which decisions were
measured, what they were measured against, and which promising ideas turned out not to work. It is
also the reference record for the numbers. Every figure quoted anywhere in this repository comes from
this file. Nothing here is estimated: each number is a leaderboard result, a logged holdout
evaluation, or a corpus build statistic. If a fact is not recorded here, it must not be asserted in
the documentation.

The headline result is the recipe `p2n_mbr`: public leaderboard **0.766791** (CER 0.108130,
WER 0.358288), private leaderboard **0.772552** (CER 0.110664, WER 0.344233), **second place** of
the competition. It is a per-clip selection across seven complete ensembles, each a 26 to 38 member
character-level ROVER vote over the arms listed below, each arm decoded at one of several blank
penalties; for every clip the transcript kept is the one closest in mean normalised character edit
distance to the 26 members of the corrected vote `p2n_ens_masked`. The mechanism is described under
[Selection](#selection-the-last-mechanism-that-worked). A second recipe, `p2n_ens_distil`,
reproduces an earlier 26-member configuration at 0.764915 and is kept as a fixed reference point
for checking an installation.

Step-by-step instructions for running the pipeline are in [INFERENCE.md](INFERENCE.md). This document
explains what was done and why it worked.

## Background

### The challenge

The Google WAXAL ASR Challenge targets Lingala, Shona and Luganda. The corrected Phase 2 test set
released on 2026-08-02 contains 892 clips, and open-set language identification over all of them
found it to be roughly half Lingala and half Shona, with no Luganda. The solution is therefore
optimised for Lingala and Shona, while the training code and several arms retain Luganda support.

Score: `1 - 0.5 * (WER + CER)`, higher is better. The public leaderboard covers about 30 per cent of
the test set (roughly 268 clips); the remaining 70 per cent decides the final ranking.

### How the metric actually behaves

Two controlled experiments pinned this down, and the second corrected an earlier assumption.

1. Two submissions identical except for the capitalisation of 50 rows returned **WER identical to
   nine decimal places** (0.382307342 both times) while CER moved from 0.109473 to 0.109366. If both
   halves normalised, CER could not have moved; if neither did, WER could not have stayed fixed.
   Therefore WER lowercases and CER does not.
2. Two submissions identical except for 87 added commas returned **different WER** (0.361457 against
   0.362198). Therefore WER does **not** strip punctuation.

**WER is computed on lowercased but otherwise unmodified text. CER is computed on the raw string.**
Punctuation errors are charged twice, once to each metric. A lowercase, punctuation-free model is
capped on CER no matter how accurate its words are. This is why every CTC model here uses a character
vocabulary that includes capitals and the punctuation CER scores.

### Test set composition

The competition was advertised as three languages. Rather than trust that, the 892 test clips were
passed through two independent open-set language identification models.

| model | Lingala | Shona | other |
|---|---|---|---|
| [`facebook/mms-lid-4017`](https://huggingface.co/facebook/mms-lid-4017) | 437 (49.0 per cent), confidence 0.990 | 423 (47.4 per cent), confidence 0.980 | 32 low confidence, one of which it called Luganda |
| [`facebook/mms-lid-256`](https://huggingface.co/facebook/mms-lid-256) | 440 (49.3 per cent), confidence 0.996 | 440 (49.3 per cent), confidence 0.996 | 12 low confidence |

Both agree: the set is half Lingala and half Shona, and Luganda is effectively absent. Every
subsequent decision about data weighting and routing follows from this measurement.

Audio: 48 kHz WAV, mean 20.2 seconds, range 1.01 to 35.2 seconds, 3.3 per cent longer than 30 seconds.

The 34 clips that either model declined to call Lingala or Shona were inspected individually. All are
neighbouring-language confusions rather than coverage gaps: clips labelled Chichewa, Ndau or Venda
are transcribed as Shona by every arm, and clips labelled Swahili or Umbundu are transcribed as
Lingala. The single clip labelled Luganda (`ID_REPXZM`) is Lingala, and all eight arms agree on it.

The word "open-set" matters. An earlier closed-set run, restricted to the three advertised languages,
was structurally incapable of reporting anything else and returned a confident but wrong answer.
Restricting the candidate set to what you expect to find is not a way to reduce noise; it is a way to
guarantee you cannot be surprised by the truth.

### Hardware and environment

One Linux workstation: NVIDIA RTX 5090 (32 GB VRAM), 32 CPU cores, 92 GB RAM, under a SLURM batch
scheduler with a four hour wall-clock limit per job and checkpoint resume. Ensembling and
post-processing are CPU only and ran on a laptop. No cloud or paid compute was used.

Python 3.12.13, torch 2.11.0+cu128, transformers 4.57.6, datasets 3.6.0, soundfile 0.14.0,
librosa 0.11.0, numpy 2.4.6; the full pin set recorded from that environment is
`requirements.txt`. Seeds are fixed per arm (42, 43, 44, 46).

### Training data

Every source is publicly available. Licences are as published by each dataset.

| source | rows | hours | licence |
|---|---|---|---|
| [`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona train and validation | 33,002 | about 179 | CC-BY-4.0 |
| [`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP), Lingala and Shona **test** split | 3,615 (1,866 lin, 1,749 sna) | 19.2 | CC-BY-4.0 |
| [`KasuleTrevor/Lingala_100hrs`](https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs), AfriVoice Lingala | 22,131 | 104.5 | CC-BY-4.0 |
| [`realtime-speech/shona1`](https://huggingface.co/datasets/realtime-speech/shona1), AfriVoice Shona | 15,923 | 91.0 | CC-BY-4.0 |
| [`shunyalabs/lingala-speech-dataset`](https://huggingface.co/datasets/shunyalabs/lingala-speech-dataset) | 4,341 | 21.5 | see dataset card |
| [`google/fleurs`](https://huggingface.co/datasets/google/fleurs), config `ln_cd` | 2,847 | 15.0 | CC-BY-4.0 |
| [`google/fleurs`](https://huggingface.co/datasets/google/fleurs), config `sn_zw` | 3,704 | 15.0 | CC-BY-4.0 |
| [`asr-africa/ASRAfricaDataEfficiencyBenchmark`](https://huggingface.co/datasets/asr-africa/ASRAfricaDataEfficiencyBenchmark), config `Shona` | 1,055 | 6.0 | see dataset card |
| Pseudo-labels over `google/WaxalNLP` **unlabeled** Lingala | 22,986, top 60 per cent kept | not measured | derived |
| Pseudo-labels over `google/WaxalNLP` **unlabeled** Shona | 22,396, top 60 per cent kept | not measured | derived |
| Pseudo-labels over the 892 Phase 2 test clips | 892 | 5.0 | derived |

FLEURS is read using its `raw_transcription` field, which preserves casing and punctuation. The
lowercased `transcription` field would cap CER.

The non-derived sources above sum to **about 451 hours**: 179 + 19.2 + 104.5 + 91.0 + 21.5 + 15.0 +
15.0 + 6.0. No single arm trains on all of them; the largest single corpus is s44's 42,934 rows.

## The models

All CTC arms fine-tune [`facebook/w2v-bert-2.0`](https://huggingface.co/facebook/w2v-bert-2.0) with a
raw character vocabulary that includes capitals and the punctuation CER scores. Holdout is a
speaker-disjoint carve of WAXAL train, reported as error (lower is better).

| arm | base | corpus | epochs and learning rate | holdout |
|---|---|---|---|---|
| `s43` | w2v-bert-2.0, seed 43 | WAXAL lin/sna, plus Phase 1 test gold (30,987 rows) | 8 at 1.0e-4 | 0.2862 |
| `s44` | w2v-bert-2.0, seed 44 | as s43, plus FLEURS, Shunya, ASR Africa (42,934 rows) | 8 at 1.0e-4 | 0.2809 |
| `s44r` | continues s44 | WAXAL refresh | 1 at 1.0e-5 | 0.2812 |
| `soup5` | weight average | p1raw, linspec_r, snaspec_r, linspec_p2, snaspec_p2 | not applicable | 0.1610 on Shona |
| `p1raw` | w2v-bert-2.0 | WAXAL lin, sna and lug, raw vocabulary | 12 at 1.0e-4 | 0.3131 on Lingala |
| `linspec_r` | continues p1raw | AfriVoice Lingala, then WAXAL refresh | 3 then 1, at 3e-5 then 1e-5 | 0.2785 on Lingala |
| `snaspec_r` | continues p1raw | AfriVoice Shona, then WAXAL refresh | 3 then 1, at 3e-5 then 1e-5 | 0.1616 on Shona |
| `p1av` | continues p1raw | WAXAL and both AfriVoice corpora mixed throughout | 2 at 2e-5 | 0.2905 |
| `distil` | continues soup5 | WAXAL gold, plus ensemble transcripts for the 892 test clips | 3 at 8e-6 | **0.2746**, the best single arm on the leaderboard (0.746787) |
| `s46` | w2v-bert-2.0, seed 46 | as s43 (the seed-43 corpus rerun at seed 46) | 8 at 1.0e-4 | 0.2588, the best holdout of any arm, a gain that never appeared in any ensemble |
| `soup6` | weight average | the five soup5 members plus distil | not applicable | not measured |
| `turbo_linsna_r` | [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | WAXAL and Phase 1 test gold, then refresh | 3 then 1, at 1e-5 then 5e-6 | 0.2772 |
| Sunbird 51 | [third party](https://huggingface.co/Sunbird/asr-whisper-51-african-languages), zero-shot | not fine-tuned; beam 8, language-routed | not applicable | 0.7372 on the leaderboard |

Sunbird 51 is pinned to revision `5d4f0038`. Its card states the weights will be replaced, so an
unpinned load would silently change the submission.

### The soup members

`soup5` averages five checkpoints. All five share the `p1raw` vocabulary and CTC token order, which
is what makes them averageable. Settings are read from the configs of record.

| member | config | continues | epochs and learning rate |
|---|---|---|---|
| `p1raw` | `w2vbert_p1raw.yaml` | w2v-bert-2.0 | 12 at 1.0e-4 |
| `linspec_r` | `w2vbert_linspec_r.yaml` | `linspec` | 1 at 1.0e-5 |
| `snaspec_r` | `w2vbert_snaspec_r.yaml` | `snaspec` | 1 at 1.0e-5 |
| `linspec_p2` | `w2vbert_linspec_p.yaml`, then `_p2` | `linspec_r` | 2 at 1.5e-5, then 1 at 1.0e-5 |
| `snaspec_p2` | `w2vbert_snaspec_p.yaml`, then `_p2` | `snaspec_r` | 2 at 1.5e-5, then 1 at 1.0e-5 |

### Vocabulary and model size

The raw vocabulary is 97 tokens: the letters, the 26 capitals, and the punctuation `! , . : ; ?`.
The earlier stripped vocabulary was 81 tokens. `facebook/w2v-bert-2.0` is a 600M parameter encoder;
`openai/whisper-large-v3-turbo` is 809M.

### Reference punctuation rates

Measured over the WAXAL reference transcripts, per row: **0.630 commas** and **1.402 full stops**.
These are the targets the four punctuation experiments aimed at. Our ensemble under-produces commas
(individual high-blank-penalty arms reach 0.490) and, at high blank penalty, over-produces full stops
(1.994 per row). Matching the reference rate did not improve the score in any of the four attempts,
which is what established that the punctuation errors are misplaced rather than merely miscounted.

## What worked

1. **Raw capitals and punctuation vocabulary, +0.0171.** The largest single modelling gain. CER is
   scored on raw text and WER does not strip punctuation, so a lowercase model is capped on both.
   Rebuilding the CTC character vocabulary from 81 tokens to 97 was worth 0.0171 on the leaderboard.
2. **Blank-penalty decoding, +0.0104 in total.** Greedy CTC drops a character whenever the blank
   marginally outranks the best character, and often the whole word with it. Subtracting a constant
   from the blank logit restores them. Detail below.
3. **Character-level ROVER ensembling, +0.006 to +0.010**, provided members are strong and
   decorrelated.
4. **Source diversity beats parameter diversity.** Four members from four checkpoints gained 0.0040;
   six members from two checkpoints gained 0.0006.
5. **Fresh random seeds, about +0.0032 per member**, cheaper and more effective than new
   architectures.
6. **Curriculum order.** External audio first, in-style WAXAL gold last: mixing throughout scored
   0.2905 on the holdout, sequencing scored 0.2785. Detail below.
7. **Ensemble distillation, best single model.** Training one model on the ensemble's own transcripts
   for the test clips gave a holdout of 0.2746, better than any directly trained arm.
8. **Loop collapse**, +0.0069 when introduced.
9. **Beam 8 over greedy for Whisper**, +0.0224.
10. **Capitalisation after a sentence break**, small and CER-only.

### Blank-penalty decoding

Our error profile was unusual: the best CER of the leading entries alongside the worst WER. That
pattern means errors were spread thinly, one or two characters in many words, rather than
concentrated in a few.

Measuring against matched references showed the cause. Our hypotheses carried only 97.9 per cent of
the reference word count, 1.30 words per second against a reference rate of 1.41. We were dropping
words outright, and a dropped word is a deletion: expensive in WER, nearly free in CER.

The mechanism is greedy CTC decoding. At each frame the decoder takes the arg max, so whenever the
blank symbol marginally outranks the best character, that character disappears, and often the word
with it. Subtracting a constant from the blank logit shifts the tie-break towards emitting. The
constant was chosen so that the output word rate matched the reference rate, not by tuning against
the leaderboard.

The re-decoded models are worse on their own, 0.743 against 0.746 for the same checkpoint decoded
normally, because some recovered words are wrong. They are extremely valuable as ensemble members,
because a vote can discard a spurious word but can never recover a missing one. Deletions are
unrecoverable; insertions are filterable.

### The two-stage curriculum

External corpora help, but only in the right order. Two arms were trained on identical data:

| recipe | holdout error |
|---|---|
| WAXAL and AfriVoice mixed together throughout | 0.2905 |
| AfriVoice first, then a low-learning-rate pass over WAXAL alone | 0.2785 |

The reason is transcription convention rather than acoustics. AfriVoice is punctuation-poor, so
training on it last teaches the model to stop emitting punctuation, which CER scores. Training on it
first and finishing on WAXAL keeps the acoustic benefit and restores the output style. On the Shona
specialist the refresh moved punctuated output from 32 per cent of rows to 100 per cent and improved
the holdout from 0.2185 to 0.1616.

The refresh matters only when the external corpus has different transcription conventions. It made no
difference for FLEURS, which is already punctuated: s44 scored 0.2809 against s44r at 0.2812.

### Ensemble composition

Ensembling gave between 0.006 and 0.010, but only under conditions that took several experiments to
identify.

| change | effect |
|---|---|
| four members from four different strong checkpoints | +0.0040 |
| six members from two checkpoints, varying only the decode parameter | +0.0006 |
| adding a fresh-seed member | +0.0032 |
| adding members from weak checkpoints | -0.0003 (0.764476 to 0.764152) |
| swapping which arm anchors the vote | -0.0015 (0.764915 to 0.763419) |

The rule that emerges: members must be strong, and they must fail differently. Source diversity beats
parameter diversity, and a fresh random seed is a cheaper and more effective source of diversity than
a new architecture.

Anchor choice deserves emphasis. The anchor is the arm whose text survives unless outvoted, so it
determines word identity more than any single member does.

## What did not work

These are recorded because they cost real time, and because the reasons are more useful than the
outcomes. Each is paired with the measurement that killed it.

1. **Language model fusion (KenLM).** Every combination of weights lost, even after fixing a genuine
   bug in unigram extraction. The reason is diagnostic: 76 to 84 per cent of our word errors are
   real-word errors, valid words in the wrong place. A language model cannot repair those.
2. **Lexicon spelling correction.** Every configuration scored zero or worse. Out-of-vocabulary rates
   were nearly identical between references and hypotheses, 2.9 against 2.8 per cent, so there was
   almost nothing for a lexicon to fix.
3. **Word-level voting.** Introduced on the theory that character voting was assembling words that no
   member had proposed. It left WER unchanged (0.3813 to 0.3821) and made CER much worse (0.1109 to
   0.1261). The theory was wrong: character voting was not inventing bad words, and word voting
   imported each member's punctuation conventions into the anchor's text.
4. **Architectural diversity.** Whisper-turbo fine-tuned reached 0.7331, XLS-R was the weakest arm
   trained, and a fine-tuned Sunbird 51 lost to its own zero-shot output. All three were slower and
   worse than retraining the same architecture with a different seed.
5. **Self-training on unlabelled WAXAL audio.** Flat on the holdout in both languages. The labelled
   portion of the same collection had already saturated what that audio could teach.
6. **Punctuation correction, four separate attempts.** Comma restoration by member agreement
   (0.764314 against 0.764759), period penalty at decode time (no CER gain despite matching the
   reference rate), and a fully period-corrected ensemble (0.756972). Our punctuation errors are
   systematic across the whole model family, so neither voting nor confidence-based decoding can
   locate them.
7. **Other post-processing beyond sentence case.** Apostrophes appeared to be 250 times under-emitted
   until the statistic was broken down by language: apostrophes are a Luganda feature at 2.5 per row,
   while in Lingala and Shona they occur 0.009 times per row, where our output already matched. Short
   outputs looked like failures until 17 independent arms were found to agree on them, which means
   the audio genuinely contains little speech.
8. **Forcing capitalisation to help WER.** WER lowercases, so this cannot help by construction.
   Sentence-case repair is kept only for its CER effect.
9. **Weak checkpoints as ensemble members**, and members from the same lineage as the anchor. Both
   are measured as paired comparisons under [Results](#results).

## Results

### Leaderboard history

| submission | score | CER | WER |
|---|---|---|---|
| Phase 1 champion, stripped vocabulary | 0.724984 | 0.130719 | 0.419312 |
| Sunbird 51 zero-shot, beam 8, routed | 0.737168 | 0.124049 | 0.401616 |
| **p1raw, raw capitals and punctuation vocabulary** | **0.742142** | 0.115951 | 0.399766 |
| per-language specialists, routed | 0.744835 | 0.115141 | 0.395189 |
| soup5, five-way weight average | 0.746054 | 0.114864 | 0.393027 |
| **distil, best single model: `p2n_distil_nl_f`** | **0.746787** | 0.113117 | 0.393308 |
| first three-family ensemble | 0.747041 | 0.114054 | not recorded |
| ensemble, four members | 0.751181 | 0.112669 | 0.384969 |
| ensemble, seven members | 0.754847 | 0.109473 | 0.380833 |
| **first blank-penalty member** | **0.759338** | 0.108620 | 0.372703 |
| **17 members, four new blank-penalty sources: `p2n_ens_bp10`** | **0.764476** | 0.108578 | 0.362471 |
| 25 members, all strong sources: `p2n_ens_bp25` | 0.764759 | 0.109025 | 0.361457 |
| 26 members, adds distil: `p2n_ens_distil` | 0.764915 | 0.108961 | 0.361209 |
| **26 members, corrected decoding: `p2n_ens_masked`** | **0.766563** | **0.108364** | **0.358509** |
| the two weakest members at half weight: `p2n_ens_weighted` | 0.766580 | not recorded | not recorded |
| a vote over five complete ensembles: `p2n_meta` | 0.766683 | 0.108386 | 0.358249 |
| **per-clip selection across seven ensembles: `p2n_mbr`** | **0.766791** | **0.108130** | 0.358288 |

Single-ensemble variants measured on the way: seed 46 in the weakest slot (`p2n_ens_s46swap`)
0.766477, the six-way soup in place of the five-way (`p2n_ens_soup6`) 0.766226, seed 46 as the
anchor (`p2n_ens_s46anchor`) 0.766083, the penalty bracket widened to 38 members (`p2n_ens_wide`)
0.765960.

Negative results on the same board: 21 members including weak checkpoints 0.764152; comma
restoration 0.764314; period-corrected ensemble 0.756972; distil as anchor rather than member
0.763419.

### Final standing

The scored pick was `p2n_mbr` (file `cand_mbr.csv`). On the private 70 per cent of the test set it
scored **0.772552** (CER 0.110664, WER 0.344233), finishing **second** of the competition; the
winner scored 0.780944 and third place 0.771866. The second pick was `p2n_ens_distil`. The private
score running 0.006 above the public one is expected behaviour, not luck: a 624-clip split is
kinder to an averaged system than a 268-clip split, and the selection recipe was chosen for exactly
that property.

### The paired comparisons

Two of those negative results are paired comparisons, and the comparator matters:

- **Weak members cost 0.0003.** The 21-member vote at 0.764152 is the 17-member vote at 0.764476
  plus blank-penalty members derived from the stripped-vocabulary `robust` arm (0.724984) and from
  `snaspec_p2`. Same anchor, same threshold, four extra weak members, 0.000324 lost.
- **Anchor choice costs 0.0015.** Making `distil` the anchor rather than a member scored 0.763419
  against 0.764915 for the same set of checkpoints, a difference of 0.001496.

Both of these are near or below the 0.0027 noise band described at the end of this file, and should
be read as directional rather than precise.

### Reproduction, verified on the leaderboard

The recipe in `configs/ensembles.yaml` was rebuilt from the per-arm outputs and resubmitted. It
returned **0.76491506, CER 0.108961108, WER 0.36120877**: identical to the original submission on
all three figures to every recorded digit. The rebuild differs from the archived file on 5 of 892
rows, and because a single differing row in the public split would have moved CER, all 5 fall in the
private 70 per cent. The public portion is an exact match.

This is the strongest statement available about reproducibility: the documented recipe regenerates
the submitted result, not merely something close to it.

| verification submission | score | CER | WER |
|---|---|---|---|
| recipe rebuilt from the per-arm outputs | **0.764915** | 0.108961 | 0.361209 |
| the same recipe with the retrained p1av arm | 0.764774 | 0.109046 | 0.361406 |
| retrained p1av alone | 0.726089 | 0.127544 | 0.420278 |

Seeds are fixed per arm and recorded in the configs. The one genuine source of non-determinism in
training is GPU kernel selection in cuDNN, which can move a metric in the fourth decimal place. The
published weights remove even that: inference from them is deterministic, and it is the inference
path that reproduces the submitted score.

### The republished p1av arm

The original `p1av` checkpoint was deleted during a disk cleanup and no copy existed locally, on the
Hub, or in the backup repository. It was retrained from `configs/w2vbert_p1av.yaml` at the same seed
and republished. The retrained arm is **not** the original weights, and the difference is large:

| comparison | result |
|---|---|
| retrained arm against the original, per clip | 766 of 892 transcripts differ, 85.9 per cent |
| the 26-member ensemble, same comparison | 50 of 892 rows differ, 5.6 per cent |
| retrained arm alone, public leaderboard | 0.726089 (CER 0.127544, WER 0.420278) |
| retrained arm holdout | 0.2599 |
| ensemble with the retrained arm | **0.764774** (CER 0.109046, WER 0.361406) |
| ensemble with the original arm | **0.764915** |

Three things follow, and all matter.

**The published checkpoints reproduce 0.764774, not 0.764915.** The 0.000141 gap is the retrained
arm, and it is about a tenth of the 0.0013 standard error, so the two are indistinguishable. But the
honest statement is that running `p2n_ens_distil` from the published weights gives 0.764774.

**The holdout was misleading here.** The retrained arm scored 0.2599 on the holdout against 0.2905
recorded for the original, which reads as a large improvement, while the ensemble containing it
scored fractionally *worse* on the leaderboard. Part of that is that the two holdout figures may not
share a carve, and part is that a single member's solo quality does not predict its contribution to
a vote. Holdout movements below roughly 0.03 should not be treated as evidence about the submission.

**Ensemble robustness.** An arm that disagrees with its predecessor on 86 per cent of clips moved the
final output by 5.6 per cent. The vote absorbs almost all of a single member's variation, which is
the clearest measurement in this project of why the ensemble is worth its cost.

## Disclosures

- The WAXAL Phase 1 test split is used for training. Phase 2 permits this and the organisers
  confirmed it publicly.
- Pseudo-labels were used, generated by our own models. Two kinds: over WAXAL's unlabeled pools, and
  over the 892 Phase 2 test clips.
- **One arm (`distil`) is trained on the test audio using our own ensemble's transcripts as targets.**
  This is self-training on unlabeled test audio, which the organisers explicitly permit subject to
  disclosure. No reference transcript for any test clip was used, and none exists publicly, as the
  fingerprint check below establishes independently.
- Blank-penalty and token-penalty decoding are decoding parameters, not adaptation.

### Provenance of the test clips

The 892 test clips were fingerprinted against every transcribed AfriVoice clip: 16,815 rows across
all splits of `realtime-speech/shona1` and 23,540 rows of `KasuleTrevor/Lingala_100hrs`. Two passes
were run, an exact one (duration within 80 ms, two second waveform correlation) and a tolerant one
(duration within 2 seconds, excerpt slid across the first 8 seconds). The ASR Africa Shona benchmark
was scanned as well. **No matches in any pass.** No test clip has a public transcription.

## Attention masking, and decoding a sequence to sequence arm

Two properties of the inference path are worth stating on their own, because each is worth more
score than most modelling changes and neither is obvious.

**A batched CTC forward pass must carry the attention mask.** Clips in a batch have different
lengths, so the shorter ones are zero-padded. Without the mask, self-attention treats that padding
as audio, and the encoder output changes for every frame rather than only the padded tail. It
alters roughly half of all transcripts. Masking is correct and is the default in this package;
`attention_mask: false` exists in the recipes only to rebuild members that were produced without it.

**A sequence to sequence arm cannot be decoded like a CTC arm.** `turbo_linsna_r` is a Whisper
checkpoint: it generates, and there is no per-frame arg max and no blank to penalise. Sent through
the CTC path it emits a stream of punctuation, roughly 1,376 characters per clip, and a
character-level vote absorbs that silently rather than failing. It has its own path in
`waxal_asr/modeling/whisper.py`.

Both effects are measurable end to end on the same 26-member ensemble:

| ensemble | score | CER | WER |
|---|---|---|---|
| unmasked blank-penalty members, Whisper arm decoded as CTC | 0.764915 | 0.108961 | 0.361209 |
| masked, Whisper arm still decoded as CTC | 0.766253 | 0.108407 | 0.359087 |
| masked, Whisper arm decoded as a generator | 0.766563 | 0.108364 | 0.358509 |
| **a vote over five such ensembles** | **0.766683** | **0.108386** | **0.358249** |

Together they are worth 0.001648 with no additional training. That is about 1.3 standard errors on
the public split, so the size is not decisive on its own. Two things argue the effect is real:
both corrections moved CER and WER in the same direction, and the masking effect is systematic,
altering roughly half of all transcripts rather than a handful.

`p2n_ens_masked` is the recipe to prefer. `p2n_ens_distil` is retained because it reproduces an
earlier configuration exactly, which is useful for verifying the pipeline against a known result.

## Averaging over ensembles rather than over models

A recipe here combines either models or other recipes. The second kind builds several complete
ensembles and votes over their finished transcripts, which is a different operation from adding more
members to one vote and it is worth separating.

Adding members reduces bias: more opinions on each character slot. That has a limit, and this
solution reached it. Twenty-six members is an optimum, measured from both directions:

| members | score |
|---|---|
| 17 | 0.764476 |
| 25 | 0.766374 |
| **26** | **0.766580** |
| 38 | 0.765960 |
| 42 | 0.765247 |
| 43 | 0.764912 |

Past 26 the additions are weaker decodes of checkpoints already present, and a weak member does not
merely fail to help, it votes. Below 26 the ensemble loses sources it needs.

Averaging over finished ensembles reduces variance instead. Five 26-to-38 member ensembles, each
using the correct forward pass but differing in which checkpoints fill the slots and in how much
weight the two weakest members carry, were combined by the same character vote. That is `p2n_meta`,
at 0.766683, and it has the lowest word error rate of anything measured here.

Two properties matter more than the score. The inputs were included regardless of their own results,
two of the five being worse than the best single ensemble, so the average selects nothing. And the
operation converged: extending the vote from five ensembles to seven produced a byte-identical file.
That convergence is the clearest evidence available that the configuration space is exhausted.

Eleven configuration changes were measured after the decoding corrections, and every one landed
inside the noise band described below. A system whose every remaining lever moves it by less than
its measurement error is finished, and further search on a 268 clip public split is more likely to
fit noise than to find anything.

## Selection, the last mechanism that worked

Voting exhausted itself: after the decoding corrections, eleven configuration changes in a row
landed inside the noise band. What remained was the character half of the metric, and one mechanism
still targets it directly. For each clip, build several complete ensembles, then keep the transcript
with the smallest mean normalised character edit distance to the 26 members. Selection cannot
synthesise a string no ensemble produced, and it minimises expected character error, which
positional voting does not.

| configuration | score | CER | WER |
|---|---|---|---|
| best single vote, `p2n_ens_weighted` | 0.766580 | 0.108428 | 0.358411 |
| vote over five ensembles, `p2n_meta` | 0.766683 | 0.108386 | 0.358249 |
| **selection across seven ensembles, `p2n_mbr`** | **0.766791** | **0.108130** | **0.358288** |
| the same selection over a ten candidate pool | 0.766773 | 0.108194 | 0.358260 |

The last row is the reason to believe the mechanism. A second, independently composed pool scored
within 0.000018 of the first, and both produced the two best character error rates recorded in this
work. The improvement is small in total score but it is exactly where it was aimed.

Two test time ideas from the same family were tried on the final day and are recorded as negative:
frame-level posterior averaging across the six shared-vocabulary checkpoints (0.764248; averaging
lets the blank symbol dominate wherever the models disagree on spike timing, shortening transcripts)
and single-utterance entropy-minimisation adaptation of the strongest arm (0.765910).

A third final-day negative deserves its own paragraph. A second round of ensemble distillation,
trained on the transcripts of the best available ensemble rather than the earlier weaker one,
produced the lowest holdout error ever recorded in this work, 0.2429 against 0.2746 for its
predecessor. Substituted into the vote it scored 0.766401, slightly below the ensemble it was meant
to improve. That makes three arms in a row whose holdout suggested a clear win and whose ensemble
contribution was nil or negative: a retrained checkpoint at 0.2599, a fresh seed at 0.2588, and this
one at 0.2429. The holdout is a speaker-disjoint carve of the training distribution; it measures
similarity to that distribution, and past a certain strength every new arm is optimising that
similarity rather than anything the vote lacks. In this work, no holdout improvement below roughly
0.03 ever predicted an ensemble improvement, and decisions were made accordingly.

## Measurement discipline

The public leaderboard covers roughly 268 clips. For a paired comparison between two similar systems
the standard error on the score difference is about 0.0013, so **differences below roughly 0.0027
cannot be distinguished from noise**. The final eight submissions span 0.7643 to 0.7649, which is
inside that band. The final two were therefore chosen on the principle of maximum averaging rather
than on ranking within the noise.
