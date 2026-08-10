"""Inference entry point: raw test audio to a submission CSV.

Recipes are defined in ``configs/ensembles.yaml`` rather than in this file, so a new combination of
arms and decoding settings needs no code change:

    python -m waxal_asr.modeling.predict --recipe p2n_distil_nl_f  # one arm, fast, verifies setup
    python -m waxal_asr.modeling.predict --recipe p2n_mbr          # the scored submission, 0.766791
    python -m waxal_asr.modeling.predict --recipe p2n_ens_distil   # the fixed reference ensemble
    python -m waxal_asr.modeling.predict --list                    # show available recipes

Recipes are named after the submission file each one produced, so a recipe maps to exactly one
leaderboard row.

Output is written to ``data/processed/submission.csv`` and validated before the process exits: the
row count must match the test list, the identifier sets must be identical, and no cell may be empty.
Zindi rejects a submission containing an empty cell, and one clip in the test set is short enough to
provoke one.

Per-arm transcripts are cached under ``data/interim`` so that an interrupted run resumes instead of
restarting, and so that recipes sharing an arm decode it only once.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from waxal_asr.config import (
    CONFIGS_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    resolve_model,
)
from waxal_asr.ensemble import _combine
from waxal_asr.postprocess import postprocess

RECIPES_FILE = CONFIGS_DIR / "ensembles.yaml"


def load_recipes(path: Path = RECIPES_FILE) -> tuple[dict, dict]:
    """Read the recipe file.

    Args:
        path: YAML file defining ``defaults`` and ``recipes``.

    Returns:
        A tuple of (recipes, defaults).

    Raises:
        SystemExit: if the file is missing or malformed.
    """
    if not path.exists():
        raise SystemExit(f"{path} not found")
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    recipes = doc.get("recipes")
    if not recipes:
        raise SystemExit(f"{path} defines no recipes")
    return recipes, doc.get("defaults", {})


def read_test_ids(test_csv: Path) -> list[str]:
    """Return the clip identifiers listed in the competition test CSV."""
    with open(test_csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "ID" not in rows[0]:
        raise SystemExit(f"{test_csv}: expected a CSV with an ID column")
    return [row["ID"] for row in rows]


def validate(rows: dict[str, str], expected_ids: list[str]) -> None:
    """Fail loudly on the three defects that make a submission unusable.

    Args:
        rows: mapping of clip identifier to transcript.
        expected_ids: identifiers the submission must cover, in order.

    Raises:
        SystemExit: on a row count mismatch, an identifier mismatch, or any empty transcript.
    """
    if len(rows) != len(expected_ids):
        raise SystemExit(f"row count {len(rows)} does not match the test list {len(expected_ids)}")
    if set(rows) != set(expected_ids):
        missing = set(expected_ids) - set(rows)
        extra = set(rows) - set(expected_ids)
        raise SystemExit(f"identifier mismatch: {len(missing)} missing, {len(extra)} unexpected")
    empty = [clip for clip, text in rows.items() if not text.strip()]
    if empty:
        raise SystemExit(f"{len(empty)} empty transcript(s), which Zindi rejects: {empty[:5]}")


def write_submission(rows: dict[str, str], ids: list[str], out: Path) -> None:
    """Write a two-column submission CSV in the order given by the test list."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Target"])
        for clip_id in ids:
            writer.writerow([clip_id, rows[clip_id]])
    print(f"[predict] wrote {len(ids)} rows to {out}")


def transcribe_arm(arm: str, penalty: float, ids: list[str], audio_dir: Path,
                   use_attention_mask: bool = True) -> dict[str, str]:
    """Decode every clip with one arm at one blank penalty, caching the result.

    The forward pass is the expensive part, so the cache means re-running a recipe, or running a
    second recipe that shares an arm, costs nothing extra.

    Args:
        arm: key from the model registry, or ``sunbird51`` for the third-party zero-shot arm.
        penalty: amount subtracted from the CTC blank logit; 0 is ordinary greedy decoding.
        ids: clip identifiers to transcribe.
        audio_dir: directory holding one audio file per identifier.
        use_attention_mask: CTC arms only. See transcribe_ctc; False reproduces the submitted
            blank-penalty members, which were decoded without the mask.

    Returns:
        Mapping of clip identifier to post-processed transcript.
    """
    tag = f"{arm}_bp{penalty:g}".replace(".", "")
    if not use_attention_mask:
        tag += "_nomask"          # a different forward pass is a different member, so a different cache
    cache = INTERIM_DATA_DIR / f"{tag}.json"
    if cache.exists():
        print(f"[predict] {tag}: reusing cache")
        return json.loads(cache.read_text(encoding="utf-8"))

    if arm == "sunbird51":
        from waxal_asr.modeling.sunbird import load_lang_map, transcribe_sunbird

        # The submitted member was language routed: the per-clip label picks the Whisper language
        # token. Running it unrouted produces a different member and therefore a different ensemble,
        # so a missing map is an error rather than something to shrug off.
        lang_json = INTERIM_DATA_DIR / "lid.json"
        if not lang_json.exists():
            raise SystemExit(
                f"{lang_json} not found, and the sunbird51 arm is language routed. "
                "Run `make lid` (or python -m waxal_asr.lid) first."
            )
        predictions = transcribe_sunbird(ids, audio_dir, lang_map=load_lang_map(lang_json))
    elif arm == "turbo_linsna_r":
        # A sequence to sequence model: it generates, so there is no per-frame arg max and no blank
        # to penalise. Decoding it through the CTC path yields a stream of punctuation.
        from waxal_asr.modeling.whisper import transcribe_whisper

        predictions = transcribe_whisper(resolve_model(arm), ids, audio_dir)
    else:
        from waxal_asr.modeling.ctc import transcribe_ctc

        predictions = transcribe_ctc(resolve_model(arm), ids, audio_dir, penalty=penalty,
                                     use_attention_mask=use_attention_mask)

    predictions = {clip: postprocess(text) for clip, text in predictions.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    return predictions


def run_recipe(recipe: dict, defaults: dict, ids: list[str], audio_dir: Path,
               recipes: dict | None = None, _seen: tuple = ()) -> dict[str, str]:
    """Build one recipe and return its transcripts.

    A recipe combines either models or other recipes:

    * ``members`` lists arms. Each is decoded, then combined by character-level ROVER.
    * ``ensembles`` lists other recipe names. Each is built in full, then their finished
      transcripts are combined by the same vote. This second level averages over complete
      ensembles rather than over models, which reduces variance rather than bias.

    Args:
        recipe: one entry from the ``recipes`` mapping.
        defaults: the ``defaults`` mapping, supplying the vote threshold and skeleton.
        ids: clip identifiers to transcribe.
        audio_dir: directory holding the audio.
        recipes: the full recipe mapping, required only when ``ensembles`` is used.
        _seen: recipe names already being built, used to reject a cycle.

    Returns:
        Mapping of clip identifier to the combined, post-processed transcript.

    Raises:
        SystemExit: if a recipe names itself in a cycle, or defines neither key.
    """
    threshold = float(recipe.get("vote_threshold", defaults.get("vote_threshold", 2.0)))
    skeleton = recipe.get("skeleton", defaults.get("skeleton", "anchor"))

    if "select" in recipe:
        # Per-clip minimum-Bayes-risk selection. Each candidate recipe is built in full; for every
        # clip the candidate whose transcript has the smallest mean normalised character edit
        # distance to the judging members is kept. Selection can only choose a string a candidate
        # actually produced, so unlike positional voting it cannot synthesise an unsupported one,
        # and it optimises expected character error directly.
        from rapidfuzz.distance import Levenshtein

        judge = recipe.get("judged_by")
        if not recipes or judge not in recipes or "members" not in recipes[judge]:
            raise SystemExit("a select recipe needs judged_by naming a members recipe")
        electorate = [
            transcribe_arm(m["arm"], float(m.get("blank_penalty", 0.0)), ids, audio_dir,
                           use_attention_mask=bool(m.get("attention_mask", True)))
            for m in recipes[judge]["members"]
        ]
        candidates = []
        for name in recipe["select"]:
            if name in _seen:
                raise SystemExit(f"recipe cycle: {' -> '.join((*_seen, name))}")
            if name not in recipes:
                raise SystemExit(f"unknown recipe {name!r} in select")
            candidates.append(run_recipe(recipes[name], defaults, ids, audio_dir, recipes,
                                         (*_seen, name)))
        print(f"[predict] select recipe over {len(candidates)} candidates, "
              f"judged by the {len(electorate)} members of {judge}")
        chosen = {}
        for clip in ids:
            texts = [e[clip] for e in electorate]
            best, best_i = None, 0
            for i, cand in enumerate(candidates):
                h = cand[clip]
                d = sum(Levenshtein.distance(h, x) / max(len(x), 1) for x in texts)
                if best is None or d < best:
                    best, best_i = d, i
            chosen[clip] = candidates[best_i][clip]
        return chosen

    if "ensembles" in recipe:
        names = recipe["ensembles"]
        print(f"[predict] meta recipe over {len(names)} ensembles: {', '.join(names)}")
        outputs, weights = [], []
        for name in names:
            if name in _seen:
                raise SystemExit(f"recipe cycle: {' -> '.join((*_seen, name))}")
            if not recipes or name not in recipes:
                raise SystemExit(f"unknown ensemble {name!r} referenced by a meta recipe")
            outputs.append(run_recipe(recipes[name], defaults, ids, audio_dir, recipes,
                                      (*_seen, name)))
            weights.append(1.0)
        combined = {c: _combine([o[c] for o in outputs], weights, threshold, skeleton) for c in ids}
        return {c: postprocess(t) for c, t in combined.items()}

    if "members" not in recipe:
        raise SystemExit("a recipe must define members, ensembles or select")

    members = recipe["members"]
    print(f"[predict] recipe with {len(members)} member(s), anchor is {members[0]['arm']}")
    arms = [
        transcribe_arm(m["arm"], float(m.get("blank_penalty", 0.0)), ids, audio_dir,
                       use_attention_mask=bool(m.get("attention_mask", True)))
        for m in members
    ]
    if len(arms) == 1:
        return dict(arms[0])

    # A member may carry less than a full vote. Down-weighting a weak arm keeps the decorrelation
    # it contributes while limiting the damage it can do, which removing it entirely would lose.
    weights = [float(m.get("weight", 1.0)) for m in members]
    combined = {
        clip: _combine([arm[clip] for arm in arms], weights, threshold, skeleton) for clip in ids
    }
    return {clip: postprocess(text) for clip, text in combined.items()}


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Produce a submission CSV from released weights.")
    parser.add_argument("--recipe", default="p2n_distil_nl_f",
                        help="recipe name from configs/ensembles.yaml")
    parser.add_argument("--list", action="store_true", help="list available recipes and exit")
    parser.add_argument("--audio-dir", type=Path, default=RAW_DATA_DIR / "test_audio")
    parser.add_argument("--test-csv", type=Path, default=RAW_DATA_DIR / "Test.csv")
    parser.add_argument("--out", type=Path, default=PROCESSED_DATA_DIR / "submission.csv")
    parser.add_argument("--recipes-file", type=Path, default=RECIPES_FILE)
    args = parser.parse_args()

    recipes, defaults = load_recipes(args.recipes_file)

    if args.list:
        print(f"recipes in {args.recipes_file}:")
        for name, recipe in recipes.items():
            score = recipe.get("public_score")
            score_text = f"public {score}" if score else "not submitted"
            if "ensembles" in recipe:
                size = f"{len(recipe['ensembles']):2} ensembles"
            elif "select" in recipe:
                size = f"{len(recipe['select']):2} candidates"
            else:
                size = f"{len(recipe['members']):2} member(s)"
            print(f"  {name:18} {size}  {score_text}"
                  f"  {recipe.get('description', '')}")
        return

    if args.recipe not in recipes:
        raise SystemExit(f"unknown recipe {args.recipe!r}; available: {', '.join(recipes)}")
    if not args.test_csv.exists():
        raise SystemExit(f"{args.test_csv} not found. See the README for the expected layout.")
    if not args.audio_dir.exists():
        raise SystemExit(f"{args.audio_dir} not found. See the README for the expected layout.")

    ids = read_test_ids(args.test_csv)
    print(f"[predict] {len(ids)} clips, recipe {args.recipe}")

    rows = run_recipe(recipes[args.recipe], defaults, ids, args.audio_dir, recipes)
    validate(rows, ids)
    write_submission(rows, ids, args.out)


if __name__ == "__main__":
    main()
