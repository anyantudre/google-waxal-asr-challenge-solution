"""End-to-end test of the submission path, using cached arm outputs instead of a GPU.

The expensive part of inference is the forward pass, and `predict.py` already caches its result per
arm under data/interim. Writing those caches directly lets this test exercise the real ensemble
vote, the real post-processing and the real validation, with no model weights and no audio.

This is what catches the failure that actually bit us: a submission containing an empty cell, which
Zindi rejects outright.
"""

import csv

import pytest
import yaml

from waxal_asr.modeling.predict import load_recipes, read_test_ids, validate, write_submission

CLIP_IDS = ["ID_AAA", "ID_BBB", "ID_CCC"]


@pytest.fixture
def test_csv(tmp_path):
    path = tmp_path / "Test.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ID"])
        for clip_id in CLIP_IDS:
            writer.writerow([clip_id])
    return path


class TestReadTestIds:
    def test_reads_the_id_column(self, test_csv):
        assert read_test_ids(test_csv) == CLIP_IDS

    def test_rejects_a_csv_without_an_id_column(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("Name\nfoo\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            read_test_ids(bad)


class TestValidate:
    def test_accepts_a_correct_submission(self):
        rows = {c: "some text" for c in CLIP_IDS}
        validate(rows, CLIP_IDS)  # must not raise

    def test_rejects_an_empty_transcript(self):
        # The exact failure that blocks a Zindi submission.
        rows = {c: "some text" for c in CLIP_IDS}
        rows["ID_BBB"] = "   "
        with pytest.raises(SystemExit, match="empty"):
            validate(rows, CLIP_IDS)

    def test_rejects_a_missing_clip(self):
        rows = {c: "text" for c in CLIP_IDS[:-1]}
        with pytest.raises(SystemExit):
            validate(rows, CLIP_IDS)

    def test_rejects_a_wrong_id_at_the_right_count(self):
        # Same number of rows, one identifier substituted. This is the case a row count check
        # cannot catch, and it is how a misaligned join would show up.
        rows = {c: "text" for c in CLIP_IDS[:-1]}
        rows["ID_WRONG"] = "text"
        with pytest.raises(SystemExit, match="identifier mismatch"):
            validate(rows, CLIP_IDS)

    def test_rejects_an_extra_clip(self):
        rows = {c: "text" for c in CLIP_IDS}
        rows["ID_EXTRA"] = "text"
        with pytest.raises(SystemExit):
            validate(rows, CLIP_IDS)


class TestWriteSubmission:
    def test_writes_the_expected_header_and_order(self, tmp_path):
        rows = {"ID_CCC": "third", "ID_AAA": "first", "ID_BBB": "second"}
        out = tmp_path / "submission.csv"
        write_submission(rows, CLIP_IDS, out)

        with open(out, encoding="utf-8") as fh:
            written = list(csv.reader(fh))
        assert written[0] == ["ID", "Target"]
        # Row order follows the test list, not dictionary insertion order.
        assert [r[0] for r in written[1:]] == CLIP_IDS
        assert written[1][1] == "first"


class TestRecipes:
    """The recipe file is the reproducibility contract, so its contents are asserted here."""

    def test_the_recipe_file_parses(self):
        recipes, defaults = load_recipes()
        assert recipes
        assert defaults["vote_threshold"] == 2.0
        assert defaults["skeleton"] == "anchor"

    def test_the_submitted_recipes_are_present(self):
        # Recipes are named after the submission file each produced, so these names double as the
        # link between this repository and a specific leaderboard row.
        recipes, _ = load_recipes()
        for name in ("p2n_mbr", "p2n_ens_masked", "p2n_meta",
                     "p2n_ens_distil", "p2n_ens_bp25", "p2n_distil_nl_f"):
            assert name in recipes, f"{name} must stay in the recipe file"

    def test_the_scored_recipe_matches_what_was_submitted(self):
        # p2n_mbr produced cand_mbr.csv, the scored final pick (public 0.766791, private 0.772552).
        recipes, _ = load_recipes()
        scored = recipes["p2n_mbr"]
        assert scored["public_score"] == 0.766791
        assert scored["judged_by"] == "p2n_ens_masked", "the electorate is the corrected 26-member vote"
        assert scored["select"] == [
            "p2n_meta", "p2n_ens_weighted", "p2n_ens_masked", "p2n_ens_s46swap",
            "p2n_ens_soup6", "p2n_ens_wide", "p2n_ens_s46anchor",
        ], "the seven candidates, in tie-break order"

    def test_the_reference_recipe_matches_what_was_submitted(self):
        # p2n_ens_distil is the fixed reference for checking an installation: it rebuilds the
        # historical 26-member vote exactly.
        recipes, _ = load_recipes()
        best = recipes["p2n_ens_distil"]
        assert len(best["members"]) == 26, "the 0.764915 submission had 26 members"
        assert best["members"][0]["arm"] == "s43", "s43 anchors the submitted ensemble"
        assert "blank_penalty" not in best["members"][0], "the anchor is decoded greedily"
        assert best["public_score"] == 0.764915

    def test_every_recipe_draws_on_several_checkpoints(self):
        # Source diversity was worth far more than penalty diversity: four arms from four
        # checkpoints gained 0.0040, six arms from two checkpoints gained 0.0006.
        recipes, _ = load_recipes()
        for name in ("p2n_ens_distil", "p2n_ens_bp25", "p2n_ens_bp10"):
            arms = {m["arm"] for m in recipes[name]["members"]}
            assert len(arms) >= 6, f"{name} draws on only {len(arms)} checkpoints"

    def test_every_arm_used_is_documented_and_matches_the_registry(self):
        # The recipe file names the Hugging Face repository for each arm so a reader can fetch the
        # weights without reading Python. That block is only useful if it cannot drift from the
        # registry the code actually loads from.
        from waxal_asr.config import ENSEMBLE_ARMS, SUNBIRD_51
        from waxal_asr.modeling.predict import RECIPES_FILE

        doc = yaml.safe_load(RECIPES_FILE.read_text(encoding="utf-8"))
        arms, recipes = doc["arms"], doc["recipes"]

        used = {m["arm"] for r in recipes.values() for m in r.get("members", [])}
        undocumented = used - set(arms)
        assert not undocumented, f"arms used but not described: {sorted(undocumented)}"

        # A meta recipe combines other recipes rather than models, so its references must resolve
        # or the recipe cannot be built at all.
        for name, r in recipes.items():
            kinds = sum(k in r for k in ("members", "ensembles", "select"))
            assert kinds == 1, f"{name} must define exactly one of members, ensembles or select"
            for ref in [*r.get("ensembles", []), *r.get("select", [])]:
                assert ref in recipes, f"{name} references unknown recipe {ref}"
                assert ref != name, f"{name} references itself"
            if "select" in r:
                judge = r.get("judged_by")
                assert judge in recipes and "members" in recipes[judge], (
                    f"{name} needs judged_by naming a members recipe"
                )

        for name, entry in arms.items():
            assert entry.get("repo"), f"{name} has no repo"
            assert entry.get("note"), f"{name} has no note"
            if name == "sunbird51":
                assert entry["repo"] == SUNBIRD_51
            else:
                assert name in ENSEMBLE_ARMS, f"{name} is not in the code registry"
                assert entry["repo"] == ENSEMBLE_ARMS[name], (
                    f"{name}: recipe file says {entry['repo']}, "
                    f"registry says {ENSEMBLE_ARMS[name]}"
                )


class TestFullPipelineWithCachedArms:
    def test_produces_a_valid_submission_from_cached_arm_outputs(self, tmp_path, monkeypatch):
        from waxal_asr.ensemble import _combine
        from waxal_asr.postprocess import postprocess

        # Three arms: two agree on "cat", the anchor says "dog". A loop is planted in one arm to
        # confirm post-processing runs, and one arm is silent to confirm abstention.
        arms = [
            {"ID_AAA": "the dog sat", "ID_BBB": "hello " * 6, "ID_CCC": "ok"},
            {"ID_AAA": "the cat sat", "ID_BBB": "hello", "ID_CCC": ""},
            {"ID_AAA": "the cat sat", "ID_BBB": "hello", "ID_CCC": "ok"},
        ]
        weights = [1.0] * len(arms)
        rows = {
            clip: postprocess(_combine([a[clip] for a in arms], weights, 2.0, "anchor"))
            for clip in CLIP_IDS
        }

        validate(rows, CLIP_IDS)
        out = tmp_path / "submission.csv"
        write_submission(rows, CLIP_IDS, out)

        assert "cat" in rows["ID_AAA"], "two agreeing members should overrule the anchor"
        assert rows["ID_BBB"].lower().count("hello") <= 2, "the repetition loop should be collapsed"
        assert rows["ID_CCC"].strip(), "a silent member must not empty the cell"
        assert out.exists()
