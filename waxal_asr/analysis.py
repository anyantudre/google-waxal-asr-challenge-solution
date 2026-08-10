"""Regenerate the data insights the solution was built on, from whatever the checkout holds.

    python -m waxal_asr.analysis --audio-dir data/raw/test_audio --out reports/
    python -m waxal_asr.analysis --audio-dir data/raw/test_audio --submission sub.csv

Four independent sections are written to ``reports/data_insights.md``: clip durations, the language
mix, the reference word and punctuation rates, and the same rates for any submission given with
``--submission``. Every section degrades on its own. A section whose input is absent prints why it
was skipped and says so in the report rather than filling the gap with an estimate, so the script
runs on a fresh checkout that holds nothing but the code.

Two of these measurements decided how the pipeline decodes. The fraction of clips longer than 30
seconds decides whether a Whisper arm needs chunking, because its encoder window is fixed at 30
seconds. The word rate decides the blank penalty: greedy CTC ran at 1.30 words per second against a
reference rate of 1.41, which is under-generation, and a missing word is a deletion that costs word
error heavily and character error almost nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from waxal_asr.config import INTERIM_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR, REPORTS_DIR
from waxal_asr.decode import REFERENCE_WORDS_PER_SECOND

# Whisper's encoder consumes a fixed 30 second window, so a longer clip loses its tail unless it is
# chunked. CTC arms have no such limit, which is why this fraction is reported rather than assumed.
WHISPER_WINDOW_S = 30.0

# Rates measured over the Lingala and Shona references. They are the yardstick a submission is read
# against: a word rate below the reference means the decoder is dropping words, and a punctuation
# rate away from the reference costs twice, since character error is scored on the raw string and
# word error does not strip punctuation. The word rate (REFERENCE_WORDS_PER_SECOND, imported above)
# lives in waxal_asr.decode, which calibrates the blank penalty against the same yardstick.
REFERENCE_COMMAS_PER_ROW = 0.630
REFERENCE_PERIODS_PER_ROW = 1.402

# The corrected test set ships WAV, earlier phases shipped MP3, and the corpora built here are FLAC.
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac")

# Searched, in order, for the two optional json inputs.
SEARCH_DIRS = (PROJ_ROOT / "results" / "eval", INTERIM_DATA_DIR)

# Top level keys of a language identification document that are metadata, not clip identifiers.
LID_METADATA_KEYS = frozenset({"model", "n", "failed", "distribution"})


def _rel(path: Path) -> str:
    """Path relative to the project root when possible, so the report carries no absolute paths."""
    try:
        return Path(path).resolve().relative_to(PROJ_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _skipped(title: str, reason: str) -> list[str]:
    """Markdown for a section whose input is absent, also reported on standard output."""
    print(f"[analysis] {title}: skipped, {reason}")
    return [f"## {title}", "", f"Skipped: {reason}.", ""]


def _search_dirs_text() -> str:
    """The directories searched for optional inputs, for a skip message."""
    return " or ".join(_rel(directory) for directory in SEARCH_DIRS)


def find_json(patterns: tuple[str, ...]) -> Path | None:
    """First json matching any of the patterns under the search directories, or None."""
    for directory in SEARCH_DIRS:
        if not directory.is_dir():
            continue
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
    return None


def clip_durations(audio_dir: Path) -> tuple[dict[str, float], int]:
    """Read the duration of every audio file in a directory without decoding it.

    ``soundfile.info`` parses the header only, so this stays fast over the whole test set and needs
    no resampling. The file extension is discovered rather than assumed, because the competition
    shipped WAV for the corrected test set and MP3 earlier.

    Args:
        audio_dir: directory holding one audio file per clip.

    Returns:
        A tuple of (mapping of file stem to duration in seconds, count of files whose header could
        not be read). Both are empty or zero if the directory holds no audio.
    """
    durations: dict[str, float] = {}
    unreadable = 0
    for path in sorted(audio_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        try:
            durations[path.stem] = float(sf.info(path).duration)
        except Exception:
            unreadable += 1
    return durations, unreadable


def describe_durations(durations: dict[str, float]) -> dict[str, float]:
    """Summarise clip durations, including the fraction above Whisper's 30 second window.

    Args:
        durations: mapping of clip identifier to duration in seconds, which must not be empty.

    Returns:
        Mapping with the clip count, total hours, mean, median, minimum and maximum in seconds, and
        the fraction of clips longer than the window as a value between 0 and 1.
    """
    values = np.asarray(list(durations.values()), dtype=float)
    return {
        "clips": int(values.size),
        "total_hours": float(values.sum() / 3600.0),
        "mean_s": float(values.mean()),
        "median_s": float(np.median(values)),
        "min_s": float(values.min()),
        "max_s": float(values.max()),
        "frac_over_window": float((values > WHISPER_WINDOW_S).mean()),
    }


def read_lid(path: Path) -> tuple[dict[str, tuple[str, float | None]], str | None]:
    """Read a language identification json as clip identifier to language and confidence.

    Three layouts are accepted, because the file may come from the routing step of an inference run
    or from a stand-alone evaluation: a document carrying a ``per_clip`` mapping, a flat mapping of
    clip to a record with a ``lang`` field, or a flat mapping of clip straight to a language code.

    Args:
        path: json file to read.

    Returns:
        A tuple of (mapping of clip identifier to (language, confidence or None), the identifier of
        the model that produced the file or None). The mapping is empty when no layout matched.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return {}, None
    per_clip = doc.get("per_clip")
    records = per_clip if isinstance(per_clip, dict) else doc
    model = doc.get("model") if isinstance(doc.get("model"), str) else None

    labels: dict[str, tuple[str, float | None]] = {}
    for clip, value in records.items():
        # In the flat layout the metadata keys sit alongside the clips and would otherwise be read
        # as clips whose language is the model name.
        if records is doc and clip in LID_METADATA_KEYS:
            continue
        if isinstance(value, str):
            labels[clip] = (value, None)
        elif isinstance(value, dict) and isinstance(value.get("lang"), str):
            confidence = value.get("conf")
            usable = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            labels[clip] = (value["lang"], float(confidence) if usable else None)
    return labels, model


def summarise_lid(labels: dict[str, tuple[str, float | None]]) -> list[dict]:
    """Per language clip count, share of the set and mean confidence, most frequent first.

    Args:
        labels: mapping of clip identifier to (language, confidence or None).

    Returns:
        One row per language, each with ``lang``, ``clips``, ``share`` between 0 and 1, and
        ``mean_conf``, which is None when the source file recorded no confidence.
    """
    clips: dict[str, int] = {}
    confidences: dict[str, list[float]] = {}
    for language, confidence in labels.values():
        clips[language] = clips.get(language, 0) + 1
        if confidence is not None:
            confidences.setdefault(language, []).append(confidence)
    total = len(labels)
    rows = [
        {
            "lang": language,
            "clips": count,
            "share": count / total,
            "mean_conf": float(np.mean(confidences[language])) if language in confidences else None,
        }
        for language, count in clips.items()
    ]
    return sorted(rows, key=lambda row: row["clips"], reverse=True)


def read_holdout(path: Path) -> list[dict]:
    """Read a holdout predictions json, or return an empty list if the layout is unrecognised.

    Args:
        path: json file expected to hold a list of records, each with a hypothesis and a reference.

    Returns:
        The records that carry both a ``hyp`` field and at least one reference field, which is
        ``raw_ref`` when casing and punctuation are preserved and ``ref`` when they are not.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, list):
        return []
    return [
        record
        for record in doc
        if isinstance(record, dict)
        and isinstance(record.get("hyp"), str)
        and (isinstance(record.get("raw_ref"), str) or isinstance(record.get("ref"), str))
    ]


def text_rates(texts: list[str]) -> dict[str, float]:
    """Word and punctuation rates for a set of transcripts.

    Commas and full stops are counted literally rather than through the normaliser: character error
    is scored on the raw string and word error does not strip punctuation, so both marks are scored
    and both are worth measuring.

    Args:
        texts: transcripts, one per row.

    Returns:
        Mapping with the row count, the total word count, and the per row word, comma and full stop
        rates. An empty input returns zeroes.
    """
    rows = len(texts)
    if rows == 0:
        return {"rows": 0, "words": 0, "words_per_row": 0.0, "commas": 0.0, "periods": 0.0}
    words = sum(len(text.split()) for text in texts)
    return {
        "rows": rows,
        "words": words,
        "words_per_row": words / rows,
        "commas": sum(text.count(",") for text in texts) / rows,
        "periods": sum(text.count(".") for text in texts) / rows,
    }


def section_durations(audio_dir: Path, durations: dict[str, float], unreadable: int) -> list[str]:
    """Markdown for the clip duration section, or a skip note when no audio was read.

    Args:
        audio_dir: directory the durations came from, quoted in the report and the skip message.
        durations: mapping of clip identifier to seconds, empty when nothing was read.
        unreadable: number of files whose header could not be parsed.

    Returns:
        Markdown lines.
    """
    title = "1. Clip durations"
    if not durations:
        return _skipped(title, f"no readable audio in {_rel(audio_dir)}")

    stats = describe_durations(durations)
    lines = [
        f"## {title}",
        "",
        f"Read from `{_rel(audio_dir)}`, headers only.",
        "",
        "| statistic | value |",
        "|---|---|",
        f"| clips | {stats['clips']} |",
        f"| total | {stats['total_hours']:.2f} hours |",
        f"| mean | {stats['mean_s']:.2f} s |",
        f"| median | {stats['median_s']:.2f} s |",
        f"| minimum | {stats['min_s']:.2f} s |",
        f"| maximum | {stats['max_s']:.2f} s |",
        f"| longer than {WHISPER_WINDOW_S:.0f} s | "
        f"{stats['frac_over_window'] * 100:.1f} per cent |",
        "",
        f"Whisper's encoder consumes a fixed {WHISPER_WINDOW_S:.0f} second window, so every clip "
        "above that line loses its tail unless the arm chunks it. A CTC arm has no such limit, "
        "which is one reason the ensemble is built on CTC arms and uses Whisper as a minority "
        "member.",
        "",
    ]
    if unreadable:
        lines += [f"{unreadable} file(s) could not be read and are excluded from the table.", ""]
    return lines


def section_langid() -> list[str]:
    """Markdown for the language mix, from the first language identification json found."""
    title = "2. Language identification"
    path = find_json(("*lid*.json", "*langid*.json"))
    if path is None:
        return _skipped(title, f"no language identification json in {_search_dirs_text()}")

    labels, model = read_lid(path)
    if not labels:
        return _skipped(title, f"{_rel(path)} is not in a recognised layout")

    source = f"Source: `{_rel(path)}`"
    lines = [
        f"## {title}",
        "",
        f"{source}, model `{model}`." if model else f"{source}.",
        "",
        "| language | clips | share | mean confidence |",
        "|---|---|---|---|",
    ]
    for row in summarise_lid(labels):
        confidence = f"{row['mean_conf']:.3f}" if row["mean_conf"] is not None else "not measured"
        lines.append(
            f"| {row['lang']} | {row['clips']} | {row['share'] * 100:.1f} per cent | "
            f"{confidence} |"
        )
    lines += [
        "",
        "Languages outside the target set are listed exactly as the model returned them. "
        "docs/SOLUTION.md records that every such clip in the corrected test set was inspected "
        "individually and found to be a neighbouring-language confusion rather than a coverage "
        "gap.",
        "",
    ]
    return lines


def section_reference_text() -> list[str]:
    """Markdown for the reference and hypothesis text rates, from a holdout predictions json."""
    title = "3. Reference text rates"
    path = find_json(("*holdout_preds*.json", "*_preds*.json"))
    if path is None:
        return _skipped(title, f"no holdout predictions json in {_search_dirs_text()}")

    records = read_holdout(path)
    if not records:
        return _skipped(title, f"{_rel(path)} is not in a recognised layout")

    # raw_ref keeps casing and punctuation; ref has been through the normaliser, so its comma and
    # full stop counts would be zero and must not be reported as reference rates.
    raw_refs = [r["raw_ref"] for r in records if isinstance(r.get("raw_ref"), str)]
    punctuated = len(raw_refs) == len(records)
    references = raw_refs if punctuated else [r.get("raw_ref") or r["ref"] for r in records]
    hypotheses = [r["hyp"] for r in records]

    reference = text_rates(references)
    hypothesis = text_rates(hypotheses)
    carried = hypothesis["words"] / reference["words"] if reference["words"] else 0.0

    # A decoder that emits punctuation at all emits it somewhere across a thousand rows, so a total
    # of zero means the column was stored after normalisation and its rate is unmeasurable here,
    # which is a different statement from a rate of zero.
    hyp_punctuated = hypothesis["commas"] > 0 or hypothesis["periods"] > 0

    def rate(stats: dict[str, float], key: str, measured: bool) -> str:
        """Format a punctuation rate, or mark it unmeasurable on normalised text."""
        return f"{stats[key]:.3f}" if measured else "not measured"

    lines = [
        f"## {title}",
        "",
        f"Source: `{_rel(path)}`, {len(records)} rows.",
        "",
        "| statistic | reference | hypothesis |",
        "|---|---|---|",
        f"| words per row | {reference['words_per_row']:.2f} | "
        f"{hypothesis['words_per_row']:.2f} |",
        f"| commas per row | {rate(reference, 'commas', punctuated)} | "
        f"{rate(hypothesis, 'commas', hyp_punctuated)} |",
        f"| full stops per row | {rate(reference, 'periods', punctuated)} | "
        f"{rate(hypothesis, 'periods', hyp_punctuated)} |",
        "",
        f"The hypothesis carries {carried * 100:.1f} per cent of the reference word count. A value "
        "below 100 per cent is under-generation: greedy CTC drops a character, and often the whole "
        "word with it, whenever the blank marginally outranks the best character. Subtracting a "
        "constant from the blank logit at decode time restores them. This figure depends on the "
        "arm that produced the file: docs/SOLUTION.md records 97.9 per cent for the arm that "
        "motivated blank-penalty decoding.",
        "",
        "Per second rates are not measured in this section: the holdout predictions file records "
        "no clip durations.",
        "",
    ]
    if not punctuated:
        lines += [
            "Reference punctuation is not measured here because this file stores normalised "
            "references, from which punctuation has already been removed.",
            "",
        ]
    if not hyp_punctuated:
        lines += [
            "Hypothesis punctuation is not measured here for the same reason: this file stores "
            "the hypotheses after normalisation, so their punctuation rate is unknown rather "
            "than zero.",
            "",
        ]
    return lines


def submission_rates(path: Path, durations: dict[str, float]) -> tuple[list[str], str | None]:
    """Read one submission CSV and measure its word and punctuation rates.

    The word rate is computed over the rows whose audio is present, so it stays correct when the
    submission and the audio directory cover different clip sets.

    Args:
        path: submission CSV with an ``ID`` column and a ``Target`` column.
        durations: mapping of clip identifier to seconds, used to convert words into a word rate.

    Returns:
        A tuple of (markdown lines, reason the file was skipped). Exactly one of the two is set: on
        success the reason is None, and on failure the lines are empty.
    """
    if not path.exists():
        return [], "not found"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "ID" not in frame.columns or "Target" not in frame.columns:
        return [], f"columns are {list(frame.columns)}, expected ID and Target"

    texts = frame["Target"].tolist()
    rates = text_rates(texts)
    matched = [(clip, text) for clip, text in zip(frame["ID"], texts) if clip in durations]
    seconds = sum(durations[clip] for clip, _ in matched)
    words = sum(len(text.split()) for _, text in matched)

    def row(name: str, value: float | None, reference: float, decimals: int) -> str:
        """One comparison row: measured value, recorded reference, and the difference."""
        if value is None:
            return f"| {name} | not measured | {reference:.{decimals}f} | not measured |"
        return (
            f"| {name} | {value:.{decimals}f} | {reference:.{decimals}f} | "
            f"{value - reference:+.{decimals}f} |"
        )

    words_per_second = words / seconds if seconds > 0 else None
    lines = [
        f"### `{_rel(path)}`",
        "",
        f"{rates['rows']} rows, {rates['words']} words.",
        "",
        "| statistic | submission | reference | difference |",
        "|---|---|---|---|",
        row("words per second", words_per_second, REFERENCE_WORDS_PER_SECOND, 2),
        row("commas per row", rates["commas"], REFERENCE_COMMAS_PER_ROW, 3),
        row("full stops per row", rates["periods"], REFERENCE_PERIODS_PER_ROW, 3),
        "",
    ]
    if words_per_second is None:
        lines += [
            "The word rate needs clip durations, and none of these identifiers were found in the "
            "audio directory.",
            "",
        ]
    else:
        lines += [
            f"The word rate covers the {len(matched)} of {rates['rows']} rows "
            "whose audio was read.",
            "",
        ]
    return lines, None


def section_submissions(paths: list[Path], durations: dict[str, float]) -> list[str]:
    """Markdown comparing every submission given on the command line against the reference rates.

    Args:
        paths: submission CSV paths from ``--submission``.
        durations: mapping of clip identifier to seconds, empty when no audio was read.

    Returns:
        Markdown lines, or a skip note when no submission was given.
    """
    title = "4. Submission text rates"
    if not paths:
        return _skipped(title, "no submission given, pass one with --submission")

    lines = [
        f"## {title}",
        "",
        "Rates measured over the Lingala and Shona references, for comparison: "
        f"{REFERENCE_WORDS_PER_SECOND:.2f} words per second, "
        f"{REFERENCE_COMMAS_PER_ROW:.3f} commas per row, "
        f"{REFERENCE_PERIODS_PER_ROW:.3f} full stops per row. A word rate below the reference "
        "means deletions, which cost word error heavily and character error almost nothing; a "
        "rate above it means the decoder is inserting.",
        "",
    ]
    for path in paths:
        rows, reason = submission_rates(path, durations)
        if reason is not None:
            print(f"[analysis] {title}: skipped {_rel(path)}, {reason}")
            lines += [f"### `{_rel(path)}`", "", f"Skipped: {reason}.", ""]
            continue
        lines += rows
    return lines


def build_report(audio_dir: Path, submissions: list[Path]) -> str:
    """Assemble the whole report, running each section independently of the others.

    Args:
        audio_dir: directory holding the test audio.
        submissions: submission CSV paths to profile, possibly empty.

    Returns:
        The markdown document as a single string.
    """
    durations: dict[str, float] = {}
    unreadable = 0
    if audio_dir.is_dir():
        durations, unreadable = clip_durations(audio_dir)

    lines = [
        "# Data insights",
        "",
        "Regenerated by `python -m waxal_asr.analysis`. Each section is independent, and a section "
        "whose input is absent is reported as skipped rather than estimated.",
        "",
    ]
    lines += section_durations(audio_dir, durations, unreadable)
    lines += section_langid()
    lines += section_reference_text()
    lines += section_submissions(submissions, durations)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Measure the data behind the solution.")
    parser.add_argument("--audio-dir", type=Path, default=RAW_DATA_DIR / "test_audio")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR, help="output directory")
    parser.add_argument(
        "--submission", type=Path, nargs="*", default=[], help="submission CSV(s) to profile"
    )
    args = parser.parse_args()

    report = build_report(args.audio_dir, list(args.submission))
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "data_insights.md"
    destination.write_text(report, encoding="utf-8")
    print(f"[analysis] wrote {destination}")


if __name__ == "__main__":
    main()
