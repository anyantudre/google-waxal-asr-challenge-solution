# Contributing

The project is a working ASR pipeline as well as a competition solution, and it is meant to be forked
for other languages and other challenges. These conventions keep it readable as it grows.

## Running the checks

```bash
make dev        # install dependencies plus test and lint tools
make test       # 90 tests, no GPU, no downloads, a few seconds
make lint       # ruff check
```

The test suite enforces the style rules below mechanically, so a violation fails the build rather
than waiting for review.

## Docstring conventions

**Module level.** Every module opens with a docstring: one summary line, a blank line, then the
context a reader needs to use or modify it. For modules that implement a measured decision, state the
measurement, because the number is the reason the code looks the way it does.

```python
"""Short summary of what this module provides.

Longer explanation: what problem it solves, and any measurement or constraint that determined the
approach. Keep it to what a maintainer needs; put narrative in docs/SOLUTION.md.
"""
```

**Class level.** One summary line. Add attribute documentation only when the attributes are not
obvious from their names and types.

**Function level.** Scale the docstring to the function:

- Short, self-evident helpers get a single line.

  ```python
  def norm_word(word: str) -> str:
      """Comparison form of a word: letters and digits only, lowercased."""
  ```

- Anything with several parameters, a non-obvious contract, or the ability to raise gets the full
  form. Use Google style, which renders correctly in most tooling.

  ```python
  def validate(rows: dict[str, str], expected_ids: list[str]) -> None:
      """Fail loudly on the three defects that make a submission unusable.

      Args:
          rows: mapping of clip identifier to transcript.
          expected_ids: identifiers the submission must cover, in order.

      Raises:
          SystemExit: on a row count mismatch, an identifier mismatch, or any empty transcript.
      """
  ```

Document the reason rather than the mechanism. `# add the attention mask` restates the code;
`# without the mask, padding is treated as speech and half the transcripts change` explains why the
line cannot be removed.

## Formatting rules

These are enforced by `tests/test_formatting.py`:

- No em dash or en dash. Use a comma, a colon, parentheses, or a plain hyphen.
- No curly quotes or curly apostrophes. Straight ASCII only.
- No Unicode arrows and no emoji, anywhere in code, comments or documentation.
- No decorative banners. A section opens with a plain `# Title` comment, never a row of dashes or
  equals signs.
- Line length 100, checked by ruff.

Non-ASCII characters are allowed only where they carry meaning, such as a Lingala or Shona example.
Where a non-ASCII character is functional rather than illustrative, for example inside a regular
expression that strips curly quotes, write it as a backslash-u escape so the source stays portable.

## Adding an ensemble recipe

Recipes live in `configs/ensembles.yaml` and need no code change:

```yaml
  my_recipe:
    description: what this combination is testing
    members:
      - {arm: s43}                      # the first entry is the anchor
      - {arm: s44, blank_penalty: 1.5}
```

```bash
python -m waxal_asr.modeling.predict --recipe my_recipe
```

Two findings are worth knowing before designing one. Members must be strong and must fail
differently: adding four members drawn from weak checkpoints took the vote from 0.764476 to
0.764152. Anchor choice matters more than any single member, because the anchor's text survives
unless it is outvoted; swapping the anchor to the strongest arm scored 0.763419 against 0.764915.

## Adding a language

Nothing in the package hardcodes Lingala or Shona. To target another language:

1. Add a corpus builder following the recipe recorded in `docs/dataset_card.md` (the builder
   scripts themselves are not shipped in this package); any Hugging Face dataset with an audio
   column and a text column fits the manifest format.
2. Copy a config from `configs/` and change `data.languages` and the external manifests.
3. Train with `make train ARM=<stem>`, where ARM is the config name after `w2vbert_`, or call
   `python -m waxal_asr.modeling.train --config configs/<file>.yaml` directly.

The character vocabulary is built from the training transcripts, so a new script or a new set of
diacritics is handled automatically. Keep the raw vocabulary setting: it is worth 0.0171 on this
metric and the reasoning applies to any metric that scores characters on unmodified text.
