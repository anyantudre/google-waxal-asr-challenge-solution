"""Tests for the text normalisation policy.

The policy is deliberately conservative. Three characters that a generic normaliser would strip are
lexical in these languages and must survive, and getting that wrong inflates both error rates.
"""

from waxal_asr.normalize import normalize_text


class TestLexicalCharactersSurvive:
    def test_apostrophe_is_kept(self):
        # Lingala elision (n'a), Shona n'anga, Luganda ng'. Stripping these merges distinct words.
        assert "'" in normalize_text("n'anga ayenda")

    def test_hyphen_is_kept(self):
        # Reduplication is productive in Bantu languages and is written with a hyphen.
        assert "-" in normalize_text("mwana-mwasi")

    def test_accents_are_kept(self):
        # Lingala orthography carries French accents; NFKC or accent stripping would destroy them.
        assert "\u00e9" in normalize_text("caf\u00e9")


class TestNormalisation:
    def test_lowercases_by_default(self):
        assert normalize_text("Bato Ya Masano") == "bato ya masano"

    def test_strips_sentence_punctuation(self):
        assert normalize_text("Hello, world.") == "hello world"

    def test_strips_curly_quotes(self):
        # Curly quotes appear in some sources and must normalise away like their ASCII forms.
        assert normalize_text("he said \u201chello\u201d") == "he said hello"

    def test_collapses_whitespace(self):
        assert normalize_text("too   many    spaces") == "too many spaces"

    def test_handles_none_and_empty(self):
        assert normalize_text(None) == ""
        assert normalize_text("") == ""

    def test_is_idempotent(self):
        once = normalize_text("Hello,  World!  ")
        assert normalize_text(once) == once
