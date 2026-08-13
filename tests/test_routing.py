"""Wake-word / route parsing — the logic that decides which window an utterance
is aimed at, and what gets typed once the wake word is eaten."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import routing  # noqa: E402


def test_plain_dictation_is_not_routed():
    target, payload = routing.split_route("שלום עידו מה שלומך")
    assert target is None
    assert payload == "שלום עידו מה שלומך"


def test_claude_wake_word_is_consumed():
    target, payload = routing.split_route("קלוד תכתוב לי תוכנית עבודה")
    assert target == "claude"
    assert payload == "תכתוב לי תוכנית עבודה"


def test_grok_wake_word():
    target, payload = routing.split_route("גרוק מה השעה")
    assert target == "grok"
    assert payload == "מה השעה"


def test_latin_and_mishear_aliases_route():
    for word in ("claude", "clod", "כלוד", "קולד"):
        target, payload = routing.split_route(f"{word} hello there")
        assert target == "claude", word
        assert payload == "hello there"


def test_punctuation_after_wake_word_is_stripped():
    target, payload = routing.split_route("קלוד, תפתח את הקובץ")
    assert target == "claude"
    assert payload == "תפתח את הקובץ"


def test_wake_word_alone_leaves_empty_payload():
    target, payload = routing.split_route("קלוד")
    assert target == "claude"
    assert payload == ""


def test_word_prefix_is_not_a_wake_word():
    """'קלודיה' starts with 'קלוד' but is a name, not a route."""
    target, payload = routing.split_route("קלודיה הגיעה")
    assert target is None
    assert payload == "קלודיה הגיעה"


def test_longest_alias_wins():
    aliases = {"claude": ["קלוד", "היי קלוד"]}
    target, payload = routing.split_route("היי קלוד מה קורה", aliases)
    assert target == "claude"
    assert payload == "מה קורה"


def test_empty_input():
    assert routing.split_route("") == (None, "")
    assert routing.split_route(None) == (None, "")


def test_aliases_load_from_keywords_file(tmp_path):
    path = tmp_path / "kw.json"
    path.write_text(json.dumps({"keywords": [
        {"phrase": "קלוד", "action": "dictate", "enabled": True},
        {"phrase": "גרוק", "action": "dictate", "enabled": True},
        {"phrase": "כבה נורות", "action": "assistant", "enabled": True},
        {"phrase": "cloud", "action": "dictate", "enabled": False},
    ]}, ensure_ascii=False), encoding="utf-8")
    aliases = routing.load_aliases(str(path))
    assert "קלוד" in aliases["claude"]
    assert "גרוק" in aliases["grok"]
    # non-dictate and disabled entries are not route words
    assert "כבה נורות" not in aliases["claude"] + aliases["grok"]
    assert "cloud" not in aliases["claude"]


def test_missing_keywords_file_falls_back_to_defaults():
    aliases = routing.load_aliases("/nonexistent/clod_keywords.json")
    assert aliases == routing.DEFAULT_ALIASES


def test_broken_file_never_disables_routing(tmp_path):
    path = tmp_path / "kw.json"
    path.write_text('{"keywords": []}', encoding="utf-8")
    aliases = routing.load_aliases(str(path))
    assert aliases["claude"], "an empty keyword list must not kill routing"


def test_repo_keywords_file_parses():
    """The file shipped in the repo must actually produce route words."""
    aliases = routing.load_aliases()
    assert "קלוד" in aliases["claude"]
    assert "גרוק" in aliases["grok"]
