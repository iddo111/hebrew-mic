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
    """Both aliases match at position 0, so a first-match-wins parser would fail here."""
    aliases = {"claude": ["היי", "היי קלוד"]}
    target, payload = routing.split_route("היי קלוד מה קורה", aliases)
    assert target == "claude"
    assert payload == "מה קורה"


def test_longest_alias_wins_regardless_of_declaration_order():
    aliases = {"claude": ["היי קלוד", "היי"]}
    assert routing.split_route("היי קלוד מה קורה", aliases) == ("claude", "מה קורה")


def test_dictated_punctuation_is_preserved():
    """Only the separator attached to the wake word is eaten — quotes and a leading
    minus are things the speaker actually said."""
    assert routing.split_route('קלוד "שלום עולם"') == ("claude", '"שלום עולם"')
    assert routing.split_route("קלוד -5 מעלות") == ("claude", "-5 מעלות")


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


def test_shipped_file_carries_every_builtin_alias():
    """clod_keywords.json is the source of truth, so it must not silently lose the
    Whisper mishears the engine used to hardcode."""
    aliases = routing.load_aliases()
    for target, words in routing.DEFAULT_ALIASES.items():
        for word in words:
            assert word in aliases[target], f"{word!r} missing from clod_keywords.json"


def test_mishears_route_through_the_shipped_file():
    aliases = routing.load_aliases()
    for word in ("כלוד", "קולד", "קלאוד", "קלוו"):
        assert routing.split_route(f"{word} שלום", aliases)[0] == "claude", word
    for word in ("גראוק", "גורוק"):
        assert routing.split_route(f"{word} שלום", aliases)[0] == "grok", word
