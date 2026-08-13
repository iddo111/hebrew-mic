# routing.py — pure wake-word / route parsing for the dictation pipeline.
#
# Kept dependency-free (stdlib only) so it can be unit-tested on any OS without
# a mic, a GPU or Windows. main.py owns the window-focusing side effects; this
# module only answers "which target did the speaker name, and what's left?".
# לזכרה של עלמה ז״ל
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KEYWORDS_PATH = os.path.join(HERE, "clod_keywords.json")

# Fallback aliases, used when clod_keywords.json is missing or unreadable.
# The file is the source of truth (README documents it as such).
DEFAULT_ALIASES: dict[str, list[str]] = {
    "claude": ["קלוד", "claude", "clod", "cloud", "כלוד", "קולד", "קלאוד", "קלוו"],
    "grok": ["גרוק", "grok", "גראוק", "גורוק", "גרוֹק"],
}

# Which logical target each keyword phrase in clod_keywords.json belongs to.
_TARGET_HINTS = {
    "claude": ("קלוד", "claude", "clod", "cloud", "כלוד", "קולד", "קלאוד", "קלוו"),
    "grok": ("גרוק", "grok", "גראוק", "גורוק"),
}

_STRIP = ".,!:;?—־-'\"“”׳״"


def load_aliases(path: str = KEYWORDS_PATH) -> dict[str, list[str]]:
    """Read route aliases from clod_keywords.json, falling back to DEFAULT_ALIASES.

    Only entries with action 'dictate' and enabled=true are considered a route
    word; a phrase is assigned to the target whose hint it contains."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {k: list(v) for k, v in DEFAULT_ALIASES.items()}

    out: dict[str, list[str]] = {k: [] for k in _TARGET_HINTS}
    for kw in data.get("keywords") or []:
        if not kw.get("enabled") or kw.get("action") != "dictate":
            continue
        phrase = (kw.get("phrase") or "").strip().lower()
        if not phrase:
            continue
        for target, hints in _TARGET_HINTS.items():
            if any(h in phrase for h in hints):
                if phrase not in out[target]:
                    out[target].append(phrase)
                break
    # Never return an empty alias set — a broken/edited file must not silently
    # disable routing.
    for target, fallback in DEFAULT_ALIASES.items():
        if not out.get(target):
            out[target] = list(fallback)
    return out


def split_route(text: str, aliases: dict[str, list[str]] | None = None):
    """If `text` starts with a route word return (target, payload-after-word);
    else (None, text). The route word itself is consumed, never typed.

    Multi-word aliases ("היי קלוד") are matched before single-word ones."""
    aliases = aliases if aliases is not None else DEFAULT_ALIASES
    s = (text or "").strip()
    if not s:
        return None, ""
    low = s.lower()

    best: tuple[str, int] | None = None  # (target, matched-length)
    for target, words in aliases.items():
        for alias in words:
            a = (alias or "").strip().lower()
            if not a or not low.startswith(a):
                continue
            rest = s[len(a):]
            # The alias must end on a word boundary, so "קלודיה" isn't a route.
            if rest and rest[0] not in " \t\n" and rest[0] not in _STRIP:
                continue
            if best is None or len(a) > best[1]:
                best = (target, len(a))
    if best is None:
        return None, s
    payload = s[best[1]:].lstrip(_STRIP + " \t\n").strip()
    return best[0], payload
