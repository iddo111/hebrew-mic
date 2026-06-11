"""corrections.py — teachable correction map for dictation ("learn while using").

A simple {wrong: right} dictionary in corrections.json, applied to EVERY
transcription after Whisper + HaikuBrainz. Iddo (or a voice 'teach' command)
adds entries, so the app learns his words over time — e.g. Whisper hears 'העיכוב',
the map fixes it to 'Haiku'. Deterministic, instant, no model retrain needed.
Hot-reloads the file each call so edits take effect without restarting the widget.
"""
from __future__ import annotations
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "corrections.json")


def _load() -> dict:
    try:
        d = json.load(open(PATH, encoding="utf-8"))
        return {k: v for k, v in d.items() if not k.startswith("_")}
    except Exception:
        return {}


def fix(text: str) -> str:
    """Apply learned corrections to `text`. Longest patterns first so multi-word
    fixes win; case-insensitive match for Latin, exact for Hebrew."""
    if not text:
        return text
    m = _load()
    for wrong in sorted(m, key=len, reverse=True):
        right = m[wrong]
        try:
            text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
        except Exception:
            text = text.replace(wrong, right)
    return text


def teach(wrong: str, right: str) -> bool:
    """Add a learned correction (used by a future voice 'תלמד X זה Y' command)."""
    try:
        try:
            d = json.load(open(PATH, encoding="utf-8"))
        except Exception:
            d = {}
        d[wrong.strip()] = right.strip()
        json.dump(d, open(PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for t in ["תפעיל את העיכוב", "אני בונה אג'נטים על אופן קלו", "שלום רגיל בלי תיקון"]:
        print(f"{t!r} -> {fix(t)!r}")
