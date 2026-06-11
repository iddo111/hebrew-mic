"""haiku_correct.py — HaikuBrainz: a fast Haiku-powered autocorrect for dictation.

Whisper gives raw voice→text with predictable errors (homophones, missing punctuation,
run-on spacing, mis-segmented Hebrew/English). This runs the raw line through Claude
HAIKU (fast + cheap) with the recent context, so the text that lands in the prompt is
clean — "autocorrect backwards and forwards" using context, like smart typing.

Design rules:
- NEVER blocks the dictation loop: short timeout, and on ANY failure returns the RAW
  text unchanged (dictation keeps working even if Haiku is down/slow).
- Haiku only (model configurable). Key-free via the Max-sub claude.exe, stdin UTF-8.
- Output is ONLY the corrected text — the persona forbids preamble/quotes.
- Idempotent-ish: if raw is already clean, Haiku returns it ~unchanged.
"""
from __future__ import annotations
import os, subprocess, sys

# Path to the Claude CLI binary. Override with the CLAUDE_CLI env var.
# Default resolves "claude" on PATH (works once the Claude CLI is installed).
CLAUDE = os.environ.get("CLAUDE_CLI", "claude")
HAIKU_MODEL = "haiku"   # claude.exe alias (use --model, NOT -m)
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

_SYS = (
    "You are a SUPER-SMART dictation autocorrector for Hebrew (and English). The text below "
    "was produced by voice→text (Whisper) and may have errors. Rewrite it into clean, natural "
    "text in the SAME language. Rules:\n"
    "1) Fix wrong homophones, punctuation, run-on words, mis-segmented Hebrew/English, doubled "
    "words (e.g. 'the the' -> 'the', 'ל משה' -> 'למשה', 'ה חוזה' -> 'החוזה').\n"
    "2) LETTER-BY-LETTER SPELLING: if the speaker spelled a word out letter by letter — "
    "separated letters like 'h a y k u', 'o p u s', or spaced Hebrew letters 'ק ל ו ד' — "
    "ASSEMBLE the letters into the intended word ('Haiku', 'Opus', 'קלוד').\n"
    "3) Fix obvious mis-hears of names & technical terms using context, ESPECIALLY AI model "
    "names: 'oppose'/'a pose' -> 'Opus'; 'sonic'/'sonnet' -> 'Sonnet'; 'hi cool'/'haiku brains' "
    "-> 'Haiku'; 'clode'/'cloud' (when addressing the assistant) -> 'Claude'; 'grok'.\n"
    "4) Use the CONTEXT (already-typed text before it) to disambiguate words both BACKWARDS and "
    "FORWARDS. Preserve the speaker's meaning and wording — fix, don't rephrase or add.\n"
    "Output ONLY the corrected text: no quotes, no preamble, no explanation, no markdown. "
    "If it is already clean, return it unchanged."
)


def correct(raw: str, context: str = "", timeout: float = 12.0, model: str = HAIKU_MODEL) -> str:
    """Return a cleaned version of `raw` (or `raw` itself on any failure). Never raises."""
    raw = (raw or "").strip()
    if len(raw) < 2:
        return raw
    ctx = (context or "")[-400:]
    payload = (
        _SYS
        + "\n\nCONTEXT (already typed, for disambiguation only — do NOT repeat it):\n<<<\n"
        + ctx + "\n>>>\n\nRAW (correct THIS):\n<<<\n" + raw + "\n>>>\n\nCORRECTED:"
    ).encode("utf-8")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        r = subprocess.run(
            [CLAUDE, "--model", model, "-p"],
            input=payload, capture_output=True, timeout=timeout, env=env,
            creationflags=_NO_WINDOW,
        )
        out = (r.stdout or b"").decode("utf-8", "replace").strip()
        # sanity: a good correction is roughly the same length; if Haiku rambled or
        # returned empty/echoed the prompt, fall back to raw.
        if not out:
            return raw
        if len(out) > max(80, len(raw) * 4):
            return raw
        # strip accidental wrapping quotes/markers
        out = out.strip().strip('"').strip("'").strip()
        for marker in ("CORRECTED:", "<<<", ">>>"):
            out = out.replace(marker, "").strip()
        return out or raw
    except Exception:
        return raw


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import time
    tests = [
        ("שלום קלוד תכתוב לי תוכנית עבודה למחר", ""),
        ("אני רוצה לקנות דירה ב תל אביב כמה זה אולה", ""),     # 'אולה' homophone of 'עולה'
        ("the the meeting is at five oclock pm", ""),
        ("תשלח הודעה ל משה פרץ על ה חוזה", "אנחנו עובדים על עסקת מקרקעין"),
    ]
    for raw, ctx in tests:
        t0 = time.monotonic()
        out = correct(raw, ctx)
        dt = time.monotonic() - t0
        print(f"[{dt:.1f}s]\n  RAW: {raw!r}\n  FIX: {out!r}\n")
