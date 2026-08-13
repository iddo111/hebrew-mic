# postprocess.py — the one text/audio polishing stage, shared by BOTH front-ends.
#
# Before this module existed, audio_clean / haiku_correct / corrections were wired
# into clod_widget.py only, so anyone running `python main.py` — the entry point the
# README documents — silently got none of them. Both pipelines now call in here.
# לזכרה של עלמה ז״ל
from __future__ import annotations

import logging

log = logging.getLogger("dictation.postprocess")


def clean_audio(audio, sr: int, enabled: bool = True):
    """AGC + spectral denoise + soft limit. Returns the input unchanged on any failure."""
    if not enabled:
        return audio
    try:
        import audio_clean
        return audio_clean.clean(audio, sr)
    except Exception as e:  # noqa: BLE001
        log.warning(f"audio_clean skipped: {e}")
        return audio


def fix_text(text: str, context: str = "", haiku: bool = False,
             timeout: float = 12.0) -> str:
    """Polish a raw transcript: optional HaikuBrainz pass, then the teachable
    {wrong: right} map, which always gets the last word.

    Fails open at every step — a missing Claude CLI or an unreadable corrections
    file must never cost the speaker their sentence."""
    out = (text or "").strip()
    if not out:
        return out

    if haiku:
        try:
            import haiku_correct
            fixed = haiku_correct.correct(out, context, timeout=timeout)
            if fixed:
                out = fixed
        except Exception as e:  # noqa: BLE001
            log.warning(f"haiku_correct skipped: {e}")

    try:
        import corrections
        out = corrections.fix(out)
    except Exception as e:  # noqa: BLE001
        log.warning(f"corrections skipped: {e}")

    return out


def push_context(context: str, text: str, limit: int = 400) -> str:
    """Rolling context window handed to HaikuBrainz for disambiguation."""
    return ((context or "") + " " + (text or "")).strip()[-limit:]
