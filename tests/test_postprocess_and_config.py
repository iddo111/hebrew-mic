"""Shared polish stage, the teachable corrections map, config defaults, and the
window-title gates that decide where text and Enter may land."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import corrections  # noqa: E402
import keytyper  # noqa: E402
import postprocess  # noqa: E402


# ── corrections map ───────────────────────────────────────────────────────────
def test_corrections_apply(monkeypatch, tmp_path):
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps({"_note": "ignored", "העיכוב": "Haiku"},
                               ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(corrections, "PATH", str(path))
    assert corrections.fix("תפעיל את העיכוב") == "תפעיל את Haiku"


def test_corrections_leave_clean_text_alone(monkeypatch, tmp_path):
    path = tmp_path / "corrections.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(corrections, "PATH", str(path))
    assert corrections.fix("שלום רגיל בלי תיקון") == "שלום רגיל בלי תיקון"


def test_corrections_survive_a_missing_file(monkeypatch):
    monkeypatch.setattr(corrections, "PATH", "/nonexistent/corrections.json")
    assert corrections.fix("טקסט") == "טקסט"


def test_longer_patterns_win(monkeypatch, tmp_path):
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps({"הי קו": "X", "הי קו בראנץ": "Haiku Brainz"},
                               ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(corrections, "PATH", str(path))
    assert corrections.fix("הי קו בראנץ") == "Haiku Brainz"


def test_teach_writes_an_entry(monkeypatch, tmp_path):
    path = tmp_path / "corrections.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(corrections, "PATH", str(path))
    assert corrections.teach("פייבל", "Fable")
    assert corrections.fix("פייבל") == "Fable"


# ── postprocess ───────────────────────────────────────────────────────────────
def test_fix_text_never_calls_haiku_when_disabled(monkeypatch):
    import haiku_correct

    def explode(*a, **kw):
        raise AssertionError("haiku must not run when haiku=False")

    monkeypatch.setattr(haiku_correct, "correct", explode)
    assert postprocess.fix_text("שלום", haiku=False) == "שלום"


def test_fix_text_fails_open_when_haiku_breaks(monkeypatch):
    import haiku_correct
    monkeypatch.setattr(haiku_correct, "correct",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no CLI")))
    assert postprocess.fix_text("טקסט גולמי", haiku=True) == "טקסט גולמי"


def test_fix_text_uses_haiku_result(monkeypatch):
    import haiku_correct
    monkeypatch.setattr(haiku_correct, "correct", lambda raw, ctx="", **kw: raw + " מתוקן")
    assert postprocess.fix_text("טקסט", haiku=True).startswith("טקסט מתוקן")


def test_fix_text_handles_empty():
    assert postprocess.fix_text("") == ""
    assert postprocess.fix_text(None) == ""


def test_clean_audio_disabled_is_identity():
    sentinel = object()
    assert postprocess.clean_audio(sentinel, 16000, enabled=False) is sentinel


def test_push_context_keeps_a_bounded_window():
    ctx = postprocess.push_context("א" * 500, "סוף")
    assert len(ctx) <= 400
    assert ctx.endswith("סוף")


# ── window-title gates ────────────────────────────────────────────────────────
def test_title_matches():
    assert keytyper.title_matches("Claude — Dictation", ["claude"])
    assert keytyper.title_matches("CLAUDE CODE", ["claude"])
    assert not keytyper.title_matches("Windows PowerShell", ["claude"])
    assert not keytyper.title_matches("anything", [])
    assert not keytyper.title_matches("", ["claude"])


def test_submit_refuses_off_windows():
    """On Linux/macOS the submit path must refuse, not raise."""
    if keytyper.IS_WINDOWS:
        return
    assert keytyper.submit_to_claude(0) == keytyper.UNSUPPORTED


def test_type_text_is_a_noop_off_windows():
    if keytyper.IS_WINDOWS:
        return
    keytyper.type_text("שלום")  # must not raise


# ── config ────────────────────────────────────────────────────────────────────
def test_safe_defaults():
    cfg = config.DEFAULTS
    assert cfg["mode"] == "ptt", "push-to-talk must stay the default"
    assert cfg["auto_enter"] is False, "auto-submit must be opt-in"
    assert cfg["submit_title_patterns"] == ["claude"]


def test_every_documented_key_is_present():
    """Keys the README and dictation.json.example promise."""
    for key in ("default_route", "submit_title_patterns", "auto_enter_blocklist",
                "type_blocklist", "audio_clean", "haiku_brainz", "enter_delay"):
        assert key in config.DEFAULTS, key


def test_example_config_only_uses_known_keys():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "dictation.json.example"), encoding="utf-8") as fh:
        example = json.load(fh)
    unknown = [k for k in example
               if k not in config.DEFAULTS and not k.startswith("_")
               and k not in ("listening",)]
    assert not unknown, f"example config has keys the loader ignores: {unknown}"


def test_type_text_guarded_refuses_a_blocklisted_window(monkeypatch):
    monkeypatch.setattr(keytyper, "foreground_title", lambda: "Bank Login — Chrome")
    typed = []
    monkeypatch.setattr(keytyper, "type_text", lambda t, d=0: typed.append(t))
    assert keytyper.type_text_guarded("סיסמה", blocklist=["bank"]) == "skipped-blocklisted"
    assert not typed, "nothing may reach a blocklisted window"


def test_type_text_guarded_types_when_allowed(monkeypatch):
    monkeypatch.setattr(keytyper, "foreground_title", lambda: "Claude — Dictation")
    typed = []
    monkeypatch.setattr(keytyper, "type_text", lambda t, d=0: typed.append(t))
    assert keytyper.type_text_guarded("שלום", blocklist=["bank"]) == "typed"
    assert typed == ["שלום"]


def test_type_text_guarded_with_no_blocklist_never_queries_the_window(monkeypatch):
    def explode():
        raise AssertionError("must not look at the foreground window for an empty blocklist")

    monkeypatch.setattr(keytyper, "foreground_title", explode)
    monkeypatch.setattr(keytyper, "type_text", lambda t, d=0: None)
    assert keytyper.type_text_guarded("שלום") == "typed"


def test_phone_submit_default_is_documented():
    """clod_mic_server reads phone_submit; the example config must mention the gates
    the phone path now honours."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "clod_mic_server.py"), encoding="utf-8") as fh:
        src = fh.read()
    assert "type_text_guarded" in src, "the phone path must cross the typing gate"
    assert "submit_title_patterns" in src, "the phone path must honour the Enter gate"
