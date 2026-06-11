# dictation/main.py — entry point: wires capture → STT → typer + control server
# לזכרה של עלמה ז״ל
import sys, os, logging, time, signal, threading
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg_mod
from capture import MicCapture
from stt import WhisperSTT
import typer as typer_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dictation")

# ── smart window routing (Iddo 2026-06-11) ─────────────────────────────────────
# Say "קלוד ..." -> type into the Claude window; "גרוק ..." -> the Grok window;
# otherwise type into whatever window is focused. Reuses the proven find/focus
# helpers from router.py (ctypes user32, AttachThreadInput foreground dance).
# router.py ships in this same directory, so the same-dir sys.path insert above covers it.
try:
    import router as _winrouter
except Exception as _e:  # noqa: BLE001
    _winrouter = None
    log.warning(f"window router unavailable ({_e}); routing disabled, typing into focused window")

_WIN_TARGETS = {
    "claude": {"titles": ["claude", "claude code", "windows terminal", "windows powershell",
                           "powershell", "pwsh", "cmd"],
               "procs": ["claude.exe", "windowsterminal.exe", "powershell.exe", "pwsh.exe", "cmd.exe"]},
    "grok":   {"titles": ["grok"], "procs": ["grok.exe"]},
}
_ROUTE_ALIASES = {
    "claude": ["קלוד", "claude", "clod", "כלוד", "קולד", "קלאוד", "קלוו"],
    "grok":   ["גרוק", "grok", "גראוק", "גורוק", "גרוֹק"],
}


def _route(text: str):
    """If the utterance starts with a route word return (target, payload-after-word);
    else (None, full text). The route word itself is consumed, never typed."""
    s = (text or "").strip()
    words = s.split()
    if not words:
        return None, s
    first = words[0].lower().strip(".,!:—־-'\"")
    for tgt, aliases in _ROUTE_ALIASES.items():
        if first in aliases:
            return tgt, s[len(words[0]):].lstrip(" ,!:—־-'\"").strip()
    return None, s


# ── shared state ──────────────────────────────────────────────────────────────
_state: dict = {
    "listening": False,   # SAFETY (Iddo 2026-06-10): start inert; PTT-hold / explicit /listen turns it on
    "model_loaded": False,
    "utterances_total": 0,
    "last_text": "",
    "last_lang": "",
    "mode": "ptt",           # "ptt" | "always_on"
    "ptt_held": False,       # True while PTT hotkey is held
    "level_rms": 0.0,        # live mic level (updated by capture thread)
    "quit": False,
}

_capture: MicCapture = None
_stt: WhisperSTT = None


def utterance_handler(audio: np.ndarray, sr: int):
    """Called by capture thread when an utterance is ready."""
    # In PTT mode, only process if button is held OR we're in always_on
    mode = _state.get("mode", "ptt")
    # SAFETY (Iddo 2026-06-10): type ONLY on explicit activation — PTT held in ptt mode,
    # or listening explicitly turned on in always_on mode. Ambient audio is dropped, so a
    # revived/idle process can never auto-type into the focused window.
    if mode == "ptt":
        if not _state.get("ptt_held", False):
            return
    elif not _state.get("listening", False):
        return

    t0 = time.monotonic()
    try:
        text, lang = _stt.transcribe(audio, sr)
    except Exception as e:
        log.error(f"STT error: {e}")
        return

    latency_ms = (time.monotonic() - t0) * 1000
    if not text:
        return

    log.info(f"[{lang}] {text!r}  ({latency_ms:.0f}ms)")
    _state["utterances_total"] += 1
    _state["last_text"] = text
    _state["last_lang"] = lang

    cfg = cfg_mod.load()

    # ── Smart routing + submit (Iddo 2026-06-11) ──
    # "קלוד ..." → type into the Claude window; "גרוק ..." → the Grok window;
    # else → the currently-focused window. Then press Enter at the END of the
    # utterance to SUBMIT. This Enter is per-utterance (fires once, right after the
    # text), NOT a wall-clock silence timer — so there is no runaway auto-Enter.
    stripped = text.strip()
    if not stripped:
        return
    target, payload = _route(stripped)
    if target and not payload:
        return  # just the route word ("קלוד"/"גרוק") with nothing after — ignore
    if target and _winrouter is not None:
        try:
            ok, info = _winrouter.focus_target(target, _WIN_TARGETS)
            if ok:
                _state["last_target"] = target
                log.info(f"routed → {target} window ({info})")
                time.sleep(0.12)  # let the focus switch settle before typing
            else:
                log.warning(f"route {target}: {info} — typing into the focused window instead")
        except Exception as e:
            log.warning(f"route {target} error: {e}")
    typer_mod.type_text(payload + " ")
    # ── SMART auto-Enter (Iddo 2026-06-11) ──
    # Press Enter at the end of the utterance ONLY when (a) auto_enter is on, (b) there is
    # real text, (c) the mic is listening (not paused), AND (d) the FOREGROUND window is
    # Claude. submit_to_claude() is the screen-recording-verified LKG path: it re-checks the
    # foreground is a Claude window (title contains "claude") and refuses otherwise — so a
    # stray Enter never lands in a non-Claude window. Typing still happens everywhere; only
    # the auto-submit is Claude-gated, exactly the smart toggle requested.
    if cfg.get("auto_enter", True) and payload.split() and _state.get("listening", False):
        time.sleep(float(cfg.get("enter_delay", 0.07)))  # tunable via dictation.json; let text commit then submit
        m = typer_mod.submit_to_claude(0)
        log.info(f"auto-enter -> {m}")


def _level_update_loop():
    """Periodically copy capture level_rms to shared state for widget polling."""
    while not _state.get("quit", False):
        if _capture is not None:
            _state["level_rms"] = _capture.level_rms
        time.sleep(0.05)


def main():
    global _stt, _capture

    cfg = cfg_mod.load()
    log.info("=== Dictation v2 starting ===")
    log.info(f"Language: {cfg['language']} | Mode: {cfg['mode']} | Gain: {cfg.get('software_gain', 14.0)}")

    # load STT model
    _stt = WhisperSTT(cfg)
    summary = _stt.load()
    log.info(f"STT ready: {summary}")
    _state["model_loaded"] = True
    _state["mode"] = cfg.get("mode", "ptt")
    # always_on = listen immediately (Iddo wants the mic live); ptt stays inert until key held
    _state["listening"] = (_state["mode"] == "always_on")

    # start mic capture
    _capture = MicCapture(cfg, utterance_handler)
    _capture.start()
    log.info(f"Mic: [{_capture.device_index}] {_capture.device_name} | SR={cfg['sample_rate']} | Gain={cfg.get('software_gain',14.0)}x")

    # config-change callback (called by ControlServer when /config POST arrives)
    def on_config_change(new_cfg: dict):
        _state["mode"] = new_cfg.get("mode", "ptt")
        _state["listening"] = new_cfg.get("listening", _state["listening"])
        _capture.reload_config(new_cfg)
        _stt.reload_config(new_cfg)
        log.info(f"Config reloaded: lang={new_cfg.get('language')} mode={new_cfg.get('mode')} gain={new_cfg.get('software_gain')}")

    # start control server
    from control import ControlServer
    ctrl = ControlServer(_state, cfg, on_config_change=on_config_change)
    ctrl.set_capture(_capture)  # so /devices endpoint works
    ctrl.start()

    # level update thread
    lv_thread = threading.Thread(target=_level_update_loop, daemon=True, name="dictation-level")
    lv_thread.start()

    log.info(f"Ready. Hotkey: {cfg['hotkey']}  |  HTTP: http://127.0.0.1:{cfg['http_port']}")
    log.info("Speak into the mic — text will be typed into the focused window.")

    def _sighandler(sig, frame):
        _state["quit"] = True

    signal.signal(signal.SIGINT, _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    while not _state.get("quit", False):
        time.sleep(0.2)

    log.info("Shutting down...")
    _capture.stop()
    ctrl.stop()


if __name__ == "__main__":
    main()
