# dictation/main.py — entry point: wires capture → STT → typer + control server
# לזכרה של עלמה ז״ל
import sys, os, logging, time, signal, threading
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg_mod
from capture import MicCapture
from stt import WhisperSTT
import keytyper as typer_mod
import postprocess
import routing

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
# Wake words come from clod_keywords.json (which the README documents as their home),
# with routing.DEFAULT_ALIASES as the fallback if that file is missing or edited badly.
_ROUTE_ALIASES = routing.load_aliases()


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
    "last_submit": "",       # outcome of the last auto-Enter attempt
    "hb_context": "",        # rolling context handed to HaikuBrainz
    "quit": False,
}
_state_lock = threading.Lock()

_capture: MicCapture = None
_stt: WhisperSTT = None


def is_armed() -> bool:
    """SAFETY (Iddo 2026-06-10): dictation is live ONLY on explicit activation — the PTT
    key held down in ptt mode, or listening explicitly turned on in always_on mode. A
    revived or idle process can never auto-type into the focused window.

    capture.MicCapture samples this at SPEECH ONSET and passes the answer back with the
    utterance, because by the time an utterance is flushed (after silence_ms of trailing
    silence) the PTT key has already been released."""
    if _state.get("mode", "ptt") == "ptt":
        return bool(_state.get("ptt_held", False))
    return bool(_state.get("listening", False))


def utterance_handler(audio: np.ndarray, sr: int, armed: bool = True,
                      speech_samples: int | None = None):
    """Called on the capture worker thread when an utterance is ready.
    `armed` is the gate state sampled when the speaker STARTED talking;
    `speech_samples` is the buffer minus its trailing silence."""
    if not armed:
        return

    cfg = cfg_mod.load()

    t0 = time.monotonic()
    try:
        audio = postprocess.clean_audio(audio, sr, enabled=cfg.get("audio_clean", False))
        text, lang = _stt.transcribe(audio, sr, speech_samples=speech_samples)
    except Exception as e:
        log.error(f"STT error: {e}")
        return

    if not text:
        return

    # Shared polish stage: HaikuBrainz (opt-in) + the teachable corrections map.
    text = postprocess.fix_text(
        text, _state.get("hb_context", ""),
        haiku=bool(cfg.get("haiku_brainz", False)),
        timeout=float(cfg.get("haiku_timeout", 12.0)),
    )
    if not text:
        return
    _state["hb_context"] = postprocess.push_context(_state.get("hb_context", ""), text)

    latency_ms = (time.monotonic() - t0) * 1000
    log.info(f"[{lang}] {text!r}  ({latency_ms:.0f}ms)")
    with _state_lock:
        _state["utterances_total"] += 1
        _state["last_text"] = text
        _state["last_lang"] = lang

    # ── Smart routing + submit (Iddo 2026-06-11) ──
    # "קלוד ..." → type into the Claude window; "גרוק ..." → the Grok window;
    # else → the currently-focused window. Then press Enter at the END of the
    # utterance to SUBMIT. This Enter is per-utterance (fires once, right after the
    # text), NOT a wall-clock silence timer — so there is no runaway auto-Enter.
    stripped = text.strip()
    if not stripped:
        return
    if cfg.get("default_route", "smart") == "smart":
        target, payload = routing.split_route(stripped, _ROUTE_ALIASES)
    else:
        target, payload = None, stripped   # 'focused': never steal focus, always type here
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

    # SAFETY: the Enter gate alone is not enough — refuse to TYPE into a window the
    # user has blocklisted (password prompts, banking, a lock screen).
    type_blocklist = cfg.get("type_blocklist", [])
    fg_title = typer_mod.foreground_title()
    if type_blocklist and typer_mod.title_matches(fg_title, type_blocklist):
        log.warning(f"foreground {fg_title!r} is blocklisted — not typing")
        _state["last_submit"] = "skipped-blocklisted"
        return

    typer_mod.type_text(payload + " ")
    # ── SMART auto-Enter (Iddo 2026-06-11) ──
    # Press Enter at the end of the utterance ONLY when (a) auto_enter is on, (b) there is
    # real text, (c) the mic is listening (not paused), AND (d) the FOREGROUND window is
    # Claude. submit_to_claude() is the screen-recording-verified LKG path: it re-checks the
    # foreground is a Claude window (title contains "claude") and refuses otherwise — so a
    # stray Enter never lands in a non-Claude window. Typing still happens everywhere; only
    # the auto-submit is Claude-gated, exactly the smart toggle requested.
    if cfg.get("auto_enter", False) and payload.split():
        time.sleep(float(cfg.get("enter_delay", 0.07)))  # tunable via dictation.json; let text commit then submit
        m = typer_mod.submit_to_claude(
            0,
            allow_titles=cfg.get("submit_title_patterns", list(typer_mod.DEFAULT_SUBMIT_TITLES)),
            blocklist=cfg.get("auto_enter_blocklist", []),
        )
        _state["last_submit"] = m
        if m.startswith("skipped"):
            # Typing succeeded but the message was NOT sent — say so plainly instead of
            # leaving the speaker to wonder why nothing happened.
            log.warning(f"typed into {fg_title!r} but did NOT press Enter ({m}); "
                        f"submit_title_patterns={cfg.get('submit_title_patterns', list(typer_mod.DEFAULT_SUBMIT_TITLES))}")
        else:
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

    # start mic capture — is_armed() is sampled at speech onset, not at flush time
    _capture = MicCapture(cfg, utterance_handler, gate=is_armed)
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
