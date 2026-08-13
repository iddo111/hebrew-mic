# dictation/control.py — HTTP control server + global hotkey toggle (OURS)
# Flask HTTP API on port 7788 (default). Global hotkey via `keyboard` lib.
# Tray icon via pystray.
# Exposes /devices for mic-selection dropdown in widget.

import threading, logging, time, os, functools
import flask
from flask import request, jsonify
import keyboard as kb

import authtoken

log = logging.getLogger("dictation.control")


def _require_token(fn):
    """Guard for state-changing routes. Binding to 127.0.0.1 is NOT authentication:
    any page in any browser tab on this machine can POST to a localhost port."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        if not authtoken.check(authtoken.extract(request.headers, request.args)):
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return fn(*a, **kw)
    return wrapper


class ControlServer:
    """
    HTTP API contract:
      POST /listen  {on: true|false}
      POST /config  {auto_enter, language, mode, software_gain, mic_device, ...}
      GET  /state
      GET  /health  (includes level_rms, last_text, mode)
      GET  /devices  → list of safe input devices for dropdown
    Also wires a global hotkey toggle + PTT hotkey (hold) + optional tray icon.
    """

    def __init__(self, app_state: dict, cfg: dict, on_config_change=None):
        self._state = app_state
        self._cfg = cfg
        self._on_config_change = on_config_change
        self._port = cfg.get("http_port", 7788)
        self._hotkey = cfg.get("hotkey", "ctrl+shift+d")
        self._ptt_hotkey = cfg.get("ptt_hotkey", "ctrl+shift+space")
        self._flask = flask.Flask("dictation")
        self._flask.logger.setLevel(logging.ERROR)
        self._tray = None
        self._capture_ref = None  # set by main.py after capture starts
        self._setup_routes()

    def set_capture(self, capture):
        """Give the server a reference to MicCapture so /devices works."""
        self._capture_ref = capture

    # ── Flask routes ──────────────────────────────────────────────────────────
    def _setup_routes(self):
        app = self._flask

        @app.route("/listen", methods=["POST"])
        @_require_token
        def listen_toggle():
            data = request.get_json(force=True, silent=True) or {}
            on = bool(data.get("on", True))
            self._state["listening"] = on
            self._update_tray()
            return jsonify({"ok": True, "listening": on})

        @app.route("/config", methods=["POST"])
        @_require_token
        def config_update():
            data = request.get_json(force=True, silent=True) or {}
            self._cfg.update(data)
            from config import save
            save(self._cfg)
            if self._on_config_change:
                self._on_config_change(self._cfg)
            # If mic_device changed, switch capture LIVE (re-open the stream)
            if self._capture_ref and "mic_device" in data and data["mic_device"] is not None:
                try:
                    self._capture_ref.switch_device(
                        self._cfg.get("mic_device"), self._cfg.get("mic_blocklist"))
                except Exception as e:
                    import logging
                    logging.getLogger("dictation.control").error(f"switch_device failed: {e}")
            return jsonify({"ok": True, "config": self._cfg})

        @app.route("/state", methods=["GET"])
        def state():
            return jsonify(dict(self._state, config=self._cfg))

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "ok": True,
                "listening": self._state.get("listening", False),
                "model_loaded": self._state.get("model_loaded", False),
                "utterances_total": self._state.get("utterances_total", 0),
                "last_text": self._state.get("last_text", ""),
                "last_lang": self._state.get("last_lang", ""),
                "level_rms": round(float(self._state.get("level_rms", 0.0)), 5),
                "mode": self._state.get("mode", "ptt"),
                "ptt_held": self._state.get("ptt_held", False),
            })

        @app.route("/devices", methods=["GET"])
        def devices():
            if self._capture_ref:
                try:
                    devs = self._capture_ref.list_safe_devices()
                    return jsonify(devs)
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
            return jsonify([])

        @app.route("/dispatch", methods=["POST"])
        @_require_token
        def dispatch():
            """Receive text from Clod Mic PWA and pipe to claude CLI."""
            import subprocess, os as _os
            data = request.get_json(force=True, silent=True) or {}
            text = (data.get("text") or "").strip()
            if not text:
                return jsonify({"ok": False, "error": "empty text"}), 400
            log.info(f"[dispatch] '{text}'")
            try:
                subprocess.Popen(
                    ["claude", "--print", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
            except Exception as e:
                log.error(f"dispatch claude CLI failed: {e}")
                return jsonify({"ok": False, "error": str(e)}), 500
            return jsonify({"ok": True, "text": text})

    # ── hotkeys ───────────────────────────────────────────────────────────────
    def _toggle_hotkey(self):
        self._state["listening"] = not self._state.get("listening", True)
        status = "ON" if self._state["listening"] else "OFF"
        log.info(f"Hotkey: dictation {status}")
        self._update_tray()

    def _ptt_press(self):
        """PTT key down — enable listening."""
        self._state["ptt_held"] = True
        self._state["listening"] = True
        self._update_tray()

    def _ptt_release(self):
        """PTT key up — close the utterance immediately, then disable listening.

        Flushing first matters: capture would otherwise hold the audio until
        `silence_ms` of trailing silence elapses, adding almost a second of lag to
        every push-to-talk phrase.

        `keyboard.on_release_key` fires on EVERY release of the combo's last key
        (a bare space, with no modifiers, in any mode), so flush only when a real PTT
        hold is being ended — `ptt_held` is set exclusively by `_ptt_press`, which needs
        the whole combo. Without this check a stray spacebar tap would cut whatever
        sentence is in flight in half."""
        was_held = bool(self._state.get("ptt_held", False))
        self._state["ptt_held"] = False
        if not was_held:
            return
        if self._capture_ref is not None:
            try:
                self._capture_ref.flush()
            except Exception as e:
                log.warning(f"PTT flush failed: {e}")
        if self._state.get("mode", "ptt") == "ptt":
            self._state["listening"] = False
            self._update_tray()

    # ── tray icon ─────────────────────────────────────────────────────────────
    def _update_tray(self):
        if self._tray is None:
            return
        try:
            listening = self._state.get("listening", True)
            icon = self._make_icon(listening)
            self._tray.icon = icon
            self._tray.title = f"Dictation: {'ON' if listening else 'OFF'}"
        except Exception as e:
            log.debug(f"tray update: {e}")

    @staticmethod
    def _make_icon(listening: bool):
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        color = (0, 200, 50) if listening else (120, 120, 120)
        draw.ellipse([4, 4, 60, 60], fill=color)
        draw.rectangle([26, 16, 38, 38], fill=(255, 255, 255))
        draw.ellipse([22, 30, 42, 42], fill=(255, 255, 255))
        draw.line([32, 42, 32, 50], fill=(255, 255, 255), width=3)
        draw.line([24, 50, 40, 50], fill=(255, 255, 255), width=3)
        return img

    def _build_tray_menu(self):
        try:
            import pystray
            def on_toggle(icon, item):
                self._toggle_hotkey()
            def on_quit(icon, item):
                self._state["quit"] = True
                icon.stop()
            return pystray.Menu(
                pystray.MenuItem("Toggle dictation", on_toggle),
                pystray.MenuItem("Quit", on_quit),
            )
        except Exception:
            return None

    def _start_tray(self):
        try:
            import pystray
            icon_img = self._make_icon(self._state.get("listening", True))
            menu = self._build_tray_menu()
            self._tray = pystray.Icon(
                "dictation",
                icon_img,
                "Dictation: ON",
                menu=menu,
            )
            self._tray.run()
        except Exception as e:
            log.info(f"Tray not available: {e} (headless mode OK)")

    # ── start everything ──────────────────────────────────────────────────────
    def start(self):
        # Toggle hotkey (press to flip on/off)
        try:
            kb.add_hotkey(self._hotkey, self._toggle_hotkey, suppress=False)
            log.info(f"Toggle hotkey: {self._hotkey}")
        except Exception as e:
            log.warning(f"Toggle hotkey registration failed: {e}")

        # PTT hotkey (hold while speaking in ptt mode)
        try:
            kb.add_hotkey(self._ptt_hotkey, self._ptt_press,
                          suppress=False, trigger_on_release=False)
            # Release detection: monitor the last key in the combo
            last_key = self._ptt_hotkey.split("+")[-1].strip()
            kb.on_release_key(last_key, lambda e: self._ptt_release(), suppress=False)
            log.info(f"PTT hotkey: {self._ptt_hotkey} (release key: {last_key})")
        except Exception as e:
            log.warning(f"PTT hotkey registration failed: {e}")

        # Flask in daemon thread
        t = threading.Thread(
            target=lambda: self._flask.run(
                host="127.0.0.1",
                port=self._port,
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
            name="dictation-http",
        )
        t.start()
        log.info(f"HTTP control on http://127.0.0.1:{self._port}")
        log.info(f"Access token for POST routes: {authtoken.get_token()} "
                 f"(also in {authtoken.TOKEN_FILE}; override with ${authtoken.ENV_VAR})")

        # Tray in daemon thread
        tray_t = threading.Thread(target=self._start_tray, daemon=True, name="dictation-tray")
        tray_t.start()

    def stop(self):
        try:
            kb.remove_hotkey(self._hotkey)
        except Exception:
            pass
        if self._tray:
            try:
                self._tray.stop()
            except Exception:
                pass
