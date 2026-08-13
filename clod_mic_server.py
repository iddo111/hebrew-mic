# clod_mic_server.py — Clod Mic PWA server + /dispatch + /transcribe + /type endpoints
#
# SECURITY: every endpoint here can type into the PC and press Enter, so all of them
# require the shared token (see authtoken.py). The bind address defaults to loopback —
# set CLOD_MIC_HOST to your Tailscale address (preferred) or 0.0.0.0 to let the phone
# reach it. There is no wildcard CORS: the page is served from this same origin, and a
# cross-origin caller has no business driving the keyboard.
#
#   CLOD_MIC_HOST=100.x.y.z python clod_mic_server.py
#
# לזכרה של עלמה ז״ל

import functools, os, subprocess, logging, threading, tempfile, numpy as np
from flask import Flask, request, jsonify, Response

import authtoken

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clod-mic")

app = Flask("clod-mic")
app.logger.setLevel(logging.ERROR)

_HERE = os.path.dirname(os.path.abspath(__file__))
_HTML = os.path.join(_HERE, "android_mic.html")

_typer_mod = None


def _require_token(fn):
    """Every route that can reach the keyboard goes through this."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        if not authtoken.check(authtoken.extract(request.headers, request.args)):
            log.warning("rejected %s %s from %s — bad or missing token",
                        request.method, request.path, request.remote_addr)
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return fn(*a, **kw)
    return wrapper


def _load_typer():
    """Load the local keytyper module (works as package OR standalone)."""
    global _typer_mod
    if _typer_mod is not None:
        return _typer_mod
    try:
        from . import keytyper as t  # package import
        _typer_mod = t
    except Exception:
        import importlib.util, pathlib, sys
        p = pathlib.Path(__file__).parent / "keytyper.py"
        spec = importlib.util.spec_from_file_location("keytyper", p)
        t = importlib.util.module_from_spec(spec)
        # Register under its own name only — never claim the `typer` module slot,
        # which belongs to the unrelated CLI package on PyPI.
        sys.modules["keytyper"] = t
        spec.loader.exec_module(t)
        _typer_mod = t
    return _typer_mod


def _type_and_submit(text: str) -> str:
    """Type `text` into the PC's focused window, then SMART-Enter: submit_to_claude only
    presses Enter when the foreground window is Claude (else types-only). This makes A70
    voice land in Claude exactly like the local PC mic (the verified LKG behavior)."""
    import time
    t = _load_typer()
    t.type_text(text + " ")
    time.sleep(0.07)
    return t.submit_to_claude(0)


@app.route("/", methods=["GET"])
@app.route("/android_mic.html", methods=["GET"])
def serve_pwa():
    """Serve the phone page. When the URL carries a valid ?token=…, the token is
    baked into the page so the phone doesn't have to be typed at again; otherwise
    the page loads and asks for it."""
    with open(_HTML, encoding="utf-8") as f:
        html = f.read()
    presented = authtoken.extract(request.headers, request.args)
    baked = authtoken.get_token() if authtoken.check(presented) else ""
    html = html.replace("__CLOD_TOKEN__", baked)
    return Response(html, mimetype="text/html")


@app.route("/dispatch", methods=["POST"])
@_require_token
def dispatch():
    """Receive text from Clod Mic PWA on phone → pipe to claude CLI."""
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "empty text"}), 400
    log.info(f"[dispatch] '{text}'")
    try:
        m = _type_and_submit(text)   # type into focused window + smart-Enter (Claude-gated)
    except Exception as e:
        log.error(f"dispatch type+submit failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "text": text, "submit": m})


@app.route("/transcribe", methods=["POST"])
@_require_token
def transcribe():
    """Accept multipart audio file from A70 widget → run Whisper → return {text}."""
    if "audio" not in request.files:
        return jsonify({"ok": False, "error": "no audio field"}), 400
    audio_file = request.files["audio"]
    suffix = ".wav"
    # Accept any audio format (m4a, wav, etc.) — soundfile/ffmpeg handles conversion
    orig_name = audio_file.filename or ""
    if "." in orig_name:
        suffix = "." + orig_name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)
    try:
        text = _whisper_transcribe(tmp_path)
        log.info(f"[transcribe] '{text}' ({suffix})")
        return jsonify({"ok": True, "text": text})
    except Exception as e:
        log.error(f"[transcribe] error: {e}")
        return jsonify({"ok": False, "error": str(e), "text": ""}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _whisper_transcribe(audio_path: str) -> str:
    """Load audio with soundfile/librosa → run WhisperSTT → return text string."""
    global _whisper_model
    if _whisper_model is None:
        import importlib.util, pathlib, sys
        _dir = pathlib.Path(__file__).parent
        # Load stt and config modules directly (works both as package and standalone)
        def _load_local(name):
            p = _dir / f"{name}.py"
            spec = importlib.util.spec_from_file_location(name, p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
        try:
            from .stt import WhisperSTT
            from .config import load as load_cfg
        except ImportError:
            stt_mod = _load_local("stt")
            cfg_mod = _load_local("config")
            WhisperSTT = stt_mod.WhisperSTT
            load_cfg = cfg_mod.load
        cfg = load_cfg()
        _whisper_model = WhisperSTT(cfg)
        _whisper_model.load()

    # Read audio — soundfile → librosa → ffmpeg raw PCM
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
    except Exception:
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception:
            raw = subprocess.check_output([
                "ffmpeg", "-loglevel", "quiet", "-i", audio_path,
                "-f", "s16le", "-ar", "16000", "-ac", "1", "-"
            ])
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000
    text, _lang = _whisper_model.transcribe(audio, sr)
    return text


_whisper_model = None


@app.route("/type", methods=["POST"])
@_require_token
def type_text_endpoint():
    """Accept {text} → inject as Unicode keystrokes into the focused Windows window."""
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "empty text"}), 400
    log.info(f"[type] '{text}'")
    try:
        _load_typer().type_text(text)
    except Exception as e:
        log.error(f"[type] error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "text": text})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "clod-mic"})


@app.after_request
def security_headers(resp):
    """No wildcard CORS. The phone page is same-origin with this server, so it needs
    no CORS at all; set CLOD_MIC_ORIGIN only if you deliberately serve the page from
    somewhere else (e.g. a Tailscale Serve HTTPS front-end)."""
    origin = os.environ.get("CLOD_MIC_ORIGIN", "").strip()
    if origin and request.headers.get("Origin") == origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Clod-Token"
        resp.headers["Vary"] = "Origin"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


if __name__ == "__main__":
    # Loopback by default: opening the keyboard to the whole LAN has to be a
    # deliberate act, not what happens when you double-click the launcher.
    host = os.environ.get("CLOD_MIC_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("CLOD_MIC_PORT", "7789"))
    token = authtoken.get_token()
    log.info(f"=== Clod Mic server starting on {host}:{port} ===")
    if host in ("0.0.0.0", "::"):
        log.warning("bound to ALL interfaces — every device on this network can reach it. "
                    "Prefer CLOD_MIC_HOST=<your-tailscale-ip>.")
    log.info(f"Open on the phone:  http://{host}:{port}/?token={token}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
