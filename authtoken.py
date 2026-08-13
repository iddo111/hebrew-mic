# authtoken.py — shared bearer token for the two local HTTP servers.
#
# Why this exists: both servers expose endpoints that TYPE INTO THE MACHINE. Without a
# token, anything that can reach the port owns the keyboard:
#   * clod_mic_server binds 0.0.0.0 by design (the phone needs to reach it), so that was
#     every device on the LAN.
#   * control.py binds 127.0.0.1, which sounds safe but isn't — any web page open in a
#     browser on this machine can POST to http://127.0.0.1:7788/ (a form/plain-text POST
#     needs no CORS preflight), so a random tab could flip the config or fire the CLI.
#
# The token is read from CLOD_MIC_TOKEN, else from .clod_token next to this file, else
# generated and written there with 0600 permissions on first run.
# לזכרה של עלמה ז״ל
from __future__ import annotations

import hmac
import logging
import os
import secrets
import stat

log = logging.getLogger("dictation.auth")

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(HERE, ".clod_token")
ENV_VAR = "CLOD_MIC_TOKEN"

_cached: str | None = None


def get_token() -> str:
    """Return the shared token, creating one on first use."""
    global _cached
    if _cached:
        return _cached

    env = (os.environ.get(ENV_VAR) or "").strip()
    if env:
        _cached = env
        return _cached

    try:
        with open(TOKEN_FILE, encoding="utf-8") as fh:
            existing = fh.read().strip()
        if existing:
            _cached = existing
            return _cached
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning(f"could not read {TOKEN_FILE}: {e}")

    token = secrets.token_urlsafe(24)
    try:
        with open(TOKEN_FILE, "w", encoding="utf-8") as fh:
            fh.write(token)
        os.chmod(TOKEN_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0600 — owner only
        log.info(f"generated a new access token in {TOKEN_FILE}")
    except Exception as e:  # noqa: BLE001
        log.warning(f"could not persist token ({e}); using an in-memory one")
    _cached = token
    return _cached


def extract(headers, args) -> str:
    """Pull a presented token out of a request: Authorization: Bearer …,
    X-Clod-Token: …, or ?token= (the last one is for the phone's first load)."""
    auth = headers.get("Authorization", "") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (headers.get("X-Clod-Token") or args.get("token") or "").strip()


def check(presented: str) -> bool:
    """Constant-time comparison against the shared token."""
    return bool(presented) and hmac.compare_digest(presented, get_token())
