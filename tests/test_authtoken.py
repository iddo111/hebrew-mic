"""The shared token. Both HTTP servers can type into the machine, so an absent or
wrong token has to be rejected — including the empty-string case."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import authtoken  # noqa: E402


class _Headers(dict):
    """Minimal stand-in for a Flask headers object."""


def _reset(monkeypatch, tmp_path, token=None):
    monkeypatch.setattr(authtoken, "_cached", None)
    monkeypatch.setattr(authtoken, "TOKEN_FILE", str(tmp_path / ".clod_token"))
    if token is None:
        monkeypatch.delenv(authtoken.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(authtoken.ENV_VAR, token)


def test_env_var_wins(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path, token="sekrit")
    assert authtoken.get_token() == "sekrit"


def test_token_is_generated_and_persisted(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    first = authtoken.get_token()
    assert len(first) >= 16
    monkeypatch.setattr(authtoken, "_cached", None)
    assert authtoken.get_token() == first, "token must survive a restart"


def test_token_file_is_owner_only(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    authtoken.get_token()
    mode = os.stat(authtoken.TOKEN_FILE).st_mode & 0o777
    assert mode == 0o600


def test_check_accepts_the_real_token(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path, token="abc123")
    assert authtoken.check("abc123")


def test_check_rejects_everything_else(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path, token="abc123")
    for bad in ("", None, "abc12", "ABC123", "abc1234"):
        assert not authtoken.check(bad), repr(bad)


def test_extract_reads_bearer_header():
    assert authtoken.extract(_Headers({"Authorization": "Bearer xyz"}), {}) == "xyz"
    assert authtoken.extract(_Headers({"Authorization": "bearer xyz"}), {}) == "xyz"


def test_extract_reads_custom_header_and_query():
    assert authtoken.extract(_Headers({"X-Clod-Token": "xyz"}), {}) == "xyz"
    assert authtoken.extract(_Headers(), {"token": "xyz"}) == "xyz"


def test_extract_returns_empty_when_absent():
    assert authtoken.extract(_Headers(), {}) == ""
