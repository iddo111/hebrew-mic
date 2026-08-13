"""The push-to-talk regression.

An utterance is only released after `silence_ms` of trailing silence — long after the
PTT key is let go. The old code read the "am I armed?" flag at that moment, so in the
default ptt mode every utterance was discarded and nothing was ever typed. The gate
must be sampled at SPEECH ONSET and travel with the audio.

sounddevice is stubbed out so this runs on CI with no audio hardware.
"""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def capture_mod(monkeypatch):
    """Import capture.py with a fake sounddevice (no PortAudio needed)."""
    fake_sd = types.ModuleType("sounddevice")
    fake_sd.query_devices = lambda *a, **kw: []
    fake_sd.default = types.SimpleNamespace(device=(None, None))
    fake_sd.InputStream = object
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    sys.modules.pop("capture", None)
    import capture
    yield capture
    sys.modules.pop("capture", None)


CFG = {
    "sample_rate": 16000,
    "silence_ms": 300,
    "energy_threshold": 0.01,
    "software_gain": 1.0,
    "dsp_enabled": False,
    "mic_device": None,
}


def _loud(n=512):
    return np.full(n, 0.5, dtype=np.float32)


def _quiet(n=512):
    return np.zeros(n, dtype=np.float32)


def _make(capture_mod, gate):
    return capture_mod.MicCapture(dict(CFG), on_utterance=lambda *a: None, gate=gate)


def test_gate_is_sampled_at_speech_onset_not_at_flush(capture_mod):
    """The key is held while speaking and released before the silence runs out —
    the utterance must still be delivered as armed."""
    held = {"down": True}
    cap = _make(capture_mod, gate=lambda: held["down"])

    cap._process_block(_loud())          # speech starts while the key is held
    held["down"] = False                 # ...user releases the key here
    for _ in range(20):                  # trailing silence flushes the utterance
        cap._process_block(_quiet())

    audio, sr, armed = cap._work.get_nowait()
    assert armed is True, "utterance recorded while armed must not be dropped later"
    assert sr == 16000
    assert len(audio) > 0


def test_speech_starting_while_disarmed_is_marked_unarmed(capture_mod):
    """Ambient talk with no key held stays unarmed even if the key is hit mid-sentence."""
    held = {"down": False}
    cap = _make(capture_mod, gate=lambda: held["down"])

    cap._process_block(_loud())
    held["down"] = True                  # too late — this utterance was never armed
    for _ in range(20):
        cap._process_block(_quiet())

    _audio, _sr, armed = cap._work.get_nowait()
    assert armed is False


def test_flush_emits_immediately_without_waiting_out_the_silence(capture_mod):
    cap = _make(capture_mod, gate=lambda: True)
    cap._process_block(_loud())
    assert cap._work.empty(), "nothing should be emitted while speech is still open"

    cap.flush()                          # what the PTT key-up handler calls
    audio, _sr, armed = cap._work.get_nowait()
    assert armed is True
    assert len(audio) > 0


def test_flush_with_no_speech_is_harmless(capture_mod):
    cap = _make(capture_mod, gate=lambda: True)
    cap.flush()
    assert cap._work.empty()


def test_silence_alone_never_emits(capture_mod):
    cap = _make(capture_mod, gate=lambda: True)
    for _ in range(50):
        cap._process_block(_quiet())
    assert cap._work.empty()


def test_utterances_stay_in_capture_order(capture_mod):
    """One worker, one queue — two quick sentences cannot overtake each other."""
    cap = _make(capture_mod, gate=lambda: True)
    for marker in (0.2, 0.4, 0.6):
        cap._process_block(np.full(512, marker, dtype=np.float32))
        cap.flush()

    firsts = [cap._work.get_nowait()[0][0] for _ in range(3)]
    assert firsts == pytest.approx([0.2, 0.4, 0.6])


def test_a_raising_gate_fails_closed(capture_mod):
    def boom():
        raise RuntimeError("state unavailable")

    cap = _make(capture_mod, gate=boom)
    cap._process_block(_loud())
    cap.flush()
    _audio, _sr, armed = cap._work.get_nowait()
    assert armed is False, "if we can't tell whether we're armed, we are not"
