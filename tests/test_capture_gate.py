"""The push-to-talk regression.

An utterance is only released after `silence_ms` of trailing silence — long after the
PTT key is let go. The old code read the "am I armed?" flag at that moment, so in the
default ptt mode every utterance was discarded and nothing was ever typed. The gate
must be sampled at SPEECH ONSET and travel with the audio.

sounddevice is stubbed out so this runs on CI with no audio hardware.
"""
import os
import sys
import time
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

    audio, sr, armed, _speech = cap._work.get_nowait()
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

    _audio, _sr, armed, _speech = cap._work.get_nowait()
    assert armed is False


def test_flush_emits_immediately_without_waiting_out_the_silence(capture_mod):
    cap = _make(capture_mod, gate=lambda: True)
    cap._process_block(_loud())
    assert cap._work.empty(), "nothing should be emitted while speech is still open"

    cap.flush()                          # what the PTT key-up handler calls
    audio, _sr, armed, _speech = cap._work.get_nowait()
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
    _audio, _sr, armed, _speech = cap._work.get_nowait()
    assert armed is False, "if we can't tell whether we're armed, we are not"


def test_restart_does_not_inherit_a_poisoned_queue(capture_mod):
    """stop() then start() — as switch_device does — must leave a live worker.

    The regression: stop() pushed a None sentinel that a worker busy inside Whisper
    never consumed, so the NEXT worker read it and exited immediately, wedging
    dictation permanently.
    """
    handled = []
    cap = capture_mod.MicCapture(dict(CFG), on_utterance=lambda *a: handled.append(a),
                                 gate=lambda: True)
    cap.start()
    cap.switch_device(None)
    try:
        assert cap._worker is not None and cap._worker.is_alive()

        cap._process_block(_loud())
        cap.flush()
        for _ in range(100):                      # give the worker up to ~1s
            if handled:
                break
            time.sleep(0.01)
        assert handled, "the worker after a restart must still drain the queue"
    finally:
        cap.stop()


def test_speech_length_excludes_trailing_silence(capture_mod):
    """stt.py decides whether a one-word transcript is real from the SPEECH portion.
    The buffer always ends with silence_ms of silence, so it must not be counted."""
    cap = _make(capture_mod, gate=lambda: True)
    cap._process_block(_loud(1600))               # 0.1s of speech
    for _ in range(10):
        cap._process_block(_quiet(1600))          # 0.6s of silence, flushes at 0.3s

    audio, _sr, _armed, speech = cap._work.get_nowait()
    assert speech == 1600, "only the speech blocks count"
    assert len(audio) > speech, "the buffer still carries its trailing silence"


def test_gate_rising_mid_segment_keeps_the_armed_phrase(capture_mod):
    """A cough, then the key goes down and the user speaks — the phrase survives and
    the un-armed prefix is dropped."""
    held = {"down": False}
    cap = _make(capture_mod, gate=lambda: held["down"])

    cap._process_block(np.full(512, 0.9, dtype=np.float32))   # ambient noise, unarmed
    held["down"] = True
    cap._process_block(np.full(512, 0.3, dtype=np.float32))   # the actual phrase
    cap.flush()

    audio, _sr, armed, _speech = cap._work.get_nowait()
    assert armed is True
    assert len(audio) == 512, "the pre-arm prefix must be discarded"
    assert audio[0] == pytest.approx(0.3)
