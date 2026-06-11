"""audio_clean.py — auto gain + auto volume + noise reduction for dictation.

Runs on each captured utterance (float32 mono, 16 kHz) BEFORE Whisper, so the STT
hears a clean, well-leveled signal — which directly improves Hebrew accuracy:
  1) high-pass        — kill <70 Hz rumble / mains hum / desk thump
  2) spectral denoise — estimate the noise floor from the quietest frames and soft-
                        gate it out (spectral gating; no external libs, scipy only)
  3) auto-gain (AGC)  — normalize loudness to a target RMS so near/far speech is even
  4) soft limiter     — prevent the post-gain peaks from clipping

scipy.signal is already a dictation dep; nothing new to install. Pure functions,
never raise — on any error return the input unchanged so dictation never breaks.
"""
from __future__ import annotations
import numpy as np

try:
    import scipy.signal as ss
except Exception:  # pragma: no cover
    ss = None

TARGET_RMS = 0.12          # desired loudness after AGC (0..1)
MAX_GAIN = 12.0            # don't amplify pure silence into noise
NOISE_OVERSUB = 1.5        # how aggressively to subtract the estimated noise floor


def _highpass(audio: np.ndarray, sr: int, cut: float = 70.0) -> np.ndarray:
    if ss is None:
        return audio
    sos = ss.butter(4, cut / (sr / 2.0), btype="highpass", output="sos")
    return ss.sosfiltfilt(sos, audio).astype(np.float32)


def _spectral_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Spectral gating: estimate per-bin noise from the quietest 15% of frames,
    then soft-subtract it. Classic, robust, library-free."""
    if ss is None or len(audio) < sr // 4:
        return audio
    nper = 512
    f, t, Z = ss.stft(audio, fs=sr, nperseg=nper, noverlap=nper * 3 // 4)
    mag = np.abs(Z)
    phase = np.angle(Z)
    # noise profile = mean magnitude over the lowest-energy frames
    frame_energy = mag.mean(axis=0)
    if frame_energy.size == 0:
        return audio
    thresh = np.quantile(frame_energy, 0.15)
    quiet = mag[:, frame_energy <= thresh]
    if quiet.shape[1] < 2:
        return audio
    noise = quiet.mean(axis=1, keepdims=True) * NOISE_OVERSUB
    # soft mask: keep magnitude above the noise floor, attenuate below
    clean_mag = np.maximum(mag - noise, 0.0)
    mask = clean_mag / (mag + 1e-8)
    mask = np.clip(mask, 0.0, 1.0)
    # smooth the mask a touch over time to avoid musical-noise artifacts
    if mask.shape[1] >= 3:
        mask = ss.medfilt(mask, kernel_size=(1, 3))
    Z2 = mask * mag * np.exp(1j * phase)
    _, rec = ss.istft(Z2, fs=sr, nperseg=nper, noverlap=nper * 3 // 4)
    out = rec[: len(audio)].astype(np.float32)
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out


def _auto_gain(audio: np.ndarray) -> np.ndarray:
    """AGC: scale to TARGET_RMS (auto volume), capped so silence isn't blown up."""
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms < 1e-4:                      # essentially silence — leave it
        return audio
    gain = min(TARGET_RMS / rms, MAX_GAIN)
    return (audio * gain).astype(np.float32)


def _soft_limit(audio: np.ndarray) -> np.ndarray:
    """Tanh soft-clip so AGC peaks don't hard-clip (which Whisper hates)."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.95:
        return audio
    return np.tanh(audio * 0.9).astype(np.float32)


def clean(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Full chain: high-pass -> denoise -> auto-gain -> soft-limit. Never raises."""
    try:
        a = np.ascontiguousarray(audio, dtype=np.float32)
        if a.size < sr // 8:           # <125ms: too short to denoise, just AGC
            return _soft_limit(_auto_gain(a))
        a = _highpass(a, sr)
        a = _spectral_denoise(a, sr)
        a = _auto_gain(a)
        a = _soft_limit(a)
        return a
    except Exception:
        return audio


if __name__ == "__main__":
    import sys, wave
    sys.stdout.reconfigure(encoding="utf-8")
    # Build a noisy version of a real clip, clean it, compare levels + STT.
    # Usage: python audio_clean.py path/to/sample.wav
    sample = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    w = wave.open(sample, "rb")
    sr = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    if sr != 16000 and ss is not None:
        a = ss.resample_poly(a, 16000, sr).astype(np.float32); sr = 16000
    rng = np.random.RandomState(0)
    noisy = (a * 0.4 + rng.randn(len(a)).astype(np.float32) * 0.03)  # quieter + hiss
    cleaned = clean(noisy, sr)

    def rms(x): return float(np.sqrt(np.mean(x ** 2)))
    print(f"noisy   RMS={rms(noisy):.4f} peak={np.max(np.abs(noisy)):.3f}")
    print(f"cleaned RMS={rms(cleaned):.4f} peak={np.max(np.abs(cleaned)):.3f}  (target {TARGET_RMS})")

    try:
        from stt import WhisperSTT
        s = WhisperSTT({"whisper_model": "medium", "whisper_device": "cpu",
                        "language": "auto", "no_speech_threshold": 0.8})
        s.load()
        tn, _ = s.transcribe(noisy.copy(), sr)
        tc, _ = s.transcribe(cleaned.copy(), sr)
        print(f"\nNOISY  -> {tn!r}")
        print(f"CLEAN  -> {tc!r}")
    except Exception as e:
        print("STT compare skipped:", e)
