# dictation/stt.py — STT engine wrapper around faster-whisper
# Our code: model selection, language logic, hallucination filter, result post-processing.
# Their code: CTranslate2 inference (the trained weights — the real moat).

import re
import numpy as np
import logging

log = logging.getLogger("dictation.stt")

# Known Whisper hallucinations — these are noise artifacts at ANY utterance length.
_ALWAYS_PATTERNS = [
    r"^\s*$",
    r"^[\s\.\!\?]+$",
    r"^\s*you[\.\s]*$",
    r"^\s*thanks for watching[\.\s]*$",
    r"^\s*\*\*.*\*\*\s*$",
    r"^\s*\[.*\]\s*$",
    r"^\s*\(.*\)\s*$",
    r"^\s*\.+\s*$",
    r"^\s*(?:uh+|um+|eh+|hmm+)\s*$",
    r"^\s*תודה\s+רבה\s+על\s+הצפייה[\.\s,!]*$",
    r"^\s*תודה\s+שצפיתם[\.\s,!]*$",
]

# Words Whisper invents out of silence — but which a human also genuinely says.
# Dropping these unconditionally made it impossible to dictate "כן", "לא" or "רגע",
# so they are only rejected when the audio was too short to BE a real word
# (see SHORT_UTTERANCE_SEC). A deliberate one-word answer clears that bar easily.
_SHORT_ONLY_PATTERNS = [
    r"^\s*thank you[\.\s]*$",
    # 1-3 Hebrew letters only — usually a noise artifact
    r"^[\sא-ת]{1,3}\s*[.\s]*$",
    r"^\s*תודה[\.\s,!]*$",
    r"^\s*תודה\s+רבה[\.\s,!]*$",
    r"^\s*בבקשה[\.\s,!]*$",
    r"^\s*כן[\.\s,!]*$",
    r"^\s*לא[\.\s,!]*$",
    r"^\s*סליחה[\.\s,!]*$",
    r"^\s*שלום[\.\s,!]*$",
    r"^\s*אמן[\.\s,!]*$",
    r"^\s*ביי[\.\s,!]*$",
]

# Audio shorter than this can't hold a deliberate spoken word, so the
# short-only patterns above stay armed for it.
SHORT_UTTERANCE_SEC = 0.6

_ALWAYS_RE = [re.compile(p, re.IGNORECASE) for p in _ALWAYS_PATTERNS]
_SHORT_ONLY_RE = [re.compile(p, re.IGNORECASE) for p in _SHORT_ONLY_PATTERNS]

# Repeated token detection (e.g. "כתוביות כתוביות כתוביות")
def _has_repeated_tokens(text: str, max_repeats: int = 3) -> bool:
    tokens = text.split()
    if len(tokens) < max_repeats:
        return False
    # Check if any token repeats max_repeats times consecutively
    for i in range(len(tokens) - max_repeats + 1):
        window = tokens[i:i + max_repeats]
        if len(set(w.lower() for w in window)) == 1:
            return True
    return False


def _is_repeated_char_garbage(text: str) -> bool:
    """Catch Whisper silence-hallucinations that are a single space-less token of
    one character repeated (e.g. 'חחחחחח…') — _has_repeated_tokens misses these
    because there are no spaces to split on."""
    s = re.sub(r"\s", "", text)
    if len(s) < 4:
        return False
    most = max((s.count(c) for c in set(s)), default=0)
    if most / len(s) > 0.6:           # one char dominates the string
        return True
    if re.search(r"(.)\1{5,}", s):    # any char run of 6+
        return True
    return False


class WhisperSTT:
    """
    Wraps faster-whisper for Hebrew/English dictation.
    model: "medium" or "large-v3"
    Falls back to CPU if CUDA unavailable.
    """

    def __init__(self, cfg: dict):
        self._model_name = cfg.get("whisper_model", "medium")
        self._device = cfg.get("whisper_device", "cuda")
        self._language = cfg.get("language", "he")
        self._no_speech_threshold = float(cfg.get("no_speech_threshold", 0.80))
        self._model = None

    def reload_config(self, cfg: dict):
        """Hot-reload language and threshold settings."""
        self._language = cfg.get("language", "he")
        self._no_speech_threshold = float(cfg.get("no_speech_threshold", 0.80))

    def load(self) -> str:
        try:
            import torch
            if self._device == "cuda" and not torch.cuda.is_available():
                log.warning("CUDA not available — falling back to CPU")
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"

        from faster_whisper import WhisperModel
        # ivrit-ai turbo = a local CT2 dir path (best Hebrew). If it's a path that
        # doesn't exist, fall back to the bundled 'medium' so the mic never dies.
        import os as _os
        if (_os.sep in self._model_name or "/" in self._model_name) and not _os.path.isdir(self._model_name):
            log.warning(f"model dir not found: {self._model_name} -> falling back to 'medium'")
            self._model_name = "medium"
        compute = "float16" if self._device == "cuda" else "int8"
        log.info(f"Loading Whisper {self._model_name} on {self._device} ({compute})")
        try:
            self._model = WhisperModel(
                self._model_name, device=self._device, compute_type=compute,
            )
        except Exception as e:
            # cuda chosen but cuDNN/CT2 libs missing -> never kill the mic, fall back to CPU
            if self._device == "cuda":
                log.warning(f"CUDA Whisper init failed ({e}); falling back to CPU/int8")
                self._device, compute = "cpu", "int8"
                self._model = WhisperModel(
                    self._model_name, device=self._device, compute_type=compute,
                )
            else:
                raise
        return f"Whisper {self._model_name} / {self._device} / {compute}"

    def _run_once(self, audio: np.ndarray, lang_hint, use_vad: bool,
                  allow_short_words: bool = False) -> tuple[str, str]:
        """One faster-whisper pass. VAD optional (Silero-VAD can throw a torch
        tensor error under GPU pressure — the caller retries without it)."""
        kw = dict(
            language=lang_hint, beam_size=5,
            without_timestamps=False, word_timestamps=False,
            no_speech_threshold=self._no_speech_threshold,
            condition_on_previous_text=False,
        )
        if use_vad:
            kw.update(vad_filter=True, vad_parameters=dict(
                min_silence_duration_ms=700, speech_pad_ms=200, threshold=0.50))
        segs, info = self._model.transcribe(audio, **kw)
        good_parts = []
        for s in segs:
            if s.no_speech_prob > self._no_speech_threshold:
                continue
            t = s.text.strip()
            if (self._is_hallucination(t, allow_short_words)
                    or _has_repeated_tokens(t) or _is_repeated_char_garbage(t)):
                continue
            good_parts.append(t)
        return " ".join(good_parts).strip(), (info.language or "unknown")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> tuple[str, str]:
        """
        Returns (text, detected_language). ("", "") when no real speech / on failure.
        Self-healing: VAD error -> retry without VAD; CUDA/torch error under GPU
        pressure -> reload on CPU once and retry. Never raises to the widget loop.
        """
        if self._model is None:
            raise RuntimeError("call load() first")
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
        # contiguous float32 — a non-contiguous/sliced array trips "expected tensor" errors
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        lang_hint = None if self._language == "auto" else self._language
        # Long enough to be a deliberate word? Then keep short answers ("כן"/"לא").
        allow_short = (len(audio) / 16000.0) >= SHORT_UTTERANCE_SEC

        try:
            return self._run_once(audio, lang_hint, use_vad=True, allow_short_words=allow_short)
        except Exception as e:
            log.warning(f"transcribe failed ({type(e).__name__}: {e}); retry without VAD")
            try:
                return self._run_once(audio, lang_hint, use_vad=False, allow_short_words=allow_short)
            except Exception as e2:
                # CUDA/torch error (e.g. OOM on a crowded 8GB GPU) -> heal to CPU.
                if self._device == "cuda":
                    log.warning(f"GPU transcribe failed ({type(e2).__name__}: {e2}); reloading on CPU")
                    try:
                        from faster_whisper import WhisperModel
                        self._device = "cpu"
                        self._model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
                        return self._run_once(audio, lang_hint, use_vad=False,
                                              allow_short_words=allow_short)
                    except Exception as e3:
                        log.error(f"CPU fallback also failed: {type(e3).__name__}: {e3}")
                return "", ""

    @staticmethod
    def _is_hallucination(text: str, allow_short_words: bool = False) -> bool:
        """True when `text` looks like a Whisper artifact rather than speech.

        `allow_short_words=True` (set for utterances longer than SHORT_UTTERANCE_SEC)
        keeps genuine one-word answers like "כן" / "לא" / "תודה"."""
        for pat in _ALWAYS_RE:
            if pat.match(text):
                return True
        if not allow_short_words:
            for pat in _SHORT_ONLY_RE:
                if pat.match(text):
                    return True
        return False

    @staticmethod
    def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        try:
            import scipy.signal as signal
            ratio = to_sr / from_sr
            n_out = int(len(audio) * ratio)
            return signal.resample(audio, n_out).astype(np.float32)
        except ImportError:
            indices = np.linspace(0, len(audio) - 1, int(len(audio) * to_sr / from_sr))
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
