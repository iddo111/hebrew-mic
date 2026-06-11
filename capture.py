# dictation/capture.py — continuous mic capture + DSP chain + energy VAD
# Yields complete utterance numpy arrays to a callback when trailing silence detected.
# Integrates mic_guard firewall: BT/loopback/virtual devices are NEVER selectable.

import threading, numpy as np, sounddevice as sd
import scipy.signal as sig_proc

from mic_guard import (
    pick_mic, assert_safe_to_open, query_devices_live, DeviceEntry,
    DEFAULT_BLOCKLIST, DEFAULT_PREFER_NAMES,
)


class MicCapture:
    """
    Streams mic audio with DSP chain, detects utterances via energy threshold + trailing silence.
    Calls `on_utterance(audio_np_float32, sample_rate)` in a background thread.
    Mic selection is enforced through mic_guard: BT/loopback/virtual devices are rejected.
    """

    def __init__(self, cfg: dict, on_utterance):
        self._cfg = cfg
        self._on_utterance = on_utterance
        self._sr = cfg["sample_rate"]
        self._silence_frames = int(cfg["silence_ms"] / 1000 * self._sr)
        self._energy_thresh = cfg.get("energy_threshold", 0.005)
        self._gain = float(cfg.get("software_gain", 14.0))
        self._dsp_enabled = cfg.get("dsp_enabled", True)
        self._device = self._pick_safe_device(cfg)
        self._running = False
        self._thread = None
        self._frames: list = []
        self._silent_count = 0
        self._in_speech = False
        self._stop_evt = threading.Event()
        # PTT state
        self._ptt_active = False  # controlled by main.py
        # Level meter (RMS of last block, for widget)
        self.level_rms: float = 0.0
        self._build_dsp_filters()

    # ── device selection via mic_guard ────────────────────────────────────────
    def _pick_safe_device(self, cfg) -> int | None:
        # If explicit device index given, validate it through mic_guard
        explicit = cfg.get("mic_device")
        devices = query_devices_live()

        if explicit is not None:
            try:
                chosen = devices[int(explicit)]
                assert_safe_to_open(chosen, blocklist=cfg.get("mic_blocklist", DEFAULT_BLOCKLIST))
                return chosen.index
            except (IndexError, ValueError) as e:
                import logging
                logging.getLogger("dictation.capture").warning(
                    f"Explicit mic_device={explicit} rejected: {e} — falling back to auto-select"
                )

        # Auto-select via mic_guard
        prefer = cfg.get("mic_prefer_name", "")
        prefer_list = [prefer] + list(DEFAULT_PREFER_NAMES) if prefer else list(DEFAULT_PREFER_NAMES)
        blocklist = cfg.get("mic_blocklist", DEFAULT_BLOCKLIST)

        result = pick_mic(devices, prefer_names=prefer_list, blocklist=blocklist)
        import logging
        log = logging.getLogger("dictation.capture")
        if not result.ok:
            log.error(f"mic_guard: {result.reason}")
            return None
        log.info(f"mic_guard selected: [{result.device.index}] {result.device.name} — {result.reason}")
        return result.device.index

    # ── DSP filter construction ───────────────────────────────────────────────
    def _build_dsp_filters(self):
        """Build filters AND initialise persistent streaming state (zi arrays).
        Critical: sosfilt/lfilter MUST use zi state across blocks or filter
        memory is lost at every 32ms boundary → transient distortion on each block.
        """
        sr = self._sr
        hpf_hz = self._cfg.get("dsp_hpf_hz", 110.0)
        notch_freqs = self._cfg.get("dsp_notch_freqs", [50, 100, 150])
        notch_q = self._cfg.get("dsp_notch_q", 30.0)

        # High-pass filter (Butterworth 2nd order, SOS form)
        self._sos_hp = sig_proc.butter(2, hpf_hz, btype="high", fs=sr, output="sos")
        self._zi_hp = sig_proc.sosfilt_zi(self._sos_hp) * 0.0  # init at 0

        # Notch filters for hum (IIR, b/a form)
        self._notch_ba: list = []
        self._zi_notch: list = []
        for f0 in notch_freqs:
            if 0 < f0 < sr / 2:
                b, a = sig_proc.iirnotch(f0, Q=notch_q, fs=sr)
                self._notch_ba.append((b, a))
                # lfilter_zi expects initial conditions length = max(len(a),len(b))-1
                self._zi_notch.append(sig_proc.lfilter_zi(b, a) * 0.0)

        self._gate_thresh = self._cfg.get("dsp_noise_gate", 0.01)

    # ── DSP chain ─────────────────────────────────────────────────────────────
    def _apply_dsp(self, block: np.ndarray) -> np.ndarray:
        """Apply gain + HPF + hum notches + noise gate, maintaining filter state
        across blocks so there is no transient distortion at block boundaries."""
        # Software gain
        if self._gain != 1.0:
            block = block * self._gain

        # High-pass filter (stateful)
        block, self._zi_hp = sig_proc.sosfilt(self._sos_hp, block, zi=self._zi_hp)

        # Hum notch filters (stateful)
        for idx, (b, a) in enumerate(self._notch_ba):
            block, self._zi_notch[idx] = sig_proc.lfilter(b, a, block, zi=self._zi_notch[idx])

        # Noise gate: zero out 10ms frames below gate threshold
        frame_size = max(64, int(self._sr * 0.01))
        for i in range(0, len(block) - frame_size, frame_size):
            frame = block[i:i + frame_size]
            if np.sqrt(np.mean(frame ** 2)) < self._gate_thresh:
                block[i:i + frame_size] = 0.0

        return np.clip(block, -1.0, 1.0).astype(np.float32)

    # ── public API ────────────────────────────────────────────────────────────
    def start(self):
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="dictation-capture")
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=3)

    def reload_config(self, cfg: dict):
        """Hot-reload config (gain, thresholds) without restarting the stream."""
        self._cfg = cfg
        self._gain = float(cfg.get("software_gain", 14.0))
        self._energy_thresh = cfg.get("energy_threshold", 0.005)
        self._dsp_enabled = cfg.get("dsp_enabled", True)
        self._silence_frames = int(cfg["silence_ms"] / 1000 * self._sr)
        self._build_dsp_filters()

    def switch_device(self, index, blocklist=None):
        """Live device switch: stop the stream, repoint, restart — no process
        restart needed. Used by the picker dropdown so changing mic takes effect
        immediately. Explicit index is used as-is (validated in _run)."""
        import logging
        log = logging.getLogger("dictation.capture")
        if blocklist is not None:
            self._cfg["mic_blocklist"] = blocklist
        self._device = int(index) if index is not None else None
        if self._thread is not None and self._thread.is_alive():
            self.stop()
        self._frames.clear(); self._in_speech = False; self._silent_count = 0
        self.level_rms = 0.0
        self.start()
        log.info(f"switch_device -> {self._device}")

    @property
    def device_name(self) -> str:
        try:
            idx = self._device if self._device is not None else sd.default.device[0]
            return sd.query_devices(idx)["name"]
        except Exception:
            return "unknown"

    @property
    def device_index(self) -> int | None:
        return self._device

    def list_safe_devices(self) -> list[dict]:
        """Return list of safe input devices for UI mic-selection dropdown."""
        blocklist = self._cfg.get("mic_blocklist", DEFAULT_BLOCKLIST)
        from mic_guard import is_blocked
        result = []
        for d in query_devices_live():
            if d.max_input_channels > 0 and not is_blocked(d.name, blocklist):
                result.append({"index": d.index, "name": d.name,
                                "channels": d.max_input_channels})
        return result

    # ── internal ─────────────────────────────────────────────────────────────
    def _run(self):
        import logging
        log = logging.getLogger("dictation.capture")

        # LEGO-4: validate device before opening
        if self._device is not None:
            devices = query_devices_live()
            try:
                dev = next(d for d in devices if d.index == self._device)
                assert_safe_to_open(dev, blocklist=self._cfg.get("mic_blocklist", DEFAULT_BLOCKLIST))
            except StopIteration:
                log.error(f"Device index {self._device} not found; using system default")
                self._device = None
            except ValueError as e:
                log.error(f"LEGO-4 rejected device: {e}")
                self._device = None

        import queue
        blocksize = int(self._sr * 0.032)  # 32ms blocks
        latency = self._cfg.get("stream_latency", "high")
        # CALLBACK mode (NOT blocking stream.read): BT/MME devices hang on the
        # blocking read API (PaErrorCode -9987 / "Blocking API not supported"),
        # but the callback API works (this is why sd.rec worked). The callback
        # only enqueues; a consumer here does DSP/VAD. + self-heal reconnect loop.
        q: "queue.Queue" = queue.Queue(maxsize=256)

        def _cb(indata, frames, time_info, status):
            try:
                q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        backoff = 0
        while not self._stop_evt.is_set():
            try:
                with sd.InputStream(
                    device=self._device,
                    channels=1,
                    samplerate=self._sr,
                    dtype="float32",
                    blocksize=blocksize,
                    latency=latency,
                    callback=_cb,
                ):
                    log.info(f"Stream open (callback): device={self.device_name} "
                             f"sr={self._sr} blocksize={blocksize} latency={latency}")
                    backoff = 0
                    while not self._stop_evt.is_set():
                        try:
                            block = q.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        self.level_rms = float(np.sqrt(np.mean(block ** 2)))
                        if self._dsp_enabled:
                            block = self._apply_dsp(block.copy())
                        else:
                            block = np.clip(block * self._gain, -1.0, 1.0)
                        self._process_block(block)
            except Exception as e:
                self.level_rms = 0.0
                if self._stop_evt.is_set():
                    break
                backoff = min(backoff + 1, 5)
                log.error(f"Audio stream error: {e} — reconnecting in {backoff}s")
                self._stop_evt.wait(backoff)

    def _process_block(self, block: np.ndarray):
        rms = float(np.sqrt(np.mean(block ** 2)))
        if rms >= self._energy_thresh:
            self._frames.append(block.copy())
            self._in_speech = True
            self._silent_count = 0
        elif self._in_speech:
            self._frames.append(block.copy())
            self._silent_count += len(block)
            if self._silent_count >= self._silence_frames:
                audio = np.concatenate(self._frames)
                self._frames.clear()
                self._in_speech = False
                self._silent_count = 0
                t = threading.Thread(
                    target=self._on_utterance,
                    args=(audio, self._sr),
                    daemon=True,
                )
                t.start()
