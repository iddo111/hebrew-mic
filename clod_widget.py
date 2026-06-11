"""Clod Dictation Widget — standalone, self-contained GUI.

Everything in one window, no background service required:
  * MIC PICKER  — choose ANY input device (never locked to one mic).
  * WAVEFORM    — live oscilloscope of the mic so you SEE your voice.
  * LEVEL       — peak-hold bar.
  * Types the Hebrew/English transcription into the FOREGROUND window.
  * AUTO-ENTER  — presses Enter after N minutes of silence (default 5 min).

Capture is done the right way: open at the device's NATIVE samplerate, then
resample the whole utterance to 16 kHz in software (proven clean) — no forced
16 kHz at the driver. The widget sets WS_EX_NOACTIVATE so it never steals focus,
so keystrokes land in whatever app you were typing in.

    python clod_widget.py

Window title: "Claude — Dictation".  לזכרה של עלמה ז״ל — iddoperez.ai
"""
from __future__ import annotations

import os, sys, time, json, re, unicodedata, threading, queue, ctypes, ctypes.wintypes
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd
import scipy.signal as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stt import WhisperSTT          # faster-whisper wrapper (he/en, hallucination filter)
import typer as typer_mod           # SendInput Unicode typing + Enter
import audio_clean                   # auto-gain + auto-volume + denoise (better Hebrew)
import haiku_correct                  # HaikuBrainz — smart autocorrect (per-utterance)
import corrections                    # teachable {wrong:right} map — learns Iddo's words
from mic_guard import pick_mic, query_devices_live, DEFAULT_PREFER_NAMES

TARGET_SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
KEYWORDS_PATH = os.path.join(HERE, "clod_keywords.json")

try:
    import winsound
except Exception:  # non-Windows / no audio
    winsound = None


def _dbg(msg):
    # pythonw has no console → sys.stderr/stdout are None; an unguarded print()
    # crashes the widget on the first utterance. Guard it.
    try:
        if sys.stderr is not None:
            print(msg, file=sys.stderr, flush=True)
    except Exception:
        pass

# ── colours (2026-06 facelift — Grok-spec near-black + indigo accent) ───────
BG="#0a0a0c"; BAR="#1a1b20"; PANEL="#131316"; CARD="#131316"; EDGE="#24252b"
FG="#f1f5f9"; MUT="#7c8089"
GREEN="#22c55e"; RED="#ef4444"; AMBER="#f59e0b"; WAVE="#a5b4fc"; ACC="#6366f1"
ARMED="#9ca3af"; ROUTE="#60a5fa"
WAVE_BG="#0f0f12"; WAVE_GLOW="#312e81"; WAVE_GLOW2="#4f46e5"; GRID="#16171c"; MIDL="#1f2027"
BTN_ON="#6366f1";  BTN_ON_H="#787bf5"      # listening — calm live indigo
BTN_OFF="#b45309"; BTN_OFF_H="#d97706"     # paused — amber warning
BTN_BLUE="#24252b"; BTN_BLUE_H="#313340"   # neutral pill (language)
BTN_MODE="#24252b"; BTN_MODE_H="#313340"   # wake-word mode — neutral
BTN_ALWAYS="#15803d"; BTN_ALWAYS_H="#16a34a"  # continuous mode — hot green
LV_BG="#0f0f12"; LV_H=12
W, H = 380, 418
TITLE_H = 36


# ── Hebrew normalize + fuzzy wake-word matching (Gemini-reviewed) ───────────
_NIKUD = re.compile(r"[֑-ׇ]")

def _norm(s: str) -> str:
    """Strip nikud + punctuation, collapse whitespace, lower — for matching only."""
    s = unicodedata.normalize("NFKC", s or "")
    s = _NIKUD.sub("", s)
    s = re.sub(r"[^\w\s֐-׿]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()

def _lev(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def _wake_remainder(text: str, phrase: str, max_ratio: float = 0.34):
    """If `text` STARTS with `phrase` (fuzzy), return the remainder of the ORIGINAL
    text after the wake-word, else None. Anchored at start + edit-distance gate to
    cut false triggers on a short word like 'קלוד' (Gemini)."""
    pn = _norm(phrase)
    tn = _norm(text)
    if not pn or not tn:
        return None
    pwords = pn.split()
    twords = tn.split()
    head = " ".join(twords[:len(pwords)]) if len(twords) >= len(pwords) else tn
    if _lev(head, pn) <= max(1, int(len(pn) * max_ratio)):
        orig = text.strip().split()
        return " ".join(orig[len(pwords):]) if len(orig) >= len(pwords) else ""
    return None


# ════════════════════════════════════════════════════════════════════════════
#  Audio engine — native-rate capture, software resample, energy VAD
# ════════════════════════════════════════════════════════════════════════════
class AudioEngine:
    def __init__(self, on_utterance, on_block):
        self.on_utterance = on_utterance      # (audio16k: np, sr=16000)
        self.on_block = on_block              # (gained_native_block: np)  — for waveform/level
        self.device = None
        self.native_sr = 48000
        self.gain = 6.0
        self.energy_threshold = 0.010         # post-gain RMS to count as speech
        self.silence_ms = 900                 # trailing silence ends an utterance
        self.listening = True
        self._q: "queue.Queue" = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._thread = None
        self.level_rms = 0.0
        self.last_error = ""

    def set_device(self, index):
        self.device = index
        if self._thread and self._thread.is_alive():
            self.stop()
        self.start()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="clod-capture")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        try:
            info = sd.query_devices(self.device if self.device is not None else sd.default.device[0])
            self.native_sr = int(info["default_samplerate"]) or 48000
        except Exception:
            self.native_sr = 48000
        silence_samples = int(self.silence_ms / 1000 * self.native_sr)
        frames, in_speech, silent = [], False, 0

        def cb(indata, n, t, status):
            try: self._q.put_nowait(indata[:, 0].copy())
            except queue.Full: pass

        backoff = 0
        while not self._stop.is_set():
            try:
                with sd.InputStream(device=self.device, channels=1, samplerate=self.native_sr,
                                    dtype="float32", blocksize=int(self.native_sr*0.032),
                                    latency="high", callback=cb):
                    self.last_error = ""
                    backoff = 0
                    while not self._stop.is_set():
                        try: raw = self._q.get(timeout=0.4)
                        except queue.Empty: continue
                        block = raw * self.gain
                        np.clip(block, -1.0, 1.0, out=block)
                        self.level_rms = float(np.sqrt(np.mean(block**2)))
                        self.on_block(block)
                        if not self.listening:
                            frames, in_speech, silent = [], False, 0
                            continue
                        rms = self.level_rms
                        if rms >= self.energy_threshold:
                            frames.append(block); in_speech = True; silent = 0
                        elif in_speech:
                            frames.append(block); silent += len(block)
                            if silent >= silence_samples:
                                utt = np.concatenate(frames); frames, in_speech, silent = [], False, 0
                                a16 = ss.resample_poly(utt, TARGET_SR, self.native_sr).astype(np.float32) \
                                      if self.native_sr != TARGET_SR else utt
                                self.on_utterance(a16, TARGET_SR)
            except Exception as e:
                self.level_rms = 0.0
                if self._stop.is_set(): break
                backoff = min(backoff+1, 5)
                self.last_error = str(e)[:80]
                self._stop.wait(backoff)


# ════════════════════════════════════════════════════════════════════════════
#  Widget
# ════════════════════════════════════════════════════════════════════════════
class ClodWidget:
    def __init__(self, silence_enter_sec: int = 2):
        self.silence_enter_sec = silence_enter_sec
        self.root = tk.Tk()
        self.root.title("Claude — Dictation")
        self.root.attributes("-topmost", True)
        self.root.configure(bg=BG)
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"{W}x{H}+{sw - W - 24}+60")

        self._wave = np.zeros(TARGET_SR // 2, np.float32)   # ring buffer of recent (gained) samples
        self._wave_lock = threading.Lock()
        self._enter_after = None
        self._stt = None
        self._model_ready = False
        self._typed_since_enter = False
        self._self_hwnd = 0
        self._target_hwnd = 0
        self._target_title = ""
        self._last_shown_title = None

        # wake-word state
        self._mode = "wakeword"          # "wakeword" | "always"
        self._dictating = False
        self._disarm_after = None
        self._dictate_timeout = 30
        self._stop_phrases = ["סיום", "תפסיק"]
        self._submit_phrases = ["קלוד שלח", "שירי שלחי", "send it"]
        self._keywords = []
        self._dictate_to_claude = True
        self._claude_patterns = ["Claude"]
        self._load_keywords()

        self.engine = AudioEngine(self._on_utterance, self._on_block)
        self._build_ui()
        self._refresh_keys_label()
        self._no_activate()
        self._load_stt_async()
        self._populate_devices()
        self.engine.start()
        self._draw_loop()

    # ── make the window never steal focus (so typing goes to the real app) ──
    def _no_activate(self):
        try:
            self.root.update_idletasks()
            GWL_EXSTYLE=-20; WS_EX_NOACTIVATE=0x08000000; WS_EX_TOOLWINDOW=0x00000080
            u = ctypes.windll.user32
            hwnd = (u.FindWindowW(None, "Claude — Dictation")
                    or u.GetParent(self.root.winfo_id()) or self.root.winfo_id())
            self._self_hwnd = hwnd
            ex = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
            u.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        except Exception:
            pass

    # ── target window: type into whatever app Iddo is using (never ourselves) ──
    #   Self-exclusion is by PID (robust) — any window of OUR process is never a
    #   target, so the widget can never receive its own dictation.
    def _track_target(self):
        try:
            u = ctypes.windll.user32
            fg = u.GetForegroundWindow()
            if not fg:
                return
            pid = ctypes.wintypes.DWORD()
            u.GetWindowThreadProcessId(fg, ctypes.byref(pid))
            if pid.value == os.getpid():
                return  # our own window — keep the previous real target
            # When dictating after the 'קלוד' wake-word, the target is LOCKED to the
            # Claude window (set in _start_dictating) — don't let the foreground tracker
            # steal it back to whatever app Iddo clicks. "Say קלוד → types to Claude."
            if self._dictating and self._dictate_to_claude:
                return
            self._target_hwnd = fg
            buf = ctypes.create_unicode_buffer(160)
            u.GetWindowTextW(fg, buf, 160)
            self._target_title = buf.value or ""
        except Exception:
            pass

    def _focus_target(self):
        """Force the tracked target window to the foreground before typing, using
        the AttachThreadInput trick so it works past Windows' foreground lock."""
        hwnd = self._target_hwnd
        if not hwnd:
            return
        try:
            u = ctypes.windll.user32; k = ctypes.windll.kernel32
            fg = u.GetForegroundWindow()
            # NOTE: no early-out when fg == hwnd. Re-running the activation dance even when
            # Claude is already foreground re-asserts its last-active control and pairs with
            # the caret-focus click in submit_to_claude — this is what fixes the idle-Enter
            # case where window focus is held but the renderer caret has drifted off the box.
            # ALT tap unlocks Windows' SetForegroundWindow restriction, then attach
            # input queues and pull the target to the front — robust across apps.
            u.keybd_event(0x12, 0, 0, 0); u.keybd_event(0x12, 0, 0x0002, 0)
            t_fg = u.GetWindowThreadProcessId(fg, None)
            t_tg = u.GetWindowThreadProcessId(hwnd, None)
            cur = k.GetCurrentThreadId()
            u.AttachThreadInput(t_fg, cur, True)
            u.AttachThreadInput(t_tg, cur, True)
            u.SetForegroundWindow(hwnd)
            u.BringWindowToTop(hwnd)
            u.AttachThreadInput(t_fg, cur, False)
            u.AttachThreadInput(t_tg, cur, False)
            time.sleep(0.06)
        except Exception:
            pass

    def _find_claude_window(self):
        """Find the Claude window (Desktop/Code) to aim dictation at, regardless of
        which app currently has focus. Excludes our own widget (by PID and title).
        Returns (hwnd, title) or (0, "")."""
        try:
            u = ctypes.windll.user32
            mypid = os.getpid()
            found = []
            EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

            def cb(hwnd, _):
                try:
                    if not u.IsWindowVisible(hwnd):
                        return True
                    pid = ctypes.wintypes.DWORD()
                    u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    if pid.value == mypid:
                        return True
                    n = u.GetWindowTextLengthW(hwnd)
                    if n == 0:
                        return True
                    buf = ctypes.create_unicode_buffer(n + 1)
                    u.GetWindowTextW(hwnd, buf, n + 1)
                    title = buf.value or ""
                    low = title.lower()
                    if "dictation" in low or "הכתבה" in title:
                        return True
                    for pat in self._claude_patterns:
                        if pat.lower() in low:
                            score = 0 if low.strip() == pat.lower() else 1
                            found.append((score, hwnd, title))
                            break
                except Exception:
                    pass
                return True

            u.EnumWindows(EnumProc(cb), 0)
            if found:
                found.sort(key=lambda x: x[0])
                return found[0][1], found[0][2]
        except Exception:
            pass
        return 0, ""

    # ── button styling helpers (visual only) ────────────────────────────────
    def _pill(self, parent, text, bg, hbg, command, fg="#ffffff", font=("Segoe UI", 9, "bold"), padx=12, pady=4):
        b = tk.Button(parent, text=text, font=font, bg=bg, fg=fg, bd=0, relief="flat",
                      padx=padx, pady=pady, cursor="hand2",
                      activebackground=hbg, activeforeground=fg, command=command)
        b._base, b._hover = bg, hbg
        b.bind("<Enter>", lambda e: b.config(bg=b._hover))
        b.bind("<Leave>", lambda e: b.config(bg=b._base))
        return b

    def _restyle(self, b, text, bg, hbg):
        b._base, b._hover = bg, hbg
        b.config(text=text, bg=bg, activebackground=hbg)

    def _build_ui(self):
        r = self.root
        r.option_add("*TCombobox*Listbox.background", PANEL)
        r.option_add("*TCombobox*Listbox.foreground", FG)
        r.option_add("*TCombobox*Listbox.selectBackground", ACC)
        r.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        r.option_add("*TCombobox*Listbox.borderWidth", 0)

        bar = tk.Frame(r, bg=BAR, height=TITLE_H); bar.pack(fill="x"); bar.pack_propagate(False)
        tk.Label(bar, text="🎙️", font=("Segoe UI Emoji", 11), fg=ACC, bg=BAR).pack(side="left", padx=(12,5))
        tk.Label(bar, text="הכתבה בעברית", font=("Segoe UI Semibold", 11),
                 fg=FG, bg=BAR).pack(side="left")
        btn_close = tk.Button(bar, text="✕", font=("Segoe UI", 10), bg=BAR, fg=MUT, bd=0,
                              relief="flat", padx=12, cursor="hand2",
                              activebackground="#e5484d", activeforeground="#ffffff",
                              command=self._close)
        btn_close.pack(side="right", fill="y")
        btn_close.bind("<Enter>", lambda e: btn_close.config(bg="#e5484d", fg="#ffffff"))
        btn_close.bind("<Leave>", lambda e: btn_close.config(bg=BAR, fg=MUT))
        self.btn_min = tk.Button(bar, text="—", font=("Segoe UI", 10), bg=BAR, fg=MUT, bd=0,
                                 relief="flat", padx=12, cursor="hand2",
                                 activebackground=PANEL, activeforeground=FG,
                                 command=self._toggle_minimize)
        self.btn_min.pack(side="right", fill="y")
        self.btn_min.bind("<Enter>", lambda e: self.btn_min.config(bg=PANEL, fg=FG))
        self.btn_min.bind("<Leave>", lambda e: self.btn_min.config(bg=BAR, fg=MUT))
        tk.Frame(r, bg=EDGE, height=1).pack(fill="x")

        self._body = body = tk.Frame(r, bg=BG); body.pack(fill="both", expand=True, padx=14, pady=(10,12))

        # mic picker
        row = tk.Frame(body, bg=BG); row.pack(fill="x", pady=(0,8))
        tk.Label(row, text="מיקרופון", font=("Segoe UI", 9), fg=MUT, bg=BG).pack(side="right", padx=(8,0))
        self._mic_var = tk.StringVar(value="…")
        st = ttk.Style(); st.theme_use("clam")
        st.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG,
                     selectbackground=PANEL, selectforeground=FG, arrowcolor=ACC,
                     bordercolor=EDGE, lightcolor=PANEL, darkcolor=PANEL,
                     insertcolor=FG, padding=4, relief="flat", arrowsize=14)
        st.map("TCombobox",
               fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)],
               background=[("readonly", PANEL)], bordercolor=[("focus", ACC)])
        self.combo = ttk.Combobox(row, textvariable=self._mic_var, font=("Segoe UI", 8),
                                  state="readonly", width=30)
        self.combo.pack(side="left", fill="x", expand=True)
        self.combo.bind("<<ComboboxSelected>>", self._on_pick_mic)

        # waveform
        self.canvas = tk.Canvas(body, height=104, bg=WAVE_BG, highlightthickness=1,
                                highlightbackground=EDGE, bd=0)
        self.canvas.pack(fill="x", pady=(0,8))
        self.canvas.bind("<Configure>", self._on_wave_resize)

        # level bar
        lv = tk.Frame(body, bg=BG); lv.pack(fill="x", pady=(0,10))
        tk.Label(lv, text="עוצמה", font=("Segoe UI", 8), fg=MUT, bg=BG).pack(side="right", padx=(8,0))
        self.level_canvas = tk.Canvas(lv, height=LV_H, bg=LV_BG, highlightthickness=1,
                                      highlightbackground=EDGE, bd=0)
        self.level_canvas.pack(side="left", fill="x", expand=True)
        self._level_bar = self.level_canvas.create_rectangle(0,0,0,LV_H, fill=GREEN, outline="")
        self._level_peak = self.level_canvas.create_rectangle(0,0,0,LV_H, fill="#e4e7ee", outline="")
        self._lv_peak = 0.0
        self.level_canvas.bind("<Configure>", self._on_level_resize)

        # controls: listen toggle, gain, language
        c = tk.Frame(body, bg=BG); c.pack(fill="x", pady=(0,8))
        self.btn_listen = self._pill(c, "‖ השהה", BTN_ON, BTN_ON_H, self._toggle_listen)
        self.btn_listen.pack(side="left")
        self.btn_lang = self._pill(c, "HE", BTN_BLUE, BTN_BLUE_H, self._cycle_lang)
        self.btn_lang.pack(side="left", padx=8)
        tk.Label(c, text="Gain", font=("Segoe UI", 8), fg=MUT, bg=BG).pack(side="left", padx=(8,2))
        self._gain = tk.DoubleVar(value=self.engine.gain)
        tk.Scale(c, variable=self._gain, from_=1, to=30, resolution=0.5, orient="horizontal",
                 length=104, bg=BG, fg=MUT, troughcolor=EDGE, activebackground=ACC,
                 highlightthickness=0, bd=0, sliderrelief="flat", width=10,
                 showvalue=True, font=("Segoe UI", 7),
                 command=lambda v: setattr(self.engine, "gain", float(v))).pack(side="left")

        # mode toggle + HaikuBrainz toggle + active wake-words
        m = tk.Frame(body, bg=BG); m.pack(fill="x")
        self.btn_mode = self._pill(m, "● מילת-הפעלה", BTN_MODE, BTN_MODE_H, self._toggle_mode)
        self.btn_mode.pack(side="left")
        _hb_on = getattr(self, "_haiku_brainz", False)
        self.btn_haiku = self._pill(
            m, "🧠 הייקו" if _hb_on else "🧠 הייקו ✕",
            BTN_ON if _hb_on else BTN_OFF, BTN_ON_H if _hb_on else BTN_OFF_H,
            self._toggle_haiku)
        self.btn_haiku.pack(side="left", padx=8)
        self.lbl_keys = tk.Label(m, text="", font=("Segoe UI", 7), fg=MUT, bg=BG, anchor="e")
        self.lbl_keys.pack(side="right", fill="x", expand=True)

        # status card: accent strip + status / last-text / auto-enter info
        card = tk.Frame(body, bg=CARD, highlightthickness=1, highlightbackground=EDGE)
        card.pack(fill="both", expand=True, pady=(10,0))
        tk.Frame(card, bg=ACC, width=3).pack(side="right", fill="y")
        inner = tk.Frame(card, bg=CARD); inner.pack(fill="both", expand=True, padx=10, pady=8)
        self.lbl_status = tk.Label(inner, text="טוען מנוע…", font=("Segoe UI", 10, "bold"),
                                   fg=AMBER, bg=CARD, anchor="e", justify="right")
        self.lbl_status.pack(fill="x")
        self.lbl_last = tk.Label(inner, text="", font=("Segoe UI", 10), fg="#cbd5e1", bg=CARD,
                                 anchor="e", justify="right", wraplength=W-64)
        self.lbl_last.pack(fill="x", pady=(3,0))
        self.lbl_enter = tk.Label(inner, text=f"אנטר אוטומטי אחרי {self.silence_enter_sec} שנ׳ שקט",
                                  font=("Segoe UI", 7), fg=MUT, bg=CARD, anchor="e")
        self.lbl_enter.pack(fill="x", pady=(3,0))

    # ── static canvas dressing (grid + LED ticks), visual only ──────────────
    def _on_wave_resize(self, e=None):
        try:
            cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
            self.canvas.delete("grid")
            if cw > 4 and ch > 4:
                for fy in (0.25, 0.75):
                    self.canvas.create_line(0, ch*fy, cw, ch*fy, fill=GRID, tags="grid")
                for fx in (0.25, 0.5, 0.75):
                    self.canvas.create_line(cw*fx, 0, cw*fx, ch, fill=GRID, tags="grid")
            self.canvas.tag_lower("grid")
        except Exception:
            pass

    def _on_level_resize(self, e=None):
        try:
            lw = self.level_canvas.winfo_width()
            self.level_canvas.delete("tick")
            for x in range(7, max(lw, 8), 7):
                self.level_canvas.create_line(x, 0, x, LV_H, fill=LV_BG, width=2, tags="tick")
            self.level_canvas.tag_raise("tick")
        except Exception:
            pass

    # ── devices ──────────────────────────────────────────────────────────────
    def _list_inputs(self):
        out = []
        try:
            apis = sd.query_hostapis()
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    api = apis[d["hostapi"]]["name"]
                    out.append((i, d["name"], api))
        except Exception:
            pass
        return out

    def _populate_devices(self):
        self._inputs = self._list_inputs()
        labels = [f"[{i}] {name}  ·{api}" for (i, name, api) in self._inputs]
        self.combo["values"] = labels or ["(אין מיקרופונים)"]
        # default = wired USB mic on the most stable host API (WASAPI > WDM-KS > MME).
        # MME throws "Unanticipated host error" + its indices shift with BT churn.
        def find(api_sub, name_sub):
            for (i, n, a) in self._inputs:
                if name_sub.lower() in n.lower() and api_sub.lower() in a.lower():
                    return i
            return None
        # Iddo's mic is literally named "USB Audio 1.0" (with a space) on MME and he
        # wants THAT one — the old search used "USBAudio" (no space) so it never matched
        # and fell through to device[0] (the headset). Match "usb audio" first, MME-preferred.
        default_idx = (find("MME", "USB Audio") or find("WASAPI", "USB Audio")
                       or find("WDM-KS", "USB Audio") or find("MME", "USB")
                       or find("WASAPI", "USBAudio") or find("WDM-KS", "USBAudio")
                       or find("MME", "USBAudio") or find("WASAPI", "Microphone"))
        if default_idx is None:
            try:
                r = pick_mic(query_devices_live(), prefer_names=["USBAudio"] + list(DEFAULT_PREFER_NAMES))
                if r.ok:
                    default_idx = r.device.index
            except Exception:
                pass
        if default_idx is None and self._inputs:
            default_idx = self._inputs[0][0]
        if default_idx is not None:
            for lbl, (i, _, _) in zip(labels, self._inputs):
                if i == default_idx:
                    self._mic_var.set(lbl); break
            self.engine.set_device(default_idx)

    def _on_pick_mic(self, _=None):
        sel = self._mic_var.get()
        try:
            idx = int(sel[1:sel.index("]")])
        except Exception:
            return
        self.engine.set_device(idx)
        self._set_status(f"מיק: {sel.split(']')[1].strip()}", ACC)
        self.root.after(40, self._no_activate)   # keep no-focus after combobox use

    # ── STT ────────────────────────────────────────────────────────────────────
    def _load_stt_async(self):
        def work():
            cfg = dict(getattr(self, "_stt_cfg", None) or
                       {"whisper_model": "medium", "whisper_device": "cuda",
                        "language": "auto", "no_speech_threshold": 0.8})
            self._stt = WhisperSTT(cfg)
            self._stt.load()
            self._model_ready = True
            self.root.after(0, self._set_idle_status)
        threading.Thread(target=work, daemon=True, name="clod-stt-load").start()

    def _on_utterance(self, audio16k, sr):
        """Capture-thread callback. Transcribe, then route by mode/state.
        Typing/focus happen here (off the UI thread); UI updates via root.after."""
        if not self._model_ready or self._stt is None:
            return
        try:
            if getattr(self, "_audio_clean", True):
                audio16k = audio_clean.clean(audio16k, sr)   # AGC + denoise -> better Hebrew
            text, _ = self._stt.transcribe(audio16k, sr)
        except Exception as e:
            self.root.after(0, lambda: self._set_status(f"שגיאת STT: {e}", RED)); return
        _dbg(f"[utt] mode={self._mode} dict={self._dictating} -> {text!r}")
        if not text:
            return

        # HaikuBrainz — smart autocorrect per utterance (Iddo's choice: smartest, ~10s).
        # ONLY when armed/always (Iddo: "בהתחלה עד שמתחמש אל תפעיל מוח הייקו") — never
        # waste 10s polishing a wake-word-detection utterance.
        if getattr(self, "_haiku_brainz", False) and (self._dictating or self._mode == "always"):
            self.root.after(0, lambda: self._set_status("🧠 מלטש (HaikuBrainz)…", AMBER))
            fixed = haiku_correct.correct(text, getattr(self, "_hb_context", ""))
            if fixed:
                text = fixed
            self._hb_context = ((getattr(self, "_hb_context", "") + " " + text)[-400:])
            _dbg(f"[haiku] -> {text!r}")

        # teachable corrections — deterministic overrides win (learns Iddo's words)
        text = corrections.fix(text)

        if self._mode == "always":
            self._do_type(text); return

        # ── wake-word mode: ARMED — keep typing turn after turn until stop phrase ──
        if self._dictating:
            prefix, stopped = self._stop_split(text)
            submit = False
            if not stopped:
                prefix, submit = self._submit_split(prefix)   # 'קלוד שלח' / 'send' -> Enter
            if prefix.strip():
                self._do_type(prefix)
            if stopped:
                _dbg(f"[armed] stop phrase heard -> disarm (not typed): {text!r}")
                self._stop_dictating()
            else:
                if submit:
                    _dbg(f"[armed] submit phrase heard -> Enter (stay armed): {text!r}")
                    self.root.after(140, self._submit_now)
                self.root.after(0, self._reschedule_disarm)
            return

        # armed — look for an enabled wake-word at the start of the utterance
        for kw in self._keywords:
            if not kw.get("enabled"):
                continue
            rem = _wake_remainder(text, kw["phrase"])
            if rem is None:
                continue
            if kw["action"] == "dictate":
                self._start_dictating()
                if rem.strip():
                    self._do_type(rem)
            elif kw["action"] == "assistant":
                self.root.after(0, lambda r=rem: self._route_assistant(r))
            elif kw["action"] == "council":
                self.root.after(0, lambda r=rem: self._route_council(r))
            return
        self.root.after(0, lambda: self._set_status("⚪ ממתין ל‏'קלוד'…", ARMED))

    def _do_type(self, text):
        self._focus_target()
        typer_mod.type_text(text + " ")
        self._typed_since_enter = True
        self.root.after(0, lambda t=text: self._after_typed(t))

    def _after_typed(self, text):
        self.lbl_last.config(text=text[-60:])
        if self._mode == "always" or self._dictating:
            self._set_status("🎙️ מקליד", GREEN)
        self._reschedule_enter()

    # ── wake-word state machine ─────────────────────────────────────────────
    def _start_dictating(self):
        self._dictating = True
        if self._dictate_to_claude:
            hwnd, title = self._find_claude_window()
            if hwnd:
                self._target_hwnd = hwnd
                self._target_title = title
                _dbg(f"[wake] locked target -> Claude window {title!r}")
        self._beep(1180)
        self.root.after(0, lambda: self._set_status("🎙️ חמוש — מקליד הכל (אמור 'קלוד סוף' לעצירה)", GREEN))
        self.root.after(0, self._reschedule_disarm)

    def _stop_dictating(self):
        self._dictating = False
        self._beep(560)
        self.root.after(0, self._cancel_disarm)
        self.root.after(0, self._set_idle_status)

    def _reschedule_disarm(self):
        self._cancel_disarm()
        # dictate_timeout_sec <= 0 → ARMED MODE: stay armed forever, only a stop
        # phrase ('קלוד סוף' / 'שירי סוף' / 'סוף הקלטה') disarms.
        if self._dictate_timeout > 0:
            self._disarm_after = self.root.after(int(self._dictate_timeout * 1000), self._stop_dictating)

    def _cancel_disarm(self):
        if self._disarm_after:
            try: self.root.after_cancel(self._disarm_after)
            except Exception: pass
            self._disarm_after = None

    def _route_assistant(self, remainder):
        try:
            import sys
            p = os.environ.get("CLOD_ASSISTANT_DIR", "")  # optional external assistant integration; unset = disabled
            if not p:
                self._set_status("שירי: assistant disabled (set CLOD_ASSISTANT_DIR)", ROUTE)
                return
            if p not in sys.path:
                sys.path.insert(0, p)
            from assistant_brain import respond
            from verify_gate import confirm_action
            q = (remainder or "").strip()
            if not q:
                self._set_status("🔵 שירי: מה תרצה?", ROUTE)
                return
            reply = respond(q, trusted=True)
            self._set_status(f"שירי: {reply[:70]}", ROUTE)
            _dbg(f"[שירי] q={q} reply={reply[:200]}")
            if any(k in q.lower() for k in ("send", "שלח", "call", "התקשר", "book", "קבע")):
                if confirm_action(reply, target="contact", channel="assistant"):
                    self._set_status("שירי: ✓ אושר", GREEN)
                else:
                    self._set_status("שירי: מבוטל", RED)
            if any(k in q.lower() for k in ("ram", "led", "light", "rgb", "נורות", "ramstick")):
                try:
                    import sys
                    p = os.environ.get("CLOD_ASSISTANT_DIR", "")  # optional external assistant integration; unset = disabled
                    if not p: return
                    if p not in sys.path: sys.path.insert(0, p)
                    from lights import kill_ram_leds
                    from verify_gate import confirm_action
                    if confirm_action("Disable all RAM stick RGB LEDs (dark/quiet/sleep guardian mode)", target="PC hardware", channel="lights"):
                        res = kill_ram_leds()
                        self._set_status(f"שירי: {res[:60]}", GREEN)
                    else:
                        self._set_status("שירי: lights unchanged", RED)
                except Exception as e:
                    self._set_status(f"lights error: {e}", RED)
        except Exception as e:
            self._set_status(f"שירי שגיאה: {e}", RED)

    def _route_council(self, remainder):
        self._set_status(f"גרוק: multi-AI council (QUORUM :8099) {remainder[:30] if remainder else ''}", ROUTE)
        _dbg(f"[גרוק] remainder={remainder}")

    def _is_stop(self, text):
        return self._stop_split(text)[1]

    def _stop_split(self, text):
        """Detect a stop phrase ('קלוד סוף' / 'שירי סוף' / 'סוף הקלטה' …) at the END
        of an utterance (fuzzy, Levenshtein-gated). Returns (prefix_to_type, stopped).
        The stop phrase itself is never typed; speech before it still is."""
        twords = _norm(text).split()
        owords = (text or "").strip().split()
        for p in self._stop_phrases:
            pn = _norm(p)
            pw = pn.split()
            if not pw or len(twords) < len(pw):
                continue
            tail = " ".join(twords[-len(pw):])
            if _lev(tail, pn) <= max(1, int(len(pn) * 0.25)):
                prefix = " ".join(owords[:-len(pw)]) if len(owords) > len(pw) else ""
                return prefix, True
        return text, False

    def _submit_split(self, text):
        """Detect a submit phrase ('קלוד שלח' / 'send it' …) at the END of an utterance
        (fuzzy). Returns (prefix_to_type, submit). The submit phrase itself isn't typed."""
        twords = _norm(text).split()
        owords = (text or "").strip().split()
        for p in self._submit_phrases:
            pn = _norm(p)
            pw = pn.split()
            if not pw or len(twords) < len(pw):
                continue
            tail = " ".join(twords[-len(pw):])
            if _lev(tail, pn) <= max(1, int(len(pn) * 0.25)):
                prefix = " ".join(owords[:-len(pw)]) if len(owords) > len(pw) else ""
                return prefix, True
        return text, False

    def _set_idle_status(self):
        if self._mode == "wakeword":
            self._set_status("⚪ ממתין ל‏'קלוד'…", ARMED)
        else:
            self._set_status("🎙️ מצב רצוף — מקליד הכל", GREEN)

    def _beep(self, hz):
        if winsound is None:
            return
        threading.Thread(target=lambda: winsound.Beep(int(hz), 120), daemon=True).start()

    # ── keyword registry ────────────────────────────────────────────────────
    def _load_keywords(self):
        try:
            with open(KEYWORDS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            self._mode = data.get("mode", "wakeword")
            self._dictate_timeout = int(data.get("dictate_timeout_sec", 30))
            self._stop_phrases = data.get("stop_phrases") or self._stop_phrases
            self._submit_phrases = data.get("submit_phrases") or self._submit_phrases
            self._keywords = data.get("keywords") or []
            self._dictate_to_claude = bool(data.get("dictate_to_claude", True))
            self._claude_patterns = data.get("claude_window_patterns") or self._claude_patterns
            self._audio_clean = bool(data.get("audio_clean", True))
            self._haiku_brainz = bool(data.get("haiku_brainz", False))
            self._stt_cfg = {
                "whisper_model": data.get("whisper_model", "medium"),
                "whisper_device": data.get("whisper_device", "cuda"),
                "language": data.get("language", "auto"),
                "no_speech_threshold": float(data.get("no_speech_threshold", 0.8)),
            }
        except Exception:
            self._keywords = [{"phrase": "קלוד", "action": "dictate", "enabled": True}]

    def _refresh_keys_label(self):
        on = [k["phrase"] for k in self._keywords if k.get("enabled")]
        self.lbl_keys.config(text=("מילים: " + " · ".join(on)) if on else "—")
        if hasattr(self, "btn_mode"):
            if self._mode == "always":
                self._restyle(self.btn_mode, "∞ רצוף", BTN_ALWAYS, BTN_ALWAYS_H)
            else:
                self._restyle(self.btn_mode, "● מילת-הפעלה", BTN_MODE, BTN_MODE_H)

    def _toggle_haiku(self):
        """Toggle HaikuBrainz polish on/off live (Iddo wanted a button to turn it off)."""
        self._haiku_brainz = not getattr(self, "_haiku_brainz", False)
        on = self._haiku_brainz
        self._restyle(self.btn_haiku, "🧠 הייקו" if on else "🧠 הייקו ✕",
                      BTN_ON if on else BTN_OFF, BTN_ON_H if on else BTN_OFF_H)
        self._set_status("🧠 ליטוש Haiku פעיל" if on else "ליטוש Haiku כבוי (raw מהיר)",
                         GREEN if on else AMBER)

    def _toggle_mode(self):
        self._mode = "always" if self._mode == "wakeword" else "wakeword"
        if self._mode == "wakeword":
            self._restyle(self.btn_mode, "● מילת-הפעלה", BTN_MODE, BTN_MODE_H)
            self._dictating = False
        else:
            self._restyle(self.btn_mode, "∞ רצוף", BTN_ALWAYS, BTN_ALWAYS_H)
        self._set_idle_status()

    # ── auto-Enter after N seconds of silence — DELETED (Iddo 2026-06-10) ───────
    # A wall-clock silence timer must NEVER submit on its own: it fired Enter into the
    # focused window (Claude Code) with no live mic turn. Submit happens ONLY on the
    # explicit spoken phrase 'קלוד שלח'/'send' via _submit_now. Permanent no-op that
    # also cancels any stray pending timer.
    def _reschedule_enter(self):
        if self._enter_after:
            try: self.root.after_cancel(self._enter_after)
            except Exception: pass
            self._enter_after = None

    def _ensure_claude_target(self):
        """Re-resolve the Claude window right before submitting. After a 5s pause the
        locked target can be stale/minimized, so Enter would go nowhere — re-find it."""
        if not self._dictate_to_claude:
            return
        try:
            low = (self._target_title or "").lower()
            has = self._target_hwnd and any(p.lower() in low for p in self._claude_patterns)
            if not has:
                hwnd, title = self._find_claude_window()
                if hwnd:
                    self._target_hwnd = hwnd
                    self._target_title = title
                    _dbg(f"[enter] re-resolved Claude target -> {title!r}")
        except Exception:
            pass

    def _fire_enter(self):
        # DELETED (Iddo 2026-06-10): silence-timer auto-Enter removed. Hard no-op so any
        # stale scheduled callback can never inject a global Enter. Submit only via the
        # explicit spoken phrase path (_submit_now).
        self._enter_after = None
        return

    def _submit_now(self):
        """Force-submit immediately (voice command 'קלוד שלח' / 'send') — stays armed."""
        self._ensure_claude_target()
        m = typer_mod.submit_to_claude(self._target_hwnd, focus_cb=self._focus_target)
        _dbg(f"[send] submit_to_claude -> {m}")
        self._typed_since_enter = False
        if self._enter_after:
            try: self.root.after_cancel(self._enter_after)
            except Exception: pass
            self._enter_after = None
        self._set_status("↵ נשלח (פקודה קולית)", AMBER)

    # ── controls ────────────────────────────────────────────────────────────────
    def _toggle_minimize(self):
        """Collapse the widget to just its title bar (borderless window, so native
        iconify doesn't apply). Click ➖ again to restore."""
        mini = not getattr(self, "_minimized", False)
        self._minimized = mini
        try:
            if mini:
                self._body.pack_forget()
                self.root.geometry(f"{W}x{TITLE_H}")
                self.btn_min.config(text="🔼")
            else:
                self._body.pack(fill="both", expand=True, padx=14, pady=(10,12))
                self.root.geometry(f"{W}x{H}")
                self.btn_min.config(text="—")
        except Exception as e:
            _dbg(f"[minimize] {e}")

    def _toggle_listen(self):
        self.engine.listening = not self.engine.listening
        if self.engine.listening:
            self._restyle(self.btn_listen, "‖ השהה", BTN_ON, BTN_ON_H); self._set_status("🎙️ מאזין", GREEN)
        else:
            self._restyle(self.btn_listen, "▶ הפעל", BTN_OFF, BTN_OFF_H); self._set_status("מושהה", RED)

    _LANGS = [("he","HE"),("en","EN"),("auto","AUTO")]
    def _cycle_lang(self):
        cur = self._stt._language if self._stt else "he"
        i = next((k for k,(c,_) in enumerate(self._LANGS) if c==cur), 0)
        code, label = self._LANGS[(i+1) % len(self._LANGS)]
        if self._stt: self._stt.reload_config({"language": code, "no_speech_threshold": 0.8})
        self.btn_lang.config(text=label)

    def _set_status(self, text, color):
        self.lbl_status.config(text=text, fg=color)

    def _on_block(self, block):
        n = len(block)
        with self._wave_lock:
            self._wave = np.roll(self._wave, -n)
            self._wave[-n:] = block

    # ── draw loop (waveform + level) ────────────────────────────────────────────
    def _draw_loop(self):
        try:
            self._track_target()
            if self._target_title != self._last_shown_title:
                self._last_shown_title = self._target_title
                tgt = (self._target_title[:28] + "…") if len(self._target_title) > 28 else self._target_title
                self.lbl_enter.config(text=f"→ יעד: {tgt or '—'}   ·  אנטר אוטו׳ {self.silence_enter_sec}שנ׳")
            cw = self.canvas.winfo_width(); ch = self.canvas.winfo_height()
            if cw > 4 and ch > 4:
                with self._wave_lock:
                    w = self._wave.copy()
                pts_n = min(cw, 380)
                seg = max(1, len(w)//pts_n)
                ds = w[::seg][:pts_n]
                mid = ch/2; amp = ch*0.46
                coords = []
                for x, v in enumerate(ds):
                    coords += [x*cw/len(ds), mid - float(v)*amp]
                self.canvas.delete("mid")
                self.canvas.create_line(0, mid, cw, mid, fill=MIDL, tags="mid")
                self.canvas.delete("wave")
                if len(coords) >= 4:
                    # layered glow: wide dark indigo -> mid indigo -> bright core
                    self.canvas.create_line(*coords, fill=WAVE_GLOW, width=5, tags="wave", smooth=True)
                    self.canvas.create_line(*coords, fill=WAVE_GLOW2, width=2, tags="wave", smooth=True)
                    self.canvas.create_line(*coords, fill=WAVE, width=1, tags="wave", smooth=True)
            # level bar + decaying peak-hold
            lw = self.level_canvas.winfo_width()
            if lw > 2:
                norm = min(1.0, self.engine.level_rms / 0.25)
                color = GREEN if norm < 0.6 else (AMBER if norm < 0.9 else RED)
                self.level_canvas.coords(self._level_bar, 0, 0, int(lw*norm), LV_H)
                self.level_canvas.itemconfig(self._level_bar, fill=color)
                self._lv_peak = max(norm, self._lv_peak * 0.96)
                px = int(lw * self._lv_peak)
                self.level_canvas.coords(self._level_peak, max(0, px-2), 0, px, LV_H)
                self.level_canvas.tag_raise("tick")
            if self.engine.last_error:
                self._set_status(f"⚠ {self.engine.last_error}", RED)
        except Exception:
            pass
        self.root.after(33, self._draw_loop)

    def _close(self):
        try: self.engine.stop()
        except Exception: pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    sec = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ClodWidget(silence_enter_sec=sec).run()
