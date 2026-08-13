# dictation/config.py — configuration loader/saver (OURS)
import json, os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "dictation.json")

DEFAULTS = {
    "auto_enter": False,          # SAFETY (Iddo 2026-06-10): never auto-submit unless explicitly enabled
    "language": "he",             # "auto" | "he" | "en"  — default Hebrew
    "hotkey": "ctrl+shift+d",
    "ptt_hotkey": "ctrl+shift+space",  # hold for PTT mode
    "mode": "ptt",                # SAFETY: push-to-talk by default; "always_on" only if user opts in
    "mic_device": None,           # None = mic_guard picks by name
    "mic_prefer_name": "USBAudio1.0",  # wired USB mic (BRIO not detected, fallback)
    "mic_blocklist": [            # device name substrings that are FORBIDDEN
        "WH-1000XM4", "Headset", "Hands-Free", "Loopback", "Virtual", "AudioRelay",
    ],
    "whisper_model": "medium",
    "whisper_device": "cuda",
    "silence_ms": 900,
    "energy_threshold": 0.005,    # RMS threshold (lower = more sensitive)
    "software_gain": 14.0,        # Strong gain for weak USB mic (AB13X/USBAudio1.0)
    # DSP chain (applied in capture before STT)
    "dsp_enabled": True,
    "dsp_hpf_hz": 110.0,          # High-pass cutoff
    "dsp_notch_freqs": [50, 100, 150],  # Hum notch frequencies (Hz)
    "dsp_notch_q": 30.0,          # Notch filter Q
    "dsp_noise_gate": 0.01,       # Noise gate RMS threshold (post-gain)
    # STT quality
    "no_speech_threshold": 0.65,  # Tighter — reject more silence hallucinations
    "sample_rate": 16000,
    "http_port": 7788,
    "enter_delay": 0.07,          # pause between the last character and the Enter
    # Routing: "smart" = a leading "קלוד"/"גרוק" focuses that window first;
    # "focused" = never steal focus, always type into the foreground window.
    "default_route": "smart",
    # Window-title substrings. Enter is pressed ONLY when the foreground title matches
    # submit_title_patterns, and never when it matches auto_enter_blocklist.
    "submit_title_patterns": ["claude"],
    "auto_enter_blocklist": [],
    # Refuse to TYPE at all when the foreground title matches — password prompts,
    # banking, a lock screen. The Enter gate alone does not protect these.
    "type_blocklist": [],
    # Shared text/audio polish stage (postprocess.py), off by default:
    # audio_clean costs a little latency, haiku_brainz needs the Claude CLI.
    "audio_clean": False,
    "haiku_brainz": False,
    # The phone companion presses Enter after typing — tapping send on the phone is
    # the intent to submit. Set false for type-only. It still obeys the gates above.
    "phone_submit": True,
    "haiku_timeout": 12.0,
}

def load() -> dict:
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg

def save(cfg: dict) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
