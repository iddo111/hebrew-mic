# Hebrew Mic — always-on Hebrew voice dictation

Always-on Hebrew (and English) voice dictation that types straight into whatever
window you have focused. Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
with the [ivrit-ai](https://huggingface.co/ivrit-ai) Hebrew Whisper models. It adds
**smart routing** (say "קלוד…" to send the rest into the Claude window, "גרוק…" for
Grok), **smart auto-Enter** (it only presses Enter to submit when the foreground
window is Claude — never in a stray window), and an **Android PWA companion** so your
phone's mic can dictate into the PC over your LAN / Tailscale.

> Part of the [ALMAWARE](https://iddoperez.ai) toolset · לזכרה של עלמה ז״ל

## What it does

- **Hebrew-first STT.** Defaults to the `ivrit-ai/whisper-large-v3-turbo-ct2`
  CTranslate2 model; GPU (CUDA) or CPU. Aggressive hallucination filtering kills the
  classic Whisper "תודה רבה על הצפייה" / "thanks for watching" noise artifacts.
- **Types into the focused window** via Windows SendInput Unicode injection
  (`typer.py`), with a UIAutomation paste fallback for stubborn apps.
- **Smart window routing** (`router.py`): an utterance beginning with a wake word
  (`קלוד`/`claude`, `גרוק`/`grok`, plus common Whisper mishears) is focused into that
  app's window before typing; the wake word itself is consumed, not typed. Anything
  else goes to whatever window is already focused.
- **Smart auto-Enter (LKG behavior).** When `auto_enter` is on, Enter is pressed to
  submit **only** if the foreground window is a Claude window — re-checked at submit
  time (`typer.submit_to_claude`). It's a per-utterance Enter, not a wall-clock timer,
  so there is no runaway auto-submit.
- **Safe by default.** Push-to-talk mode (`ptt`) on by default: nothing is typed
  unless you hold the PTT hotkey. `always_on` is opt-in. A revived/idle process can
  never silently type.
- **Mic firewall** (`mic_guard.py`): Bluetooth headset / loopback / virtual devices
  are blocklisted so dictation always grabs your real wired mic.
- **Optional HaikuBrainz autocorrect** (`haiku_correct.py`): runs raw transcripts
  through the Claude CLI (`haiku`) to fix homophones, spelling-out, and AI model-name
  mishears — fails open (returns raw text) if the CLI is missing or slow.
- **Teachable corrections** (`corrections.json`): a simple `{wrong: right}` map applied
  after STT, so you can teach it your own vocabulary.
- **Android PWA companion** (`android_mic.html` + `clod_mic_server.py`): open the page
  on your phone, speak, and the text is transcribed on the PC's Whisper and typed into
  the focused PC window with the same smart-Enter behavior.

## Requirements

- Python 3.10+
- A microphone (wired recommended)
- For GPU Whisper: an NVIDIA GPU + a CUDA-enabled `torch` build (CPU also works)
- **Keystroke injection (`typer.py`) is Windows-only.** The STT engine and the PWA
  server run cross-platform; auto-typing into other apps is implemented with Windows
  SendInput.
- (Optional) the [Claude CLI](https://docs.claude.com/en/docs/claude-code) on PATH for
  HaikuBrainz autocorrect — set `CLAUDE_CLI` to override the binary path.

## Install

```bash
git clone https://github.com/iddo111/hebrew-mic
cd hebrew-mic
pip install -r requirements.txt
cp dictation.json.example dictation.json   # then tune for your machine
```

Edit `dictation.json`: set `whisper_device` to `cuda` or `cpu`, pick a
`whisper_model` (a Hugging Face faster-whisper / CT2 model id or a local folder path),
and choose `mode` (`ptt` or `always_on`).

## Usage

**Desktop dictation:**

```bash
python main.py        # or: run.cmd (Windows) / run.sh (Linux/macOS)
```

- In `ptt` mode, hold **Ctrl+Shift+Space** and speak; release to stop.
- In `always_on` mode it listens continuously (turn on via config or the HTTP control
  endpoint). Say **"קלוד …"** to route into Claude, **"גרוק …"** for Grok, or just
  speak to type into the focused window.
- Stop dictating by saying a stop phrase (e.g. "קלוד סגור מיקרופון").
- HTTP control + status: `http://127.0.0.1:7788/state`.

**Phone companion (Android PWA):**

```bash
python clod_mic_server.py     # or: start-clod-mic.cmd  (serves on 0.0.0.0:7789)
```

Open `http://<your-pc-ip>:7789/` on your phone (best over HTTPS via Tailscale Serve so
the browser grants mic access). Speak; the audio is transcribed on the PC and typed
into the focused PC window. For an installable PWA / HTTPS, point the page's server at
the PC's LAN or Tailscale address (see the comment at the top of `android_mic.html`).

## Configuration knobs (`dictation.json`)

| key | meaning |
|-----|---------|
| `mode` | `ptt` (hold to talk) or `always_on` |
| `auto_enter` | press Enter to submit — only fires when Claude is foreground |
| `whisper_model` / `whisper_device` | HF model id or local path / `cuda`\|`cpu` |
| `software_gain` | input gain for weak USB mics |
| `mic_prefer_name` / `mic_blocklist` | mic selection / Bluetooth-loopback firewall |
| `default_route` | `smart` enables the "קלוד…"/"גרוק…" window routing |

Wake words, stop/submit phrases and routing keywords live in `clod_keywords.json`.

## File map

| file | role |
|------|------|
| `main.py` | entry point: wires capture → STT → typer + smart routing/Enter |
| `capture.py` | mic capture + DSP (HPF, hum-notch, noise gate, gain) |
| `stt.py` | faster-whisper wrapper + Hebrew hallucination filter |
| `typer.py` | Windows Unicode keystroke injection + Claude-gated submit |
| `router.py` | resolve/focus a target window by title/process |
| `control.py` | HTTP control server + tray icon |
| `config.py` | config loader/saver (`dictation.json`) |
| `mic_guard.py` | mic device firewall (rejects BT/loopback/virtual) |
| `haiku_correct.py` | optional Claude-CLI autocorrect |
| `corrections.py` / `corrections.json` | teachable `{wrong:right}` map |
| `clod_mic_server.py` / `android_mic.html` | phone PWA companion + server |
| `clod_widget.py` | optional Tk floating widget UI |
| `audio_clean.py` | optional auto-gain/denoise stage |

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
