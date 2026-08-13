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
  (`keytyper.py`), with a UIAutomation paste fallback for stubborn apps. Windows can
  be refused per-window: `type_blocklist` stops dictation landing in a password prompt.
- **Smart window routing** (`router.py`): an utterance beginning with a wake word
  (`קלוד`/`claude`, `גרוק`/`grok`, plus common Whisper mishears) is focused into that
  app's window before typing; the wake word itself is consumed, not typed. Anything
  else goes to whatever window is already focused.
- **Smart auto-Enter (LKG behavior).** When `auto_enter` is on, Enter is pressed to
  submit **only** if the foreground window title matches `submit_title_patterns`
  (default: Claude only) and does not match `auto_enter_blocklist` — re-checked at submit
  time (`keytyper.submit_to_claude`). It's a per-utterance Enter, not a wall-clock timer,
  so there is no runaway auto-submit. When text is typed but the Enter is refused, the
  log says so explicitly rather than leaving you guessing.
- **Safe by default.** Push-to-talk mode (`ptt`) on by default: nothing is typed
  unless you hold the PTT hotkey. `always_on` is opt-in. A revived/idle process can
  never silently type. The armed state is sampled when you *start* speaking and travels
  with the utterance, so releasing the key doesn't swallow the sentence you just said;
  releasing it also flushes the utterance immediately instead of waiting out `silence_ms`.
- **Mic firewall** (`mic_guard.py`): Bluetooth headset / loopback / virtual devices
  are blocklisted so dictation always grabs your real wired mic.
- **Optional HaikuBrainz autocorrect** (`haiku_correct.py`): runs raw transcripts
  through the Claude CLI (`haiku`) to fix homophones, spelling-out, and AI model-name
  mishears — fails open (returns raw text) if the CLI is missing or slow. Enable with
  `haiku_brainz`.
- **Teachable corrections** (`corrections.json`): a simple `{wrong: right}` map applied
  after STT, so you can teach it your own vocabulary.
- **One polish stage for both front-ends** (`postprocess.py`): denoise/AGC, HaikuBrainz
  and the corrections map run identically whether you launch `main.py` or the widget.
- **Android PWA companion** (`android_mic.html` + `clod_mic_server.py`): open the page
  on your phone, speak, and the text is transcribed on the PC's Whisper and typed into
  the focused PC window with the same smart-Enter behavior. Token-protected — see
  [Security](#security).

## Requirements

- Python 3.10+
- A microphone (wired recommended)
- For GPU Whisper: an NVIDIA GPU + a CUDA-enabled `torch` build (CPU also works)
- **Keystroke injection (`keytyper.py`) is Windows-only.** The STT engine and the PWA
  server run cross-platform — `keytyper` imports fine everywhere and degrades to a
  logged no-op off Windows; auto-typing into other apps is implemented with Windows
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
# loopback by default — reaching it from the phone is a deliberate act
CLOD_MIC_HOST=100.x.y.z python clod_mic_server.py    # your Tailscale address
```

On startup the server prints the exact URL to open on the phone, token included:
`http://<host>:7789/?token=…`. Open it once and the phone remembers the token. Speak;
the audio is transcribed on the PC and typed into the focused PC window. Serving over
HTTPS (Tailscale Serve) is still the best route — browsers only grant mic access on a
secure origin.

## Security

Every endpoint on both servers can type into your machine and press Enter, so they all
require a shared token:

- The token is read from `$CLOD_MIC_TOKEN`, else from `.clod_token` (created 0600 on
  first run, git-ignored). `control.py` prints it at startup.
- `clod_mic_server.py` binds `127.0.0.1` unless you set `CLOD_MIC_HOST`. Prefer your
  Tailscale address over `0.0.0.0` — `0.0.0.0` hands the keyboard to the whole LAN.
- No wildcard CORS. The phone page is same-origin with its server; set
  `CLOD_MIC_ORIGIN` only if you deliberately serve the page from elsewhere.
- `control.py` binds loopback **and** requires the token: any web page in any browser
  tab on the machine can POST to `127.0.0.1`, so loopback alone is not authentication.
- The phone endpoints obey the same window gates as the local mic — `type_blocklist`,
  `submit_title_patterns` and `auto_enter_blocklist` — so a request off the network can
  never do more than the mic on your desk.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

No GPU, microphone or Windows needed — the suite covers the pure logic (routing, the
Hebrew hallucination filter, the mic firewall, the push-to-talk gate, corrections, the
auth token) and runs on every push via GitHub Actions.

## Configuration knobs (`dictation.json`)

| key | meaning |
|-----|---------|
| `mode` | `ptt` (hold to talk) or `always_on` |
| `auto_enter` | press Enter to submit — only fires when Claude is foreground |
| `whisper_model` / `whisper_device` | HF model id or local path / `cuda`\|`cpu` |
| `software_gain` | input gain for weak USB mics |
| `mic_prefer_name` / `mic_blocklist` | mic selection / Bluetooth-loopback firewall |
| `default_route` | `smart` enables the "קלוד…"/"גרוק…" window routing; `focused` disables it |
| `submit_title_patterns` | window titles allowed to receive the auto-Enter (default: `claude`) |
| `auto_enter_blocklist` / `type_blocklist` | window titles that may never be submitted to / typed into |
| `audio_clean` / `haiku_brainz` | enable the denoise+AGC and Claude-CLI autocorrect stages |
| `phone_submit` | the Android companion presses Enter after typing (set false for type-only) |

Wake words live in `clod_keywords.json` and are read by both front-ends
(`routing.load_aliases`); stop/submit phrases in the same file drive the widget's
armed-mode state machine.

## File map

| file | role |
|------|------|
| `main.py` | entry point: wires capture → STT → polish → typing + smart routing/Enter |
| `capture.py` | mic capture + DSP (HPF, hum-notch, noise gate, gain) |
| `stt.py` | faster-whisper wrapper + Hebrew hallucination filter |
| `keytyper.py` | Windows Unicode keystroke injection + title-gated submit |
| `routing.py` | wake-word parsing (pure, cross-platform, unit-tested) |
| `router.py` | resolve/focus a target window by title/process (Windows) |
| `postprocess.py` | the one polish stage shared by `main.py` and the widget |
| `authtoken.py` | shared bearer token for both HTTP servers |
| `control.py` | HTTP control server + tray icon |
| `config.py` | config loader/saver (`dictation.json`) |
| `mic_guard.py` | mic device firewall (rejects BT/loopback/virtual) |
| `haiku_correct.py` | optional Claude-CLI autocorrect |
| `corrections.py` / `corrections.json` | teachable `{wrong:right}` map |
| `clod_mic_server.py` / `android_mic.html` | phone PWA companion + server |
| `clod_widget.py` | optional Tk floating widget UI |
| `audio_clean.py` | optional auto-gain/denoise stage |
| `tests/` | pytest suite — no hardware required |

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
