# dictation/keytyper.py — Windows Unicode keystroke injection + auto-Enter (OURS)
# Uses SendInput with KEYEVENTF_UNICODE (0x0004) so Hebrew codepoints type correctly.
# Does NOT use the cp1252 path — that garbles non-ASCII. Pure ctypes, no external libs.
#
# Named keytyper (not typer) so it can never shadow the `typer` package on PyPI —
# importing this module must not change what `import typer` means for anything else
# in the environment.
#
# Importable on every OS: the Win32 surface is resolved lazily and every public
# entry point degrades to a logged no-op off Windows, so the STT engine and the
# PWA server can run on Linux/macOS (README promises exactly that).

import ctypes, time, os, sys, logging

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import ctypes.wintypes as wintypes
else:  # minimal stand-ins so the ctypes.Structure definitions below still build
    class _Types:
        WORD = ctypes.c_uint16
        DWORD = ctypes.c_uint32
        LONG = ctypes.c_int32

    wintypes = _Types()

log = logging.getLogger("dictation.keytyper")

UNSUPPORTED = "unsupported-platform"

# ── Win32 structs ─────────────────────────────────────────────────────────────
INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP   = 0x0002
VK_RETURN = 0x0D

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",         wintypes.WORD),
        ("wScan",       wintypes.WORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

# NOTE: the union must be sized for the LARGEST INPUT member (MOUSEINPUT), not just
# KEYBDINPUT. On 64-bit Windows MOUSEINPUT is 32 bytes (its trailing ULONG_PTR forces
# 8-byte alignment), so sizeof(INPUT) must be 40. A KEYBDINPUT-only union yields
# sizeof(INPUT)=32, which SendInput rejects with ERROR_INVALID_PARAMETER (87) and
# silently sends 0 events. Declaring the real MOUSEINPUT member lets ctypes size the
# union from the running ABI — 40 on Win64, 28 on 32-bit Python — instead of a
# hard-coded pad that is only right on one of them.
class _INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("_input", _INPUT_UNION)]

class _NoWin32:
    """Stand-in for user32 off Windows: every call is a logged no-op returning 0,
    so a stray code path can never raise AttributeError mid-utterance."""

    def __getattr__(self, name):
        def _noop(*_a, **_kw):
            log.debug("keytyper: %s() ignored — not running on Windows", name)
            return 0
        return _noop


user32 = ctypes.windll.user32 if IS_WINDOWS else _NoWin32()

# ── helpers ───────────────────────────────────────────────────────────────────
def _make_unicode_input(codepoint: int, keyup: bool) -> INPUT:
    flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if keyup else 0)
    ki = KEYBDINPUT(wVk=0, wScan=codepoint & 0xFFFF, dwFlags=flags, time=0, dwExtraInfo=None)
    inp = INPUT(type=INPUT_KEYBOARD, _input=_INPUT_UNION(ki=ki))
    return inp

def _make_vk_input(vk: int, keyup: bool, scan: int = 0) -> INPUT:
    # Include the hardware scan code: Chromium/Electron apps (Claude Desktop) read the
    # scan code off the event and IGNORE a pure virtual-key Enter (wScan=0). With the
    # real scan code present they treat it as a genuine keypress -> the message sends.
    flags = KEYEVENTF_KEYUP if keyup else 0
    ki = KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=None)
    inp = INPUT(type=INPUT_KEYBOARD, _input=_INPUT_UNION(ki=ki))
    return inp

def _send(inputs: list[INPUT]):
    arr = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        log.warning(f"SendInput: sent {sent}/{len(inputs)} events")

# ── public API ────────────────────────────────────────────────────────────────
def type_text(text: str, delay_ms: int = 0) -> None:
    """
    Inject `text` as Unicode keystrokes into the focused window.
    Handles BMP codepoints (covers all Hebrew + ASCII). Surrogate pairs for SMP chars.
    No-op off Windows.
    """
    if not text:
        return
    if not IS_WINDOWS:
        log.warning("keystroke injection needs Windows — not typing %d chars", len(text))
        return
    inputs: list[INPUT] = []
    for ch in text:
        cp = ord(ch)
        if cp <= 0xFFFF:
            # BMP: single Unicode keystroke
            inputs.append(_make_unicode_input(cp, False))
            inputs.append(_make_unicode_input(cp, True))
        else:
            # SMP (emoji etc.): encode as surrogate pair
            cp -= 0x10000
            high = 0xD800 + (cp >> 10)
            low  = 0xDC00 + (cp & 0x3FF)
            for surr in (high, low):
                inputs.append(_make_unicode_input(surr, False))
                inputs.append(_make_unicode_input(surr, True))
        if delay_ms:
            time.sleep(delay_ms / 1000)
    # send in batches of 64 to avoid SendInput limit on some systems
    batch = 64
    for i in range(0, len(inputs), batch):
        _send(inputs[i:i + batch])

SC_RETURN = 0x1C  # hardware scan code for Enter
KEYEVENTF_EXTENDEDKEY = 0x0001

def press_enter() -> None:
    """Send a REAL Enter keypress (vk + scan code) so Electron/Chromium apps
    (Claude Desktop) actually submit. Down, brief gap, up — like a physical key.

    Hardening (KF_REPEAT bug): Chromium drops a VK_RETURN whose lparam carries the
    key-repeat flag, which SendInput sets if Enter's async down-state is stale. So we
    first clear any lingering down-state, then send a clean down/up pair. The scan code
    is included because CEF reads it; KEYEVENTF_EXTENDEDKEY makes it an unambiguous Enter."""
    if not IS_WINDOWS:
        log.warning("press_enter needs Windows — ignored")
        return
    # 1) clear stale physical/async down-state so the synthesized event is not a repeat
    try:
        if user32.GetAsyncKeyState(VK_RETURN) & 0x8000:
            _send([_make_vk_input(VK_RETURN, True, SC_RETURN)])
            time.sleep(0.02)
    except Exception:
        pass
    # 2) clean down + up pair (SendInput, scan-code)
    _send([_make_vk_input(VK_RETURN, False, SC_RETURN)])
    time.sleep(0.02)
    _send([_make_vk_input(VK_RETURN, True, SC_RETURN)])
    # 3) ALSO fire the legacy keybd_event Enter — the most broadly-compatible path;
    #    some composers (Claude Code app, non-maximized) accept this when SendInput
    #    alone doesn't register. Harmless double-Enter is avoided because the composer
    #    is empty after the first submit (a 2nd Enter on an empty box is a no-op).
    try:
        KEYEVENTF_KEYUP_ = 0x0002
        user32.keybd_event(VK_RETURN, SC_RETURN, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(VK_RETURN, SC_RETURN, KEYEVENTF_KEYUP_, 0)
    except Exception:
        pass


# ── caret-focus click + robust submit pipeline ─────────────────────────────────
# Why: UIAutomation can't see inside the Claude Desktop CEF renderer (verified), and
# the Cowork composer has NO send button — Enter submits. Window-foreground != textarea
# caret-focus, so after an idle pause a bare Enter lands on a drifted element and does
# nothing. A real left-click into the input box restores DOM caret focus reliably.

MOUSEEVENTF_MOVE      = 0x0001
MOUSEEVENTF_LEFTDOWN  = 0x0002
MOUSEEVENTF_LEFTUP    = 0x0004
MOUSEEVENTF_ABSOLUTE  = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
INPUT_MOUSE = 0
SM_XVIRTUALSCREEN = 76; SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78; SM_CYVIRTUALSCREEN = 79

class _MINPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("padding", ctypes.c_byte * 24)]

class MINPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("_input", _MINPUT_UNION)]

def _set_dpi_aware():
    """Per-monitor DPI awareness so GetWindowRect / cursor math match real pixels."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

def _click_screen(x: int, y: int) -> None:
    """Absolute left-click at virtual-desktop screen coords (x, y), normalized 0..65535."""
    vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    if vw <= 1 or vh <= 1:
        return
    nx = int((x - vx) * 65535 / (vw - 1))
    ny = int((y - vy) * 65535 / (vh - 1))
    base = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    for fl in (base, base | MOUSEEVENTF_LEFTDOWN, base | MOUSEEVENTF_LEFTUP):
        mi = MOUSEINPUT(dx=nx, dy=ny, mouseData=0, dwFlags=fl, time=0, dwExtraInfo=None)
        inp = MINPUT(type=INPUT_MOUSE, _input=_MINPUT_UNION(mi=mi))
        arr = (MINPUT * 1)(inp)
        user32.SendInput(1, arr, ctypes.sizeof(MINPUT))
        time.sleep(0.02)

def _input_box_point(hwnd: int):
    """Compute a screen point inside the Claude composer input box for the given window.

    The CEF renderer doesn't expose the textbox to UIA, so we target it geometrically:
    horizontally centered, vertically pinned a fixed fraction up from the window bottom
    (the composer sits ~120-200px above the bottom on a maximized 4K window). This adapts
    to any window size/position instead of hard-coding session-specific coordinates."""
    class RECT(ctypes.Structure):
        _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                    ("right", wintypes.LONG), ("bottom", wintypes.LONG)]
    r = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        return None
    w = r.right - r.left
    h = r.bottom - r.top
    if w <= 0 or h <= 0:
        return None
    cx = r.left + w // 2
    # Composer input row sits low; ~7% of height up from the bottom lands in the textbox,
    # clear of the toolbar row beneath it. Clamp so a small/restored window still hits text.
    off = max(70, min(int(h * 0.07), 160))
    cy = r.bottom - off
    return (cx, cy)

def focus_input_box(hwnd: int) -> bool:
    """Give the Claude composer textbox real DOM caret focus via a left-click inside it.
    Returns True if a click was issued. This is the durable fix for idle caret-drift."""
    if not hwnd:
        return False
    _set_dpi_aware()
    pt = _input_box_point(hwnd)
    if not pt:
        return False
    _click_screen(pt[0], pt[1])
    time.sleep(0.05)
    return True


# ── UIAutomation Invoke fallback (only if a real Send button is exposed) ───────────
# Claude Desktop's Cowork composer has no send button and exposes no a11y tree, so this
# almost never fires — but for a future build / a different Claude window that DOES expose
# a "Send"/"Send Message" button, Invoke() submits position-independently. Degrades to
# False silently if comtypes/UIA or the button is unavailable (no hard dep).
def _uia_invoke_send(hwnd: int) -> bool:
    try:
        import comtypes.client
        UIA = comtypes.client.GetModule("UIAutomationCore.dll")
        from comtypes.gen.UIAutomationClient import (
            IUIAutomation, IUIAutomationInvokePattern, CUIAutomation,
        )
    except Exception:
        return False
    try:
        uia = comtypes.client.CreateObject(CUIAutomation, interface=IUIAutomation)
        elem = uia.ElementFromHandle(hwnd) if hwnd else uia.GetRootElement()
        if not elem:
            return False
        UIA_NamePropertyId = 30005
        TreeScope_Descendants = 4
        for name in ("Send Message", "Send", "שלח"):
            try:
                cond = uia.CreatePropertyCondition(UIA_NamePropertyId, name)
                btn = elem.FindFirst(TreeScope_Descendants, cond)
            except Exception:
                btn = None
            if not btn:
                continue
            try:
                pat = btn.GetCurrentPattern(UIA.UIA_InvokePatternId)
                pat.QueryInterface(IUIAutomationInvokePattern).Invoke()
                return True
            except Exception:
                continue
    except Exception:
        return False
    return False


# The geometric caret-click is OFF by default. VERIFIED 2026-06-10 (screen recording):
# _input_box_point clicks the WINDOW's horizontal center, but in Claude Code the composer
# lives in the LEFT panel — so on a maximized/wide window the click lands in DEAD SPACE to
# the right of the textbox, BLURS it, and the following Enter submits nothing (dictation
# then piles up unsent). The robust path is: foreground the target (AttachThreadInput) +
# press Enter while the caret is still fresh from typing — no fragile coordinate guessing.
CLICK_CARET = False

def foreground_title() -> str:
    """Lower-cased title of the foreground window ('' when unavailable)."""
    if not IS_WINDOWS:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(user32.GetForegroundWindow(), buf, 256)
        return (buf.value or "").lower()
    except Exception:
        return ""


def title_matches(title: str, patterns) -> bool:
    """True if `title` contains any of `patterns` (case-insensitive)."""
    t = (title or "").lower()
    return any(p and p.lower() in t for p in (patterns or ()))


DEFAULT_SUBMIT_TITLES = ("claude",)


def type_text_guarded(text: str, blocklist=(), delay_ms: int = 0) -> str:
    """type_text, but refuses when the foreground window title is blocklisted.

    Lives here rather than in main.py so EVERY front-end — the local engine, the widget
    and the phone server — crosses the same gate. Returns 'typed' or 'skipped-blocklisted'."""
    if blocklist:
        fg = foreground_title()
        if title_matches(fg, blocklist):
            log.warning("foreground %r is blocklisted -> not typing", fg)
            return "skipped-blocklisted"
    type_text(text, delay_ms)
    return "typed"


def submit_to_claude(hwnd: int = 0, focus_cb=None,
                     allow_titles=DEFAULT_SUBMIT_TITLES, blocklist=()) -> str:
    """Robust submit into the Claude composer.

    Steps:
      1) Bring the Claude window to the foreground (caller supplies focus_cb — the
         widget's AttachThreadInput dance).
      2) Press Enter with scan code (down ~20ms up) + legacy keybd_event — submits in the
         composer. The caret is already in the textbox from the just-typed text, so NO
         coordinate click is needed (and the old click MISSED on Claude Code's left-panel
         layout — see CLICK_CARET note above).
      3) Fallback: UIAutomation Invoke on a Send button (no-op for the buttonless composer;
         works if a future/alternate Claude window exposes one).

    `allow_titles` is the whitelist of foreground-window title substrings that may
    receive the Enter (default: Claude only); `blocklist` always wins over it.

    Returns the method used: 'enter-only', 'click+enter', or 'uia-invoke'. Never raises."""
    if not IS_WINDOWS:
        log.warning("submit needs Windows — refusing Enter")
        return UNSUPPORTED
    method = "enter-only"
    try:
        if focus_cb:
            try:
                focus_cb()
            except Exception:
                pass
        # SAFETY (Iddo 2026-06-10): never a blind global Enter. Only submit when the REAL
        # foreground window is the intended target AND its title is whitelisted — else
        # refuse, so a stray Enter can never land in a terminal, an editor or a password box.
        fg = user32.GetForegroundWindow()
        if hwnd and fg != hwnd:
            log.warning("submit_to_claude: foreground %s != target %s -> refusing Enter", fg, hwnd)
            return "skipped-not-foreground"
        try:
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(fg, buf, 256)
            ftitle = (buf.value or "").lower()
        except Exception:
            ftitle = ""
        if title_matches(ftitle, blocklist):
            log.warning("submit_to_claude: foreground title %r is blocklisted -> refusing Enter", ftitle)
            return "skipped-blocklisted"
        if not title_matches(ftitle, allow_titles):
            log.warning("submit_to_claude: foreground title %r not in %s -> refusing Enter",
                        ftitle, tuple(allow_titles))
            return "skipped-not-foreground"
        if CLICK_CARET and hwnd and focus_input_box(hwnd):
            method = "click+enter"
        press_enter()
    except Exception:
        pass
    # Belt-and-suspenders fallback: if a real Send button exists, Invoke it too. Harmless
    # for the buttonless Cowork composer (returns False); decisive for a future button.
    try:
        if _uia_invoke_send(hwnd):
            method = "uia-invoke"
    except Exception:
        pass
    return method

def get_foreground_exe() -> str:
    """Return the exe name of the foreground window (for blocklist check)."""
    try:
        hwnd = user32.GetForegroundWindow()
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        import psutil
        return psutil.Process(pid.value).name().lower()
    except Exception:
        return ""
