"""router.py - voice-controlled dictation TARGET routing for the Hebrew mic.

Two jobs:

1) Parse the wake-phrase "שירי" (Shiri) + a routing command out of a transcript,
   e.g.  "שירי עכשיו הקלדה בקלוד"  ->  switch target to the Claude Code window.
   The routing command itself is CONSUMED (never typed as text).

2) Resolve a logical target ("claude" / "eran" / "whatsapp" / "notepad") to a real
   top-level window (by title / process name) and focus it, so dictation is typed
   ONLY into that window — never leaked into whatever happens to be focused.

Targets are matched by window-title substrings + process names (config-driven).

לזכרה של עלמה ז"ל - iddoperez.ai
"""
from __future__ import annotations

import ctypes
import logging
import re
from ctypes import wintypes

logger = logging.getLogger("voiceinput.router")

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# ── target definitions ───────────────────────────────────────────────────────
# logical name -> {title substrings, process exe substrings, Hebrew+EN aliases}
DEFAULT_TARGETS = {
    "claude": {
        "titles": ["claude", "claude code", "windows powershell", "powershell",
                   "cmd", "windows terminal"],
        "procs": ["windowsterminal.exe", "powershell.exe", "pwsh.exe",
                  "cmd.exe", "claude.exe", "conhost.exe"],
        "aliases": ["קלוד", "claude", "קולד", "כלוד", "קלוו"],
    },
    "eran": {
        "titles": ["eran", "ערן", "eran console"],
        "procs": ["python.exe", "eran.exe"],
        "aliases": ["ערן", "eran", "עירן"],
    },
    "whatsapp": {
        "titles": ["whatsapp", "וואטסאפ"],
        "procs": ["whatsapp.exe"],
        "aliases": ["וואטסאפ", "ווצאפ", "whatsapp", "ואטסאפ", "וטסאפ"],
    },
    "notepad": {
        "titles": ["notepad", "פנקס רשימות", "- notepad"],
        "procs": ["notepad.exe", "notepad++.exe"],
        "aliases": ["נוטפד", "notepad", "פנקס", "פנקס רשימות"],
    },
}

# Hebrew wake-word variants Whisper may produce for "שירי"
_WAKE = ["שירי", "שירה", "שיר י", "שירית", "שירל", "siri", "shiri", "שרי"]
# routing verb cues that precede the target ("הקלדה ב...", "תקלידי ל...", "מצב ...")
_ROUTE_CUE = re.compile(
    r"(?:עכשיו\s+)?(?:הקלד[הי]?|תקליד[יה]?|כתב[הי]?|תכתב[יה]?|מצב|עבר[יה]?\s+ל|"
    r"דבר[יה]?\s+ל|type|target|switch\s+to|go\s+to)",
    re.I)


# ── wake + route parsing ──────────────────────────────────────────────────────
def _norm(s: str) -> str:
    # strip nikud + collapse whitespace + drop trailing punctuation
    s = re.sub(r"[֑-ׇ]", "", s or "")
    return re.sub(r"\s+", " ", s).strip(" .,!?־-")


def parse_command(text: str, targets: dict | None = None):
    """If `text` begins with the wake-phrase 'שירי' AND names a known target,
    return (logical_target, leftover_text). leftover_text is anything the user
    said AFTER the routing command (usually '' — to be discarded, not typed).

    Returns (None, text) when this is NOT a routing command (normal dictation)."""
    targets = targets or DEFAULT_TARGETS
    t = _norm(text)
    low = t.lower()

    # must start with a wake variant within the first few words
    head = " ".join(low.split()[:2])
    if not any(w in head for w in _WAKE):
        return None, text

    # find which target alias appears
    chosen = None
    pos = -1
    for name, spec in targets.items():
        for alias in spec.get("aliases", []):
            a = _norm(alias).lower()
            # allow a glued Hebrew preposition prefix (ב/ל/ו/מ/ה) e.g. "בקלוד",
            # "לוואטסאפ" — Whisper writes these as one token with no boundary.
            m = re.search(r"(?:^|\s|[בלומה])" + re.escape(a) + r"(?:\W|$)", low)
            if m:
                # prefer the LAST cue+alias so "שירי עכשיו הקלדה בקלוד" wins
                if m.start() > pos:
                    chosen, pos = name, m.start()
    if not chosen:
        # wake word but no recognised target -> treat as command (consume),
        # don't type "שירי ..." as text
        logger.info("wake heard but no known target in: %r", t)
        return "__no_target__", ""

    # leftover after the alias = anything still meant as dictation (rare)
    after = t[pos + 1:]
    # drop the alias word itself + a leading 'ב'/'ל' preposition
    after = re.sub(r"^\W*", "", after)
    leftover = ""  # default: routing command line carries no dictation payload
    return chosen, leftover


def is_route_only(text: str, targets: dict | None = None) -> bool:
    tgt, _ = parse_command(text, targets)
    return tgt is not None


# ── window discovery + focus ──────────────────────────────────────────────────
def _proc_name(pid: int) -> str:
    PROCESS_QUERY_LIMITED = 0x1000
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED, False, pid)
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(512)
        size = wintypes.DWORD(512)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value.split("\\")[-1].lower()
    finally:
        kernel32.CloseHandle(h)
    return ""


def _enum_windows():
    out = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND,
                                         wintypes.LPARAM)

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        out.append((hwnd, buf.value, _proc_name(pid.value)))
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    return out


def find_window(target: str, targets: dict | None = None):
    """Return (hwnd, title) of the best window matching logical `target`, or None.
    Title-substring match is preferred; process-name match is a fallback."""
    targets = targets or DEFAULT_TARGETS
    spec = targets.get(target)
    if not spec:
        return None
    wins = _enum_windows()
    title_subs = [s.lower() for s in spec.get("titles", [])]
    procs = [p.lower() for p in spec.get("procs", [])]

    # 1) title substring match (strongest signal)
    for hwnd, title, proc in wins:
        tl = title.lower()
        if any(s in tl for s in title_subs):
            return hwnd, title
    # 2) process-name match
    for hwnd, title, proc in wins:
        if proc in procs:
            return hwnd, title
    return None


def focus_window(hwnd) -> bool:
    """Bring hwnd to the foreground (restore if minimised). Best-effort."""
    SW_RESTORE = 9
    try:
        user32.ShowWindow(hwnd, SW_RESTORE)
        # AttachThreadInput trick to beat foreground-lock
        fg = user32.GetForegroundWindow()
        cur_tid = kernel32.GetCurrentThreadId()
        tgt_tid = user32.GetWindowThreadProcessId(hwnd, None)
        fg_tid = user32.GetWindowThreadProcessId(fg, None)
        user32.AttachThreadInput(cur_tid, fg_tid, True)
        user32.AttachThreadInput(cur_tid, tgt_tid, True)
        user32.BringWindowToTop(hwnd)
        ok = bool(user32.SetForegroundWindow(hwnd))
        user32.AttachThreadInput(cur_tid, fg_tid, False)
        user32.AttachThreadInput(cur_tid, tgt_tid, False)
        return ok
    except Exception as e:  # noqa: BLE001
        logger.warning("focus_window failed: %s", e)
        return False


def focus_target(target: str, targets: dict | None = None):
    """Find + focus the window for a logical target. Returns (ok, title|err)."""
    found = find_window(target, targets)
    if not found:
        return False, f"no window for target {target!r}"
    hwnd, title = found
    ok = focus_window(hwnd)
    return ok, title


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    tests = [
        "שירי עכשיו הקלדה בקלוד",
        "שירי עכשיו הקלדה בערן",
        "שירי תקלידי בוואטסאפ",
        "שלום עידו מה שלומך",
        "שירי מצב נוטפד",
    ]
    for t in tests:
        tgt, leftover = parse_command(t)
        print(f"{t!r:45} -> target={tgt!r} leftover={leftover!r}")
    print("\n--- live windows for known targets ---")
    for name in DEFAULT_TARGETS:
        f = find_window(name)
        print(f"  {name:10} -> {f[1] if f else '(none)'}")
