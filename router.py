"""router.py — resolve a logical dictation target to a real window, and focus it.

Given a logical target ("claude" / "grok") this finds the matching top-level window
by title substring or process name and brings it to the foreground, so dictation is
typed ONLY into that window instead of leaking into whatever happened to be focused.

Wake-word parsing lives in routing.py (pure, testable, cross-platform); this module
is the Windows side-effect half. Windows-only: it is imported defensively by main.py,
which falls back to typing into the focused window when it can't be loaded.

לזכרה של עלמה ז"ל - iddoperez.ai
"""
from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes

logger = logging.getLogger("voiceinput.router")

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# ── target definitions ───────────────────────────────────────────────────────
# logical name -> {title substrings, process exe substrings}
# main.py passes its own table; this is the fallback for direct callers.
DEFAULT_TARGETS = {
    "claude": {
        "titles": ["claude", "claude code", "windows powershell", "powershell",
                   "cmd", "windows terminal"],
        "procs": ["windowsterminal.exe", "powershell.exe", "pwsh.exe",
                  "cmd.exe", "claude.exe", "conhost.exe"],
    },
    "grok": {
        "titles": ["grok"],
        "procs": ["grok.exe"],
    },
}


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
    print("--- live windows for known targets ---")
    for name in DEFAULT_TARGETS:
        f = find_window(name)
        print(f"  {name:10} -> {f[1] if f else '(none)'}")
