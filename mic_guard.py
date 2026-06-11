"""
mic_guard.py — AUDIO LEGO-2: safe microphone selection helper.

Rules enforced:
  LEGO-1 / LEGO-4: any device whose name contains a blocked keyword is
    unconditionally REJECTED — regardless of what the caller requests.
  LEGO-2: pick a stable wired input by name (prefer_name list, BRIO first).
  LEGO-3: callers must open in WASAPI SHARED mode; this module does not
    open any devices itself — it only returns safe device metadata.

This module has NO side-effects: it never opens a mic, never captures audio,
never calls sounddevice.rec(). It is pure selection logic so it can be unit-
tested with a fake device list without touching any hardware.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Constants — mirrors dictation.json and AUDIO_LEGO rules
# ---------------------------------------------------------------------------

# Default blocklist: any device whose name contains one of these (case-insensitive)
# is FORBIDDEN as a capture device.
DEFAULT_BLOCKLIST: tuple[str, ...] = (
    "WH-1000XM4",   # Sony Bluetooth headset (A2DP music — never SCO mic)
    "Headset",       # generic BT HFP headset label
    "Hands-Free",    # HFP/HSP profile label
    "Handsfree",     # alternate spelling
    "HFP",           # explicit HFP tag
    "HSP",           # Headset Profile
    "Loopback",      # virtual loopback (e.g. VB-Cable, Voicemeeter aux)
    "Virtual",       # virtual audio device
    "AudioRelay",    # AudioRelay BT streaming app
    "Stereo Mix",    # Windows WASAPI loopback — capture of render, not a real mic
    "What U Hear",   # SoundBlaster loopback alias
)

# Preference order for wired mics (first match wins)
DEFAULT_PREFER_NAMES: tuple[str, ...] = (
    "Logitech BRIO",
    "BRIO",
    "Scarlett Solo",
    "Scarlett",
    "USBAudio",   # generic wired USB fallback
    "USB Audio",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DeviceEntry:
    """Minimal device descriptor — independent of any audio backend."""
    index: int
    name: str
    max_input_channels: int   # >0 means it CAN be a capture device


@dataclass
class GuardResult:
    ok: bool
    device: Optional[DeviceEntry]
    reason: str             # human-readable verdict


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def is_blocked(name: str, blocklist: Sequence[str] = DEFAULT_BLOCKLIST) -> bool:
    """Return True if the device name contains any blocked keyword (case-insensitive)."""
    nl = name.lower()
    return any(kw.lower() in nl for kw in blocklist)


def pick_mic(
    devices: Sequence[DeviceEntry],
    prefer_names: Sequence[str] = DEFAULT_PREFER_NAMES,
    blocklist: Sequence[str] = DEFAULT_BLOCKLIST,
) -> GuardResult:
    """
    Select the safest available capture device.

    Algorithm:
      1. Filter to input-capable devices only.
      2. Hard-exclude any device matching the blocklist.
      3. Among survivors, prefer by prefer_names order (first partial match wins).
      4. If no preferred name matches, fall back to the first unblocked input.
      5. If nothing survives, return ok=False with a reason.

    Returns a GuardResult — never raises, never opens any device.
    """
    inputs = [d for d in devices if d.max_input_channels > 0]

    if not inputs:
        return GuardResult(ok=False, device=None, reason="No input devices found")

    allowed = [d for d in inputs if not is_blocked(d.name, blocklist)]

    if not allowed:
        blocked_names = [d.name for d in inputs]
        return GuardResult(
            ok=False,
            device=None,
            reason=f"All {len(inputs)} input device(s) are blocked by blocklist: {blocked_names}",
        )

    # Priority scan: prefer_names in order
    for pref in prefer_names:
        pl = pref.lower()
        for d in allowed:
            if pl in d.name.lower():
                return GuardResult(
                    ok=True,
                    device=d,
                    reason=f"Matched preferred name '{pref}' -> '{d.name}'",
                )

    # Fallback: first unblocked input
    d = allowed[0]
    return GuardResult(
        ok=True,
        device=d,
        reason=f"Fallback to first unblocked input: '{d.name}'",
    )


def assert_safe_to_open(device: DeviceEntry, blocklist: Sequence[str] = DEFAULT_BLOCKLIST) -> None:
    """
    LEGO-4 guard: raise ValueError if the device is BT/loopback.
    Call this immediately before opening sd.InputStream so that a last-minute
    device-list change can't slip a bad device past the earlier pick_mic check.
    """
    if is_blocked(device.name, blocklist):
        raise ValueError(
            f"AUDIO LEGO-4: refused to open '{device.name}' — it matches the BT/loopback blocklist. "
            "Opening a Bluetooth SCO mic would destroy music playback. "
            "Use a wired device (BRIO / USBAudio1.0 / Scarlett)."
        )


# ---------------------------------------------------------------------------
# Convenience: build DeviceEntry list from sounddevice (optional import)
# ---------------------------------------------------------------------------

def query_devices_live() -> list[DeviceEntry]:
    """
    Return live device list from sounddevice.
    Import is deferred so this module can be unit-tested without hardware.
    DO NOT call this during unit tests — use fake_device_list() instead.
    """
    import sounddevice as sd  # noqa: PLC0415
    result = []
    for i, d in enumerate(sd.query_devices()):
        result.append(DeviceEntry(
            index=i,
            name=d["name"],
            max_input_channels=d["max_input_channels"],
        ))
    return result


# ---------------------------------------------------------------------------
# Unit test (offline — no hardware) — also runnable as __main__
# ---------------------------------------------------------------------------

def fake_device_list() -> list[DeviceEntry]:
    """Synthetic device list covering all cases; no real hardware needed."""
    return [
        DeviceEntry(0, "Headphones (WH-1000XM4)",        0),   # output only
        DeviceEntry(1, "Headset (WH-1000XM4)",            1),   # BLOCKED: XM4 SCO mic
        DeviceEntry(2, "WH-1000XM4 Hands-Free",           1),   # BLOCKED: HFP/SCO
        DeviceEntry(3, "Microphone (USB Audio)",           1),   # allowed (USB fallback)
        DeviceEntry(4, "Microphone (USBAudio1.0)",         1),   # allowed (USB named)
        DeviceEntry(5, "Logitech BRIO",                    1),   # PREFERRED
        DeviceEntry(6, "Scarlett Solo (USB)",               1),   # allowed (Scarlett)
        DeviceEntry(7, "CABLE Output (VB-Audio Virtual)",  1),   # BLOCKED: virtual loopback
        DeviceEntry(8, "Stereo Mix (Realtek)",             1),   # BLOCKED: loopback
        DeviceEntry(9, "AudioRelay Source",                1),   # BLOCKED: AudioRelay
        DeviceEntry(10,"Headset (SomeOtherBT)",            1),   # BLOCKED: generic BT headset
    ]


def run_unit_tests() -> int:
    """Run offline unit tests; return number of failures."""
    failures = 0
    devs = fake_device_list()

    def check(label: str, condition: bool, detail: str = ""):
        nonlocal failures
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
        if not condition:
            failures += 1

    print("\n=== mic_guard.py offline unit tests ===\n")

    # 1. is_blocked() basics
    check("XM4 Headset is blocked", is_blocked("Headset (WH-1000XM4)"))
    check("XM4 Hands-Free is blocked", is_blocked("WH-1000XM4 Hands-Free"))
    check("Loopback CABLE is blocked", is_blocked("CABLE Output (VB-Audio Virtual Cable)"))
    check("Stereo Mix is blocked", is_blocked("Stereo Mix (Realtek)"))
    check("AudioRelay is blocked", is_blocked("AudioRelay Source"))
    check("BRIO is NOT blocked", not is_blocked("Logitech BRIO"))
    check("USBAudio is NOT blocked", not is_blocked("Microphone (USBAudio1.0)"))
    check("Scarlett Solo is NOT blocked", not is_blocked("Scarlett Solo (USB)"))

    # 2. pick_mic() happy path — BRIO is present and preferred
    r = pick_mic(devs)
    check("pick_mic selects BRIO", r.ok and r.device is not None and "BRIO" in r.device.name, r.reason)
    check("pick_mic result is input-capable", r.ok and (r.device.max_input_channels > 0 if r.device else False))

    # 3. pick_mic() without BRIO — falls back to Scarlett
    no_brio = [d for d in devs if "BRIO" not in d.name]
    r2 = pick_mic(no_brio)
    check("pick_mic without BRIO falls back to Scarlett", r2.ok and r2.device is not None and "Scarlett" in r2.device.name, r2.reason)

    # 4. pick_mic() with only blocked devices
    blocked_only = [
        DeviceEntry(0, "Headset (WH-1000XM4)", 1),
        DeviceEntry(1, "WH-1000XM4 Hands-Free", 1),
        DeviceEntry(2, "Stereo Mix (Realtek)", 1),
    ]
    r3 = pick_mic(blocked_only)
    check("pick_mic with only blocked devices returns ok=False", not r3.ok, r3.reason)
    check("pick_mic with only blocked devices has no device", r3.device is None)

    # 5. pick_mic() with no input devices at all
    no_inputs = [DeviceEntry(0, "Headphones (WH-1000XM4)", 0)]
    r4 = pick_mic(no_inputs)
    check("pick_mic with no inputs returns ok=False", not r4.ok, r4.reason)

    # 6. assert_safe_to_open() blocks BT device
    bt_dev = DeviceEntry(1, "Headset (WH-1000XM4)", 1)
    try:
        assert_safe_to_open(bt_dev)
        check("assert_safe_to_open raises for BT", False, "expected ValueError, got none")
    except ValueError as e:
        check("assert_safe_to_open raises for BT", True, str(e)[:80])

    # 7. assert_safe_to_open() allows wired device
    wired_dev = DeviceEntry(5, "Logitech BRIO", 1)
    try:
        assert_safe_to_open(wired_dev)
        check("assert_safe_to_open allows BRIO", True)
    except ValueError as e:
        check("assert_safe_to_open allows BRIO", False, str(e)[:80])

    # 8. Custom blocklist overrides
    custom_bl = ["BRIO"]
    r5 = pick_mic([DeviceEntry(0, "Logitech BRIO", 1), DeviceEntry(1, "Scarlett Solo", 1)], blocklist=custom_bl)
    check("Custom blocklist excludes BRIO, selects Scarlett", r5.ok and r5.device is not None and "Scarlett" in r5.device.name, r5.reason)

    print(f"\n{'='*40}")
    if failures == 0:
        print(f"ALL TESTS PASSED (0 failures)")
    else:
        print(f"{failures} TEST(S) FAILED")
    print()
    return failures


if __name__ == "__main__":
    import sys
    sys.exit(run_unit_tests())
