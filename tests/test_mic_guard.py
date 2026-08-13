"""Mic firewall: a Bluetooth SCO mic must never be openable, whatever is asked for."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mic_guard import (  # noqa: E402
    DeviceEntry,
    assert_safe_to_open,
    fake_device_list,
    is_blocked,
    pick_mic,
    run_unit_tests,
)


@pytest.mark.parametrize("name", [
    "Headset (WH-1000XM4)", "WH-1000XM4 Hands-Free",
    "CABLE Output (VB-Audio Virtual Cable)", "Stereo Mix (Realtek)",
    "AudioRelay Source",
])
def test_blocked_devices(name):
    assert is_blocked(name)


@pytest.mark.parametrize("name", ["Logitech BRIO", "Microphone (USBAudio1.0)", "Scarlett Solo (USB)"])
def test_allowed_devices(name):
    assert not is_blocked(name)


def test_prefers_brio():
    r = pick_mic(fake_device_list())
    assert r.ok and "BRIO" in r.device.name


def test_falls_back_to_scarlett_without_brio():
    devs = [d for d in fake_device_list() if "BRIO" not in d.name]
    r = pick_mic(devs)
    assert r.ok and "Scarlett" in r.device.name


def test_all_blocked_returns_failure():
    r = pick_mic([DeviceEntry(0, "Headset (WH-1000XM4)", 1)])
    assert not r.ok and r.device is None


def test_no_inputs_returns_failure():
    r = pick_mic([DeviceEntry(0, "Headphones (WH-1000XM4)", 0)])
    assert not r.ok


def test_assert_safe_to_open_rejects_bluetooth():
    with pytest.raises(ValueError):
        assert_safe_to_open(DeviceEntry(1, "Headset (WH-1000XM4)", 1))


def test_assert_safe_to_open_allows_wired():
    assert_safe_to_open(DeviceEntry(5, "Logitech BRIO", 1))


def test_custom_blocklist_overrides_preference():
    r = pick_mic([DeviceEntry(0, "Logitech BRIO", 1), DeviceEntry(1, "Scarlett Solo", 1)],
                 blocklist=["BRIO"])
    assert r.ok and "Scarlett" in r.device.name


def test_builtin_self_test_still_passes():
    assert run_unit_tests() == 0
