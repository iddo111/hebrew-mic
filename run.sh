#!/usr/bin/env bash
# run.sh - launch the dictation engine (Linux/macOS). Note: keystroke injection in
# typer.py is Windows-only; on other OSes the STT + PWA server still run.
cd "$(dirname "$0")"
echo "Starting Hebrew dictation. HTTP control: http://127.0.0.1:7788/state"
python3 main.py
