@echo off
REM run.cmd - launch always-on Hebrew/English dictation (main.py)
REM Paths are relative to this script (%~dp0), so the repo runs from anywhere.
echo Starting Hebrew dictation (HE+EN)...
echo Toggle hotkey: Ctrl+Shift+D   PTT: hold Ctrl+Shift+Space
echo HTTP control:  http://127.0.0.1:7788/state
echo Press Ctrl+C to quit.
echo.
python "%~dp0main.py"
