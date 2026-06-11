@echo off
REM Launch the phone/PWA companion server (clod_mic_server.py) on 0.0.0.0:7789.
REM Open http://<this-pc-ip>:7789/ on your phone to use the Android mic companion.
cd /d "%~dp0"
start "" /min python clod_mic_server.py
