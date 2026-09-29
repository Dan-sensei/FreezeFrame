@echo off
rem Starts the capture daemon. Launch your game through Ninja Ripper, then press
rem the rip key (PrintScreen by default) in-game. Each capture is processed,
rem built into a .blend and exposure-matched automatically.
cd /d "%~dp0"
python gtb.py watch
pause
