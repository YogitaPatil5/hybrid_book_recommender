@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  BookMind Pro — one-click launcher for Windows.
REM  Double-click this file, or run "run.bat" in a Command Prompt.
REM  Uses the project's .venv directly (no activation needed) and opens the
REM  browser. Extra options are passed through, e.g.  run.bat --port 9000
REM ─────────────────────────────────────────────────────────────────────────
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [BookMind] Virtual environment not found. Run the setup steps in README.md first.
    pause
    exit /b 1
)

if not exist "artifacts\manifest.json" (
    echo [BookMind] Search index not built yet. Building it now ^(one-time^)...
    ".venv\Scripts\python.exe" -m bookmind.pipeline.cli download || goto :fail
    ".venv\Scripts\python.exe" -m bookmind.pipeline.cli all --classifier keyword || goto :fail
)

".venv\Scripts\python.exe" -m bookmind --open %*
exit /b %errorlevel%

:fail
echo [BookMind] Setup step failed - see the messages above.
pause
exit /b 1
