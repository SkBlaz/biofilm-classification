@echo off
setlocal
cd /d "%~dp0"

where powershell >nul 2>&1
if errorlevel 1 (
    echo Windows PowerShell is required to create the desktop shortcut.
    echo.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_microics_shortcut.ps1"
if errorlevel 1 (
    echo.
    echo Could not create the MicroICS desktop shortcut.
    pause
    exit /b 1
)

echo.
echo The MicroICS GUI shortcut is now on your desktop.
pause
endlocal
