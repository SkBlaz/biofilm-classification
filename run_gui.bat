@echo off
setlocal

cd /d "%~dp0"

where docker >nul 2>&1
if errorlevel 1 (
    echo Docker Desktop is required. Install it from https://docs.docker.com/desktop/install/windows-install/
    echo.
    pause
    exit /b 1
)

where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py -3"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo Python 3 is required. Install it from https://www.python.org/downloads/windows/
        echo.
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

echo Starting MicroICS GUI...
%PYTHON% gui\app.py %*
if errorlevel 1 (
    echo.
    echo MicroICS GUI stopped with an error.
    pause
)

endlocal
