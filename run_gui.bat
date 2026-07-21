@echo off
setlocal
set "PYTHONUTF8=1"

cd /d "%~dp0"

where docker >nul 2>&1
if errorlevel 1 (
    echo Docker Desktop is required. Install it from https://docs.docker.com/desktop/install/windows-install/
    echo.
    pause
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo Docker Desktop is installed but is not running or is not available to this user.
    echo Start Docker Desktop, wait until it reports that Docker is running, and try again.
    echo.
    pause
    exit /b 1
)

set "PYTHON="
where py >nul 2>&1
if not errorlevel 1 (
    py -3 --version >nul 2>&1
    if not errorlevel 1 set "PYTHON=py -3"
)
if not defined PYTHON (
    where python >nul 2>&1
    if errorlevel 1 (
        echo Python 3 is required. Install it from https://www.python.org/downloads/windows/
        echo.
        pause
        exit /b 1
    )
    python --version >nul 2>&1
    if not errorlevel 1 set "PYTHON=python"
)
if not defined PYTHON (
    echo Python 3 is required. Install it from https://www.python.org/downloads/windows/
    echo.
    pause
    exit /b 1
)

echo Starting MicroICS GUI...
%PYTHON% gui\app.py %*
if errorlevel 1 (
    echo.
    echo MicroICS GUI stopped with an error.
    pause
)

endlocal
