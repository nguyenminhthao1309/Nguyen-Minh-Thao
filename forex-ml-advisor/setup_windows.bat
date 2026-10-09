@echo off
rem ===========================================================================
rem  fxquant one-click setup for Windows
rem  Double-click this file, or in VS Code: Terminal > Run Task > "1. Setup".
rem  Creates a .venv folder with Python packages and builds the C++ core.
rem ===========================================================================
setlocal
cd /d "%~dp0"
echo.
echo === fxquant setup (Windows) ===
echo Folder: %CD%
echo.

rem --- 1. Find Python 3.10 or newer -----------------------------------------
set "PY="
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
        if not errorlevel 1 set "PY=python"
    )
)
if not defined PY (
    echo [error] Python 3.10 or newer was not found.
    echo         Install it from https://www.python.org/downloads/ and tick
    echo         "Add python.exe to PATH" in the installer, then run this again.
    goto :fail
)
for /f "delims=" %%v in ('%PY% -c "import sys; print(sys.version.split()[0])"') do echo Using Python %%v

rem --- 2. Virtual environment ---------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PY% -m venv .venv
    if errorlevel 1 goto :fail
)
set "VPY=%CD%\.venv\Scripts\python.exe"

rem --- 3. Packages --------------------------------------------------------------
echo Installing Python packages (this can take a few minutes the first time) ...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
"%VPY%" -m pip install yfinance
if errorlevel 1 echo [warn] yfinance could not be installed: live Yahoo data will not work, everything else will.

rem --- 4. C++ core (optional: falls back to NumPy without a compiler) ----------
echo Building the C++ core ...
"%VPY%" -m pip install -e .
if errorlevel 1 echo [warn] Installing the fxquant package failed; you can still run it from this folder.
"%VPY%" -c "from fxquant import core; print('Numeric core in use:', core.BACKEND)"
"%VPY%" -c "from fxquant import core; import sys; sys.exit(core.BACKEND != 'c++')" >nul 2>nul
if errorlevel 1 (
    echo [info] Running on the NumPy fallback: same results, slower.
    echo        For the fast C++ core install "Build Tools for Visual Studio" with the
    echo        "Desktop development with C++" workload, then run this setup again:
    echo        https://visualstudio.microsoft.com/visual-cpp-build-tools/
)

echo.
echo === Setup complete ===
echo In VS Code: press Ctrl+Shift+P, run "Python: Select Interpreter" and choose .venv,
echo then press F5 and pick "1. Demo - synthetic market (offline)".
if "%~1"=="" pause
exit /b 0

:fail
echo.
echo === Setup failed - see the messages above ===
if "%~1"=="" pause
exit /b 1
