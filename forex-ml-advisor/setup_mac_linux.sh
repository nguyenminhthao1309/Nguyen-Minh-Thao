#!/usr/bin/env bash
# ===========================================================================
#  fxquant one-click setup for macOS and Linux
#  Run:  bash setup_mac_linux.sh   (or in VS Code: Terminal > Run Task > "1. Setup")
#  Creates a .venv folder with Python packages and builds the C++ core.
# ===========================================================================
set -u
cd "$(dirname "$0")"
echo
echo "=== fxquant setup (macOS / Linux) ==="
echo "Folder: $(pwd)"
echo

fail() { echo; echo "=== Setup failed: $1 ==="; exit 1; }

# --- 1. Find Python 3.10 or newer ------------------------------------------
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PY="$c"; break
    fi
done
[ -n "$PY" ] || fail "Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/"
echo "Using $("$PY" --version)"

# --- 2. Virtual environment -------------------------------------------------
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    echo "The existing .venv is incomplete - recreating it ..."
    rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment in .venv ..."
    if ! "$PY" -m venv .venv; then
        rm -rf .venv
        fail "could not create .venv (on Debian/Ubuntu: sudo apt install python3-venv, then run this setup again)"
    fi
fi
VPY="$(pwd)/.venv/bin/python"

# --- 3. Packages ------------------------------------------------------------
echo "Installing Python packages (this can take a few minutes the first time) ..."
"$VPY" -m pip install --upgrade pip || fail "pip upgrade"
"$VPY" -m pip install -r requirements.txt || fail "installing requirements.txt"
"$VPY" -m pip install yfinance || echo "[warn] yfinance could not be installed: live Yahoo data will not work, everything else will."

# --- 4. C++ core (optional: falls back to NumPy without a compiler) --------
echo "Building the C++ core ..."
"$VPY" -m pip install -e . || echo "[warn] Installing the fxquant package failed; you can still run it from this folder."
"$VPY" tools/configure_vscode_cpp.py >/dev/null 2>&1 || true
"$VPY" -c "from fxquant import core; print('Numeric core in use:', core.BACKEND)"
if ! "$VPY" -c "from fxquant import core; import sys; sys.exit(core.BACKEND != 'c++')" 2>/dev/null; then
    echo "[info] Running on the NumPy fallback: same results, slower."
    if [ "$(uname)" = "Darwin" ]; then
        echo "       For the fast C++ core run: xcode-select --install   then run this setup again."
    elif ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
        echo "       For the fast C++ core install a compiler and the Python headers:"
        echo "         sudo apt install g++ python3-dev     (Fedora: sudo dnf install gcc-c++ python3-devel)"
        echo "       then run this setup again."
    elif ! "$VPY" -c "import os, sys, sysconfig; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))"; then
        echo "       A C++ compiler is installed but the Python headers (Python.h) are missing:"
        echo "         sudo apt install python3-dev     (Fedora: sudo dnf install python3-devel)"
        echo "       then run this setup again."
    else
        echo "       The C++ build failed; run  .venv/bin/python -m pip install -v -e .  to see why."
    fi
fi

echo
echo "=== Setup complete ==="
echo "In VS Code: press Ctrl+Shift+P (Cmd+Shift+P on Mac), run \"Python: Select Interpreter\" and choose .venv,"
echo "then open Run and Debug, pick \"1. Demo - synthetic market (offline)\" and press Ctrl+F5 (Cmd+F5 on Mac)."
