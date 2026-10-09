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
if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment in .venv ..."
    "$PY" -m venv .venv || fail "could not create .venv (on Debian/Ubuntu: sudo apt install python3-venv)"
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
"$VPY" -c "from fxquant import core; print('Numeric core in use:', core.BACKEND)"
if ! "$VPY" -c "from fxquant import core; import sys; sys.exit(core.BACKEND != 'c++')" 2>/dev/null; then
    echo "[info] Running on the NumPy fallback: same results, slower."
    if [ "$(uname)" = "Darwin" ]; then
        echo "       For the fast C++ core run: xcode-select --install   then run this setup again."
    else
        echo "       For the fast C++ core install g++ (e.g. sudo apt install g++), then run this setup again."
    fi
fi

echo
echo "=== Setup complete ==="
echo "In VS Code: press Ctrl+Shift+P (Cmd+Shift+P on Mac), run \"Python: Select Interpreter\" and choose .venv,"
echo "then press F5 and pick \"1. Demo - synthetic market (offline)\"."
