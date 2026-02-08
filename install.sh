#!/bin/bash
#
# Alternating Word Chat — One-Step Installer for macOS
#
# Usage:
#   chmod +x install.sh && ./install.sh
#
# What this does:
#   1. Installs Homebrew (if missing)
#   2. Installs Python 3 and portaudio via Homebrew
#   3. Creates a Python virtual environment
#   4. Installs all Python packages
#   5. Pre-downloads the AI models (~3.5 GB)
#   6. Creates a launcher script you can double-click
#

set -e

# ── Colors & helpers ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

step_num=0
step() {
    step_num=$((step_num + 1))
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  Step $step_num: $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
skip() { echo -e "  ${YELLOW}[SKIP]${NC} $1"; }
fail() { echo -e "  ${RED}[ERROR]${NC} $1"; exit 1; }

# ── Resolve install directory (where this script lives) ──────────────
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$INSTALL_DIR"

echo ""
echo -e "${BOLD}  Alternating Word Chat — Installer${NC}"
echo -e "  Install directory: ${INSTALL_DIR}"

# ── Check macOS ──────────────────────────────────────────────────────
if [[ "$(uname)" != "Darwin" ]]; then
    fail "This installer is for macOS only. For Linux, use the Docker method (see Dockerfile)."
fi

# ── Step 1: Homebrew ─────────────────────────────────────────────────
step "Checking for Homebrew"

if command -v brew &>/dev/null; then
    ok "Homebrew is installed ($(brew --version | head -1))"
else
    echo "  Homebrew is not installed. Installing now..."
    echo "  (You may be asked for your password)"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH for Apple Silicon
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi

    if command -v brew &>/dev/null; then
        ok "Homebrew installed successfully"
    else
        fail "Homebrew installation failed. Please install manually: https://brew.sh"
    fi
fi

# ── Step 2: Python 3 ────────────────────────────────────────────────
step "Checking for Python 3"

# Prefer Homebrew Python, fall back to system
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    PYTHON_CMD=python3

    # Check version is at least 3.8
    MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
    MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

    if [[ "$MAJOR" -ge 3 && "$MINOR" -ge 8 ]]; then
        ok "$PYTHON_VERSION found"
    else
        echo "  Python $MAJOR.$MINOR is too old (need 3.8+). Installing via Homebrew..."
        brew install python@3.11
        PYTHON_CMD="$(brew --prefix python@3.11)/bin/python3"
        ok "Python 3.11 installed via Homebrew"
    fi
else
    echo "  Python 3 not found. Installing via Homebrew..."
    brew install python@3.11
    PYTHON_CMD="$(brew --prefix python@3.11)/bin/python3"
    ok "Python 3.11 installed via Homebrew"
fi

# ── Step 3: portaudio (needed by pyaudio for voice input) ────────────
step "Checking for portaudio (needed for voice input)"

if brew list portaudio &>/dev/null; then
    ok "portaudio is installed"
else
    echo "  Installing portaudio..."
    brew install portaudio
    ok "portaudio installed"
fi

# ── Step 4: Virtual environment ──────────────────────────────────────
step "Setting up Python virtual environment"

VENV_DIR="$INSTALL_DIR/venv"

if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/python" ]]; then
    skip "Virtual environment already exists at $VENV_DIR"
else
    echo "  Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# Activate it
source "$VENV_DIR/bin/activate"
ok "Virtual environment activated"

# ── Step 5: Install Python packages ─────────────────────────────────
step "Installing Python packages"

echo "  Upgrading pip..."
pip install --upgrade pip --quiet

echo "  Installing PyTorch, transformers, and other dependencies..."
echo "  (This may take a few minutes)"
pip install -r "$INSTALL_DIR/requirements.txt"
ok "All Python packages installed"

# ── Step 6: Download AI models ───────────────────────────────────────
step "Downloading AI models (~3.5 GB total — this will take a while)"

echo ""
echo "  Downloading Qwen2.5-1.5B-Instruct (~2.9 GB)..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('    Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('    Downloading model weights...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('    Done.')
"
ok "Qwen2.5-1.5B-Instruct downloaded"

echo ""
echo "  Downloading TinyStories-33M (~558 MB)..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('    Downloading tokenizer...')
AutoTokenizer.from_pretrained('roneneldan/TinyStories-33M')
print('    Downloading model weights...')
AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
print('    Done.')
"
ok "TinyStories-33M downloaded"

# ── Step 7: Create launcher script ───────────────────────────────────
step "Creating launcher"

LAUNCHER="$INSTALL_DIR/start.command"
cat > "$LAUNCHER" << 'LAUNCHER_SCRIPT'
#!/bin/bash
# Alternating Word Chat — Launcher
# Double-click this file to start the app.

DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/venv/bin/activate"

# Resize terminal window for best display
printf '\e[8;30;80t'

cd "$DIR"
python alternating_word_chat.py "$@"
LAUNCHER_SCRIPT

chmod +x "$LAUNCHER"
ok "Launcher created: start.command"

# Also create a plain shell launcher
LAUNCHER_SH="$INSTALL_DIR/start.sh"
cat > "$LAUNCHER_SH" << 'LAUNCHER_SH_SCRIPT'
#!/bin/bash
# Alternating Word Chat — Launcher
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/venv/bin/activate"
cd "$DIR"
python alternating_word_chat.py "$@"
LAUNCHER_SH_SCRIPT

chmod +x "$LAUNCHER_SH"
ok "Shell launcher created: start.sh"

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Installation complete!${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  To start the app, either:"
echo ""
echo -e "    ${BOLD}Option 1:${NC} Double-click ${YELLOW}start.command${NC} in Finder"
echo ""
echo -e "    ${BOLD}Option 2:${NC} Run from Terminal:"
echo -e "             ${YELLOW}${INSTALL_DIR}/start.sh${NC}"
echo ""
echo -e "    ${BOLD}With voice input:${NC}"
echo -e "             ${YELLOW}${INSTALL_DIR}/start.sh --voice${NC}"
echo ""
echo -e "    ${BOLD}Type /help${NC} inside the app for all commands."
echo ""
