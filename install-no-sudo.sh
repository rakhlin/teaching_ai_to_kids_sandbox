#!/bin/bash
#
# Alternating Word Chat — No-Sudo Installer for macOS
#
# Usage:
#   ./install-no-sudo.sh
#
# This installer does NOT require root/sudo access.
# It installs Miniconda (Python) into the project folder and
# keeps everything self-contained. Voice input is not available
# without sudo (it requires system-level portaudio).
#

set -e

# ── Colors & helpers ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

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

CONDA_DIR="$INSTALL_DIR/miniconda"
ENV_NAME="wordchat"

echo ""
echo -e "${BOLD}  Alternating Word Chat — No-Sudo Installer${NC}"
echo -e "  Install directory: ${INSTALL_DIR}"
echo ""
echo -e "  ${YELLOW}Note: Voice input (--voice) is not available with this installer.${NC}"
echo -e "  ${YELLOW}Use install.sh (requires sudo) for voice input support.${NC}"

# ── Check macOS ──────────────────────────────────────────────────────
if [[ "$(uname)" != "Darwin" ]]; then
    fail "This installer is for macOS only."
fi

# ── Step 1: Install Miniconda ────────────────────────────────────────
step "Installing Miniconda (local, no sudo)"

if [[ -f "$CONDA_DIR/bin/conda" ]]; then
    skip "Miniconda already installed at $CONDA_DIR"
else
    # Detect architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    else
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    fi

    echo "  Detected architecture: $ARCH"
    echo "  Downloading Miniconda..."

    INSTALLER="$INSTALL_DIR/miniconda_installer.sh"
    curl -fsSL "$MINICONDA_URL" -o "$INSTALLER"

    echo "  Installing to $CONDA_DIR (no sudo needed)..."
    bash "$INSTALLER" -b -p "$CONDA_DIR"

    # Clean up installer
    rm -f "$INSTALLER"

    if [[ -f "$CONDA_DIR/bin/conda" ]]; then
        ok "Miniconda installed"
    else
        fail "Miniconda installation failed"
    fi
fi

# Accept conda Terms of Service (required by newer conda versions)
"$CONDA_DIR/bin/conda" config --set auto_activate_base false 2>/dev/null || true
yes | "$CONDA_DIR/bin/conda" tos accept 2>/dev/null || true

# ── Step 2: Create conda environment ─────────────────────────────────
step "Creating Python environment"

if "$CONDA_DIR/bin/conda" env list | grep -q "$ENV_NAME"; then
    skip "Environment '$ENV_NAME' already exists"
else
    echo "  Creating environment '$ENV_NAME' with Python 3.11..."
    "$CONDA_DIR/bin/conda" create -n "$ENV_NAME" python=3.11 -y --quiet
    ok "Environment created"
fi

# Activate environment by sourcing its activate script directly
source "$CONDA_DIR/bin/activate" "$ENV_NAME"
ok "Environment activated (Python $(python --version 2>&1 | cut -d' ' -f2))"

# ── Step 3: Install Python packages ─────────────────────────────────
step "Installing Python packages"

echo "  Installing PyTorch, transformers, and dependencies..."
echo "  (This may take a few minutes)"

# Install everything except pyaudio (requires system portaudio)
pip install --quiet torch transformers accelerate

ok "All Python packages installed"

# ── Step 4: Download AI models ───────────────────────────────────────
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

# ── Step 5: Create launcher scripts ──────────────────────────────────
step "Creating launchers"

# Double-clickable .command launcher (uses /bin/bash explicitly)
LAUNCHER="$INSTALL_DIR/start.command"
cat > "$LAUNCHER" << 'LAUNCHER_SCRIPT'
#!/bin/bash
# Alternating Word Chat — Launcher (no-sudo install)
# Double-click this file to start the app.

DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/miniconda/bin/activate" wordchat

# Resize terminal window for best display
printf '\e[8;30;80t'

cd "$DIR"
python alternating_word_chat.py "$@"
LAUNCHER_SCRIPT

chmod +x "$LAUNCHER"
ok "Launcher created: start.command"

# Shell launcher
LAUNCHER_SH="$INSTALL_DIR/start.sh"
cat > "$LAUNCHER_SH" << 'LAUNCHER_SH_SCRIPT'
#!/bin/bash
# Alternating Word Chat — Launcher (no-sudo install)
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/miniconda/bin/activate" wordchat
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
echo -e "    ${YELLOW}Note:${NC} Voice input (--voice) is not available with this installer."
echo ""
echo -e "    ${BOLD}Type /help${NC} inside the app for all commands."
echo ""
