#!/usr/bin/env bash
# setup.sh – Full environment setup for the Research Assistant
# Usage: bash scripts/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Python version check ──────────────────────────────────────────────────────
info "Checking Python version …"
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PY_VER=$("$cmd" -c "import sys; print(sys.version_info[:2])")
        if [[ "$PY_VER" > "(3, 9)" ]]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done
[[ -z "$PYTHON_CMD" ]] && error "Python 3.10+ is required. Please install it and retry."
success "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# ── Virtual environment ───────────────────────────────────────────────────────
cd "$PROJECT_DIR"
info "Creating virtual environment at $PROJECT_DIR/venv …"
if [[ ! -d venv ]]; then
    "$PYTHON_CMD" -m venv venv
    success "Virtual environment created."
else
    warn "venv/ already exists – skipping creation."
fi

# Activate
# shellcheck disable=SC1091
source venv/bin/activate
success "Virtual environment activated."

# ── Dependencies ──────────────────────────────────────────────────────────────
info "Installing Python dependencies …"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
success "Dependencies installed."

# ── Data directories ─────────────────────────────────────────────────────────
info "Creating data directories …"
for dir in data/papers data/chroma_db data/lora_adapters data/cache logs; do
    mkdir -p "$dir"
    touch "$dir/.gitkeep" 2>/dev/null || true
done
success "Directories ready."

# ── Config ────────────────────────────────────────────────────────────────────
if [[ ! -f config/config.yaml ]]; then
    warn "config/config.yaml not found."
else
    success "config/config.yaml present."
fi

# ── Ollama check ──────────────────────────────────────────────────────────────
info "Checking Ollama installation …"
if command -v ollama &>/dev/null; then
    success "Ollama is installed: $(ollama --version 2>&1 | head -1)"
else
    warn "Ollama not found. Install from https://ollama.ai/download"
    echo "  Linux/macOS: curl -fsSL https://ollama.ai/install.sh | sh"
fi

echo ""
echo -e "${GREEN}✅  Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Start Ollama:             ollama serve"
echo "  3. Pull a model:             ollama pull llama2"
echo "  4. Launch Web UI:            streamlit run ui/web.py"
echo "  5. Or use the CLI:           python ui/cli.py --help"
