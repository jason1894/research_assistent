#!/usr/bin/env bash
# install_deps.sh – Install Python dependencies from requirements.txt
# Usage: bash scripts/install_deps.sh [--no-torch]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

[[ -f "$REQUIREMENTS" ]] || error "requirements.txt not found at $REQUIREMENTS"

# Detect active virtual environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d "$PROJECT_DIR/venv" ]]; then
        # shellcheck disable=SC1091
        source "$PROJECT_DIR/venv/bin/activate"
        info "Activated venv at $PROJECT_DIR/venv"
    else
        info "No virtual environment detected – installing into system Python."
    fi
fi

info "Upgrading pip …"
pip install --upgrade pip

info "Installing dependencies from $REQUIREMENTS …"
if [[ "${1:-}" == "--no-torch" ]]; then
    info "Skipping torch/transformers/peft/datasets (--no-torch flag set)."
    grep -v -E "^(torch|transformers|peft|datasets|accelerate|bitsandbytes)" "$REQUIREMENTS" \
        | pip install -r /dev/stdin
else
    pip install -r "$REQUIREMENTS"
fi

success "All dependencies installed successfully."
echo ""
echo "Installed packages:"
pip list --format=columns | grep -E "langchain|chromadb|sentence|torch|streamlit|click|rich|pyyaml|sqlalchemy"
