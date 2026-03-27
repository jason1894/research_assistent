#!/usr/bin/env bash
# download_models.sh – Download recommended LLM and embedding models
# Usage: bash scripts/download_models.sh [--models "llama2 mistral qwen"]

set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Default LLM models to pull
LLM_MODELS="${MODELS:-llama2 mistral}"

# ── Ollama LLM models ─────────────────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    info "Ollama found – pulling LLM models: $LLM_MODELS"

    # Ensure Ollama server is running
    if ! curl -sf "http://localhost:11434/api/tags" &>/dev/null; then
        info "Starting Ollama server in the background …"
        ollama serve &>/dev/null &
        OLLAMA_PID=$!
        sleep 5
        trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT
    fi

    for MODEL in $LLM_MODELS; do
        info "Pulling $MODEL …"
        if ollama pull "$MODEL"; then
            success "Downloaded: $MODEL"
        else
            error "Failed to pull $MODEL – skipping."
        fi
    done

    # Optional: Qwen (good for Chinese research)
    if [[ "${DOWNLOAD_QWEN:-0}" == "1" ]]; then
        info "Pulling qwen:7b …"
        ollama pull qwen:7b && success "Downloaded: qwen:7b" || warn "Failed to pull qwen:7b"
    fi
else
    warn "Ollama not found. Install from https://ollama.ai/download"
    echo "  Linux/macOS: curl -fsSL https://ollama.ai/install.sh | sh"
fi

# ── Sentence-transformers embedding model ─────────────────────────────────────
info "Downloading sentence-transformers embedding model …"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [[ -n "$PYTHON_CMD" ]]; then
    "$PYTHON_CMD" - <<'EOF'
import sys
try:
    from sentence_transformers import SentenceTransformer
    print("Downloading all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    test = model.encode(["test sentence"])
    print(f"  Embedding dim: {test.shape[1]}")
    print("OK  Embedding model ready.")
except ImportError:
    print("WARN sentence-transformers not installed – run install_deps.sh first.")
    sys.exit(0)
except Exception as e:
    print(f"ERROR {e}")
    sys.exit(1)
EOF
    success "Embedding model downloaded."
else
    warn "Python not found – skipping embedding model download."
fi

echo ""
success "Model downloads complete!"
echo ""
echo "Available Ollama models:"
ollama list 2>/dev/null || echo "  (ollama not available)"
