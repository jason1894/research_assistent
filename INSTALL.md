# Installation Guide

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB | 100 GB SSD |
| GPU | Optional | NVIDIA 8GB+ VRAM (for LoRA) |

## Software Prerequisites

- Python 3.10 or higher
- Git
- curl
- (Optional) NVIDIA CUDA 11.8+ for GPU acceleration

---

## Linux / macOS Installation

### Step 1 – Clone the repository

```bash
git clone https://github.com/example/research_assistent.git
cd research_assistent
```

### Step 2 – Run the automated setup script

```bash
bash scripts/setup.sh
```

This script:
- Checks your Python version
- Creates a virtual environment in `./venv`
- Installs all Python dependencies
- Creates required data directories

### Step 3 – Install Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 4 – Start Ollama and pull a model

```bash
ollama serve &          # Start in background
ollama pull llama2      # Pull default model (~4 GB)
```

### Step 5 – Activate environment and launch

```bash
source venv/bin/activate
streamlit run ui/web.py          # Web UI at http://localhost:8501
# OR
python ui/cli.py --help           # CLI
```

---

## Windows Installation

### Prerequisites

1. Install [Python 3.10+](https://www.python.org/downloads/windows/) – check "Add Python to PATH"
2. Install [Git for Windows](https://git-scm.com/download/win)
3. Install [Ollama for Windows](https://ollama.ai/download/windows)

### Setup

```powershell
git clone https://github.com/example/research_assistent.git
cd research_assistent
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## GPU Setup (CUDA)

For LoRA fine-tuning with GPU acceleration:

```bash
# Install CUDA-enabled PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available())"
```

For Ollama GPU support, install the NVIDIA container toolkit if using Docker:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

---

## Docker Installation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 20.10+
- [Docker Compose](https://docs.docker.com/compose/install/) v2+

### Start

```bash
git clone https://github.com/example/research_assistent.git
cd research_assistent
docker compose up -d
```

Access the web UI at **http://localhost:8501**.

---

## Troubleshooting

### "Cannot connect to Ollama"

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check logs
journalctl -u ollama -f      # Linux systemd
```

### "ImportError: No module named 'langchain'"

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### ChromaDB "sqlite3 version" error on older Linux

```bash
pip install pysqlite3-binary
```

Then add to the top of your script:
```python
import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
```

### LoRA training out-of-memory error

Reduce batch size in `config/config.yaml`:
```yaml
lora:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  load_in_4bit: true
```

### Slow embedding generation

Use a GPU or a smaller embedding model:
```yaml
rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"  # fastest
```
