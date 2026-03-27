# Deployment Guide

## Local Deployment

### 1. Standard Setup

```bash
git clone https://github.com/example/research_assistent.git
cd research_assistent
bash scripts/setup.sh
source venv/bin/activate
ollama serve &
streamlit run ui/web.py
```

The web UI is available at **http://localhost:8501**.

### 2. Running as a Background Service (Linux/systemd)

Create `/etc/systemd/system/research-assistant.service`:

```ini
[Unit]
Description=Research Assistant Web UI
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/research_assistent
Environment=PATH=/path/to/research_assistent/venv/bin
ExecStart=/path/to/research_assistent/venv/bin/streamlit run ui/web.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable research-assistant
sudo systemctl start research-assistant
```

---

## Docker Deployment

### Quick Start

```bash
docker compose up -d
```

Services started:
- `research_assistant` – Streamlit app on port 8501
- `ollama` – LLM server on port 11434

### Check status

```bash
docker compose ps
docker compose logs -f research_assistant
```

### Pull models inside Docker

```bash
docker compose exec ollama ollama pull llama2
docker compose exec ollama ollama pull mistral
```

### With GPU (NVIDIA)

Uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Also install NVIDIA Container Toolkit:
```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Hardware Recommendations

### Minimum (CPU only)

- 16 GB RAM
- 50 GB free disk space
- Any modern quad-core CPU

### Recommended (GPU inference)

- 32 GB RAM
- NVIDIA GPU with 8 GB VRAM (RTX 3080 or better)
- 200 GB SSD

### For LoRA Fine-Tuning

- 32+ GB RAM
- NVIDIA GPU with 16+ GB VRAM (A100, RTX 4090)
- Enable 4-bit quantisation: `load_in_4bit: true` in config

---

## Performance Tuning

### LLM Inference

```yaml
# config/config.yaml
llm:
  num_ctx: 4096        # Context window – reduce for speed
  temperature: 0.7
  max_tokens: 1024     # Reduce for faster responses
```

### RAG

```yaml
rag:
  chunk_size: 500      # Smaller chunks = faster retrieval
  top_k: 3             # Fewer retrieved docs = faster LLM prompt
```

### ChromaDB

For large collections (>100k documents), increase ChromaDB batch size:

```python
store = VectorStore(persist_directory="./data/chroma_db")
store._get_store()._collection.modify(
    metadata={"hnsw:space": "cosine", "hnsw:M": 32}
)
```

---

## Backup and Restore

### Backup

```bash
# Backup ChromaDB and SQLite DB
tar -czf backup_$(date +%Y%m%d).tar.gz \
    data/chroma_db/ \
    data/memory.db \
    data/knowledge_graph.json \
    data/papers/ \
    config/
```

### Restore

```bash
tar -xzf backup_20240101.tar.gz
```

---

## Security Considerations

- The web UI does not include authentication by default. For public deployment, add authentication (e.g., via nginx + basic auth or a reverse proxy with OAuth).
- API keys and secrets should be stored in `.env` (not committed to git).
- The `.gitignore` already excludes `.env` files.

### Example nginx reverse proxy

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```
