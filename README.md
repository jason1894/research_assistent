# 🔬 Research Assistant | 研究助手

<p align="center">
  <strong>An AI-powered research assistant with RAG, LoRA fine-tuning, and multi-model support</strong><br>
  <em>基于RAG检索增强和LoRA微调的智能研究助手</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/LLM-Ollama-orange" alt="Ollama">
  <img src="https://img.shields.io/badge/vectorDB-ChromaDB-purple" alt="ChromaDB">
</p>

---

## ✨ Features | 功能特性

| Feature | Description |
|---------|-------------|
| 🤖 **Multi-Model LLM** | Supports llama2, mistral, qwen via Ollama |
| 📚 **RAG Pipeline** | Index and query your paper collection |
| 🎯 **LoRA Fine-Tuning** | Fine-tune models on your research data |
| 💬 **Conversation Memory** | Persistent session history with SQLite |
| 🕸️ **Knowledge Graph** | Track concepts and their relationships |
| 🖥️ **Web UI** | Beautiful Streamlit interface (bilingual) |
| ⌨️ **CLI** | Powerful command-line interface |
| 🐳 **Docker** | Full container deployment |

---

## 🏗️ Architecture | 架构

```
research_assistent/
├── config/               # YAML & JSON configuration
├── src/
│   ├── llm/              # OllamaClient + ModelManager
│   ├── rag/              # Document loading, embeddings, vector store, retriever
│   ├── lora/             # LoRA fine-tuning with PEFT
│   ├── memory/           # Conversation DB + Knowledge Graph
│   └── utils/            # PDF processor, text utils, logger
├── ui/
│   ├── cli.py            # Click-based CLI
│   └── web.py            # Streamlit web app
├── scripts/              # Setup and utility scripts
├── tests/                # pytest test suite
├── data/                 # Papers, ChromaDB, LoRA adapters
└── Dockerfile + docker-compose.yml
```

**Data Flow:**
```
PDF/Text → DocumentLoader → Chunker → EmbeddingManager → VectorStore (ChromaDB)
                                                               ↓
User Query → OllamaClient ←── Retriever ←── similarity_search()
                ↓
         RAG Response + Sources
```

---

## 🚀 Quick Start | 快速开始

### Option A: Local Setup

```bash
# 1. Clone and setup
git clone https://github.com/example/research_assistent.git
cd research_assistent
bash scripts/setup.sh
source venv/bin/activate

# 2. Start Ollama and pull a model
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama2

# 3. Launch the Web UI
streamlit run ui/web.py
# OR use the CLI
python ui/cli.py --help
```

### Option B: Docker

```bash
docker compose up -d
# Open http://localhost:8501
```

---

## 📖 CLI Usage | 命令行使用

```bash
# Interactive chat
python ui/cli.py chat --model llama2 --domain machine_learning

# Review a paper
python ui/cli.py review paper.pdf

# Add paper to knowledge base
python ui/cli.py add-paper paper.pdf --domain machine_learning

# Writing assistance
python ui/cli.py write --type abstract --topic "LoRA fine-tuning"

# Fine-tune with your data
python ui/cli.py train --data training_data.json --epochs 3
```

---

## 🐍 Python API

```python
from src.llm.ollama_client import OllamaClient
from src.rag.document_loader import DocumentLoader
from src.rag.embeddings import EmbeddingManager
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever

client = OllamaClient(model="llama2")
loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
emb    = EmbeddingManager()
store  = VectorStore(persist_directory="./data/chroma_db", embedding_manager=emb)

docs = loader.load_and_split("paper.pdf")
store.add_documents(docs)

retriever = Retriever(store, client)
result = retriever.rag_query("What is the main contribution?")
print(result["answer"])
```

---

## ⚙️ Configuration | 配置

Edit `config/config.yaml` to change model, RAG settings, etc.

```yaml
llm:
  model: "llama2"
  temperature: 0.7
  max_tokens: 2048

rag:
  chunk_size: 1000
  top_k: 5
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
```

---

## 🔧 Development

```bash
pytest tests/ -v
pip install -e ".[dev]"
```

---

## 🤝 Contributing

1. Fork → feature branch → changes + tests → PR

## 📄 License

MIT License

## 🙏 Acknowledgements

[Ollama](https://ollama.ai) · [LangChain](https://langchain.com) · [ChromaDB](https://www.trychroma.com) · [PEFT](https://github.com/huggingface/peft) · [Streamlit](https://streamlit.io)
