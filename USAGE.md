# Usage Guide

## CLI Reference

All CLI commands are accessed via `python ui/cli.py <command>`.

### `chat` – Interactive Research Chat

```bash
python ui/cli.py chat [OPTIONS]

Options:
  -m, --model TEXT    LLM model override (default: from config.yaml)
  -s, --session TEXT  Session ID for conversation history
  -d, --domain TEXT   Research domain tag (machine_learning, statistics, etc.)
```

**Example:**
```bash
python ui/cli.py chat --model mistral --domain machine_learning
You: What are the key differences between LoRA and prefix tuning?
Assistant: [detailed explanation with citations]
```

Type `exit` or `quit` to end the session.

---

### `review` – Paper Review

```bash
python ui/cli.py review FILE_PATH [OPTIONS]

Arguments:
  FILE_PATH  Path to PDF or text file

Options:
  -m, --model TEXT  LLM model override
```

**Example:**
```bash
python ui/cli.py review ./papers/attention_is_all_you_need.pdf
```

Outputs a structured review covering strengths, weaknesses, and an overall assessment.

---

### `write` – Academic Writing Assistant

```bash
python ui/cli.py write [OPTIONS]

Options:
  --type TEXT   Content type: abstract, introduction, conclusion, etc.
  --topic TEXT  Research topic (prompted if not provided)
  -m, --model TEXT
```

**Example:**
```bash
python ui/cli.py write --type abstract --topic "Efficient fine-tuning of LLMs"
```

---

### `add-paper` – Index a Paper

```bash
python ui/cli.py add-paper FILE_PATH [OPTIONS]

Arguments:
  FILE_PATH  Path to the PDF file

Options:
  -d, --domain TEXT  Domain tag (default: machine_learning)
```

**Example:**
```bash
python ui/cli.py add-paper ~/papers/lora.pdf --domain transfer_learning
✓ Added 47 chunks from lora.pdf to the knowledge base.
```

---

### `list-papers` – View Indexed Papers

```bash
python ui/cli.py list-papers
```

Shows the ChromaDB collection statistics.

---

### `train` – LoRA Fine-Tuning

```bash
python ui/cli.py train [OPTIONS]

Options:
  --data PATH         Path to training data (REQUIRED) JSON/JSONL with
                      "instruction" and "response" fields
  --base-model TEXT   HuggingFace model ID
  --output PATH       Output directory for the adapter
  --epochs INT        Number of training epochs (default: 3)
```

**Training data format (JSON):**
```json
[
  {
    "instruction": "Explain the attention mechanism",
    "response": "The attention mechanism allows the model to..."
  }
]
```

**Export conversation history as training data:**
```python
from src.memory.conversation_db import ConversationDB
db = ConversationDB()
examples = db.export_for_training(min_feedback="positive")
import json
with open("training_data.json", "w") as f:
    json.dump(examples, f)
```

---

## Web UI Walkthrough

Start the web UI:
```bash
streamlit run ui/web.py
```

Navigate to **http://localhost:8501**.

### 💬 Chat Tab

1. Type your research question in the chat input
2. Toggle "Use RAG" to search the knowledge base for context
3. Conversation history is preserved across browser refreshes (same session)

### ✍️ Writing Tab

1. Select content type (Abstract, Introduction, etc.)
2. Enter your topic
3. Add additional requirements (optional)
4. Click **Generate**
5. Paste text into the Grammar Check box for corrections

### 📋 Review Tab

1. Choose review type: Full Review, Rebuttal Helper, or Summary
2. Upload a PDF or paste text
3. Click **Analyse**
4. Download the result as Markdown

### 📚 Knowledge Base Tab

1. Upload one or more PDF files
2. Select a domain tag
3. Click **Index Papers**
4. Use the search box to test retrieval

---

## Adding Papers to the Knowledge Base

**Via CLI:**
```bash
# Single paper
python ui/cli.py add-paper paper.pdf

# Batch (using shell loop)
for f in ./papers/*.pdf; do
    python ui/cli.py add-paper "$f" --domain machine_learning
done
```

**Via Python:**
```python
from src.rag.document_loader import DocumentLoader
from src.rag.embeddings import EmbeddingManager
from src.rag.vector_store import VectorStore

loader = DocumentLoader()
emb    = EmbeddingManager()
store  = VectorStore(persist_directory="./data/chroma_db", embedding_manager=emb)

docs = loader.load_directory("./data/papers")
chunks = loader.split_documents(docs)
store.add_documents(chunks)
print(f"Indexed {len(chunks)} chunks.")
```

---

## Fine-Tuning with Your Data

### 1. Prepare training data

Create a JSON file with instruction/response pairs:
```json
[
  {"instruction": "What is gradient descent?", "response": "Gradient descent is..."},
  {"instruction": "Explain backpropagation", "response": "Backpropagation computes..."}
]
```

Or export from conversation history:
```bash
python -c "
from src.memory.conversation_db import ConversationDB
import json
db = ConversationDB()
data = db.export_for_training()
json.dump(data, open('training_data.json','w'))
print(f'Exported {len(data)} examples')
"
```

### 2. Run training

```bash
python ui/cli.py train \
  --data training_data.json \
  --base-model meta-llama/Llama-2-7b-hf \
  --epochs 3 \
  --output ./data/lora_adapters/my_adapter
```

### 3. Apply the adapter

```python
from src.llm.model_manager import ModelManager
mgr = ModelManager()
mgr.apply_lora_adapter("./data/lora_adapters/my_adapter")
```

---

## Configuration Reference

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `llm` | `model` | `llama2` | Default Ollama model |
| `llm` | `temperature` | `0.7` | Sampling temperature |
| `llm` | `max_tokens` | `2048` | Max output tokens |
| `rag` | `chunk_size` | `1000` | Characters per chunk |
| `rag` | `chunk_overlap` | `200` | Overlap between chunks |
| `rag` | `top_k` | `5` | Documents to retrieve |
| `vector_store` | `persist_directory` | `./data/chroma_db` | ChromaDB path |
| `memory` | `db_path` | `./data/memory.db` | SQLite path |
| `lora` | `r` | `64` | LoRA rank |
| `lora` | `lora_alpha` | `128` | LoRA scaling |
