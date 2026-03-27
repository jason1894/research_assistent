# API Reference

## OllamaClient

```python
from src.llm.ollama_client import OllamaClient
```

### Constructor

```python
OllamaClient(
    base_url: str = "http://localhost:11434",
    model: str = "llama2",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 120,
)
```

### Methods

#### `check_connection() -> bool`
Returns `True` if the Ollama server is reachable.

#### `list_models() -> List[Dict]`
Returns a list of models available on the server.
```python
models = client.list_models()
# [{"name": "llama2:latest", "size": ...}, ...]
```

#### `pull_model(model_name: str) -> bool`
Downloads a model from the Ollama registry.
```python
client.pull_model("mistral")
```

#### `generate(prompt, model=None, stream=False, options=None) -> str | Generator`
Generate a text completion.
```python
# Non-streaming
text = client.generate("Explain LoRA in one paragraph")

# Streaming
for chunk in client.generate("Explain LoRA", stream=True):
    print(chunk, end="", flush=True)
```

#### `chat(messages, model=None, stream=False, options=None) -> str | Generator`
Send a chat request with message history.
```python
messages = [
    {"role": "system", "content": "You are a research assistant."},
    {"role": "user", "content": "What is attention?"},
]
response = client.chat(messages)
```

#### `get_langchain_llm(model=None) -> Ollama`
Returns a LangChain-compatible LLM object.

---

## VectorStore

```python
from src.rag.vector_store import VectorStore
```

### Constructor

```python
VectorStore(
    persist_directory: str = "./data/chroma_db",
    collection_name: str = "research_papers",
    embedding_manager=None,  # EmbeddingManager instance
)
```

### Methods

#### `add_documents(docs: List[Document]) -> List[str]`
Index a list of LangChain `Document` objects. Returns ChromaDB IDs.
```python
ids = store.add_documents(chunks)
```

#### `similarity_search(query, k=5, filter_metadata=None) -> List[Document]`
Find the top-k most similar documents.
```python
docs = store.similarity_search("transformer attention", k=3)
for doc in docs:
    print(doc.page_content[:200])
    print(doc.metadata)
```

#### `similarity_search_with_scores(query, k=5, filter_metadata=None) -> List[tuple]`
Same as above but returns `(Document, score)` tuples.

#### `delete_collection() -> None`
Remove the entire collection from ChromaDB.

#### `get_collection_info() -> Dict`
Returns `{"collection_name", "persist_directory", "document_count"}`.

#### `as_langchain_store() -> Chroma`
Returns the underlying LangChain Chroma instance for use in chains.

---

## ConversationDB

```python
from src.memory.conversation_db import ConversationDB
```

### Constructor

```python
ConversationDB(
    db_path: str = "./data/memory.db",
    max_history: int = 100,
)
```

### Methods

#### `save_message(session_id, role, content, domain=None, feedback=None) -> int`
Save a conversation turn. Returns the row ID.
```python
msg_id = db.save_message("session-123", "user", "What is BERT?", domain="ml")
```

#### `get_history(session_id, limit=None) -> List[Dict]`
Returns conversation history ordered oldest-first.
```python
history = db.get_history("session-123", limit=20)
# [{"id": 1, "role": "user", "content": "...", "timestamp": "...", ...}, ...]
```

#### `get_all_sessions() -> List[str]`
Returns all known session IDs.

#### `delete_session(session_id) -> int`
Deletes all messages for a session. Returns number of deleted rows.

#### `export_for_training(min_feedback=None) -> List[Dict]`
Exports conversation pairs as `{"instruction": ..., "response": ...}` dicts for LoRA training.
```python
examples = db.export_for_training(min_feedback="positive")
```

#### `save_feedback(message_id, feedback) -> bool`
Update the feedback field for a message (`"positive"` or `"negative"`).

---

## LoRATrainer

```python
from src.lora.trainer import LoRATrainer
from src.lora.config import LoRAConfig
```

### Constructor

```python
LoRATrainer(config: Optional[LoRAConfig] = None)
```

### Methods

#### `prepare_model() -> None`
Load the base model and apply LoRA adapters (with 4-bit quantisation).

#### `prepare_dataset(data_path: str) -> None`
Load and tokenise training data. Supports JSON/JSONL/CSV.

#### `train() -> None`
Run the training loop. Requires `prepare_model()` and `prepare_dataset()` first.

#### `save_adapter(path=None) -> str`
Save the trained LoRA adapter weights. Returns the saved path.

#### `load_adapter(path: str) -> None`
Load a previously saved adapter.

#### `merge_and_save(output_path=None) -> str`
Merge LoRA weights into the base model and save. Returns output path.

### LoRAConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `r` | int | 64 | LoRA rank |
| `lora_alpha` | int | 128 | Scaling factor |
| `lora_dropout` | float | 0.1 | Dropout |
| `target_modules` | List[str] | `["q_proj","v_proj"]` | Modules to adapt |
| `base_model` | str | `meta-llama/Llama-2-7b-hf` | HF model ID |
| `output_dir` | str | `./data/lora_adapters` | Output directory |
| `num_train_epochs` | int | 3 | Training epochs |
| `learning_rate` | float | 2e-4 | Peak LR |
| `load_in_4bit` | bool | True | 4-bit quantisation |

---

## KnowledgeGraph

```python
from src.memory.knowledge_graph import KnowledgeGraph
```

### Constructor

```python
KnowledgeGraph(graph_path: Optional[str] = "./data/knowledge_graph.json")
```

### Methods

#### `add_concept(name, domain="", description="", related_papers=None) -> None`
Add or update a concept node.

#### `add_relationship(source, target, relation_type="related_to") -> None`
Create a directed edge between two concepts.

#### `get_related_concepts(name, depth=2) -> List[Dict]`
BFS traversal returning concepts within `depth` hops.

#### `search_concepts(query) -> List[Dict]`
Case-insensitive substring search.

#### `export_graph() -> Dict`
Return the full graph as a dict with `nodes` and `edges`.

#### `save_graph(path=None) -> str`
Persist to JSON file.

#### `import_graph(path) -> None`
Load from a JSON file.

---

## DocumentLoader

```python
from src.rag.document_loader import DocumentLoader
```

### Constructor

```python
DocumentLoader(chunk_size: int = 1000, chunk_overlap: int = 200)
```

### Methods

| Method | Description |
|--------|-------------|
| `load_pdf(path)` | Load PDF, returns `List[Document]` |
| `load_text(path)` | Load text/markdown file |
| `load_directory(path)` | Recursively load all supported files |
| `split_documents(docs)` | Split documents into chunks |
| `load_and_split(path)` | Convenience: load + split in one call |
