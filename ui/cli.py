#!/usr/bin/env python3
"""Command-line interface for the research assistant."""

import sys
import uuid
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def _load_config() -> dict:
    """Return the merged application config."""
    import yaml  # type: ignore

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        console.print("[yellow]Warning:[/yellow] config.yaml not found; using defaults.")
        return {}


def _get_client(cfg: dict):
    """Build and return an OllamaClient from config."""
    from src.llm.ollama_client import OllamaClient  # type: ignore

    llm_cfg = cfg.get("llm", {})
    return OllamaClient(
        base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        model=llm_cfg.get("model", "llama2"),
        temperature=llm_cfg.get("temperature", 0.7),
        max_tokens=llm_cfg.get("max_tokens", 2048),
    )


@click.group()
@click.version_option("0.1.0", prog_name="research-assistant")
def main() -> None:
    """🔬 Research Assistant – AI-powered academic research tool."""


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@main.command()
@click.option("--model", "-m", default=None, help="Override the default LLM model.")
@click.option("--session", "-s", default=None, help="Session ID (auto-generated if omitted).")
@click.option("--domain", "-d", default=None, help="Research domain tag (e.g. machine_learning).")
def chat(model: Optional[str], session: Optional[str], domain: Optional[str]) -> None:
    """Start an interactive chat session with the research assistant."""
    cfg = _load_config()
    client = _get_client(cfg)

    if not client.check_connection():
        console.print(
            Panel(
                "[red]Cannot connect to Ollama.[/red]\nStart the server with: [bold]ollama serve[/bold]",
                title="Connection Error",
            )
        )
        sys.exit(1)

    session_id = session or str(uuid.uuid4())[:8]
    effective_model = model or cfg.get("llm", {}).get("model", "llama2")

    from src.memory.conversation_db import ConversationDB  # type: ignore

    db_path = cfg.get("memory", {}).get("db_path", "./data/memory.db")
    db = ConversationDB(db_path=db_path, max_history=cfg.get("memory", {}).get("max_history", 100))

    console.print(
        Panel(
            f"Model: [bold cyan]{effective_model}[/bold cyan]  |  "
            f"Session: [bold]{session_id}[/bold]  |  "
            f"Domain: [bold]{domain or 'general'}[/bold]\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to end the session.",
            title="💬 Research Assistant Chat",
        )
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI research assistant specialising in machine learning, "
                "statistics, and mathematics. Provide accurate, well-reasoned answers."
            ),
        }
    ]

    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Ending session.[/yellow]")
            break

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            console.print("[yellow]Goodbye![/yellow]")
            break
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        db.save_message(session_id, "user", user_input, domain=domain)

        with Progress(SpinnerColumn(), TextColumn("[cyan]Thinking…"), transient=True) as prog:
            prog.add_task("", total=None)
            try:
                response = client.chat(messages, model=effective_model)
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")
                messages.pop()
                continue

        messages.append({"role": "assistant", "content": response})
        db.save_message(session_id, "assistant", response, domain=domain)

        console.print(Panel(Markdown(response), title="[bold green]Assistant[/bold green]"))


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--model", "-m", default=None, help="Override LLM model.")
def review(file_path: str, model: Optional[str]) -> None:
    """Review an academic paper (PDF or text file)."""
    import json

    cfg = _load_config()
    client = _get_client(cfg)

    if not client.check_connection():
        console.print("[red]Cannot connect to Ollama. Run: ollama serve[/red]")
        sys.exit(1)

    effective_model = model or cfg.get("llm", {}).get("model", "llama2")

    path = Path(file_path)
    with Progress(SpinnerColumn(), TextColumn(f"[cyan]Loading {path.name}…"), transient=True) as prog:
        prog.add_task("", total=None)
        if path.suffix.lower() == ".pdf":
            from src.utils.pdf_processor import PDFProcessor  # type: ignore

            processor = PDFProcessor()
            data = processor.process_paper(path)
            content = data["text"][:6000]
        else:
            content = path.read_text(encoding="utf-8", errors="replace")[:6000]

    prompts_path = Path(__file__).parent.parent / "config" / "prompts.json"
    try:
        with open(prompts_path, "r", encoding="utf-8") as fh:
            prompts = json.load(fh)
        template = prompts["paper_review"]["template"]
    except Exception:
        template = "Please review the following paper:\n\n{paper_content}"

    prompt = template.format(paper_content=content)

    console.print(f"[cyan]Reviewing:[/cyan] {path.name}")
    with Progress(SpinnerColumn(), TextColumn("[cyan]Generating review…"), transient=True) as prog:
        prog.add_task("", total=None)
        response = client.generate(prompt, model=effective_model)

    console.print(Panel(Markdown(response), title=f"📋 Review: {path.name}"))


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

@main.command()
@click.option("--type", "content_type", default="abstract", show_default=True,
              help="Content type: abstract, introduction, conclusion, etc.")
@click.option("--topic", prompt="Topic", help="Research topic or title.")
@click.option("--model", "-m", default=None)
def write(content_type: str, topic: str, model: Optional[str]) -> None:
    """Get academic writing assistance."""
    import json

    cfg = _load_config()
    client = _get_client(cfg)

    if not client.check_connection():
        console.print("[red]Cannot connect to Ollama. Run: ollama serve[/red]")
        sys.exit(1)

    effective_model = model or cfg.get("llm", {}).get("model", "llama2")

    additional = Prompt.ask("[cyan]Additional requirements (optional)[/cyan]", default="")

    prompts_path = Path(__file__).parent.parent / "config" / "prompts.json"
    try:
        with open(prompts_path, "r", encoding="utf-8") as fh:
            prompts = json.load(fh)
        template = prompts["writing_assistant"]["template"]
    except Exception:
        template = "Write a {content_type} about {topic}."

    prompt = template.format(
        content_type=content_type,
        topic=topic,
        context="",
        requirements=additional,
    )

    with Progress(SpinnerColumn(), TextColumn("[cyan]Writing…"), transient=True) as prog:
        prog.add_task("", total=None)
        response = client.generate(prompt, model=effective_model)

    console.print(Panel(Markdown(response), title=f"✍️  Writing: {content_type}"))


# ---------------------------------------------------------------------------
# add-paper
# ---------------------------------------------------------------------------

@main.command("add-paper")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--domain", "-d", default="machine_learning", show_default=True)
def add_paper(file_path: str, domain: str) -> None:
    """Add a PDF paper to the RAG knowledge base."""
    cfg = _load_config()
    path = Path(file_path)

    rag_cfg = cfg.get("rag", {})
    vs_cfg = cfg.get("vector_store", {})

    from src.rag.document_loader import DocumentLoader  # type: ignore
    from src.rag.embeddings import EmbeddingManager  # type: ignore
    from src.rag.vector_store import VectorStore  # type: ignore

    loader = DocumentLoader(
        chunk_size=rag_cfg.get("chunk_size", 1000),
        chunk_overlap=rag_cfg.get("chunk_overlap", 200),
    )
    emb = EmbeddingManager(model_name=rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    store = VectorStore(
        persist_directory=vs_cfg.get("persist_directory", "./data/chroma_db"),
        collection_name=vs_cfg.get("collection_name", "research_papers"),
        embedding_manager=emb,
    )

    with Progress(SpinnerColumn(), TextColumn(f"[cyan]Processing {path.name}…"), transient=True) as prog:
        prog.add_task("", total=None)
        docs = loader.load_and_split(path)
        for doc in docs:
            doc.metadata["domain"] = domain
        ids = store.add_documents(docs)

    console.print(
        f"[green]✓[/green] Added [bold]{len(ids)}[/bold] chunks from "
        f"[bold]{path.name}[/bold] to the knowledge base."
    )


# ---------------------------------------------------------------------------
# list-papers
# ---------------------------------------------------------------------------

@main.command("list-papers")
def list_papers() -> None:
    """List papers currently indexed in the knowledge base."""
    cfg = _load_config()
    vs_cfg = cfg.get("vector_store", {})
    rag_cfg = cfg.get("rag", {})

    from src.rag.embeddings import EmbeddingManager  # type: ignore
    from src.rag.vector_store import VectorStore  # type: ignore

    emb = EmbeddingManager(model_name=rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    store = VectorStore(
        persist_directory=vs_cfg.get("persist_directory", "./data/chroma_db"),
        collection_name=vs_cfg.get("collection_name", "research_papers"),
        embedding_manager=emb,
    )

    info = store.get_collection_info()
    table = Table(title="📚 Knowledge Base")
    table.add_column("Collection", style="cyan")
    table.add_column("Documents", justify="right", style="green")
    table.add_column("Path", style="dim")
    table.add_row(
        info["collection_name"],
        str(info["document_count"]),
        info["persist_directory"],
    )
    console.print(table)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@main.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True),
              help="Path to training data (JSON/JSONL with instruction+response fields).")
@click.option("--base-model", default=None, help="HuggingFace base model ID.")
@click.option("--output", default=None, help="Output directory for the adapter.")
@click.option("--epochs", default=3, show_default=True, type=int)
def train(data_path: str, base_model: Optional[str], output: Optional[str], epochs: int) -> None:
    """Fine-tune the LLM with a LoRA adapter on your research data."""
    cfg = _load_config()
    lora_cfg_data = cfg.get("lora", {})

    from src.lora.config import LoRAConfig  # type: ignore
    from src.lora.trainer import LoRATrainer  # type: ignore

    lora_config = LoRAConfig(
        base_model=base_model or lora_cfg_data.get("base_model", "meta-llama/Llama-2-7b-hf"),
        output_dir=output or lora_cfg_data.get("output_dir", "./data/lora_adapters"),
        num_train_epochs=epochs,
        r=lora_cfg_data.get("r", 64),
        lora_alpha=lora_cfg_data.get("lora_alpha", 128),
    )

    trainer = LoRATrainer(config=lora_config)

    console.print("[cyan]Preparing model…[/cyan]")
    trainer.prepare_model()

    console.print("[cyan]Loading dataset…[/cyan]")
    trainer.prepare_dataset(data_path)

    console.print("[cyan]Starting training…[/cyan]")
    trainer.train()

    saved = trainer.save_adapter()
    console.print(f"[green]✓[/green] Adapter saved to: [bold]{saved}[/bold]")


if __name__ == "__main__":
    main()
