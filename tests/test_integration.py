"""Integration tests for the full research assistant pipeline."""

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Full RAG pipeline (mocked external services)
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    """Test document indexing and RAG query end-to-end (with mocks)."""

    @pytest.fixture()
    def text_paper(self, tmp_path) -> Path:
        paper = tmp_path / "test_paper.txt"
        paper.write_text(
            "LoRA: Low-Rank Adaptation of Large Language Models\n\n"
            "Abstract: We propose LoRA, a method that freezes the pre-trained model weights "
            "and injects trainable rank decomposition matrices into each layer of the "
            "Transformer architecture, greatly reducing the number of trainable parameters "
            "for downstream tasks.\n\n"
            "Introduction: Fine-tuning large language models is expensive...",
            encoding="utf-8",
        )
        return paper

    def test_document_load_and_split(self, text_paper):
        from src.rag.document_loader import DocumentLoader

        loader = DocumentLoader(chunk_size=200, chunk_overlap=20)
        chunks = loader.load_and_split(text_paper)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.page_content.strip()
            assert "source" in chunk.metadata

    def test_full_rag_query(self, text_paper, tmp_path):
        """Load doc -> index -> retrieve -> answer (all external calls mocked)."""
        from src.rag.document_loader import DocumentLoader
        from src.rag.retriever import Retriever

        loader = DocumentLoader(chunk_size=300, chunk_overlap=30)
        docs = loader.load_and_split(text_paper)

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = docs[:2]

        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "LoRA reduces trainable parameters by injecting low-rank matrices."
        )

        retriever = Retriever(mock_vs, mock_client, k=3)
        result = retriever.rag_query("What is LoRA?")

        assert "answer" in result
        assert "LoRA" in result["answer"]
        assert len(result["sources"]) > 0


# ---------------------------------------------------------------------------
# Conversation DB integration
# ---------------------------------------------------------------------------


class TestConversationDBIntegration:
    def test_save_and_retrieve(self, tmp_path):
        from src.memory.conversation_db import ConversationDB

        db = ConversationDB(db_path=str(tmp_path / "test.db"))
        sid = str(uuid.uuid4())

        db.save_message(sid, "user", "What is attention?", domain="machine_learning")
        db.save_message(sid, "assistant", "Attention allows models to focus on relevant tokens.", domain="machine_learning")

        history = db.get_history(sid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_export_for_training(self, tmp_path):
        from src.memory.conversation_db import ConversationDB

        db = ConversationDB(db_path=str(tmp_path / "train.db"))
        sid = str(uuid.uuid4())

        db.save_message(sid, "user", "Explain transformers", domain="ml")
        db.save_message(sid, "assistant", "Transformers use self-attention mechanisms.", domain="ml")
        db.save_message(sid, "user", "What is BERT?", domain="ml")
        db.save_message(sid, "assistant", "BERT is a pre-trained transformer.", domain="ml")

        examples = db.export_for_training()
        assert len(examples) == 2
        assert examples[0]["instruction"] == "Explain transformers"
        assert "attention" in examples[0]["response"].lower()

    def test_delete_session(self, tmp_path):
        from src.memory.conversation_db import ConversationDB

        db = ConversationDB(db_path=str(tmp_path / "del.db"))
        sid = str(uuid.uuid4())

        db.save_message(sid, "user", "Hello", domain=None)
        assert len(db.get_history(sid)) == 1

        count = db.delete_session(sid)
        assert count == 1
        assert db.get_history(sid) == []


# ---------------------------------------------------------------------------
# Knowledge Graph integration
# ---------------------------------------------------------------------------


class TestKnowledgeGraphIntegration:
    def test_add_and_retrieve_concepts(self, tmp_path):
        from src.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(graph_path=str(tmp_path / "kg.json"))

        kg.add_concept("attention", domain="ml", description="Self-attention mechanism")
        kg.add_concept("transformer", domain="ml", description="Attention-based architecture")
        kg.add_relationship("transformer", "attention", relation_type="uses")

        related = kg.get_related_concepts("transformer", depth=1)
        names = [c["name"] for c in related]
        assert "attention" in names

    def test_search_concepts(self, tmp_path):
        from src.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(graph_path=str(tmp_path / "kg.json"))
        kg.add_concept("gradient descent", domain="ml", description="Optimisation algorithm")
        kg.add_concept("stochastic gradient descent", domain="ml", description="Mini-batch variant")

        results = kg.search_concepts("gradient")
        assert len(results) == 2

    def test_save_and_load_graph(self, tmp_path):
        from src.memory.knowledge_graph import KnowledgeGraph

        kg_path = str(tmp_path / "kg.json")
        kg = KnowledgeGraph(graph_path=kg_path)
        kg.add_concept("LoRA", domain="transfer_learning", description="Low-rank adaptation")
        kg.add_concept("fine-tuning", domain="transfer_learning")
        kg.add_relationship("LoRA", "fine-tuning", "is_a")
        kg.save_graph()

        # Load into a new instance
        kg2 = KnowledgeGraph(graph_path=kg_path)
        assert kg2.get_concept("LoRA") is not None
        assert len(kg2.get_edges_for("LoRA")) == 1


# ---------------------------------------------------------------------------
# Text utilities integration
# ---------------------------------------------------------------------------


class TestTextUtils:
    def test_chunk_text_roundtrip(self):
        from src.utils.text_utils import chunk_text, clean_text

        long_text = "The quick brown fox jumps over the lazy dog. " * 100
        cleaned = clean_text(long_text)
        chunks = chunk_text(cleaned, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        reconstructed = " ".join(chunks)
        # All original words should appear somewhere in the chunks
        assert "fox" in reconstructed

    def test_format_for_training(self):
        from src.utils.text_utils import format_for_training

        result = format_for_training("What is LoRA?", "LoRA reduces trainable parameters.")
        assert "### Instruction:" in result
        assert "### Response:" in result
        assert "LoRA" in result

    def test_detect_language(self):
        from src.utils.text_utils import detect_language

        assert detect_language("This is an English sentence about machine learning.") == "en"
        assert detect_language("这是一个关于机器学习的中文句子。") == "zh"
        assert detect_language("") == "unknown"
