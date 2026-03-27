"""Unit tests for RAG components."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------


class TestDocumentLoader:
    def test_load_text_file(self, tmp_path):
        from src.rag.document_loader import DocumentLoader

        text_file = tmp_path / "paper.txt"
        text_file.write_text("This is a test paper about deep learning.", encoding="utf-8")

        loader = DocumentLoader(chunk_size=100, chunk_overlap=10)
        docs = loader.load_text(text_file)
        assert len(docs) == 1
        assert "deep learning" in docs[0].page_content

    def test_load_missing_file_raises(self, tmp_path):
        from src.rag.document_loader import DocumentLoader

        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_text(tmp_path / "missing.txt")

    def test_split_documents(self, tmp_path):
        from src.rag.document_loader import DocumentLoader
        from langchain_core.documents import Document

        long_text = "word " * 500  # 2500 chars
        docs = [Document(page_content=long_text, metadata={"source": "test"})]

        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        chunks = loader.split_documents(docs)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 600  # small buffer for splitter

    def test_load_directory_nonexistent_raises(self, tmp_path):
        from src.rag.document_loader import DocumentLoader

        loader = DocumentLoader()
        with pytest.raises(NotADirectoryError):
            loader.load_directory(tmp_path / "nonexistent")

    def test_load_directory_returns_documents(self, tmp_path):
        from src.rag.document_loader import DocumentLoader

        (tmp_path / "paper1.txt").write_text("Paper one content.", encoding="utf-8")
        (tmp_path / "paper2.txt").write_text("Paper two content.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path)
        assert len(docs) == 2


# ---------------------------------------------------------------------------
# EmbeddingManager
# ---------------------------------------------------------------------------


class TestEmbeddingManager:
    def test_embed_texts_returns_list(self):
        from src.rag.embeddings import EmbeddingManager

        mgr = EmbeddingManager.__new__(EmbeddingManager)
        mgr.model_name = "test-model"
        mgr.device = "cpu"
        mgr._model = None
        mgr._cache = {}
        mgr._cache_path = None

        mock_model = MagicMock()
        import numpy as np

        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mgr._model = mock_model

        result = mgr.embed_texts(["text one", "text two"])
        assert len(result) == 2
        assert len(result[0]) == 2

    def test_embed_query_uses_cache(self):
        from src.rag.embeddings import EmbeddingManager

        mgr = EmbeddingManager.__new__(EmbeddingManager)
        mgr.model_name = "test-model"
        mgr.device = "cpu"
        mgr._model = None
        mgr._cache = {}
        mgr._cache_path = None

        import hashlib, numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.5, 0.6])
        mgr._model = mock_model

        # First call populates cache
        r1 = mgr.embed_query("research query")
        # Second call should hit cache (no model encode)
        mock_model.encode.reset_mock()
        r2 = mgr.embed_query("research query")
        mock_model.encode.assert_not_called()
        assert r1 == r2


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class TestVectorStore:
    def test_add_documents_calls_chroma(self):
        from src.rag.vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)
        vs.persist_directory = "/tmp/test_chroma"
        vs.collection_name = "test"
        vs._embedding_manager = None
        vs._vector_store = None

        mock_store = MagicMock()
        mock_store.add_documents.return_value = ["id1", "id2"]
        vs._vector_store = mock_store

        result = vs.add_documents([MagicMock(), MagicMock()])
        assert result == ["id1", "id2"]
        mock_store.add_documents.assert_called_once()

    def test_similarity_search_forwards_to_chroma(self):
        from src.rag.vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)
        vs.persist_directory = "/tmp/test_chroma"
        vs.collection_name = "test"
        vs._embedding_manager = None

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [MagicMock()]
        vs._vector_store = mock_store

        results = vs.similarity_search("neural networks", k=3)
        assert len(results) == 1
        mock_store.similarity_search.assert_called_once_with("neural networks", k=3)

    def test_get_collection_info(self):
        from src.rag.vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)
        vs.persist_directory = "./data/chroma_db"
        vs.collection_name = "research_papers"
        vs._embedding_manager = None

        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_store = MagicMock()
        mock_store._collection = mock_collection
        vs._vector_store = mock_store

        info = vs.get_collection_info()
        assert info["collection_name"] == "research_papers"
        assert info["document_count"] == 42


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class TestRetriever:
    def _make_retriever(self):
        from src.rag.retriever import Retriever

        mock_vs = MagicMock()
        mock_client = MagicMock()
        return Retriever(mock_vs, mock_client, k=3)

    def test_retrieve_calls_similarity_search(self):
        retriever = self._make_retriever()
        retriever.vector_store.similarity_search.return_value = [MagicMock(), MagicMock()]
        docs = retriever.retrieve("test query")
        assert len(docs) == 2
        retriever.vector_store.similarity_search.assert_called_once()

    def test_rag_query_returns_answer(self):
        retriever = self._make_retriever()

        from langchain_core.documents import Document

        doc = Document(
            page_content="LoRA reduces trainable parameters.", metadata={"source": "paper.pdf", "page": 1}
        )
        retriever.vector_store.similarity_search.return_value = [doc]
        retriever.ollama_client.generate.return_value = "LoRA is efficient."

        result = retriever.rag_query("What is LoRA?")
        assert result["answer"] == "LoRA is efficient."
        assert len(result["sources"]) == 1

    def test_rag_query_no_docs(self):
        retriever = self._make_retriever()
        retriever.vector_store.similarity_search.return_value = []
        retriever.ollama_client.generate.return_value = "I don't know."

        result = retriever.rag_query("obscure query")
        assert "answer" in result
        assert result["sources"] == []
