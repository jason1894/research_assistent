"""ChromaDB-backed vector store for RAG pipelines."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """Wraps ChromaDB for document storage and similarity search.

    Documents are persisted to disk so they survive restarts.  Metadata
    filtering is forwarded directly to ChromaDB's ``where`` parameter.

    Args:
        persist_directory: Directory where ChromaDB persists its data.
        collection_name: Name of the ChromaDB collection to use.
        embedding_manager: An :class:`~src.rag.embeddings.EmbeddingManager`
            instance (or any object with a ``get_langchain_embeddings``
            method).
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "research_papers",
        embedding_manager: Optional[Any] = None,
    ) -> None:
        self.persist_directory = str(Path(persist_directory).resolve())
        self.collection_name = collection_name
        self._embedding_manager = embedding_manager
        self._vector_store: Optional[Any] = None

    # ------------------------------------------------------------------
    # Internal initialisation
    # ------------------------------------------------------------------

    def _get_embeddings(self) -> Any:
        """Return a LangChain-compatible embeddings object."""
        if self._embedding_manager is not None:
            return self._embedding_manager.get_langchain_embeddings()

        # Fallback: create a default EmbeddingManager
        from .embeddings import EmbeddingManager  # noqa: PLC0415

        return EmbeddingManager().get_langchain_embeddings()

    def _get_store(self) -> Any:
        """Lazily initialise and return the Chroma vector store."""
        if self._vector_store is None:
            from langchain_community.vectorstores import Chroma  # type: ignore

            embeddings = self._get_embeddings()
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info(
                "ChromaDB collection '%s' opened at '%s'.",
                self.collection_name,
                self.persist_directory,
            )
        return self._vector_store

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def add_documents(self, docs: List[Any]) -> List[str]:
        """Add documents to the vector store.

        Args:
            docs: List of ``langchain_core.documents.Document`` objects.

        Returns:
            List of ChromaDB document IDs that were inserted.
        """
        store = self._get_store()
        ids = store.add_documents(docs)
        logger.info("Added %d documents to collection '%s'.", len(docs), self.collection_name)
        return ids

    def delete_collection(self) -> None:
        """Delete the entire collection and all its documents."""
        store = self._get_store()
        store.delete_collection()
        self._vector_store = None
        logger.info("Deleted collection '%s'.", self.collection_name)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Return the top-k most similar documents for a query.

        Args:
            query: The search query.
            k: Number of documents to return.
            filter_metadata: Optional ChromaDB ``where`` filter dict.

        Returns:
            List of ``Document`` objects ordered by similarity.
        """
        store = self._get_store()
        kwargs: Dict[str, Any] = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata
        return store.similarity_search(query, **kwargs)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[tuple]:
        """Return top-k documents together with their similarity scores.

        Args:
            query: The search query.
            k: Number of documents to return.
            filter_metadata: Optional ChromaDB ``where`` filter dict.

        Returns:
            List of ``(Document, score)`` tuples.
        """
        store = self._get_store()
        kwargs: Dict[str, Any] = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata
        return store.similarity_search_with_score(query, **kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_collection_info(self) -> Dict[str, Any]:
        """Return basic metadata about the collection.

        Returns:
            Dict with ``collection_name``, ``persist_directory``, and
            ``document_count`` keys.
        """
        store = self._get_store()
        try:
            count: int = store._collection.count()  # type: ignore[attr-defined]
        except Exception:
            count = -1

        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": count,
        }

    def as_langchain_store(self) -> Any:
        """Return the underlying LangChain Chroma instance."""
        return self._get_store()
