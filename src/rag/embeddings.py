"""Embedding management for RAG pipelines."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages sentence-transformer embedding models.

    Provides text and query embedding with optional on-disk caching.

    Args:
        model_name: HuggingFace model identifier for the embeddings.
        cache_dir: Directory used to persist the embedding cache.  Pass
            ``None`` to disable caching.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).  Defaults
            to ``"cpu"``.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Optional[Any] = None
        self._cache: Dict[str, List[float]] = {}
        self._cache_path: Optional[Path] = None

        if cache_dir:
            self._cache_path = Path(cache_dir) / "embedding_cache.json"
            self._load_cache()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string, using the cache if available.

        Args:
            query: The query text to embed.

        Returns:
            A single embedding vector.
        """
        cache_key = hashlib.md5(f"{self.model_name}:{query}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        model = self._get_model()
        embedding: List[float] = model.encode(
            query, convert_to_numpy=True, show_progress_bar=False
        ).tolist()

        self._cache[cache_key] = embedding
        self._save_cache()
        return embedding

    # ------------------------------------------------------------------
    # LangChain integration
    # ------------------------------------------------------------------

    def get_langchain_embeddings(self) -> Any:
        """Return a LangChain-compatible embeddings object.

        Requires ``langchain-community`` and ``sentence-transformers``.

        Returns:
            A ``HuggingFaceEmbeddings`` instance configured with this
            manager's model.
        """
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError as exc:
            raise ImportError(
                "langchain-community is required. "
                "Install it with: pip install langchain-community"
            ) from exc

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> None:
        """Load the embedding cache from disk."""
        if self._cache_path and self._cache_path.exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as fh:
                    self._cache = json.load(fh)
                logger.debug("Loaded %d cached embeddings.", len(self._cache))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load embedding cache: %s", exc)

    def _save_cache(self) -> None:
        """Persist the embedding cache to disk."""
        if self._cache_path:
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._cache_path, "w", encoding="utf-8") as fh:
                    json.dump(self._cache, fh)
            except OSError as exc:
                logger.warning("Could not save embedding cache: %s", exc)

    def clear_cache(self) -> None:
        """Remove all cached embeddings from memory and disk."""
        self._cache.clear()
        if self._cache_path and self._cache_path.exists():
            self._cache_path.unlink()
