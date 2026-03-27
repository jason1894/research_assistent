"""RAG retriever combining VectorStore with an LLM."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Retriever:
    """Combines a VectorStore and an LLM for retrieval-augmented generation.

    Supports simple similarity retrieval as well as full RAG queries
    where retrieved context is injected into an LLM prompt.

    Args:
        vector_store: A :class:`~src.rag.vector_store.VectorStore` instance.
        ollama_client: An :class:`~src.llm.ollama_client.OllamaClient` instance.
        k: Default number of documents to retrieve.
        strategy: Retrieval strategy – ``"similarity"`` or ``"mmr"``.
        prompt_template: Optional prompt template string.  Must contain
            ``{context}`` and ``{question}`` placeholders.
    """

    DEFAULT_TEMPLATE = (
        "Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def __init__(
        self,
        vector_store: Any,
        ollama_client: Any,
        k: int = 5,
        strategy: str = "similarity",
        prompt_template: Optional[str] = None,
    ) -> None:
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.k = k
        self.strategy = strategy
        self.prompt_template = prompt_template or self.DEFAULT_TEMPLATE

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Retrieve the most relevant documents for a query.

        Args:
            query: The search query.
            k: Number of documents to retrieve (overrides ``self.k``).
            filter_metadata: Optional metadata filter dict.

        Returns:
            List of ``Document`` objects.
        """
        num_docs = k or self.k

        if self.strategy == "mmr":
            return self._retrieve_mmr(query, num_docs, filter_metadata)

        return self.vector_store.similarity_search(
            query, k=num_docs, filter_metadata=filter_metadata
        )

    def _retrieve_mmr(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> List[Any]:
        """Retrieve using Maximal Marginal Relevance."""
        store = self.vector_store.as_langchain_store()
        kwargs: Dict[str, Any] = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata
        return store.max_marginal_relevance_search(query, **kwargs)

    # ------------------------------------------------------------------
    # RAG query
    # ------------------------------------------------------------------

    def rag_query(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer a question using retrieved context (RAG).

        Args:
            query: The research question.
            k: Number of documents to retrieve.
            filter_metadata: Optional metadata filter.
            model: LLM model override.

        Returns:
            Dict with ``answer``, ``sources``, and ``context`` keys.
        """
        docs = self.retrieve(query, k=k, filter_metadata=filter_metadata)

        if not docs:
            logger.warning("No documents retrieved for query: %s", query)
            context = "No relevant context found."
        else:
            context = "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('source', 'unknown')}, "
                f"Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
                for doc in docs
            )

        prompt = self.prompt_template.format(context=context, question=query)

        try:
            answer = self.ollama_client.generate(prompt, model=model)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            answer = f"Error generating answer: {exc}"

        return {
            "answer": answer,
            "sources": [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page"),
                    "snippet": doc.page_content[:300],
                }
                for doc in docs
            ],
            "context": context,
            "query": query,
        }

    # ------------------------------------------------------------------
    # LangChain integration
    # ------------------------------------------------------------------

    def get_langchain_retriever(self, k: Optional[int] = None) -> Any:
        """Return a LangChain ``BaseRetriever`` for use in chains.

        Args:
            k: Number of documents to retrieve.

        Returns:
            A LangChain ``VectorStoreRetriever`` instance.
        """
        num_docs = k or self.k
        store = self.vector_store.as_langchain_store()

        if self.strategy == "mmr":
            return store.as_retriever(
                search_type="mmr", search_kwargs={"k": num_docs}
            )

        return store.as_retriever(
            search_type="similarity", search_kwargs={"k": num_docs}
        )

    def build_langchain_rag_chain(self, llm: Optional[Any] = None) -> Any:
        """Build a LangChain RAG chain (RetrievalQA).

        Args:
            llm: A LangChain-compatible LLM.  Falls back to
                ``ollama_client.get_langchain_llm()`` if not provided.

        Returns:
            A ``RetrievalQA`` chain.
        """
        from langchain.chains import RetrievalQA  # type: ignore
        from langchain.prompts import PromptTemplate  # type: ignore

        effective_llm = llm or self.ollama_client.get_langchain_llm()
        retriever = self.get_langchain_retriever()

        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"],
        )

        return RetrievalQA.from_chain_type(
            llm=effective_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
