"""Document loading and chunking for RAG pipelines."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and chunks documents for use in RAG pipelines.

    Supports PDF, plain text, and Markdown files.  Returns LangChain
    ``Document`` objects so they can be fed directly into a vector store.

    Args:
        chunk_size: Target size (in characters) of each text chunk.
        chunk_overlap: Number of overlapping characters between adjacent
            chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_pdf(self, path: str | Path) -> List[Any]:
        """Load a PDF file and return a list of LangChain Documents.

        Tries PyMuPDF first; falls back to pypdf.

        Args:
            path: Absolute or relative path to the PDF file.

        Returns:
            List of ``langchain_core.documents.Document`` objects, one per
            page.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            return self._load_pdf_pymupdf(path)
        except ImportError:
            logger.debug("PyMuPDF not available; falling back to pypdf.")
        except Exception as exc:
            logger.warning("PyMuPDF failed (%s); falling back to pypdf.", exc)

        return self._load_pdf_pypdf(path)

    def load_text(self, path: str | Path) -> List[Any]:
        """Load a plain-text or Markdown file.

        Args:
            path: Path to the text file.

        Returns:
            A list containing a single ``Document`` with the full text.
        """
        from langchain_core.documents import Document  # type: ignore

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        metadata = {"source": str(path), "file_type": path.suffix, "file_name": path.name}
        return [Document(page_content=text, metadata=metadata)]

    def load_directory(
        self,
        directory: str | Path,
        glob_pattern: str = "**/*",
    ) -> List[Any]:
        """Recursively load all supported documents from a directory.

        Args:
            directory: Path to the directory.
            glob_pattern: Glob pattern to filter files.

        Returns:
            A flat list of ``Document`` objects from all found files.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        supported = {".pdf", ".txt", ".md"}
        docs: List[Any] = []

        for file_path in sorted(directory.glob(glob_pattern)):
            if not file_path.is_file() or file_path.suffix.lower() not in supported:
                continue
            try:
                if file_path.suffix.lower() == ".pdf":
                    docs.extend(self.load_pdf(file_path))
                else:
                    docs.extend(self.load_text(file_path))
            except Exception as exc:
                logger.error("Failed to load '%s': %s", file_path, exc)

        logger.info("Loaded %d documents from '%s'.", len(docs), directory)
        return docs

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split_documents(self, docs: List[Any]) -> List[Any]:
        """Split a list of Documents into smaller chunks.

        Args:
            docs: Documents produced by any of the ``load_*`` methods.

        Returns:
            A new list of (potentially shorter) ``Document`` objects with
            inherited metadata.
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info("Split %d documents into %d chunks.", len(docs), len(chunks))
        return chunks

    def load_and_split(self, path: str | Path) -> List[Any]:
        """Convenience wrapper: load then split a single file.

        Args:
            path: Path to the file to load.

        Returns:
            List of chunk-sized ``Document`` objects.
        """
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            docs = self.load_pdf(path)
        else:
            docs = self.load_text(path)
        return self.split_documents(docs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pdf_pymupdf(self, path: Path) -> List[Any]:
        """Load PDF pages using PyMuPDF (fitz)."""
        import fitz  # type: ignore
        from langchain_core.documents import Document  # type: ignore

        docs = []
        with fitz.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    metadata: Dict[str, Any] = {
                        "source": str(path),
                        "page": page_num + 1,
                        "total_pages": len(pdf),
                        "file_name": path.name,
                        "file_type": "pdf",
                    }
                    docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def _load_pdf_pypdf(self, path: Path) -> List[Any]:
        """Load PDF pages using pypdf."""
        from pypdf import PdfReader  # type: ignore
        from langchain_core.documents import Document  # type: ignore

        docs = []
        reader = PdfReader(str(path))
        total = len(reader.pages)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                metadata: Dict[str, Any] = {
                    "source": str(path),
                    "page": page_num + 1,
                    "total_pages": total,
                    "file_name": path.name,
                    "file_type": "pdf",
                }
                docs.append(Document(page_content=text, metadata=metadata))
        return docs
