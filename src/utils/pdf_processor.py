"""PDF processing utilities using PyMuPDF."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extracts text and metadata from academic PDF papers.

    Tries PyMuPDF (``fitz``) for best results; gracefully degrades to
    ``pypdf`` if PyMuPDF is unavailable.

    Args:
        extract_references: If True, attempt to parse the reference list.
    """

    def __init__(self, extract_references: bool = False) -> None:
        self.extract_references = extract_references

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(self, path: str | Path) -> str:
        """Return the full text content of a PDF.

        Args:
            path: Path to the PDF file.

        Returns:
            Concatenated text from all pages.
        """
        path = Path(path)
        try:
            return self._text_pymupdf(path)
        except ImportError:
            return self._text_pypdf(path)

    def extract_metadata(self, path: str | Path) -> Dict[str, Any]:
        """Extract high-level metadata from a PDF.

        Attempts to parse title, authors, and abstract heuristically from
        the first page text when formal PDF metadata is missing.

        Args:
            path: Path to the PDF file.

        Returns:
            Dict with keys: ``title``, ``authors``, ``abstract``,
            ``references``, ``page_count``, ``file_name``.
        """
        path = Path(path)
        metadata: Dict[str, Any] = {
            "title": "",
            "authors": [],
            "abstract": "",
            "references": [],
            "page_count": 0,
            "file_name": path.name,
            "source": str(path),
        }

        try:
            import fitz  # type: ignore

            with fitz.open(str(path)) as pdf:
                metadata["page_count"] = len(pdf)
                info = pdf.metadata or {}
                metadata["title"] = info.get("title", "")
                if info.get("author"):
                    metadata["authors"] = [a.strip() for a in info["author"].split(";")]
                first_page_text = pdf[0].get_text() if len(pdf) > 0 else ""
        except ImportError:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            metadata["page_count"] = len(reader.pages)
            info = reader.metadata or {}
            metadata["title"] = info.get("/Title", "")
            first_page_text = reader.pages[0].extract_text() or "" if reader.pages else ""

        if not metadata["title"]:
            metadata["title"] = self._heuristic_title(first_page_text)
        if not metadata["authors"]:
            metadata["authors"] = self._heuristic_authors(first_page_text)
        metadata["abstract"] = self._extract_abstract(first_page_text)

        if self.extract_references:
            full_text = self.extract_text(path)
            metadata["references"] = self._extract_references(full_text)

        return metadata

    def process_paper(self, path: str | Path) -> Dict[str, Any]:
        """Extract both text and metadata from a PDF.

        Args:
            path: Path to the PDF file.

        Returns:
            Dict with ``text``, ``metadata``, and ``chunks`` keys.
        """
        path = Path(path)
        text = self.extract_text(path)
        meta = self.extract_metadata(path)
        return {"text": text, "metadata": meta, "source": str(path)}

    def batch_process(self, directory: str | Path) -> List[Dict[str, Any]]:
        """Process all PDF files in a directory.

        Args:
            directory: Directory containing PDF files.

        Returns:
            List of results from :meth:`process_paper`, one per PDF.
        """
        directory = Path(directory)
        results: List[Dict[str, Any]] = []
        pdf_files = sorted(directory.glob("**/*.pdf"))
        logger.info("Batch processing %d PDF files in '%s'.", len(pdf_files), directory)

        for pdf_path in pdf_files:
            try:
                results.append(self.process_paper(pdf_path))
            except Exception as exc:
                logger.error("Failed to process '%s': %s", pdf_path, exc)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _text_pymupdf(self, path: Path) -> str:
        import fitz  # type: ignore

        pages: List[str] = []
        with fitz.open(str(path)) as pdf:
            for page in pdf:
                pages.append(page.get_text())
        return "\n\n".join(pages)

    def _text_pypdf(self, path: Path) -> str:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)

    @staticmethod
    def _heuristic_title(text: str) -> str:
        """Use the first non-empty line as a proxy for the title."""
        for line in text.splitlines():
            stripped = line.strip()
            if len(stripped) > 10:
                return stripped[:200]
        return ""

    @staticmethod
    def _heuristic_authors(text: str) -> List[str]:
        """Very simple author extraction from first-page text."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for i, line in enumerate(lines[1:5], 1):
            if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", line):
                # Split on commas and "and"
                raw = re.split(r",\s*|\s+and\s+", line)
                return [a.strip() for a in raw if a.strip()]
        return []

    @staticmethod
    def _extract_abstract(text: str) -> str:
        """Try to find and return the abstract section."""
        match = re.search(
            r"(?i)\babstract\b[:\s\—–-]*(.+?)(?=\n\s*\n|\bintroduction\b|\b1\b\.)",
            text,
            re.DOTALL,
        )
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()[:1500]
        return ""

    @staticmethod
    def _extract_references(text: str) -> List[str]:
        """Extract reference entries from the references section."""
        match = re.search(r"(?i)\breferences\b[\s\n]+(.+)$", text, re.DOTALL)
        if not match:
            return []
        ref_text = match.group(1)
        # Split on numbered references [1], (1), or "1."
        entries = re.split(r"\n\s*[\[\(]?\d+[\]\)\.]\s+", ref_text)
        return [e.strip().replace("\n", " ") for e in entries if e.strip()][:100]
