"""Adapter that lets any LangChain retriever or vector store feed RAGGrounding."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from ..interfaces.retriever import Retriever

if TYPE_CHECKING:
    from langchain_core.documents import Document

_UNSAFE_ID_CHARS = re.compile(r"[\[\],\s]+")  # would break "[id]" / "[a, b]" citation parsing
_METADATA_ID_KEYS = ("id", "doc_id", "chunk_id")


def default_chunk_id(document: Document) -> str:
    """Derive a short, citable id: metadata id, else "<source name>#p<page>", else the store id, else a hash."""
    metadata = document.metadata or {}
    for key in _METADATA_ID_KEYS:
        if metadata.get(key) not in (None, ""):
            return str(metadata[key])
    if source := metadata.get("source"):
        name = PurePosixPath(str(source).replace("\\", "/")).name or str(source)
        page = metadata.get("page")
        return f"{name}#p{page}" if page is not None else name
    if store_id := getattr(document, "id", None):
        return str(store_id)
    return hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()[:10]


def documents_to_chunks(
    documents: Iterable[Document], chunk_id: Callable[[Document], str | None] = default_chunk_id
) -> list[str]:
    """Turn Documents into "[id] text" chunks with unique, citation-safe ids; empty documents are skipped."""
    seen: dict[str, int] = {}
    chunks: list[str] = []
    for document in documents:
        if not document.page_content.strip():
            continue
        base_id = _UNSAFE_ID_CHARS.sub("_", chunk_id(document) or default_chunk_id(document)).strip("_")
        seen[base_id] = seen.get(base_id, 0) + 1
        unique_id = base_id if seen[base_id] == 1 else f"{base_id}~{seen[base_id]}"
        chunks.append(f"[{unique_id}] {document.page_content.strip()}")
    return chunks


class LangChainRetriever(Retriever):
    """Wrap a LangChain retriever, vector store or document-returning Runnable as a halshield Retriever.

    Works with anything LangChain can search: `vectorstore.as_retriever()` over Chroma, PGVector,
    Pinecone, FAISS..., BM25/ensemble retrievers, or a chain that returns `list[Document]`. The vector
    database stays LangChain's concern; this class only turns Documents into citable "[id] text" chunks.

        grounding = RAGGrounding(retriever=LangChainRetriever(chroma.as_retriever(search_kwargs={"k": 4})))

    Ids come from `chunk_id(document)` (default: `default_chunk_id`), are made safe for citation
    parsing, and are de-duplicated within one result ("a", "a~2", ...).
    """

    def __init__(
        self,
        source: Any,
        k: int | None = None,
        chunk_id: Callable[[Document], str | None] | None = None,
    ) -> None:
        try:
            from langchain_core.runnables import Runnable
            from langchain_core.vectorstores import VectorStore
        except ImportError as exc:
            raise ImportError("LangChainRetriever needs langchain-core: pip install 'halshield[langchain]'") from exc
        if k is not None and k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if isinstance(source, VectorStore):
            source = source.as_retriever(search_kwargs={"k": k or 4})
        if not isinstance(source, Runnable):
            raise TypeError(f"expected a LangChain retriever, vector store or Runnable, got {type(source).__name__}")
        self.runnable, self.k = source, k
        self.chunk_id = chunk_id or default_chunk_id

    def documents(self, question: str) -> list[Document]:
        """Return the LangChain Documents for `question` (at most `k` if set)."""
        if not question.strip():
            raise ValueError("question must be a non-empty string")
        documents = self.runnable.invoke(question)
        if not isinstance(documents, list):
            raise TypeError(f"retriever must return list[Document], got {type(documents).__name__}")
        return documents[: self.k] if self.k else documents

    def retrieve(self, question: str) -> list[str]:
        """Return the retrieved documents as "[id] text" chunks with unique, citation-safe ids."""
        return documents_to_chunks(self.documents(question), self.chunk_id)
