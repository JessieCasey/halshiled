"""LangChainRetriever: reuse LangChain retrievers and vector stores without owning any infrastructure."""

from __future__ import annotations

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

from halshield import LangChainRetriever, RAGGrounding
from halshield.retrieval import default_chunk_id

from fakes import RecordingNode


class FixedRetriever(BaseRetriever):
    """A LangChain retriever that always returns the same documents."""

    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.documents


@pytest.mark.parametrize(
    ("document", "expected_id"),
    [
        (Document("t", metadata={"id": "kb-7", "source": "x.pdf"}), "kb-7"),
        (Document("t", metadata={"source": "/data/specs/K3 sheet.pdf", "page": 2}), "K3 sheet.pdf#p2"),
        (Document("t", metadata={"source": "https://kb.example.com/atlas"}), "atlas"),
        (Document("t", id="store-uuid"), "store-uuid"),
        (Document("same text"), default_chunk_id(Document("same text"))),  # stable content hash
    ],
    ids=["metadata-id", "source-and-page", "url-source", "store-id", "hash-fallback"],
)
def test_default_chunk_id(document, expected_id):
    assert default_chunk_id(document) == expected_id


def test_ids_are_citation_safe_and_unique():
    documents = [
        Document("First page.", metadata={"source": "K3 sheet.pdf", "page": 1}),
        Document("Same page again.", metadata={"source": "K3 sheet.pdf", "page": 1}),
        Document("Commas, brackets.", metadata={"id": "a,b [x]"}),
        Document("   "),  # empty chunks are skipped
    ]
    chunks = LangChainRetriever(FixedRetriever(documents=documents)).retrieve("q")
    assert chunks == [
        "[K3_sheet.pdf#p1] First page.",
        "[K3_sheet.pdf#p1~2] Same page again.",
        "[a_b_x] Commas, brackets.",
    ]


def test_vector_store_is_searched_through_langchain():
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
    store.add_documents([Document(f"chunk {n}", metadata={"id": f"c{n}"}) for n in range(5)])
    chunks = LangChainRetriever(store, k=2).retrieve("chunk 3")
    assert len(chunks) == 2 and all(chunk.startswith("[c") for chunk in chunks)


def test_grounding_accepts_langchain_objects_directly():
    retriever = FixedRetriever(documents=[Document("The K2 flies 38 minutes.", metadata={"source": "k2.md"})])
    grounding = RAGGrounding(retriever=retriever)  # wrapped automatically
    node = RecordingNode(answer="It flies 38 minutes [k2.md]. It is popular.", sources=[], confidence=0.9)
    output = grounding.run({"question": "How long does the K2 fly?"}, node)
    assert node.calls[0]["context"] == ["[k2.md] The K2 flies 38 minutes."]
    assert output["answer"] == "It flies 38 minutes [k2.md]." and output["sources"] == ["k2.md"]


def test_any_runnable_returning_documents_works():
    chain = RunnableLambda(lambda question: [Document(f"About {question}.", metadata={"id": "r1"})])
    assert LangChainRetriever(chain).retrieve("drones") == ["[r1] About drones."]


@pytest.mark.parametrize(
    ("source", "error"),
    [("not a retriever", TypeError), (RunnableLambda(lambda question: "text"), TypeError)],
    ids=["not-runnable", "returns-non-list"],
)
def test_rejects_unusable_sources(source, error):
    with pytest.raises(error):
        LangChainRetriever(source).retrieve("q")
