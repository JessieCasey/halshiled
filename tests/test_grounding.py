"""RAGGrounding: retrieval, prompt injection and the citation filter."""

from __future__ import annotations

import pytest

from halshield import ABSTAIN_MSG, HybridRetriever, RAGGrounding

from fakes import FakeLLM, RecordingNode


def test_injects_retrieved_context_and_citation_rule(grounding):
    node = RecordingNode(answer="It flies 38 minutes [d1].", sources=[], confidence=0.9)
    grounding.run({"question": "How long does the K2 drone fly?", "instructions": ["Be brief."]}, node)
    seen = node.calls[0]
    assert seen["context"][0].startswith("[d1] ")
    assert seen["instructions"][0] == "Be brief."  # existing rules are kept, the citation rule is appended
    assert "[<id>]" in seen["instructions"][1]


@pytest.mark.parametrize(
    ("answer", "expected_answer", "expected_sources", "dropped"),
    [
        pytest.param(
            "It flies 38 minutes [d1]. It is popular.",
            "It flies 38 minutes [d1].", ["d1"], ["It is popular."], id="drops-uncited-claim",
        ),
        pytest.param(
            "The K3's range is not mentioned. It is probably 12 km. However, the K2 flies 38 minutes [d1].",
            "The K3's range is not mentioned. However, the K2 flies 38 minutes [d1].", ["d1"],
            ["It is probably 12 km."], id="keeps-missing-info-drops-speculation",
        ),
        pytest.param(
            "A guess. However, it flies 38 minutes [d1].",
            "It flies 38 minutes [d1].", ["d1"], ["A guess."], id="strips-dangling-connective",
        ),
        pytest.param(
            "Founded in 2014 [d3]. The K2 flies 38 minutes [d1]. Therefore, that is 24 more. So it may be best.",
            "Founded in 2014 [d3]. The K2 flies 38 minutes [d1]. Therefore, that is 24 more [d3][d1].",
            ["d1", "d3"], ["So it may be best."], id="conclusion-inherits-citations",
        ),
        pytest.param(
            "Therefore, it is 14 minutes.", ABSTAIN_MSG, [], ["Therefore, it is 14 minutes."],
            id="orphan-conclusion-leaves-nothing",
        ),
        pytest.param(
            "Flies 38 minutes [d9].", ABSTAIN_MSG, [], ["Flies 38 minutes [d9]."], id="citation-of-unknown-chunk",
        ),
    ],
)  # fmt: skip
def test_citation_filter(grounding, answer, expected_answer, expected_sources, dropped):
    output = grounding.run({"question": "q"}, RecordingNode(answer=answer, sources=[], confidence=0.9))
    assert output["answer"] == expected_answer
    assert output["sources"] == expected_sources
    assert output["flags"] == [f"grounding: dropped uncited '{sentence}'" for sentence in dropped]


def test_abstention_and_disabled_citations_pass_through(docs):
    abstained = RAGGrounding(docs, llm=FakeLLM()).run({"question": "q"}, RecordingNode(answer=ABSTAIN_MSG))
    assert abstained["answer"] == ABSTAIN_MSG
    lenient = RAGGrounding(["alpha text", "beta text"], require_citations=False, llm=FakeLLM())
    output = lenient.run({"question": "alpha"}, RecordingNode(answer="Uncited."))
    assert output["answer"] == "Uncited." and output["context"][0].startswith("[doc1] ")


def test_keyword_overlap_finds_chunk_that_never_names_the_subject():
    docs = {"company": "Kestrel Robotics makes drones and has offices.", "leaders": "Anneli Saar became CTO in 2020."}
    retriever = HybridRetriever(docs, k=1, keyword_weight=1.0, llm=FakeLLM())
    assert retriever.retrieve("Who is Kestrel's current CTO?")[0].startswith("[leaders]")


def test_custom_retriever_is_used(docs):
    class Fixed(HybridRetriever):
        def retrieve(self, question: str) -> list[str]:
            return ["[x] fixed chunk"]

    grounding = RAGGrounding(retriever=Fixed(docs, llm=FakeLLM()))
    output = grounding.run({"question": "q"}, RecordingNode(answer="Fixed [x].", sources=[]))
    assert output["sources"] == ["x"]
    with pytest.raises(ValueError, match="not both"):
        RAGGrounding(docs, retriever=grounding.retriever)


def test_missing_info_may_repeat_numbers_from_the_question(grounding):
    answer = "Nothing says the K3 was withdrawn in 2023. It was withdrawn in 2024."
    output = grounding.run({"question": "Why was the K3 withdrawn in 2023?"}, RecordingNode(answer=answer))
    assert output["answer"] == "Nothing says the K3 was withdrawn in 2023."


class OwnRAGNode:
    """A node with its own retrieval: ignores the given context and reports the chunks it used."""

    def __init__(self, answer: str, context: list) -> None:
        self.answer, self.context = answer, context
        self.seen: list[dict] = []

    def __call__(self, state: dict) -> dict:
        self.seen.append(state)
        return {"answer": self.answer, "sources": [], "confidence": 0.9, "context": self.context}


def test_wraps_a_node_that_does_its_own_rag():
    node = OwnRAGNode("Saar is CTO [d11]. She grew revenue 40% last year.", ["[d11] Anneli Saar is CTO."])
    output = RAGGrounding().run({"question": "Who is CTO?"}, node)
    assert output["answer"] == "Saar is CTO [d11]."  # the invented sentence cites nothing it retrieved
    assert output["sources"] == ["d11"]
    assert node.seen[0]["context"] == []  # nothing retrieved on its behalf
    assert any("cite" in rule for rule in node.seen[0]["instructions"])


def test_own_rag_context_may_be_langchain_documents():
    lc_documents = pytest.importorskip("langchain_core.documents")

    documents = [lc_documents.Document("The K3 flies 52 minutes.", metadata={"source": "specs/k3.pdf", "page": 2})]
    output = RAGGrounding().run({"question": "K3?"}, OwnRAGNode("It flies 52 minutes [k3.pdf#p2].", documents))
    assert output["context"] == ["[k3.pdf#p2] The K3 flies 52 minutes."]
    assert output["sources"] == ["k3.pdf#p2"]


def test_own_rag_without_context_is_flagged_not_wiped():
    output = RAGGrounding().run({"question": "q"}, OwnRAGNode("An answer.", []))
    assert output["answer"] == "An answer."
    assert output["flags"] == ["grounding: no [id] context to check citations against; answer not filtered"]
    with pytest.raises(TypeError, match="strings or Documents"):
        RAGGrounding().run({"question": "q"}, OwnRAGNode("x", [{"text": "dict"}]))


@pytest.mark.parametrize(
    ("answer", "kept"),
    [
        pytest.param("The K3 Lite carries 0.6 kg [spec].", False, id="fact-moved-to-look-alike"),
        pytest.param("The K3 Lite carries 0.6 kg, the same as the K3 [spec].", False, id="same-as-transfer"),
        pytest.param("The K3 carries 0.6 kg [spec].", True, id="right-subject"),
        pytest.param("The K3 Lite drops the lidar [lite].", True, id="look-alike-own-chunk"),
        pytest.param("It runs 210 K3 and 60 K3 Lite drones [spec][lite].", True, id="each-name-in-some-cited-chunk"),
        pytest.param("The K2's successor, the K3, carries 0.6 kg [spec].", True, id="passing-mention-of-other-model"),
        pytest.param("Five K3s carry 3 kg [spec].", True, id="plural"),
    ],
)
def test_subject_check_drops_facts_carried_over_from_a_look_alike(answer, kept):
    context = ["[spec] K3 technical sheet: maximum payload 0.6 kg.", "[lite] The K3 Lite is a cheaper K3, no lidar."]
    grounding = RAGGrounding()
    result = grounding.filter_uncited(answer, context)
    assert (result.kept == [answer]) is kept
    assert (result.wrong_subject == (answer,)) is not kept
    assert RAGGrounding(check_subjects=False).filter_uncited(answer, context).kept == [answer]


def test_subject_drop_is_flagged():
    node = OwnRAGNode("The K3 Lite carries 0.6 kg [d16].", ["[d16] K3 technical sheet: maximum payload 0.6 kg."])
    output = RAGGrounding().run({"question": "q"}, node)
    assert output["answer"] == ABSTAIN_MSG
    assert output["flags"] == [
        "grounding: dropped 'The K3 Lite carries 0.6 kg [d16].' (cited chunks never name its subject)"
    ]


def test_chunks_of_documents_already_in_context_are_not_repeated():
    docs = {"notes": "Atlas release notes.\n\nAtlas 4.2 adds per-category thresholds.", "k3": "The K3 flies 52 min."}
    grounding = RAGGrounding(docs, k=3, llm=FakeLLM())
    node = RecordingNode(answer="Atlas 4.2 adds thresholds [notes].", sources=[])
    whole = "[notes] " + docs["notes"]
    grounding.run({"question": "Atlas 4.2 thresholds", "context": [whole]}, node)
    seen = node.calls[0]["context"]
    assert whole in seen and not any(entry.startswith("[notes#") for entry in seen)
    assert any(entry.startswith("[k3]") for entry in seen)


def test_a_reworded_restatement_of_a_carried_over_fact_is_dropped_too():
    context = ["[spec] K3 technical sheet: maximum payload 0.6 kg.", "[lite] The K3 Lite is a cheaper K3, no lidar."]
    answer = (
        "The K3 Lite carries 0.6 kg, the same as the K3 [spec]. It has no lidar [lite]. "
        "Therefore, the budget variant can carry 0.6 kg [spec]. The K3 carries 0.6 kg too [spec][lite]."
    )
    result = RAGGrounding().filter_uncited(answer, context)
    assert result.kept == ["It has no lidar [lite]."]
    assert result.wrong_subject == (
        "The K3 Lite carries 0.6 kg, the same as the K3 [spec].",
        "Therefore, the budget variant can carry 0.6 kg [spec].",
        "The K3 carries 0.6 kg too [spec][lite].",
    )
