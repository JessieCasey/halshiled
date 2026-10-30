"""CoVe, Critic, SelfConsistency and Abstain."""

from __future__ import annotations

import threading

import pytest

from halshield import ABSTAIN_MSG, Abstain, CoVe, Critic, SelfConsistency
from halshield.layers.cove import strip_findings_echo
from halshield.layers.self_consistency import samples_agree

from fakes import FakeLLM, RecordingNode


def cove_script(prompt: str) -> str | dict:
    if "List up to" in prompt:
        return {"checks": [{"claim": "flies 38 minutes", "question": "How long?"},
                           {"claim": "costs 5k", "question": "What price?"}]}  # fmt: skip
    if "Draft answer:" in prompt:
        return {"answer": "Flies 38 minutes [d1].", "confidence": 0.6}
    return {"answers": ["38 minutes."]}  # the price check gets no answer: it counts as NOT IN CONTEXT


def test_cove_verifies_independently_and_revises():
    llm = FakeLLM(cove_script)
    node = RecordingNode(
        answer="Flies 38 minutes [d1]. Costs 5k [d2].", sources=["d1", "d2"], confidence=0.9, context=["[d1] ..."]
    )
    output = CoVe(llm=llm).run({"question": "q"}, node)
    assert output["answer"] == "Flies 38 minutes [d1]."
    assert output["sources"] == ["d1"]  # the removed claim's citation goes with it
    assert output["confidence"] == 0.6  # never raised, lowered to the reviser's
    assert output["flags"] == ["cove: answer revised"]
    verification_prompts = [prompt for _, prompt, _ in llm.prompts if "Questions:" in prompt]
    assert len(verification_prompts) == 1 and "Costs 5k" not in verification_prompts[0]  # one call, no draft
    revise_prompt = next(prompt for _, prompt, _ in llm.prompts if "Draft answer:" in prompt)
    assert "Finding: NOT IN CONTEXT" in revise_prompt and len(llm.prompts) == 3


def test_cove_with_nothing_to_check_makes_one_call():
    llm = FakeLLM(lambda prompt: {"checks": []})
    output = CoVe(llm=llm).run({"question": "q"}, RecordingNode(answer="Not stated [d1].", sources=["d1"]))
    assert output["answer"] == "Not stated [d1]." and len(llm.prompts) == 1


def test_critic_uses_its_own_model_and_shared_counters():
    base = FakeLLM(lambda prompt: {"answer": "", "confidence": 0.2})
    critic = Critic(model="reviewer", llm=base)
    output = critic.run({"question": "q"}, RecordingNode(answer="Made up [d1].", sources=["d1"], confidence=0.9))
    assert critic.llm.model == "reviewer" and base.usage.calls["critic"] == 1
    assert (output["answer"], output["sources"], output["confidence"]) == ("", [], 0.2)


@pytest.mark.parametrize("layer_type", [CoVe, Critic])
def test_revision_layers_skip_abstentions(layer_type):
    llm = FakeLLM()
    output = layer_type(llm=llm).run({"question": "q"}, RecordingNode(answer=ABSTAIN_MSG))
    assert output["answer"] == ABSTAIN_MSG and llm.prompts == []


def test_self_consistency_samples_with_temperature_and_picks():
    answers = iter(["A", "B", "C"])
    lock = threading.Lock()

    def sampling_node(state):
        assert state["temperature"] == 0.8
        with lock:
            return {"answer": next(answers), "sources": ["d1"], "confidence": 0.9}

    def pick_b(prompt):  # samples run in parallel, so find where "B" landed
        index = next(line[1] for line in prompt.splitlines() if line.endswith("] B"))
        return {"best": int(index), "agreement": 0.5}

    llm = FakeLLM(pick_b)
    output = SelfConsistency(n=3, llm=llm).run({"question": "q"}, sampling_node)
    assert (output["answer"], output["confidence"]) == ("B", 0.5)
    assert llm.labels() == ["self_consistency"]  # disagreeing first pair: all 3 samples, one pick


def test_self_consistency_stops_early_when_two_samples_agree():
    answers = iter(["It flies 52 minutes [d3].", "The K3 flies for 52 minutes [d3].", "never drawn"])
    lock = threading.Lock()
    calls = []

    def sampling_node(state):
        with lock:
            calls.append(state)
            return {"answer": next(answers), "sources": ["d3"], "confidence": 0.9}

    llm = FakeLLM()
    output = SelfConsistency(n=3, llm=llm).run({"question": "q"}, sampling_node)
    assert len(calls) == 2 and llm.prompts == []  # no third sample, no pick call
    assert output["flags"] == ["self_consistency: first two samples agree"]
    assert output["answer"] != "never drawn"


@pytest.mark.parametrize(
    ("first", "second", "agree"),
    [
        ("It is 52 minutes [d3].", "52 minutes per charge [d3].", True),
        ("It is 52 minutes [d3].", "It is 45 minutes [d3].", False),
        ("It is 52 minutes [d3].", "It is 52 minutes [d13].", False),
        ("Not enough information.", "not enough  information.", True),
        ("A.", "B.", False),  # no numbers or citations to compare
        ("Total 1,200 [d1].", "Total 1200 [d1].", True),
    ],
)
def test_samples_agree(first, second, agree):
    assert samples_agree(first, second) is agree


@pytest.mark.parametrize(
    ("output", "abstains"),
    [
        ({"answer": "x [d1].", "sources": ["d1"], "confidence": 0.9}, False),
        ({"answer": "x", "sources": ["d1"], "confidence": 0.5}, True),
        ({"answer": "x", "sources": [], "confidence": 0.9}, True),
        ({"answer": " ", "sources": ["d1"], "confidence": 0.9}, True),
        ({"answer": "x", "sources": ["d1"], "confidence": None}, True),
    ],
    ids=["confident", "low-confidence", "no-sources", "blank", "null-confidence"],
)
def test_abstain(output, abstains):
    result = Abstain(0.7).run({"question": "q"}, RecordingNode(**output))
    assert (result["answer"] == ABSTAIN_MSG) is abstains


def test_layers_tolerate_null_and_garbage_numbers():
    sampler = SelfConsistency(n=2, llm=FakeLLM(lambda prompt: {"best": None, "agreement": "lots"}))
    output = sampler.run({"question": "q"}, RecordingNode(answer="A", sources=["d1"], confidence=None))
    assert output["answer"] == "A" and output["confidence"] == 1.0
    critic = Critic(llm=FakeLLM(lambda prompt: {"answer": "A", "confidence": 7}))
    output = critic.run({"question": "q"}, RecordingNode(answer="A", sources=["d1"], confidence=0.9))
    assert output["confidence"] == 0.9  # out-of-range reviser score is clamped, never raises confidence


def test_cove_drops_a_findings_block_copied_into_the_answer():
    leaked = "Range is 4.5 km [d16]. That is 2.8 miles [d16].\n\nIndependent checks (answered from ...):\n- Claim: x"
    assert strip_findings_echo(leaked) == "Range is 4.5 km [d16]. That is 2.8 miles [d16]."
    assert strip_findings_echo("- Claim: only an echo") == ""
    assert strip_findings_echo("Clean answer [d1].") == "Clean answer [d1]."
