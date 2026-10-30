"""The sentence-level rules grounding relies on, as one table each."""

from __future__ import annotations

import math

import pytest

from halshield.utils import (
    as_number,
    cited,
    context_text,
    is_conclusion,
    is_missing_info,
    resolve_citations,
    restore_citations,
    split_sentences,
    strip_leading_connective,
)


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("The weight of the K3 is not mentioned in the context.", True),
        ("The context does not specify who the CFO is.", True),
        ("The K3's read range is unknown.", True),  # digit inside a product name is fine
        ("It is not mentioned, but it is likely 55 minutes.", False),  # speculation smuggled in
        ("The price is not given; it is about 900 euros.", False),  # a quantity smuggled in
        ("The K3 weighs 2 kg.", False),
        ("Nothing says the K3 was withdrawn.", True),  # premise correction
    ],
)
def test_is_missing_info(sentence, expected):
    assert is_missing_info(sentence) is expected


def test_is_missing_info_allows_numbers_from_the_question():
    sentence = "The context does not provide any information about a recall in 2023."
    assert is_missing_info(sentence, question="Why was the K3 recalled in 2023?")
    assert not is_missing_info(sentence, question="Why was the K3 recalled?")
    assert not is_missing_info("No details are given; it cost 2,023 euros.", question="What happened in 2023?")


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("Therefore, the factory opened 8 years later.", True),
        ("So the K3 flies 12 minutes longer.", True),
        ("Therefore, it could open soon.", False),
        ("The factory opened in 2022.", False),
    ],
)
def test_is_conclusion(sentence, expected):
    assert is_conclusion(sentence) is expected


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("One [d1]. Two [d2].", ["One [d1].", "Two [d2]."]),
        ("One. [d1] Two [d2].", ["One [d1].", "Two [d2]."]),  # citation after the period stays with its sentence
        ("Range is 4.5 km [d1]. Done.", ["Range is 4.5 km [d1].", "Done."]),  # decimals don't split
    ],
)
def test_split_sentences(text, sentences):
    assert split_sentences(text) == sentences


def test_citations_and_connectives():
    assert cited("a [d1]. b [d2, d3].") == ["d1", "d2", "d3"]
    assert strip_leading_connective("However, the K2 flies.") == "The K2 flies."
    assert strip_leading_connective("The K2 flies.") == "The K2 flies."


@pytest.mark.parametrize(
    ("raw", "expected"), [("0.4", 0.4), (None, 1.0), ("high", 1.0), (math.nan, 1.0), (math.inf, 1.0)]
)
def test_as_number_tolerates_model_garbage(raw, expected):
    assert as_number(raw, 1.0) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Total 270 [doc06][doc22].", "Total 270 [d06][d22]."),  # cited in the style of the prompt's example
        ("K2 [doc2, d06].", "K2 [d02, d06]."),
        ("Notes [D21#3].", "Notes [d21#3]."),
        ("Unknown [zz].", "Unknown [zz]."),
        ("Ambiguous [doc1].", "Ambiguous [doc1]."),  # both d01 and e1 could be meant: left alone
    ],
)
def test_resolve_citations(text, expected):
    assert resolve_citations(text, ["d02", "d06", "d22", "d21#3", "d01", "e1"]) == expected


def test_restore_citations_puts_back_what_a_rewrite_dropped():
    original = "The K3 flies 52 minutes [d03]. It costs 3,500 euros [d20]."
    revised = "The K3 flies 52 minutes. It is cheap."
    assert restore_citations(revised, original) == "The K3 flies 52 minutes [d03]. It is cheap."


def test_context_text_can_keep_only_cited_chunks():
    output = {"context": ["[a] alpha", "[b] beta", "[c] gamma"]}
    assert context_text(output, {}, ["b"]) == "[b] beta"
    assert context_text(output, {}, ["zz"]) == "[a] alpha\n[b] beta\n[c] gamma"  # nothing matches: all
