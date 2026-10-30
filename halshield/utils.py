"""Text helpers shared by layers."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from .contract import ABSTAIN_MSG, State

_CITATION = re.compile(r"\[([^\[\]]+)\]")
_CHUNK_ID = re.compile(r"^\[([^\]]+)\]")
_MISSING_INFO = re.compile(
    r"\b(?:not (?:mentioned|provided|specified|stated|given|available|disclosed|included|described)"
    r"|no (?:information|details|data|mention|evidence|indication|record)"
    r"|nothing (?:says|states|indicates|mentions|shows)"
    r"|(?:does|do) not (?:say|state|mention|specify|provide|include|contain|give|describe)"
    r"|(?:is|are) unknown)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(?:likely|probably|possibly|perhaps|may|might|could|would|reasonable|expect\w*|anticipat\w*"
    r"|assum\w*|suggest\w*|impl(?:y|ies|ied)|presumabl\w*|infer\w*)\b",
    re.IGNORECASE,
)
_CONCLUSION = re.compile(r"^(?:therefore|thus|so|hence|in total|this means|that is|overall)\b,?\s", re.IGNORECASE)
_CONNECTIVE = re.compile(
    r"^(?:however|additionally|also|moreover|furthermore|but|that said|instead),?\s+", re.IGNORECASE
)
_STANDALONE_NUMBER = re.compile(r"\b\d")
# A model-style name ("K3", "F9", "X-200"), with an optional capitalised variant word ("K3 Lite").
_MODEL_NAME = re.compile(r"\b[A-Z][A-Za-z]*-?\d[\w.-]*(?:\s+[A-Z][a-z]+\b)?")
_NUMBER = re.compile(r"\b\d[\d.,]*\d\b|\b\d\b")


def cited(text: str) -> list[str]:
    """Return ids cited as [id] or [id1, id2] in `text`, in order of appearance."""
    return [chunk_id.strip() for group in _CITATION.findall(text) for chunk_id in group.split(",")]


def _id_key(chunk_id: str) -> str:
    """Return an id without its letter prefix and leading zeros: "doc06", "d6" and "D06" all give "6"."""
    match = re.fullmatch(r"[A-Za-z_-]*0*(\d.*)", chunk_id.strip())
    return match.group(1).casefold() if match else chunk_id.strip().casefold()


def resolve_citations(text: str, valid_ids: Iterable[str]) -> str:
    """Rewrite citations of ids that don't exist to the one context id they clearly mean ("[doc06]" -> "[d06]").

    Models often cite in the style of an example ("doc1") rather than the context's own ids. An id is
    rewritten only when exactly one valid id has the same number (and suffix); anything else is left as is.
    """
    valid = set(valid_ids)
    by_key: dict[str, list[str]] = {}
    for chunk_id in valid:
        by_key.setdefault(_id_key(chunk_id), []).append(chunk_id)

    def fix(group: re.Match[str]) -> str:
        ids = [chunk_id.strip() for chunk_id in group.group(1).split(",")]
        fixed = [
            chunk_id
            if chunk_id in valid or len(by_key.get(_id_key(chunk_id), [])) != 1
            else by_key[_id_key(chunk_id)][0]
            for chunk_id in ids
        ]
        return f"[{', '.join(fixed)}]"

    return _CITATION.sub(fix, text)


def strip_citations(text: str) -> str:
    """Return `text` without its [id] citations, whitespace collapsed."""
    return re.sub(r"\s+([.!?,;:])", r"\1", " ".join(_CITATION.sub("", text).split()))


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping trailing "... claim. [d1]" citations with their sentence."""
    text = re.sub(r"([.!?])((?:\s*\[[^\]]+\])+)", lambda m: f" {m.group(2).strip()}{m.group(1)}", text)
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence]


def is_missing_info(sentence: str, question: str = "") -> bool:
    """Return True for a plain "X is not mentioned" statement: no quantities, no speculation.

    Standalone numbers ("55", "2.5") disqualify it unless they are repeated from `question`
    ("nothing says it was withdrawn in 2023"); digits inside names ("K3") never do.
    """
    asked = set(_NUMBER.findall(question))
    own_numbers = _NUMBER.sub(lambda match: "" if match.group() in asked else match.group(), sentence)
    return (
        bool(_MISSING_INFO.search(sentence))
        and not _HEDGE.search(sentence)
        and not _STANDALONE_NUMBER.search(own_numbers)
    )


def model_names(sentence: str) -> set[str]:
    """Return the model-style names a sentence mentions ("K3", "K3 Lite"); a plural "K3s" counts as "K3"."""
    return {re.sub(r"(?<=\d)s$", "", name.rstrip(".-")) for name in _MODEL_NAME.findall(sentence)}


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE) is not None


def look_alike_names(sentence: str, chunks: Iterable[str]) -> list[str]:
    """Return the variant names in `sentence` ("K3 Lite") that `chunks` never mention though they name the base model.

    That is a fact carried over from a look-alike product: "The K3 Lite carries 0.6 kg [k3-spec]".
    A name the chunks lack entirely (a passing "the K2's successor") is not flagged.
    """
    text = " ".join(chunks)
    return sorted(
        name
        for name in model_names(sentence)
        if " " in name and not _mentions(text, name) and _mentions(text, name.split()[0])
    )


def numbers(text: str) -> set[str]:
    """Return the standalone numbers in `text` ("0.6", "1,200"), not digits inside names ("K3") or [d03] ids."""
    return set(_NUMBER.findall(_CITATION.sub("", text)))


def is_conclusion(sentence: str) -> bool:
    """Return True for a non-speculative "Therefore/Thus/So ..." sentence drawn from earlier ones."""
    return bool(_CONCLUSION.match(sentence.strip())) and not _HEDGE.search(sentence)


def strip_leading_connective(sentence: str) -> str:
    """Drop a dangling "However," / "Additionally," from a sentence that now opens the answer."""
    remainder = _CONNECTIVE.sub("", sentence, count=1)
    return remainder[:1].upper() + remainder[1:] if remainder != sentence else sentence


def append_citations(sentence: str, chunk_ids: Iterable[str]) -> str:
    """Return `sentence` with "[id]" citations inserted before its final punctuation."""
    body, end = (sentence[:-1], sentence[-1]) if sentence[-1:] in ".!?" else (sentence, ".")
    return f"{body} {''.join(f'[{chunk_id}]' for chunk_id in chunk_ids)}{end}"


def as_number(value: object, default: float) -> float:
    """Parse a model-produced number; return `default` for None, garbage, NaN or infinity."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def clamp_unit(value: float) -> float:
    """Clamp a score into [0, 1]."""
    return min(max(value, 0.0), 1.0)


def is_abstain_or_empty(answer: str) -> bool:
    """Return True if there is nothing for a layer to check."""
    return not answer.strip() or answer == ABSTAIN_MSG


def context_text(output: State, state: State, only_ids: Iterable[str] = ()) -> str:
    """Return the context the node actually saw (a layer may have replaced it) as one string.

    With `only_ids`, just the "[id] text" entries with those ids (all entries if none of them match).
    """
    context = list(output.get("context", state.get("context", [])))
    wanted = set(only_ids)
    chosen = [entry for entry in context if (match := _CHUNK_ID.match(entry)) and match.group(1) in wanted]
    return "\n".join(chosen or context) or "(empty)"


def restore_citations(revised: str, original: str) -> str:
    """Give back the citations a rewrite dropped from sentences it otherwise kept word for word."""
    citations_of = {strip_citations(sentence).casefold(): sentence for sentence in split_sentences(original)}
    return " ".join(
        sentence if cited(sentence) else citations_of.get(strip_citations(sentence).casefold(), sentence)
        for sentence in split_sentences(revised)
    )
