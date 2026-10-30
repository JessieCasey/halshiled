"""Every prompt halshield sends, in one place.

Templates use `string.Template` ($placeholders) so the JSON examples keep their literal braces, and
`Template.substitute` raises KeyError on a missing placeholder instead of sending a broken prompt.
Plain strings have no placeholders.
"""

from __future__ import annotations

from string import Template
from typing import Final

# --- System prompts (LLMClient) -------------------------------------------------------------------

SYSTEM_PROMPT: Final = "You are a precise assistant."
JSON_SYSTEM_PROMPT: Final = f"{SYSTEM_PROMPT} Reply in JSON."

# --- RAGGrounding: rule appended to the writer's instructions -------------------------------------

CITATION_RULE: Final = (
    "Use ONLY the context. Inside the answer text itself, end EVERY factual sentence with the [id] of the "
    "context chunk that supports it, copying the id exactly as the context writes it, e.g. 'The sky is blue "
    "[<id>].' Uncited factual sentences are deleted. For a calculation or comparison, state the result and cite "
    "every chunk it uses, e.g. 'A flies 14 minutes longer than B [<id of A's chunk>][<id of B's chunk>].' "
    "If the context does not answer the question "
    "or part of it, say so plainly in its own sentence (no citation needed) and do not guess. "
    "Announcements of planned or upcoming products are not current facts: never present them as current, "
    "and when sources disagree prefer the most recent one. "
    "A fact about one product, person or company is not a fact about a similar one: never carry it over "
    "('the same as X') unless a chunk states it for that subject. "
    "If the question assumes something the context contradicts, correct it first. "
    "Start a yes/no answer with Yes or No. In a calculation apply every stated condition (minimum terms, "
    "quantities, which model). If asked how or why and the context does not explain it, say so instead of "
    "describing related facts."
)

# --- QueryRewriter: search queries for retrieval ($question, $n) --------------------------------

QUERY_REWRITE: Final = Template(
    "Question: $question\n\nDocuments rarely use the asker's words. Write up to $n short search queries "
    "in the vocabulary the documents themselves would use, each worded differently from the question:\n"
    "- official terms instead of descriptions: job titles and their abbreviations (e.g. 'CFO' or 'chief "
    "financial officer' for 'who runs the finances'), product and model names, the measured quantity and "
    "its unit (e.g. 'battery life in minutes' for 'how long does it last');\n"
    "- if answering takes several steps, one query per fact needed;\n"
    "- if the question takes something for granted (an event, a role, a change), one plain query for the "
    "subject's actual status (e.g. 'Model X discontinued' or 'Model X on sale' for 'why was Model X recalled').\n"
    "At least one query must use only official titles, abbreviations and names, with no description.\n"
    "Also write one hypothetical sentence, phrased like a document, that would answer the question; "
    "invent placeholder details if needed (it is only used for searching, never shown). "
    'Return JSON {"queries": [str], "hypothetical": str}.'
)

# --- SelfConsistency: pick the most consistent sample ($question, $candidates) --------------------

SELF_CONSISTENCY_PICK: Final = Template(
    "Question: $question\n\nCandidate answers:\n$candidates\n\n"
    "Pick the candidate whose factual claims agree most with the other candidates, and rate "
    "how much the candidates agree overall (1 = same facts, 0 = contradictory). "
    'Return JSON {"best": int, "agreement": number 0-1}.'
)

# --- CoVe: plan ($answer, $max_questions), verify ($context, $question), revise ($answer, $findings) --

COVE_PLAN: Final = Template(
    "Answer:\n$answer\n\nList up to $max_questions factual claims this answer makes "
    "(skip statements that information is missing). Keep each claim's subject exactly as the answer "
    "states it (the product, person, company or date it is about); a fact said to be 'the same as' "
    "another product's is a claim about this product. For each claim, write one self-contained "
    "question whose answer would confirm or refute exactly that claim. Never ask about "
    'anything the answer does not state. Return JSON {"checks": [{"claim": str, "question": str}]}.'
)

COVE_VERIFY: Final = Template(
    "Context:\n$context\n\nQuestions:\n$questions\n\nAnswer each question independently, in one sentence, "
    "using ONLY the context. You may combine facts from several chunks and do simple arithmetic on them. "
    "Answer about exactly the subject asked: a fact stated for a similar product, person or company does not "
    "count. If the context does not say and it cannot be derived, answer exactly: NOT IN CONTEXT. "
    'Return JSON {"answers": [str, one per question, in order]}.'
)

COVE_REVISE: Final = Template(
    "Draft answer:\n$answer\n\nIndependent checks (answered from the source context only):\n$findings\n\n"
    "Rewrite the draft. Delete a claim ONLY if its own finding contradicts it or is NOT IN CONTEXT. "
    "Keep every other sentence verbatim, including [id] citations and statements that information is "
    "missing; do not shorten, merge or rephrase. Results computed from confirmed facts (arithmetic, unit "
    "conversions, comparisons) stay. The answer must contain only the rewritten draft: never copy the "
    "checks, claims or findings into it. If nothing survives, use an empty string. "
    'Return JSON {"answer": str, "confidence": number 0-1 that the rewrite is fully supported}.'
)

COVE_FINDING: Final = Template("- Claim: $claim\n  Check: $question\n  Finding: $finding")

# --- Critic: review ($context, $question, $answer) -------------------------------------------------

CRITIC_REVIEW: Final = Template(
    "Context:\n$context\n\nQuestion: $question\nAnswer:\n$answer\n\n"
    "You are a strict reviewer. List every claim in the answer that the context does not support, then "
    "rewrite the answer without them, keeping all other text and [id] citations verbatim "
    "(empty string if nothing remains). A claim derived from the context by combining facts or simple "
    "arithmetic IS supported. A fact the context states only for a similar product, person or company is NOT "
    "supported. Statements that information is missing are not claims: keep them. "
    'Return JSON {"issues": [str], "answer": str, "confidence": number 0-1 that the rewrite is fully supported}.'
)
