"""Every prompt template renders with exactly the arguments its call site passes."""

from __future__ import annotations

import re
from string import Template

import pytest

from halshield import prompts

from evaluation.prompts import JUDGE_PROMPT
from pipeline.prompts import WRITER_PROMPT

_IDENTIFIER = re.compile(r"\$(?:(\w+)|\{(\w+)\})")

CALL_SITE_ARGUMENTS = {
    "QUERY_REWRITE": {"question", "n"},
    "SELF_CONSISTENCY_PICK": {"question", "candidates"},
    "COVE_PLAN": {"answer", "max_questions"},
    "COVE_VERIFY": {"context", "questions"},
    "COVE_REVISE": {"answer", "findings"},
    "COVE_FINDING": {"claim", "question", "finding"},
    "CRITIC_REVIEW": {"context", "question", "answer"},
}
EXAMPLE_TEMPLATES = {
    "WRITER_PROMPT": (WRITER_PROMPT, {"context", "question", "style", "rules"}),
    "JUDGE_PROMPT": (JUDGE_PROMPT, {"documents", "question", "gold", "traps", "answer"}),
}
ALL_TEMPLATES = {
    **{name: (getattr(prompts, name), arguments) for name, arguments in CALL_SITE_ARGUMENTS.items()},
    **EXAMPLE_TEMPLATES,
}


def test_every_library_template_is_covered():
    templates = {name for name, value in vars(prompts).items() if isinstance(value, Template)}
    assert templates == CALL_SITE_ARGUMENTS.keys()


@pytest.mark.parametrize("name", ALL_TEMPLATES)
def test_template_placeholders_match_call_site(name):
    template, arguments = ALL_TEMPLATES[name]
    placeholders = {named or braced for named, braced in _IDENTIFIER.findall(template.template)}
    assert placeholders == arguments
    rendered = template.substitute(dict.fromkeys(arguments, "VALUE $notaplaceholder {x}"))
    assert "{{" not in rendered  # JSON examples keep literal braces (no leftover .format escaping)
    assert rendered.count("VALUE") == len(arguments)
