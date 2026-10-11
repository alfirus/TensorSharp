#!/usr/bin/env python3
"""Independent answer-only oracles for the three fixed Direct probe tasks.

These rules never use a model's reference answer as ground truth, remove a
Markdown/JSON wrapper, or search prose for an embedded correct answer. Callers
must additionally establish process identity, complete generation and EOS.
"""
import re


POLICY = 'fixed-answer-only-v1'
LEGACY_MAYA_POLICY = 'reference-final-text-equality-v1 (implicit historical policy)'
PROMPTS = {
    'math': 'Compute 17 + 25. Reply with only the integer.',
    'extraction': 'Inventory: cedar=3, maple=7, pine=2. Return only the names whose quantity is at least 3, comma-separated in the original order.',
    'squares': 'Return the squares of the integers 1 through 20, in order, as comma-separated integers. Return only the list.',
}


def evaluate(name, prompt, answer):
    """Bind to the exact known question; unknown tasks/prompts fail closed."""
    known = isinstance(name, str) and name in PROMPTS
    bound = known and prompt == PROMPTS[name]
    text = answer.strip() if isinstance(answer, str) else None
    valid = False
    if bound and text is not None:
        if name == 'math':
            # Matches the original Direct probe's optional sentence punctuation.
            valid = re.fullmatch(r'42[.!]?', text) is not None
        elif name == 'extraction':
            valid = re.fullmatch(r'cedar\s*,\s*maple', text, re.IGNORECASE | re.ASCII) is not None
        elif name == 'squares':
            if re.fullmatch(r'[0-9]+(?:\s*,\s*[0-9]+){19}', text, re.ASCII):
                # Bound every digit string before int(); malformed huge integer
                # input must fail, not throw or consume an unbounded conversion.
                parts = [part.strip().lstrip('0') or '0' for part in text.split(',')]
                valid = parts == [str(i * i) for i in range(1, 21)]
    return dict(policy=POLICY, case=name, known_case=known, prompt_matches=bound,
                passed=bool(valid), expected={'math': '42', 'extraction': ['cedar', 'maple'],
                    'squares': [i * i for i in range(1, 21)]}.get(name) if known else None)


def parsed_answer(name, prompt, content, thinking_closed, tool_calls):
    """Grade supplied production-parser output without trusting a saved pass."""
    answer = evaluate(name, prompt, content)
    structure = thinking_closed is True and isinstance(tool_calls, list) and not tool_calls
    return dict(**answer, answer_passed=answer['passed'], structure_passed=structure) | {
        'passed': answer['passed'] and structure}
