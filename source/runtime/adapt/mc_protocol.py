"""Strict objective-choice parsing; never mine option letters out of prose."""
import json
import re


def parse_choice(raw, letters='ABCD'):
    text = (raw or '').strip()
    fence = re.fullmatch(r'```(?:json|cot-ans)?\s*(.*?)\s*```', text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            text = obj.get('choice', '')
    except (ValueError, TypeError):
        pass
    if not isinstance(text, str):
        return -1
    match = re.fullmatch(r'([A-Da-d])[.。]?', text.strip())
    return letters.find(match.group(1).upper()) if match else -1


def parse_explicit_choice(raw, letters='ABCD'):
    """Gold-independent extraction; accept a unique explicit answer declaration.

    Old strict scores remain reproducible via parse_choice. Never choose the
    first standalone article/letter or infer an answer from an option list.
    """
    strict = parse_choice(raw, letters)
    if strict >= 0:
        return strict
    text = (raw or '').replace('**', '').replace('__', '').strip()
    declared = re.findall(
        r'\b(?:(?:the\s+)?(?:final\s+|correct\s+|best\s+)?(?:answer|choice|option)'
        r'\s*(?:is\s*:?|:|=)\s*(?:option\s*)?|I\s+(?:choose|select)\s+)'
        r'\(?([A-D])\)?(?=\s|[.,:;!)]|$)', text, re.I)
    fenced = re.findall(r'```(?:cot-ans|json)?\s*([A-D])[.]?\s*```', text, re.I)
    choices = {c.upper() for c in declared + fenced}
    if choices:
        return letters.find(next(iter(choices))) if len(choices) == 1 else -1
    # A single option-labelled answer is allowed, but not a reproduced list.
    # Requiring it to start the response avoids extracting an incidental list
    # item from reasoning that never commits to a final answer.
    labelled = re.findall(r'^\s*([A-D])[.)]\s+\S', text, re.M)
    if len(labelled) == 1 and re.match(r'^[A-D][.)]\s+\S', text):
        return letters.find(labelled[0])
    return -1
