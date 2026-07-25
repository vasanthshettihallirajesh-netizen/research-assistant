"""
decomposer.py — breaks a research question into smaller sub-questions.

This is a rule-based decomposer (no LLM call baked in) so the whole
pipeline runs offline and deterministically. If you want smarter
decomposition, swap `decompose()`'s body for a call to an LLM API —
everything downstream (db schema, API) doesn't need to change.
"""

import re

# Question templates applied based on simple keyword detection in the topic.
GENERIC_TEMPLATES = [
    "What is the current consensus on: {topic}?",
    "What evidence supports {topic}?",
    "What evidence contradicts {topic}?",
    "Who are the credible sources or experts on {topic}?",
]

CAUSAL_TEMPLATES = [
    "Is there a proven causal mechanism for {topic}?",
    "Are there controlled studies on {topic}, or only correlational data?",
    "What do skeptics or contrary studies say about {topic}?",
]

COMPARISON_TEMPLATES = [
    "What are the key differences being compared in: {topic}?",
    "What criteria matter most for this comparison: {topic}?",
    "Is there a clear consensus winner, or does it depend on use case: {topic}?",
]

CURRENT_EVENT_TEMPLATES = [
    "What is the most recent reporting on: {topic}?",
    "What do multiple independent sources say about: {topic}?",
    "Has this claim been fact-checked or disputed: {topic}?",
]

CAUSAL_TRIGGERS = ["cause", "causes", "caused", "leads to", "results in", "because of", "due to"]
COMPARISON_TRIGGERS = ["vs", "versus", "better than", "compared to", "or"]
CURRENT_EVENT_TRIGGERS = ["latest", "current", "recent", "today", "this year", "now"]


def _strip_topic(question: str) -> str:
    """Extract a bare topic phrase from a full question, for use inside templates."""
    q = question.strip().rstrip("?")
    # drop leading question words, possibly more than one in a row
    while True:
        new_q = re.sub(r"^(does|is|are|can|will|why|how|what|who)\s+", "", q, flags=re.IGNORECASE)
        if new_q == q:
            break
        q = new_q
    return q.strip()


def _contains_trigger(text: str, triggers: list) -> bool:
    """Word-boundary aware trigger matching, so short words like 'or'
    don't false-positive match inside other words like 'more'."""
    for trigger in triggers:
        if re.search(r"\b" + re.escape(trigger) + r"\b", text):
            return True
    return False


def decompose(question: str) -> list:
    """Returns a list of sub-question strings for the given research question."""
    q_lower = question.lower()
    topic = _strip_topic(question)

    if _contains_trigger(q_lower, CAUSAL_TRIGGERS):
        templates = GENERIC_TEMPLATES[:2] + CAUSAL_TEMPLATES
    elif _contains_trigger(q_lower, COMPARISON_TRIGGERS):
        templates = COMPARISON_TEMPLATES
    elif _contains_trigger(q_lower, CURRENT_EVENT_TRIGGERS):
        templates = CURRENT_EVENT_TEMPLATES
    else:
        templates = GENERIC_TEMPLATES

    sub_questions = [t.format(topic=topic) for t in templates]
    # de-duplicate while preserving order
    seen = set()
    result = []
    for sq in sub_questions:
        if sq not in seen:
            seen.add(sq)
            result.append(sq)
    return result


if __name__ == "__main__":
    samples = [
        "Does eating chocolate cause acne?",
        "Python vs JavaScript for backend development",
        "What is the latest news on the merger?",
        "Is remote work more productive?",
    ]
    for s in samples:
        print(s)
        for sq in decompose(s):
            print("  -", sq)
        print()
