"""
aggregation.py — combines sub-question answers into a topic-level
confidence score, and flags contradictions between sources instead of
silently averaging over disagreement.
"""

from dataclasses import dataclass, field


@dataclass
class TopicAssessment:
    overall_confidence: float
    resolved_ratio: float           # answered / total sub-questions
    has_contradictions: bool
    contradicted_sub_questions: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {
            "overall_confidence": round(self.overall_confidence, 3),
            "resolved_ratio": round(self.resolved_ratio, 3),
            "has_contradictions": self.has_contradictions,
            "contradicted_sub_questions": self.contradicted_sub_questions,
            "notes": self.notes,
        }


def _sub_question_has_contradiction(sub_question: dict) -> bool:
    """A sub-question is contradicted if its sources disagree — i.e. at
    least one source 'supports' and at least one 'contradicts'."""
    stances = {s["stance"] for s in sub_question.get("sources", [])}
    return "supports" in stances and "contradicts" in stances


def assess_topic(topic: dict) -> TopicAssessment:
    sub_questions = topic.get("sub_questions", [])
    if not sub_questions:
        return TopicAssessment(
            overall_confidence=0.0,
            resolved_ratio=0.0,
            has_contradictions=False,
            notes=["No sub-questions yet."],
        )

    answered = [sq for sq in sub_questions if sq["status"] == "answered" and sq.get("confidence") is not None]
    resolved_ratio = len(answered) / len(sub_questions)

    contradicted = [sq["id"] for sq in sub_questions if _sub_question_has_contradiction(sq)]
    has_contradictions = len(contradicted) > 0

    notes = []

    if not answered:
        overall_confidence = 0.0
        notes.append("No sub-questions have been answered yet.")
    else:
        # weighted average, but confidence for any contradicted sub-question
        # gets penalized rather than trusted at face value
        weighted_sum = 0.0
        weight_total = 0.0
        for sq in answered:
            weight = 1.0
            conf = sq["confidence"]
            if sq["id"] in contradicted:
                conf = conf * 0.5  # penalize confidence when sources disagree
                notes.append(
                    f"Sub-question {sq['id']} has conflicting sources; "
                    f"confidence discounted."
                )
            weighted_sum += conf * weight
            weight_total += weight
        overall_confidence = weighted_sum / weight_total if weight_total else 0.0

    unresolved = [sq for sq in sub_questions if sq["status"] == "unresolved"]
    if unresolved:
        notes.append(f"{len(unresolved)} sub-question(s) marked unresolved.")

    if resolved_ratio < 1.0:
        notes.append(
            f"{len(sub_questions) - len(answered) - len(unresolved)} sub-question(s) still open."
        )

    return TopicAssessment(
        overall_confidence=overall_confidence,
        resolved_ratio=resolved_ratio,
        has_contradictions=has_contradictions,
        contradicted_sub_questions=contradicted,
        notes=notes,
    )


if __name__ == "__main__":
    sample_topic = {
        "sub_questions": [
            {
                "id": 1, "status": "answered", "confidence": 0.8,
                "sources": [
                    {"stance": "supports"},
                    {"stance": "supports"},
                ],
            },
            {
                "id": 2, "status": "answered", "confidence": 0.6,
                "sources": [
                    {"stance": "supports"},
                    {"stance": "contradicts"},
                ],
            },
            {
                "id": 3, "status": "open", "confidence": None, "sources": [],
            },
        ]
    }
    result = assess_topic(sample_topic)
    import json
    print(json.dumps(result.to_dict(), indent=2))
