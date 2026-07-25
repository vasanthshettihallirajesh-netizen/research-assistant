"""
main.py — FastAPI backend for the Research Assistant.

Endpoints:
    POST /topics                    submit a question, get decomposed sub-questions + similar past topics
    GET  /topics                     list past topics (paginated)
    GET  /topics/{id}                full detail: sub-questions, sources, aggregated confidence assessment
    GET  /topics/search               TF-IDF similarity search against past topics
    POST /topics/{id}/answer          record an answer + confidence for a sub-question
    POST /topics/{id}/unresolved      mark a sub-question unresolved, with a note
    POST /topics/{id}/source          log a source checked for a sub-question
    PUT  /topics/{id}/status          update topic status + summary
    GET  /stats                        overall counts
    GET  /health                        liveness check

Run:
    pip install -r requirements.txt
    uvicorn app.main:app --reload
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional, Literal

from app import db
from app.decomposer import decompose
from app.aggregation import assess_topic
from app.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Research Assistant API",
    description="Decomposes research questions, tracks sources and findings, "
                 "surfaces semantically similar past research, and aggregates "
                 "confidence with contradiction detection.",
    version="2.0.0",
)


# ---------- error handling ----------

class NotFoundError(Exception):
    def __init__(self, detail: str):
        self.detail = detail


class ValidationConflictError(Exception):
    def __init__(self, detail: str):
        self.detail = detail


@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError):
    logger.warning(f"404 on {request.url.path}: {exc.detail}")
    return JSONResponse(status_code=404, content={"detail": exc.detail})


@app.exception_handler(ValidationConflictError)
async def conflict_handler(request: Request, exc: ValidationConflictError):
    logger.warning(f"400 on {request.url.path}: {exc.detail}")
    return JSONResponse(status_code=400, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url.path}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.on_event("startup")
def on_startup():
    db.init_db()
    logger.info("Database initialized, Research Assistant API starting up")


# ---------- request/response models with real validation ----------

class TopicRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        if len(v) > 2000:
            raise ValueError("question must be under 2000 characters")
        return v


class AnswerRequest(BaseModel):
    sub_question_id: int
    answer: str
    confidence: float

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("answer must not be empty")
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v


class UnresolvedRequest(BaseModel):
    sub_question_id: int
    note: Optional[str] = None


class SourceRequest(BaseModel):
    sub_question_id: int
    url: Optional[str] = None
    title: Optional[str] = None
    excerpt: Optional[str] = None
    stance: Literal["supports", "contradicts", "neutral"] = "neutral"
    reliability_note: Optional[str] = None

    @field_validator("url")
    @classmethod
    def url_looks_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.strip() and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("url must start with http:// or https://")
        return v


class StatusRequest(BaseModel):
    status: Literal["open", "in_progress", "resolved"]
    summary: Optional[str] = None


# ---------- helpers ----------

def _get_topic_or_404(topic_id: int) -> dict:
    topic = db.get_topic(topic_id)
    if not topic:
        raise NotFoundError(f"Topic {topic_id} not found")
    return topic


def _require_subq_in_topic(topic: dict, sub_question_id: int):
    if not any(sq["id"] == sub_question_id for sq in topic["sub_questions"]):
        raise ValidationConflictError(
            f"Sub-question {sub_question_id} does not belong to topic {topic['id']}"
        )


def _topic_with_assessment(topic: dict) -> dict:
    assessment = assess_topic(topic)
    result = dict(topic)
    result["assessment"] = assessment.to_dict()
    return result


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/topics")
def create_topic(req: TopicRequest):
    similar = db.find_similar_topics(req.question, limit=3)
    topic_id = db.create_topic(req.question)
    sub_questions = decompose(req.question)
    sub_question_ids = db.add_sub_questions(topic_id, sub_questions)

    logger.info(f"Created topic {topic_id} with {len(sub_question_ids)} sub-questions")

    return {
        "topic_id": topic_id,
        "question": req.question,
        "sub_questions": [
            {"id": sid, "question": sq}
            for sid, sq in zip(sub_question_ids, sub_questions)
        ],
        "similar_past_topics": [
            {
                "id": t["id"],
                "question": t["question"],
                "status": t["status"],
                "similarity_score": t["similarity_score"],
            }
            for t in similar
            if t["id"] != topic_id
        ],
    }


@app.get("/topics")
def list_topics(limit: int = Query(20, ge=1, le=500), offset: int = Query(0, ge=0)):
    all_topics = db.list_topics(limit=limit + offset)
    page = all_topics[offset:offset + limit]
    return {
        "items": page,
        "limit": limit,
        "offset": offset,
        "returned": len(page),
    }


@app.get("/topics/search")
def search_topics(q: str, limit: int = Query(10, ge=1, le=100), min_score: float = Query(0.1, ge=0.0, le=1.0)):
    if not q.strip():
        raise ValidationConflictError("query parameter 'q' must not be empty")
    return db.find_similar_topics(q, limit=limit, min_score=min_score)


@app.get("/topics/{topic_id}")
def get_topic(topic_id: int):
    topic = _get_topic_or_404(topic_id)
    return _topic_with_assessment(topic)


@app.post("/topics/{topic_id}/answer")
def answer_sub_question(topic_id: int, req: AnswerRequest):
    topic = _get_topic_or_404(topic_id)
    _require_subq_in_topic(topic, req.sub_question_id)

    db.answer_sub_question(req.sub_question_id, req.answer, req.confidence)
    logger.info(f"Answered sub-question {req.sub_question_id} (confidence={req.confidence})")

    updated = _get_topic_or_404(topic_id)
    return _topic_with_assessment(updated)


@app.post("/topics/{topic_id}/unresolved")
def mark_unresolved(topic_id: int, req: UnresolvedRequest):
    topic = _get_topic_or_404(topic_id)
    _require_subq_in_topic(topic, req.sub_question_id)

    db.mark_sub_question_unresolved(req.sub_question_id, req.note)
    logger.info(f"Marked sub-question {req.sub_question_id} unresolved")

    updated = _get_topic_or_404(topic_id)
    return _topic_with_assessment(updated)


@app.post("/topics/{topic_id}/source")
def add_source(topic_id: int, req: SourceRequest):
    topic = _get_topic_or_404(topic_id)
    _require_subq_in_topic(topic, req.sub_question_id)

    source_id = db.add_source(
        req.sub_question_id, req.url, req.title, req.excerpt, req.stance, req.reliability_note
    )
    logger.info(f"Logged source {source_id} for sub-question {req.sub_question_id} (stance={req.stance})")
    return {"source_id": source_id}


@app.put("/topics/{topic_id}/status")
def update_status(topic_id: int, req: StatusRequest):
    _get_topic_or_404(topic_id)
    db.update_topic_status(topic_id, req.status, req.summary)
    logger.info(f"Topic {topic_id} status -> {req.status}")

    updated = _get_topic_or_404(topic_id)
    return _topic_with_assessment(updated)


@app.get("/stats")
def stats():
    return db.get_stats()
