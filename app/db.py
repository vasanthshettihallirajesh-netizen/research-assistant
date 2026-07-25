"""
db.py — SQLite persistence layer for the Research Assistant.

Tracks research topics, their decomposed sub-questions, the sources
checked for each, and a running confidence/verification status — so
re-asking a similar question later can reuse prior work instead of
starting from zero.
"""

import sqlite3
import os
import time
import json
from contextlib import contextmanager

DB_PATH = os.environ.get("RESEARCH_DB_PATH", "research.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',   -- open | in_progress | resolved
    summary TEXT
);

CREATE TABLE IF NOT EXISTS sub_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',   -- open | answered | unresolved
    answer TEXT,
    confidence REAL,                        -- 0.0 - 1.0
    FOREIGN KEY (topic_id) REFERENCES topics (id)
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sub_question_id INTEGER NOT NULL,
    created_at REAL NOT NULL,
    url TEXT,
    title TEXT,
    excerpt TEXT,
    stance TEXT,                            -- supports | contradicts | neutral
    reliability_note TEXT,
    FOREIGN KEY (sub_question_id) REFERENCES sub_questions (id)
);

CREATE INDEX IF NOT EXISTS idx_subq_topic ON sub_questions (topic_id);
CREATE INDEX IF NOT EXISTS idx_sources_subq ON sources (sub_question_id);
CREATE INDEX IF NOT EXISTS idx_topics_question ON topics (question);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ---------- topics ----------

def create_topic(question: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO topics (created_at, question, status) VALUES (?, ?, 'open')",
            (time.time(), question),
        )
        return cur.lastrowid


def find_similar_topics(query: str, limit: int = 5, min_score: float = 0.1) -> list:
    """TF-IDF cosine similarity search against past topics — catches
    reworded questions that share no exact keywords but are semantically
    related, unlike plain substring matching."""
    from app.similarity import rank_by_similarity

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM topics ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
    candidates = [dict(row) for row in rows]
    if not candidates:
        return []

    ranked = rank_by_similarity(query, candidates, key=lambda t: t["question"])
    results = []
    for score, topic in ranked:
        if score < min_score:
            continue
        topic_with_score = dict(topic)
        topic_with_score["similarity_score"] = round(score, 4)
        results.append(topic_with_score)
        if len(results) >= limit:
            break
    return results


def get_topic(topic_id: int) -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
        if not row:
            return None
        topic = dict(row)
        subqs = conn.execute(
            "SELECT * FROM sub_questions WHERE topic_id = ?", (topic_id,)
        ).fetchall()
        topic["sub_questions"] = []
        for sq in subqs:
            sq_dict = dict(sq)
            sources = conn.execute(
                "SELECT * FROM sources WHERE sub_question_id = ?", (sq["id"],)
            ).fetchall()
            sq_dict["sources"] = [dict(s) for s in sources]
            topic["sub_questions"].append(sq_dict)
        return topic


def list_topics(limit: int = 50) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM topics ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def update_topic_status(topic_id: int, status: str, summary: str = None):
    with get_conn() as conn:
        if summary is not None:
            conn.execute(
                "UPDATE topics SET status = ?, summary = ? WHERE id = ?",
                (status, summary, topic_id),
            )
        else:
            conn.execute("UPDATE topics SET status = ? WHERE id = ?", (status, topic_id))


# ---------- sub-questions ----------

def add_sub_questions(topic_id: int, questions: list) -> list:
    ids = []
    with get_conn() as conn:
        for q in questions:
            cur = conn.execute(
                "INSERT INTO sub_questions (topic_id, question, status) VALUES (?, ?, 'open')",
                (topic_id, q),
            )
            ids.append(cur.lastrowid)
    return ids


def answer_sub_question(sub_question_id: int, answer: str, confidence: float):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sub_questions SET status = 'answered', answer = ?, confidence = ? WHERE id = ?",
            (answer, confidence, sub_question_id),
        )


def mark_sub_question_unresolved(sub_question_id: int, note: str = None):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sub_questions SET status = 'unresolved', answer = ? WHERE id = ?",
            (note, sub_question_id),
        )


# ---------- sources ----------

def add_source(sub_question_id: int, url: str, title: str, excerpt: str,
                stance: str, reliability_note: str = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO sources
               (sub_question_id, created_at, url, title, excerpt, stance, reliability_note)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sub_question_id, time.time(), url, title, excerpt, stance, reliability_note),
        )
        return cur.lastrowid


def get_sources_for_topic(topic_id: int) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT sources.* FROM sources
               JOIN sub_questions ON sources.sub_question_id = sub_questions.id
               WHERE sub_questions.topic_id = ?
               ORDER BY sources.created_at DESC""",
            (topic_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats() -> dict:
    with get_conn() as conn:
        topics = conn.execute("SELECT COUNT(*) as c FROM topics").fetchone()["c"]
        resolved = conn.execute(
            "SELECT COUNT(*) as c FROM topics WHERE status = 'resolved'"
        ).fetchone()["c"]
        sub_qs = conn.execute("SELECT COUNT(*) as c FROM sub_questions").fetchone()["c"]
        sources = conn.execute("SELECT COUNT(*) as c FROM sources").fetchone()["c"]
        return {
            "total_topics": topics,
            "resolved_topics": resolved,
            "total_sub_questions": sub_qs,
            "total_sources": sources,
        }
