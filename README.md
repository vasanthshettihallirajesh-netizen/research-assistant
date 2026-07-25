# Research Assistant

A backend service that takes a research question, breaks it into smaller
sub-questions, tracks every source and answer against a database, and
uses TF-IDF cosine similarity to catch when a *reworded* question is
actually the same ground already covered — so repeated or related
research doesn't start from zero.

Built with AI assistance (Claude), step by step.

## What makes this more than a CRUD wrapper

- **Real semantic similarity, not keyword matching.** `app/similarity.py`
  implements TF-IDF vectorization and cosine similarity from scratch
  (no numpy/sklearn dependency) — it correctly matches "productivity
  effects of working from home" to "Does remote work reduce
  productivity?" even though they share almost no exact words, while
  correctly rejecting unrelated topics.
- **Contradiction detection, not silent averaging.** `app/aggregation.py`
  checks whether sources logged against a sub-question disagree
  (`supports` vs `contradicts`). If they do, that sub-question's
  confidence is discounted rather than blindly averaged, and it's
  flagged explicitly in the topic assessment.
- **A pluggable research backend interface.** `app/research_backend.py`
  defines an abstract `ResearchBackend` class so a real web-search
  integration can be dropped in later to auto-populate sources, without
  touching the API or database layer.
- **A real, passing test suite.** 28 `unittest` tests across the
  decomposer, similarity engine, aggregation logic, and database layer —
  including regression tests for two actual bugs caught during
  development (a false-positive substring match, and a threshold that
  was silently dropping valid matches).
- **Structured logging and real input validation**, not prints and
  unchecked input — Pydantic validators enforce non-empty text and
  confidence bounds; a proper `logging` setup replaces ad hoc prints.

## What it does

1. Submit a question (e.g. "Does remote work reduce productivity?")
2. It's automatically broken into a handful of more specific
   sub-questions, shaped by the type of question (causal, comparison,
   current-event, or general)
3. Before starting, TF-IDF similarity search checks whether anything
   close to this has been researched before, even if worded differently
4. As sources and answers are logged against each sub-question, the
   system tracks stance (supports/contradicts/neutral) and confidence
5. `GET /topics/{id}` returns not just the raw data but an aggregated
   assessment: overall confidence (contradiction-adjusted), resolved
   ratio, and explicit notes about what's still open or disputed

This automates the repetitive bookkeeping of research — decomposition,
source tracking, avoiding duplicate work, and honest confidence
aggregation. It does not perform the actual fact-finding; pair it with
a real search tool (see "Extending it" below) to fill in sources.

## API

| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/topics` | Submit a question, get decomposed sub-questions + similar past topics |
| `GET` | `/topics` | List past topics (paginated) |
| `GET` | `/topics/{id}` | Full detail + aggregated confidence assessment |
| `GET` | `/topics/search?q=...` | TF-IDF similarity search against past topics |
| `POST` | `/topics/{id}/answer` | Record an answer + confidence for a sub-question |
| `POST` | `/topics/{id}/unresolved` | Mark a sub-question unresolved, with a note |
| `POST` | `/topics/{id}/source` | Log a source (with stance) for a sub-question |
| `PUT` | `/topics/{id}/status` | Update topic status + summary |
| `GET` | `/stats` | Overall counts |
| `GET` | `/health` | Liveness check |

### Example: a topic with conflicting sources

```bash
curl -X POST http://localhost:8000/topics \
  -d '{"question": "Does remote work reduce productivity?"}'
# -> topic_id: 1, sub_question ids returned

curl -X POST http://localhost:8000/topics/1/source \
  -d '{"sub_question_id": 1, "url": "https://a.com", "excerpt": "shows productivity gains", "stance": "supports"}'
curl -X POST http://localhost:8000/topics/1/source \
  -d '{"sub_question_id": 1, "url": "https://b.com", "excerpt": "shows productivity losses", "stance": "contradicts"}'
curl -X POST http://localhost:8000/topics/1/answer \
  -d '{"sub_question_id": 1, "answer": "Mixed evidence", "confidence": 0.7}'

curl http://localhost:8000/topics/1
```

```json
{
  "id": 1,
  "sub_questions": [ /* ... */ ],
  "assessment": {
    "overall_confidence": 0.35,
    "resolved_ratio": 0.25,
    "has_contradictions": true,
    "contradicted_sub_questions": [1],
    "notes": [
      "Sub-question 1 has conflicting sources; confidence discounted.",
      "3 sub-question(s) still open."
    ]
  }
}
```

The confidence dropped from the logged 0.7 to 0.35 — the assessment
layer caught the disagreement and discounted it automatically, rather
than reporting a falsely confident number.

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

The database file (`research.db` by default, SQLite) is created
automatically on first run. Override its location with:

```bash
export RESEARCH_DB_PATH=/path/to/research.db
```

## How the decomposition works

`app/decomposer.py` is rule-based (not an LLM call), so the whole
pipeline runs offline and deterministically. It detects question shape
via word-boundary-aware keyword triggers:

- causal ("cause", "leads to", "because of") → mechanism/evidence templates
- comparison ("vs", "better than", "compared to") → criteria/differences templates
- current-event ("latest", "recent", "current") → recency/fact-check templates
- fallback → general consensus/evidence templates

Word-boundary matching matters here — an earlier version matched "or"
as a substring inside "more" and misclassified "Is remote work more
productive?" as a comparison question. That's now a regression test in
`tests/test_decomposer.py`.

## Project structure

```
research-assistant/
├── .github/workflows/test.yml   # CI: runs the full unittest suite on every push
├── app/
│   ├── main.py                   # FastAPI backend, validation, logging, error handling
│   ├── db.py                     # SQLite persistence layer
│   ├── decomposer.py             # rule-based question decomposition
│   ├── similarity.py             # TF-IDF + cosine similarity, from scratch
│   ├── aggregation.py            # confidence aggregation + contradiction detection
│   ├── research_backend.py       # abstract interface for pluggable source-fetching
│   └── logging_config.py         # centralized structured logging
├── tests/
│   ├── test_decomposer.py
│   ├── test_similarity.py
│   ├── test_aggregation.py
│   └── test_db.py
└── requirements.txt
```

## Extending it

- **Wire in real source fetching**: implement `ResearchBackend.search()`
  in `app/research_backend.py` against a real search tool. The
  `StaticFixtureBackend` class shows the shape for testing without a
  live network call.
- **Smarter decomposition**: replace the templates in `decomposer.py`
  with an LLM call, keeping the same `decompose(question) -> list[str]`
  signature — nothing downstream needs to change.
- **Swap the database**: `app/db.py` is the only file that touches
  SQLite directly — replace its internals with Postgres and the rest of
  the app is unaffected.
- **Better similarity**: `app/similarity.py`'s TF-IDF approach is
  lightweight and dependency-free but still lexical, not truly semantic
  — swap in sentence embeddings for matching across languages or very
  differently-worded questions.

## Limitations

- The decomposer is template-based, not an LLM — it produces sub-questions
  within four fixed question-shape categories, not genuinely novel ones.
- TF-IDF similarity is lexical (shared meaningful words), not deep
  semantic understanding — two questions about the same topic using
  completely disjoint vocabulary won't be matched.
- This tracks and organizes research; it does not perform the research
  itself. Sources and answers still need to come from something else — a
  person, or a `ResearchBackend` implementation wired to a real search tool.
- SQLite is fine for a single-instance backend or local dev; for
  multi-instance production use, move to Postgres.

## License

MIT
