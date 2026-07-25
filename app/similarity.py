"""
similarity.py — TF-IDF vectorization and cosine similarity, implemented
from scratch in pure Python (no numpy/sklearn dependency).

Used to find semantically similar past research topics — genuinely
weighs rare, distinctive words more heavily than common ones, unlike
a naive keyword-overlap count.
"""

import re
import math
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "can", "may", "might", "must", "shall", "to", "of", "in",
    "on", "at", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "and", "or", "but", "if", "this",
    "that", "these", "those", "it", "its", "as", "what", "which", "who",
}


def tokenize(text: str) -> list:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def compute_tf(tokens: list) -> dict:
    """Term frequency: count normalized by document length."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def compute_idf(documents_tokens: list) -> dict:
    """Inverse document frequency across a corpus of tokenized documents."""
    n_docs = len(documents_tokens)
    if n_docs == 0:
        return {}
    df = Counter()
    for tokens in documents_tokens:
        for term in set(tokens):
            df[term] += 1
    # smoothed idf, avoids division by zero and zero-idf for terms in every doc
    return {term: math.log((1 + n_docs) / (1 + count)) + 1 for term, count in df.items()}


def compute_tfidf_vector(tf: dict, idf: dict) -> dict:
    return {term: freq * idf.get(term, 0.0) for term, freq in tf.items()}


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    if not vec_a or not vec_b:
        return 0.0
    shared_terms = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[t] * vec_b[t] for t in shared_terms)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_similarity(query: str, candidates: list, key=lambda x: x) -> list:
    """Ranks `candidates` by TF-IDF cosine similarity to `query`.

    `key` extracts the text to compare from each candidate (defaults to
    identity, i.e. candidates are already strings).

    Returns list of (score, candidate) tuples sorted descending by score,
    with score in [0, 1]. Zero-score candidates are excluded.
    """
    query_tokens = tokenize(query)
    candidate_texts = [key(c) for c in candidates]
    candidate_tokens = [tokenize(t) for t in candidate_texts]

    # build idf across query + all candidates so query terms are represented
    corpus = candidate_tokens + [query_tokens]
    idf = compute_idf(corpus)

    query_vec = compute_tfidf_vector(compute_tf(query_tokens), idf)

    scored = []
    for candidate, tokens in zip(candidates, candidate_tokens):
        cand_vec = compute_tfidf_vector(compute_tf(tokens), idf)
        score = cosine_similarity(query_vec, cand_vec)
        if score > 0:
            scored.append((score, candidate))

    scored.sort(key=lambda x: -x[0])
    return scored


if __name__ == "__main__":
    candidates = [
        "Does remote work reduce productivity?",
        "Is chocolate bad for your skin?",
        "What are the productivity effects of working from home?",
        "Python vs JavaScript for backend development",
    ]
    query = "remote work and productivity levels"
    results = rank_by_similarity(query, candidates)
    for score, text in results:
        print(f"{score:.3f}  {text}")
