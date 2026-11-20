"""Dense embeddings (bge-m3 via Ollama) and belief matching.

No keyphrase extraction. Each user turn embeds the current social-worker
utterance plus the last p trainee/client utterances, then scores them
against persona Belief vectors with weighted max-pooling.
"""

import numpy as np
import requests

from . import config

# Recency weights: current social-worker message dominates; prior trainee
# and client turns can still activate a lingering belief (e.g. "tell me more").
WEIGHT_CURRENT_TRAINEE = 1.0
WEIGHT_PRIOR_TRAINEE = 0.9
WEIGHT_PRIOR_CLIENT = 0.75


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/embed",
        json={"model": config.EMBEDDING_MODEL, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def belief_match_text(text: str, cues: str | None = None) -> str:
    """What we embed for a Belief: the statement plus optional author cues.

    Cues are retrieval hints written on the persona (not extracted from user
    input). They pull the belief vector toward therapy language like
    'failure' or 'asking for help' without a feature extractor.
    """
    text = (text or "").strip()
    cues = (cues or "").strip()
    if cues:
        return f"{text}\nRelated themes: {cues}"
    return text


def score_beliefs(
    query_vectors: np.ndarray,
    query_weights: np.ndarray,
    belief_vectors: np.ndarray,
) -> np.ndarray:
    """Weighted max-pool: score[j] = max_i weight_i * cosine(query_i, belief_j)."""
    if query_vectors.size == 0 or belief_vectors.size == 0:
        return np.zeros((belief_vectors.shape[0] if belief_vectors.size else 0,), dtype=np.float32)
    sims = cosine(query_vectors, belief_vectors)  # (n_queries, n_beliefs)
    return np.max(sims * query_weights.reshape(-1, 1), axis=0)


def select_matches(
    scores: np.ndarray,
    threshold: float | None = None,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Return (belief_index, score) pairs that clear the threshold, strongest first."""
    threshold = config.BELIEF_MATCH_THRESHOLD if threshold is None else threshold
    top_k = config.BELIEF_MATCH_TOP_K if top_k is None else top_k
    ranked = sorted(enumerate(scores.tolist()), key=lambda kv: kv[1], reverse=True)
    hits = [(i, s) for i, s in ranked if s >= threshold]
    return hits[:top_k]
