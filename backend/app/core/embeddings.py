"""Embedding function for the Embedding pipeline (design doc Section 8.2.1).

Backend selection (resolved ONCE, then fixed so all vectors share a dimension),
with RETRIES so a single transient Gemini timeout behind a corporate proxy does
NOT permanently lock the process to the near-random hashing fallback (that was
the root cause of false-positive duplicate detection):
  1. Gemini gemini-embedding-001 — real semantic vectors (FR-6/FR-7).
  2. sentence-transformers (all-MiniLM-L6-v2) if installed.
  3. Deterministic hashing fallback — LAST resort, logged loudly. Similarity is
     NOT semantic here, so DuplicateAgent is made conservative (see
     is_semantic_backend()).

Task types matter for quality: KB documents use RETRIEVAL_DOCUMENT, RAG queries
use RETRIEVAL_QUERY, and ticket-vs-ticket duplicate checks use
SEMANTIC_SIMILARITY (which separates unrelated same-domain tickets better).
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import re
import time

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger("helpdesk.embeddings")
settings = get_settings()

_HASH_DIM = 384
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_mode: str | None = None            # "gemini" | "st" | "hash"
_backend = "unresolved"
_st_model = None
_gemini_dim: int | None = None

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="embed")
_EMBED_TIMEOUT_S = 30
_RETRIES = 3

# Task types for gemini-embedding-001.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"     # KB article chunks
TASK_QUERY = "RETRIEVAL_QUERY"           # RAG search query
TASK_SIMILARITY = "SEMANTIC_SIMILARITY"  # ticket-vs-ticket duplicate check


def _resolve_backend() -> None:
    """Pick the embedding backend exactly once (with retries) and cache it."""
    global _mode, _backend, _st_model, _gemini_dim
    if _mode is not None:
        return

    # 1) Gemini — probe with retries so a transient timeout doesn't lock hashing.
    if settings.gemini_api_key:
        dim = _probe_gemini_with_retries()
        if dim:
            _mode = "gemini"
            _gemini_dim = dim
            _backend = f"gemini/{settings.gemini_embedding_model}"
            logger.info("Embedding backend ACTIVE: %s (dim=%s)", _backend, dim)
            return
        if settings.require_gemini:
            # REQUIRE_GEMINI: never fall back to hashing/ST. Stay on Gemini and
            # retry per call; a persistent failure yields a dimension-safe zero
            # vector (no false matches), never near-random hashing vectors.
            _mode = "gemini"
            _gemini_dim = 3072
            _backend = f"gemini/{settings.gemini_embedding_model} (retry-per-call, no fallback)"
            logger.error(
                "REQUIRE_GEMINI=true: Gemini embeddings not reachable at startup. Will retry "
                "each call and will NOT use the hashing fallback. Check GEMINI_API_KEY / network.")
            return
        logger.error(
            "Gemini embeddings unavailable after %d attempts; trying local backends. "
            "Check GEMINI_API_KEY, network/TLS (DISABLE_SSL_VERIFY or CA_BUNDLE).", _RETRIES,
        )

    # 2) sentence-transformers (offline real embedder).
    try:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        _mode = "st"
        _backend = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("Embedding backend ACTIVE: %s", _backend)
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("sentence-transformers unavailable (%s).", exc)

    # 3) Hashing fallback — loud, and dedup is made conservative elsewhere.
    _mode = "hash"
    _backend = "hashing-fallback-384d"
    logger.error(
        "Embedding backend ACTIVE: HASHING FALLBACK (384-d, NOT semantic). RAG grounding "
        "and duplicate similarity will be unreliable — duplicate detection is disabled to "
        "avoid false positives. Fix Gemini/network or install sentence-transformers.",
    )


def _probe_gemini_with_retries() -> int | None:
    for attempt in range(1, _RETRIES + 1):
        try:
            vec = _EXECUTOR.submit(_gemini_embed_raw, "ping", TASK_DOCUMENT).result(
                timeout=_EMBED_TIMEOUT_S + 3)
            return len(vec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini embed probe %d/%d failed (%s).", attempt, _RETRIES, exc or "timeout")
            time.sleep(1.0 * attempt)
    return None


def _gemini_embed_raw(text: str, task_type: str) -> list[float]:
    from google.genai import types

    from app.core.gemini import get_client

    resp = get_client().models.embed_content(
        model=settings.gemini_embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            http_options=types.HttpOptions(timeout=_EMBED_TIMEOUT_S * 1000),  # ms
        ),
    )
    return list(resp.embeddings[0].values)


def _gemini_embed(text: str, task_type: str) -> list[float]:
    for attempt in range(1, _RETRIES + 1):
        try:
            return _EXECUTOR.submit(_gemini_embed_raw, text, task_type).result(
                timeout=_EMBED_TIMEOUT_S + 3)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini embed %d/%d failed (%s).", attempt, _RETRIES, exc or "timeout")
            time.sleep(0.5 * attempt)
    # Dimension-safe failure: a zero vector (cosine 0 with everything -> no false
    # match), NEVER a different-dimension hash vector that would corrupt the index.
    logger.error("Gemini embed failed after %d retries; returning zero vector.", _RETRIES)
    return [0.0] * (_gemini_dim or 3072)


def _hash_embed(text: str) -> list[float]:
    """Deterministic, dependency-free embedding: hashed token frequencies, L2-normalized."""
    vec = np.zeros(_HASH_DIM, dtype=np.float32)
    for tok in _TOKEN_RE.findall(text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % _HASH_DIM] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


def embed(text: str, task_type: str = TASK_DOCUMENT) -> list[float]:
    _resolve_backend()
    if _mode == "gemini":
        return _gemini_embed(text, task_type)
    if _mode == "st" and _st_model is not None:
        emb = _st_model.encode(text, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32).tolist()
    return _hash_embed(text)


def embed_many(texts: list[str], task_type: str = TASK_DOCUMENT) -> list[list[float]]:
    return [embed(t, task_type) for t in texts]


def embedding_backend() -> str:
    _resolve_backend()
    return _backend


def is_semantic_backend() -> bool:
    """True when a REAL embedding model is active (Gemini or sentence-transformers),
    False for the hashing fallback. Callers use this to disable duplicate detection
    when similarity would be meaningless."""
    _resolve_backend()
    return _mode in ("gemini", "st")
