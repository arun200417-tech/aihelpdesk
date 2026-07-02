"""Application configuration loaded from environment variables.

No secrets are hardcoded (NFR: Security). All tunables — including the AI
confidence threshold (design doc Section 9, step 8) — are environment-driven.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = .../AI_Helpdesk_Router_Project
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Security ---
    # Opaque session tokens (no JWT/OAuth2); this controls their lifetime only.
    access_token_expire_minutes: int = 120

    # --- Storage ---
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'helpdesk.db').as_posix()}"
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma")
    upload_dir: str = str(PROJECT_ROOT / "data" / "uploads")

    # --- AI pipeline tunables (configurable, never hardcoded) ---
    ai_confidence_threshold: float = 0.70   # KB auto-resolve confidence gate (>= 70%)
    # Min RAG retrieval score to count as a real "KB match" (calibrated: genuine
    # matches score ~0.70-0.77, spurious/no-match <=0.66 with gemini embeddings).
    kb_match_min_score: float = 0.68
    # Duplicate detection uses two tiers, TUNED against real gemini-embedding-001
    # SEMANTIC_SIMILARITY scores: unrelated tickets top out ~0.80, genuine
    # duplicates score ~0.91-0.95. At/above the merge threshold the new ticket is
    # auto-resolved from the original's resolution; between suggest and merge it
    # only shows "possibly related" and continues to RAG; below suggest, nothing.
    duplicate_similarity_threshold: float = 0.90   # merge (auto-resolve as duplicate)
    duplicate_suggest_threshold: float = 0.85      # show as possibly related
    rag_top_k: int = 4

    # --- External LLM provider (Gemini only) ---
    # When true (and a key is set), embeddings use Gemini ONLY — the hashing/ST
    # fallbacks are disabled so no near-random vectors are ever used. RAG answers
    # are always Gemini-generated; there is no fallback answer generator.
    require_gemini: bool = True
    gemini_api_key: str = ""
    # flash-lite: much higher free-tier daily quota than gemini-2.5-flash.
    gemini_model: str = "gemini-2.5-flash-lite"
    # gemini-embedding-001 is available on the REST v1beta endpoint (3072-dim).
    gemini_embedding_model: str = "models/gemini-embedding-001"

    # --- Outbound TLS (backend -> external APIs such as Gemini) ---
    # Corporate proxies that intercept TLS (Zscaler/Netskope) cause
    # "CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate"
    # because Python does not trust the proxy's root CA. Choose one remedy:
    #   ca_bundle          -> RECOMMENDED: path to the corporate root CA (.pem)
    #   disable_ssl_verify -> INSECURE: skip cert verification (dev/demo only)
    ca_bundle: str = ""
    disable_ssl_verify: bool = False

    # --- Frontend ---
    api_base_url: str = "http://localhost:8000"

    def ensure_dirs(self) -> None:
        """Create runtime directories if they do not exist."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        db_path = self.database_url.replace("sqlite:///", "")
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
