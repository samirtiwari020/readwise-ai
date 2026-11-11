from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"


def normalize_database_url(raw_url: str) -> str:
    """Ensure special characters like '@' inside the DB password are percent-encoded."""
    if "://" not in raw_url:
        return raw_url
    scheme, rest = raw_url.split("://", 1)
    if "/" not in rest:
        return raw_url
    authority, path = rest.split("/", 1)
    if authority.count("@") <= 1:
        return raw_url
    creds, hostport = authority.rsplit("@", 1)
    if ":" not in creds:
        return raw_url
    user, password = creds.split(":", 1)
    encoded_user = quote(user, safe="%")
    encoded_password = quote(password, safe="%")
    return f"{scheme}://{encoded_user}:{encoded_password}@{hostport}/{path}"


class Settings(BaseSettings):
    """Application settings loaded from the root .env file."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Readwise AI Research Assistant"
    app_env: str = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/readwise_ai"

    # Storage & Vector Database
    upload_dir: str = "../data/uploads"
    chroma_persist_dir: str = "../data/chroma"
    chroma_collection_name: str = "document_chunks"

    # HuggingFace Embeddings & Reranker
    embedding_device: str = "cuda"
    hf_embedding_model: str = "BAAI/bge-base-en-v1.5"
    hf_reranker_model: str = "BAAI/bge-reranker-base"

    # LLM Provider (Gemini)
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    judge_llm_model: str = "gemini-2.5-pro"
    google_api_key: str = ""

    # Adaptive Chunking & Retrieval
    chunk_min_tokens: int = 200
    chunk_target_tokens: int = 600
    chunk_max_tokens: int = 1000
    chunk_overlap_tokens: int = 80
    retrieval_top_k: int = 20
    rerank_top_n: int = 6

    @field_validator("database_url", mode="before")
    @classmethod
    def _encode_db_url(cls, value: str) -> str:
        return normalize_database_url(value.strip())

    def _resolve_backend_relative_path(self, configured: str) -> Path:
        path = Path(configured)
        if path.is_absolute():
            return path
        return (BACKEND_DIR / path).resolve()

    @property
    def resolved_upload_dir(self) -> Path:
        """Absolute path for uploaded PDF storage."""
        return self._resolve_backend_relative_path(self.upload_dir)

    @property
    def resolved_chroma_dir(self) -> Path:
        """Absolute path for ChromaDB vector persistence."""
        return self._resolve_backend_relative_path(self.chroma_persist_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()

