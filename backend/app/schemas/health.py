from pydantic import BaseModel


class HealthResponse(BaseModel):
    """System readiness and runtime hardware status."""

    status: str
    database_connected: bool
    cuda_available: bool
    device_name: str | None = None
    llm_provider: str
    llm_model: str
    embedding_model: str

