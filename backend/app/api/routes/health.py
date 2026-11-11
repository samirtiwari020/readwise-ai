import torch
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Report database connectivity, CUDA GPU readiness, and configured models."""
    settings = get_settings()
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else None

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database_connected=db_ok,
        cuda_available=cuda_ok,
        device_name=device_name,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        embedding_model=settings.hf_embedding_model,
    )
