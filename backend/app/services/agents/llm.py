from functools import lru_cache

from fastapi import HTTPException, status
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings


def _build_gemini_model(model_name: str, api_key: str, temperature: float) -> ChatGoogleGenerativeAI:
    if not api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GOOGLE_API_KEY is not configured in .env.",
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


@lru_cache(maxsize=4)
def _cached_gemini_llm(model_name: str, api_key: str, temperature: float) -> ChatGoogleGenerativeAI:
    return _build_gemini_model(model_name=model_name, api_key=api_key, temperature=temperature)


def get_chat_llm() -> BaseChatModel:
    """FastAPI dependency returning the primary Gemini generation model."""
    settings = get_settings()
    return _cached_gemini_llm(
        model_name=settings.llm_model,
        api_key=settings.google_api_key,
        temperature=0.1,
    )


def get_judge_llm() -> BaseChatModel:
    """Return deterministic Gemini judge model for LLM-as-a-Judge evaluations."""
    settings = get_settings()
    return _cached_gemini_llm(
        model_name=settings.judge_llm_model,
        api_key=settings.google_api_key,
        temperature=0.0,
    )

