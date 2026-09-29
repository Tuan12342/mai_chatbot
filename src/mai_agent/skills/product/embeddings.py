from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from mai_agent.config import get_settings


@lru_cache
def get_product_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Tạo Gemini Embedding client dùng chung cho product retrieval."""
    settings = get_settings()

    return GoogleGenerativeAIEmbeddings(
        model=settings.google_embedding_model,
        api_key=settings.google_api_key,
        output_dimensionality=768,
    )
