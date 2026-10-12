import logging
from functools import lru_cache

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from mai_agent.config import get_settings

LOGGER = logging.getLogger(__name__)


@lru_cache
def get_langfuse_client() -> Langfuse | None:
    settings = get_settings()
    public_key = settings.langfuse_public_key.strip()
    secret_key = settings.langfuse_secret_key.strip()
    if not public_key or not secret_key:
        LOGGER.warning("Langfuse tracing is disabled because its API keys are missing.")
        return None

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=settings.langfuse_base_url.strip() or "https://cloud.langfuse.com",
        environment=settings.langfuse_tracing_environment.strip() or "development",
    )


def create_langfuse_handler() -> CallbackHandler | None:
    client = get_langfuse_client()
    if client is None:
        return None
    return CallbackHandler(public_key=get_settings().langfuse_public_key.strip())
