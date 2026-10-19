import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any

from langfuse import Langfuse, propagate_attributes
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


def langfuse_observation(
    client: Langfuse | None,
    *,
    name: str,
    as_type: str = "span",
    input: Any = None,
    metadata: Any = None,
    model: str | None = None,
):
    """Create a nested observation without making tracing a runtime dependency."""
    if client is None:
        return nullcontext(None)
    return client.start_as_current_observation(
        name=name,
        as_type=as_type,
        input=input,
        metadata=metadata,
        model=model,
    )


@contextmanager
def langfuse_trace(
    client: Langfuse | None,
    *,
    name: str,
    user_id: str,
    session_id: str,
    input: Any = None,
    tags: list[str] | None = None,
) -> Iterator[Any]:
    """Create a root trace and attach request identity attributes."""
    if client is None:
        yield None
        return

    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        tags=tags or [],
        trace_name=name,
    ):
        with client.start_as_current_observation(
            name=name,
            as_type="chain",
            input=input,
        ) as observation:
            yield observation
