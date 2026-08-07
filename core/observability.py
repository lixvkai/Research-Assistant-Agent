"""Langfuse v4 observability adapter.

Langfuse is a fail-open side channel: initialization/export failures are logged
but never change the Agent's business result.  A request root is created as a
manual observation so its lifetime may safely span Gradio generator yields.
Child observations are activated only for bounded work and never across yield.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Iterator

from config import settings

logger = logging.getLogger(__name__)

_client: Any | None = None
_client_lock = threading.Lock()
_warned_incomplete = False
_MISSING = object()

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "secret_key",
    "token",
}
_WINDOWS_USER_PATH = re.compile(r"(?i)([a-z]:\\users\\)[^\\/]+")
_BEARER_TOKEN = re.compile(r"(?i)(bearer\s+)[a-z0-9._~+/=-]+")
_SK_TOKEN = re.compile(r"\b(?:sk|pk)-[a-zA-Z0-9_-]{8,}\b")


@dataclass(frozen=True)
class TraceScope:
    """Explicit root context propagated through the project's worker threads."""

    trace_id: str
    parent_observation_id: str
    session_id: str
    tags: tuple[str, ...] = ("research-assistant", "react")


@dataclass
class AgentTrace:
    scope: TraceScope
    observation: Any


_current_scope: contextvars.ContextVar[TraceScope | None] = contextvars.ContextVar(
    "langfuse_trace_scope", default=None
)


def is_observability_enabled() -> bool:
    """Return true only when tracing is explicitly enabled and credentials exist."""
    global _warned_incomplete
    credentials_ok = bool(
        settings.LANGFUSE_PUBLIC_KEY
        and settings.LANGFUSE_SECRET_KEY
        and settings.LANGFUSE_BASE_URL
    )
    enabled = bool(settings.LANGFUSE_ENABLED and credentials_ok)
    if settings.LANGFUSE_ENABLED and not credentials_ok and not _warned_incomplete:
        logger.warning("Langfuse 已启用但凭据不完整，自动降级为关闭状态")
        _warned_incomplete = True
    return enabled


def _mask_string(value: str) -> str:
    masked = value
    for secret in (
        settings.DEEPSEEK_API_KEY,
        settings.LANGFUSE_PUBLIC_KEY,
        settings.LANGFUSE_SECRET_KEY,
    ):
        if secret:
            masked = masked.replace(secret, "***")
    masked = _BEARER_TOKEN.sub(r"\1***", masked)
    masked = _SK_TOKEN.sub("***", masked)
    return _WINDOWS_USER_PATH.sub(r"\1***", masked)


def mask_payload(
    value: Any = _MISSING, *, data: Any = _MISSING, **_kwargs: Any
) -> Any:
    """Mask credentials and local usernames before data leaves the process."""
    # Langfuse v4 invokes custom mask hooks as ``mask(data=...)``.  Keeping the
    # positional form as well lets project code reuse the exact same sanitizer.
    if data is not _MISSING:
        value = data
    elif value is _MISSING:
        value = None
    if not settings.LANGFUSE_CAPTURE_CONTENT:
        if isinstance(value, str):
            return "<content omitted>"
        if isinstance(value, (list, tuple)):
            return ["<content omitted>"] if value else []
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SENSITIVE_KEYS or normalized.endswith("_api_key"):
                result[key] = "***"
            else:
                result[key] = mask_payload(item)
        return result
    if isinstance(value, (list, tuple)):
        return [mask_payload(item) for item in value]
    if isinstance(value, str):
        return _mask_string(value)
    return value


def capture_value(value: Any, *, max_chars: int = 8_000) -> Any:
    """Prepare observation input/output without allowing unbounded tool payloads."""
    if not settings.LANGFUSE_CAPTURE_CONTENT:
        text = str(value) if value is not None else ""
        return {"content_captured": False, "characters": len(text)}
    masked = mask_payload(value)
    if isinstance(masked, str) and len(masked) > max_chars:
        return masked[:max_chars] + f"\n… <truncated {len(masked) - max_chars} chars>"
    return masked


def get_langfuse_client() -> Any | None:
    """Lazily initialize one Langfuse client after dotenv-backed settings load."""
    global _client
    if not is_observability_enabled():
        return None
    if _client is None:
        with _client_lock:
            if _client is None:
                try:
                    from langfuse import Langfuse

                    _client = Langfuse(
                        public_key=settings.LANGFUSE_PUBLIC_KEY,
                        secret_key=settings.LANGFUSE_SECRET_KEY,
                        base_url=settings.LANGFUSE_BASE_URL,
                        environment=settings.LANGFUSE_TRACING_ENVIRONMENT,
                        release=settings.LANGFUSE_RELEASE,
                        tracing_enabled=True,
                        mask=mask_payload,
                    )
                    logger.info(
                        "Langfuse tracing 已启用（environment=%s, release=%s）",
                        settings.LANGFUSE_TRACING_ENVIRONMENT,
                        settings.LANGFUSE_RELEASE,
                    )
                except Exception:
                    logger.exception("Langfuse 初始化失败，观测功能已降级")
                    return None
    return _client


def get_current_trace_scope() -> TraceScope | None:
    return _current_scope.get()


@contextlib.contextmanager
def bind_trace_scope(scope: TraceScope | None) -> Iterator[None]:
    """Bind a scope for bounded work; this context manager must not cross yield."""
    if scope is None:
        yield
        return
    token = _current_scope.set(scope)
    try:
        yield
    finally:
        _current_scope.reset(token)


def _trace_attributes(scope: TraceScope) -> dict[str, Any]:
    return {
        "trace_name": "research-assistant-chat",
        "session_id": scope.session_id,
        "tags": list(scope.tags),
        "version": settings.LANGFUSE_RELEASE,
        "environment": settings.LANGFUSE_TRACING_ENVIRONMENT,
    }


def start_agent_trace(
    *, session_id: str, input: Any, metadata: dict[str, Any] | None = None
) -> AgentTrace | None:
    """Start a root agent observation without holding an OTEL context open."""
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        attributes = {
            "trace_name": "research-assistant-chat",
            "session_id": str(session_id),
            "tags": ["research-assistant", "react"],
            "version": settings.LANGFUSE_RELEASE,
            "environment": settings.LANGFUSE_TRACING_ENVIRONMENT,
        }
        from langfuse import propagate_attributes

        # Activate the root only for creation/attribute propagation.  It remains
        # open after this bounded context and is ended by finish_agent_trace().
        # This avoids carrying an OpenTelemetry ContextVar across Gradio yields.
        with client.start_as_current_observation(
            name="research-assistant-chat",
            as_type="agent",
            input=capture_value(input),
            metadata=mask_payload(metadata or {}),
            end_on_exit=False,
        ) as observation:
            with propagate_attributes(**attributes):
                pass
        scope = TraceScope(
            trace_id=observation.trace_id,
            parent_observation_id=observation.id,
            session_id=str(session_id),
        )
        return AgentTrace(scope=scope, observation=observation)
    except Exception:
        logger.exception("创建 Langfuse 根 observation 失败，继续执行业务流程")
        return None


def finish_agent_trace(
    trace: AgentTrace | None,
    *,
    output: Any = None,
    error: BaseException | str | None = None,
) -> None:
    if trace is None:
        return
    try:
        if error is not None:
            message = _mask_string(str(error))
            trace.observation.update(
                output=capture_value(output), level="ERROR", status_message=message[:500]
            )
        else:
            trace.observation.update(output=capture_value(output))
        trace.observation.end()
    except Exception:
        logger.exception("结束 Langfuse 根 observation 失败（已忽略）")


def _active_otel_ids() -> tuple[str | None, str | None]:
    """Read the active OTel context without triggering Langfuse no-span warnings."""
    try:
        from opentelemetry import trace as otel_trace

        span_context = otel_trace.get_current_span().get_span_context()
        if not span_context.is_valid:
            return None, None
        return f"{span_context.trace_id:032x}", f"{span_context.span_id:016x}"
    except Exception:
        return None, None


def _explicit_parent(scope: TraceScope) -> dict[str, str] | None:
    """Use the active Langfuse span when available, otherwise attach to root."""
    active_trace, active_observation = _active_otel_ids()
    if active_trace == scope.trace_id and active_observation:
        return None
    return {
        "trace_id": scope.trace_id,
        "parent_span_id": scope.parent_observation_id,
    }


@contextlib.contextmanager
def observe_operation(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Create a bounded child observation and preserve business exceptions."""
    scope = get_current_trace_scope()
    client = get_langfuse_client()
    if scope is None or client is None:
        yield None
        return

    stack = contextlib.ExitStack()
    observation = None
    try:
        from langfuse import propagate_attributes

        observation = stack.enter_context(
            client.start_as_current_observation(
                trace_context=_explicit_parent(scope),
                name=name,
                as_type=as_type,
                input=capture_value(input),
                metadata=mask_payload(metadata or {}),
            )
        )
        # The span must be current before propagate_attributes is entered;
        # otherwise Langfuse correctly warns that no active span exists.
        stack.enter_context(propagate_attributes(**_trace_attributes(scope)))
    except Exception:
        logger.exception("创建 Langfuse observation '%s' 失败（已降级）", name)
        stack.close()
        yield None
        return

    try:
        yield observation
    except BaseException as exc:
        try:
            observation.update(level="ERROR", status_message=_mask_string(str(exc))[:500])
        except Exception:
            logger.debug("更新 Langfuse 错误状态失败", exc_info=True)
        raise
    finally:
        try:
            stack.close()
        except Exception:
            logger.exception("关闭 Langfuse observation '%s' 失败（已忽略）", name)


def openai_trace_kwargs(name: str = "deepseek-chat") -> dict[str, Any]:
    """Return Langfuse-only kwargs consumed by its OpenAI drop-in wrapper."""
    if not is_observability_enabled():
        return {}
    kwargs: dict[str, Any] = {"name": name}
    scope = get_current_trace_scope()
    client = get_langfuse_client()
    if scope is None or client is None:
        return kwargs
    active_trace, active_observation = _active_otel_ids()
    if active_trace != scope.trace_id or not active_observation:
        kwargs.update(
            trace_id=scope.trace_id,
            parent_observation_id=scope.parent_observation_id,
        )
    return kwargs


def flush_observability() -> None:
    client = _client
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.exception("Langfuse flush 失败（已忽略）")


def shutdown_observability() -> None:
    client = _client
    if client is None:
        return
    try:
        client.shutdown()
    except Exception:
        logger.exception("Langfuse shutdown 失败（已忽略）")


def _reset_for_tests() -> None:
    """Reset singleton state; intended only for isolated unit tests."""
    global _client, _warned_incomplete
    _client = None
    _warned_incomplete = False
