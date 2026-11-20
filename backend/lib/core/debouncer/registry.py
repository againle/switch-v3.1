"""Method registry: turns ``debounce.method`` from config.yaml into a policy.

Methods register themselves by decorating their class::

    @register_policy("typing_aware")
    class TypingAwarePolicy(DebouncePolicy):
        ...

The built-in methods live in ``methods/`` and are imported lazily so this
module never imports them at package-import time (avoids circular imports and
keeps a broken third-party method from taking the whole backend down).

An unknown or misspelled method name is **not** fatal: it logs a warning and
degrades to :data:`DEFAULT_METHOD`, so a bad config line can never leave the
session without a policy.
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .schema import DebouncePolicy, DebouncePolicyResult, PolicyContext

logger = logging.getLogger(__name__)

#: Method used when config.yaml is silent or names something unknown.
DEFAULT_METHOD = "fixed_rules"

#: name → policy class, filled by :func:`register_policy`.
_REGISTRY: dict[str, type[DebouncePolicy]] = {}

_LOADED = False


def register_policy(name: str) -> Callable[[type[DebouncePolicy]], type[DebouncePolicy]]:
    """Class decorator that registers a method under *name*.

    The decorator's name wins over the class attribute, so the config key and
    the registry key can never drift apart.
    """
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("register_policy() needs a non-empty method name")

    def decorator(cls: type[DebouncePolicy]) -> type[DebouncePolicy]:
        cls.name = key
        _REGISTRY[key] = cls
        logger.debug("[debouncer] registered method %r -> %s", key, cls.__name__)
        return cls

    return decorator


def _load_builtin_methods() -> None:
    """Import the built-in method modules once (registration side effect)."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from . import methods  # noqa: F401  (import registers every built-in method)


def available_policies() -> tuple[str, ...]:
    """Names of every registered method, sorted."""
    _load_builtin_methods()
    return tuple(sorted(_REGISTRY))


def resolve_policy_class(method: str | None) -> type[DebouncePolicy]:
    """Look up a method class by config name, falling back to the default."""
    _load_builtin_methods()
    key = (method or DEFAULT_METHOD).strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        # A typo here silently changes pacing, so name the likely intent.
        suggestion = difflib.get_close_matches(key, sorted(_REGISTRY), n=1, cutoff=0.6)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        logger.warning(
            "[debouncer] unknown debounce method %r in config.yaml (debounce.method) — "
            "falling back to %r.%s Available: %s",
            method,
            DEFAULT_METHOD,
            hint,
            ", ".join(available_policies()),
        )
        return _REGISTRY[DEFAULT_METHOD]
    return cls


def build_policy(
    method: str | None = None,
    settings: Mapping[str, Any] | None = None,
) -> DebouncePolicy:
    """Instantiate the method named *method* with its config settings.

    Raises:
        KeyError: if even the default method is missing (i.e. the built-in
            methods failed to import) — a genuinely broken installation.
    """
    return resolve_policy_class(method)(settings)


class CallablePolicyAdapter(DebouncePolicy):
    """Adapts a plain ``(messages, emotion) -> result`` callable to the port.

    Only used for legacy/test wiring (``MessageDebouncer(policy=...)``); new
    code should implement :class:`DebouncePolicy`.
    """

    def __init__(self, fn: Callable[[Sequence[str], Any], DebouncePolicyResult], name: str = "callable") -> None:
        super().__init__({})
        self._fn = fn
        self.name = name

    def evaluate(self, ctx: PolicyContext) -> DebouncePolicyResult:
        return self._fn(list(ctx.pending_messages), ctx.emotion)


__all__ = [
    "CallablePolicyAdapter",
    "DEFAULT_METHOD",
    "available_policies",
    "build_policy",
    "register_policy",
    "resolve_policy_class",
]
