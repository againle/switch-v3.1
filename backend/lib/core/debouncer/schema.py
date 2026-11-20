"""The debounce *port* — the schema every turn-taking method implements.

A turn-taking **method** answers one question: *given what has happened so
far, how long should the debouncer keep collecting user messages before the
client answers?* Every method is a small class:

    @register_policy("my_method")
    class MyMethod(DebouncePolicy):
        name = "my_method"
        DEFAULTS = {"wait_seconds": 2.0}

        def evaluate(self, ctx: PolicyContext) -> DebouncePolicyResult:
            return DebouncePolicyResult(
                wait_seconds=self.number("wait_seconds", 2.0),
            )

The debouncer (``MessageDebouncer``) never knows which method is active. It
owns the *clock* and the *queue*, exposes both through a ``PolicyContext``
snapshot, calls the hooks below, and executes the returned
``DebouncePolicyResult``. A method therefore never touches asyncio, the
websocket, or the orchestrator.

Selecting a method is a config change only::

    debounce:
      method: typing_aware

See ``README.md`` in this package for the full contract and a template.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias

from .emotions import EmotionPolarity, EmotionReading, classify_emotion

#: What the debouncer should do *immediately* (before the wait elapses).
ImmediateAction: TypeAlias = Literal["none", "silence", "micro_response"]


@dataclass
class DebouncePolicyResult:
    """A single decision returned by a policy.

    Attributes:
        wait_seconds: How long to keep collecting after the last clock
            refresh (a new message, or a typing refresh granted by
            ``on_typing``).
        after_think_seconds: Alternative wait used for messages that
            arrived *while the client was already generating a reply* (see
            ``PolicyContext.after_think``).
        immediate_action: ``"none"`` for nothing, ``"silence"`` for a
            deliberate no-op tick, or ``"micro_response"`` to emit a short
            filler now.
        micro_response_text: The filler to emit when
            ``immediate_action == "micro_response"``.
        reason: Free-form label for logs / the status payload. Purely
            diagnostic — nothing branches on it.
    """

    wait_seconds: float
    after_think_seconds: float = 1.0
    immediate_action: ImmediateAction = "none"
    micro_response_text: str | None = None
    reason: str = ""

    @property
    def has_immediate_action(self) -> bool:
        return self.immediate_action != "none"


@dataclass(frozen=True)
class PolicyContext:
    """Read-only snapshot of everything a policy may use to decide.

    The debouncer builds a fresh context for every evaluation, so a policy
    can be stateless. It is frozen: decisions may not mutate it.
    """

    #: The un-flushed user messages, oldest first.
    pending_messages: tuple[str, ...] = ()
    #: Raw emotion payload as reported by the client (typically a list of
    #: strings, e.g. ``["Defensive", "Annoyed"]``).
    emotion: Any = None
    #: Coarse negative/positive/neutral read of ``emotion``.
    emotion_reading: EmotionReading = field(default_factory=EmotionReading)
    #: True while the client is generating a reply (the debouncer keeps
    #: collecting instead of flushing).
    is_thinking: bool = False
    #: True when the pending messages arrived while the client was thinking.
    after_think: bool = False
    #: True while the user is actively typing in the frontend input box.
    typing: bool = False
    #: Seconds since the clock was last refreshed.
    idle_seconds: float = 0.0
    #: ``time.monotonic()`` at snapshot time.
    now: float = 0.0

    # ── Convenience accessors ──────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of pending messages."""
        return len(self.pending_messages)

    @property
    def last_message(self) -> str:
        """Most recent pending message, stripped (``""`` if none)."""
        return self.pending_messages[-1].strip() if self.pending_messages else ""

    @property
    def combined_text(self) -> str:
        """All pending messages joined into one string."""
        return " ".join(message.strip() for message in self.pending_messages)

    @property
    def total_chars(self) -> int:
        """Length of ``combined_text``."""
        return len(self.combined_text)

    @property
    def polarity(self) -> EmotionPolarity:
        """Shortcut for ``emotion_reading.polarity``."""
        return self.emotion_reading.polarity


class DebouncePolicy(ABC):
    """Base class for every turn-taking method.

    Subclasses implement :meth:`evaluate` and may override :meth:`on_message`,
    :meth:`on_typing` and :meth:`reset`. Settings come from
    ``backend/config.yaml`` (``debounce`` globals plus the method's own block
    under ``debounce.methods.<name>``) and are merged over :attr:`DEFAULTS`,
    so an unknown or missing key can never crash a session.
    """

    #: Method name used in ``config.yaml`` (set by ``@register_policy``).
    name: ClassVar[str] = "base"

    #: Default settings for this method; overridden by config.yaml.
    DEFAULTS: ClassVar[Mapping[str, Any]] = {}

    #: When True, the host idles its silence timer while the user is typing.
    #: This is only the *default*; a method can override it per method from
    #: config.yaml with the ``suppress_idle_timer_while_typing`` setting (see
    #: :meth:`suppresses_idle_timer`). Off = typing never touches the host's
    #: idle / adaptive-wait clock, so the client keeps its own schedule.
    suppresses_idle_timer_while_typing: ClassVar[bool] = False

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.settings: dict[str, Any] = {**dict(self.DEFAULTS), **dict(settings or {})}

    # ── Hooks a method may implement ───────────────────────────────

    @abstractmethod
    def evaluate(self, ctx: PolicyContext) -> DebouncePolicyResult:
        """Return the decision for the current context.

        Called on every clock tick while messages are pending, after every
        new message, and after every typing signal — so it must be cheap and
        must not have side effects.
        """

    def on_cycle_start(self, ctx: PolicyContext) -> None:
        """Called when a new collection cycle begins (once per client turn).

        A cycle starts with the first message queued after the previous batch
        was flushed, so this is the place for **per-turn** decisions — a
        random draw, a coin flip, a per-turn budget. ``evaluate`` runs on
        every tick and must not be used for that, since it would re-roll
        constantly.
        """
        return None

    def on_cycle_end(self, ctx: PolicyContext) -> None:
        """Called after the batch has been handed to the client.

        ``ctx.pending_messages`` is empty here. Paired with
        :meth:`on_cycle_start` for per-turn state.
        """
        return None

    def on_message(self, text: str, ctx: PolicyContext) -> None:
        """Called once when a new user message is queued.

        The clock is already refreshed when this runs; use it for
        per-message bookkeeping only.
        """
        return None

    def on_typing(self, active: bool, ctx: PolicyContext) -> bool:
        """Called on every frontend typing signal.

        Args:
            active: ``True`` = the user just typed something; ``False`` =
                the user stopped typing or cleared the box.
            ctx: Context snapshot (``typing`` already reflects *active*).

        Returns:
            ``True`` to refresh the debounce clock. ``active=False`` is an
            explicit *stop* signal — returning ``True`` there would keep the
            window open forever after the user walked away, so the default
            refuses to refresh on it.
        """
        return False

    def reset(self) -> None:
        """Drop any per-session state (called when the session stops)."""
        return None

    def suppresses_idle_timer(self) -> bool:
        """Whether the host should stand its idle timer down while typing.

        Two different clocks are in play and this setting keeps them apart:

        * the **debounce** clock — how long to keep collecting messages before
          the client answers. Typing refreshes it (``on_typing``); that is the
          "wait for the user to finish their message" behaviour.
        * the **idle / adaptive-wait** clock — how long the client has been left
          in silence (``time_threshold``, driven by AdaWait). This method
          controls whether typing touches *that* one.

        Returns False by default: typing then does not reset or pause the idle
        clock at all, so the client keeps its own fixed schedule and can speak
        while the trainee is still composing. Return True (or set
        ``suppress_idle_timer_while_typing: true``) to hold the idle timer
        until the user stops typing.
        """
        return self.flag(
            "suppress_idle_timer_while_typing",
            self.suppresses_idle_timer_while_typing,
        )

    def status(self) -> Mapping[str, Any]:
        """Extra status fields this method wants in the status payload.

        Optional. Keys are merged into ``MessageDebouncer.status_info()``, so
        a probabilistic method (say, one that sometimes ignores a typing
        signal) can make its current state visible instead of mysterious.
        """
        return {}

    # ── Settings helpers (all tolerant of bad/missing values) ──────

    def setting(self, key: str, default: Any = None) -> Any:
        """Raw setting value, or *default* when unset."""
        value = self.settings.get(key, default)
        return default if value is None else value

    def number(self, key: str, default: float) -> float:
        """Setting coerced to float, falling back to *default*."""
        value = self.settings.get(key)
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def integer(self, key: str, default: int) -> int:
        """Setting coerced to int, falling back to *default*."""
        try:
            return int(self.number(key, default))
        except (TypeError, ValueError):
            return int(default)

    def flag(self, key: str, default: bool) -> bool:
        """Setting coerced to bool.

        Accepts real booleans as well as the strings YAML users tend to
        write (``"true"``/``"yes"``/``"on"``/``"1"``).
        """
        value = self.settings.get(key)
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

    # ── Optional helper for methods that need a custom reading ─────

    def reading(self, ctx: PolicyContext, strategy: str = "any") -> EmotionReading:
        """Classify ``ctx.emotion`` with a specific strategy.

        The debouncer already classifies with the default strategy and
        exposes it as ``ctx.emotion_reading``; call this only when a method
        wants a different reading (e.g. ``strategy="majority"``).
        """
        return classify_emotion(ctx.emotion, strategy=strategy)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} settings={self.settings!r}>"


__all__ = [
    "DebouncePolicy",
    "DebouncePolicyResult",
    "ImmediateAction",
    "PolicyContext",
]
