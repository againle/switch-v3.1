"""Message debouncer — batches a burst of user messages into one client turn.

The debouncer owns the *clock* and the *queue*; which turn-taking **method**
decides how long to wait is pluggable and selected in ``backend/config.yaml``::

    debounce:
      method: typing_aware        # see methods/ and README.md

Whatever the method, the debouncer provides the same services:

* a message queue (``add``) that batches a burst into a single turn;
* a monotonic clock that method hooks may refresh (``notify_typing``);
* the after-think state (``notify_think_completed``) for messages that
  arrived while the client was already generating;
* idle-timer suppression while input is pending (``set_input_pending``);
* immediate actions (``micro_response`` / ``silence``) via
  ``on_immediate_action``.

Adding a method means adding a class under ``methods/`` — this file should not
need to change.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from .emotions import EmotionPolarity, classify_emotion
from .registry import DEFAULT_METHOD, CallablePolicyAdapter, build_policy
from .schema import DebouncePolicy, DebouncePolicyResult, PolicyContext

logger = logging.getLogger(__name__)

#: Shortest window we will ever run, so a bad setting cannot cause a busy loop.
_MIN_WAIT = 0.05


class MessageDebouncer:
    """Accumulates user messages and flushes them according to the active method.

    The debouncer is deliberately method-agnostic: it drives the clock, feeds a
    :class:`PolicyContext` snapshot to the policy, and executes the returned
    :class:`DebouncePolicyResult`.
    """

    def __init__(
        self,
        *,
        method: str = DEFAULT_METHOD,
        settings: Mapping[str, Any] | None = None,
        settings_for: Callable[[str], Mapping[str, Any]] | None = None,
        is_thinking: Callable[[], bool] | None = None,
        set_input_pending: Callable[[bool], None] | None = None,
        set_typing_active: Callable[[bool], None] | None = None,
        get_current_emotion: Callable[[], Any] | None = None,
        on_immediate_action: Callable[[str, str | None], Awaitable[None] | None] | None = None,
        policy: DebouncePolicy | Callable[[Sequence[str], Any], DebouncePolicyResult] | None = None,
        max_wait_seconds: float = 0.0,
    ) -> None:
        """
        Args:
            method: Name of the turn-taking method (``config.yaml`` →
                ``debounce.method``). Unknown names degrade to the default.
            settings: Method settings (``debounce.methods.<method>``).
            settings_for: Optional resolver ``(method_name) -> settings``.
                Used only when the requested method is unknown, so the
                fallback method runs with ITS OWN config block instead of
                the block that was meant for the missing method.
            is_thinking: Returns True while the client is generating — the
                debouncer keeps collecting instead of flushing.
            set_input_pending: Notified with True/False as the queue fills and
                drains, so the host can stand its idle timer down.
            set_typing_active: Notified with True/False while the user is
                typing — only for methods that opt into
                ``suppresses_idle_timer_while_typing``.
            get_current_emotion: Returns the client's latest emotion payload.
            on_immediate_action: Invoked with ``(action, text)`` when a method
                asks for a filler / silence tick.
            policy: Optional pre-built policy (tests, custom wiring). A plain
                callable is adapted for backwards compatibility.
            max_wait_seconds: Hard ceiling on the TOTAL collection time for one
                burst, shared by every method. 0 disables the cap.
        """
        self._is_thinking = is_thinking or (lambda: False)
        self._set_input_pending = set_input_pending
        self._set_typing_active = set_typing_active
        self._get_current_emotion = get_current_emotion or (lambda: None)
        self._on_immediate_action = on_immediate_action

        if isinstance(policy, DebouncePolicy):
            self._policy: DebouncePolicy = policy
            self._method_requested: str = self._policy.name
        elif policy is not None:
            self._policy = CallablePolicyAdapter(policy, name=f"{method}:callable")
            self._method_requested = str(method or DEFAULT_METHOD).strip().lower()
        else:
            self._policy = build_policy(method, settings)
            # What config.yaml asked for, so a typo can be reported rather
            # than silently pacing with a different method.
            self._method_requested = str(method or DEFAULT_METHOD).strip().lower()
            if self._policy.name != self._method_requested and settings_for is not None:
                self._policy = build_policy(self._policy.name, settings_for(self._policy.name))

        self._pending: list[str] = []
        self._last_event_ts: float = time.monotonic()
        # Total-collection cap (0 = disabled). `_cap_start_ts` is the baseline
        # of the hard upper timer: it is reset by every NEW message but NOT by
        # typing refreshes, so continuous typing cannot postpone the cap while a
        # fresh message legitimately restarts it. `_timed_out` records that the
        # cap (not the method's window) released the last batch, so the UI can
        # say so.
        try:
            self._max_wait: float = max(0.0, float(max_wait_seconds or 0.0))
        except (TypeError, ValueError):
            self._max_wait = 0.0
        self._cap_start_ts: float | None = None
        self._timed_out: bool = False
        self._after_think: bool = False
        self._typing: bool = False
        self._last_polarity: EmotionPolarity = "neutral"
        self._current_result: DebouncePolicyResult = self._fallback_result()

        self._flush_cb: Callable[[list[str]], Awaitable[None]] | None = None
        self._task: asyncio.Task | None = None
        self._action_tasks: set[asyncio.Task] = set()
        self._closed: bool = False

        self._refresh_decision()

    # ── Introspection ───────────────────────────────────────────────

    @property
    def method_name(self) -> str:
        """Name of the active turn-taking method."""
        return self._policy.name

    @property
    def method_requested(self) -> str:
        """Method name config.yaml asked for (may differ from the live one)."""
        return self._method_requested

    @property
    def method_mismatch(self) -> bool:
        """True when the requested method is unknown and was replaced."""
        return self._method_requested != self.method_name

    @property
    def policy(self) -> DebouncePolicy:
        """The active policy instance."""
        return self._policy

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    @property
    def pending_messages(self) -> tuple[str, ...]:
        return tuple(self._pending)

    @property
    def is_typing(self) -> bool:
        """True while the frontend reports live typing."""
        return self._typing

    @property
    def current_wait(self) -> float:
        """The window currently in force (after-think variant when applicable)."""
        return self._current_wait()

    # ── Wiring ──────────────────────────────────────────────────────

    def set_flush_callback(self, cb: Callable[[list[str]], Awaitable[None]]) -> None:
        self._flush_cb = cb

    def start(self) -> None:
        """Start the background poll loop (idempotent)."""
        if self._task is None and not self._closed:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the loop and drop every host flag this debouncer raised."""
        self._closed = True
        self._cap_start_ts = None
        self._timed_out = False
        self._sync_pending_flag()
        self._set_typing(False)
        for task in list(self._action_tasks):
            task.cancel()
        self._action_tasks.clear()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._policy.reset()

    # ── Input ───────────────────────────────────────────────────────

    def add(self, text: str) -> None:
        """Queue a user message, refresh the clock, and evaluate the method."""
        starting_cycle = not self._pending
        # Every NEW message restarts the hard upper timer (typing refreshes do
        # NOT — see _cap_reached), and clears any previous "timed out" flag.
        self._cap_start_ts = time.monotonic()
        self._timed_out = False
        self._pending.append(text)
        self._last_event_ts = time.monotonic()

        ctx = self._build_context()
        if starting_cycle:
            # Gives the method one place per client turn to draw its dice.
            self._safe_hook("on_cycle_start", ctx)
        self._safe_hook("on_message", text, ctx)

        result = self._refresh_decision()
        self._handle_immediate_action(result)
        self._sync_pending_flag()

    def notify_typing(self, active: bool) -> None:
        """Feed a frontend typing signal to the method.

        ``active=True`` means "the user just typed new content"; only then may
        a method refresh the debounce clock (see ``DebouncePolicy.on_typing``).
        ``active=False`` — stopped typing, cleared the box, or pressed send — is
        recorded but **never** refreshes the window, otherwise the turn would
        never be released once the user walked away.
        """
        active = bool(active)
        self._typing = active

        refresh = False
        try:
            refresh = bool(self._policy.on_typing(active, self._build_context()))
        except Exception:
            logger.exception("[debouncer] method %r on_typing failed", self.method_name)

        if active and refresh:
            self._last_event_ts = time.monotonic()

        self._refresh_decision()
        self._sync_typing_flag()

    def notify_think_completed(self) -> None:
        """The client finished generating — switch to the after-think window."""
        # NOTE: `_timed_out` is deliberately NOT cleared here. It is cleared when
        # the next burst starts (`add`) / on reset, so the sidebar keeps showing
        # "(DEBOUNCE TIMEOUT)" for the whole forced turn (generation + playback)
        # instead of blinking for the duration of the API call only.
        if not self._pending or self._closed:
            return
        self._after_think = True
        self._last_event_ts = time.monotonic()
        self._refresh_decision()

    def clear_pending(self) -> None:
        """Drop queued messages without flushing them.

        Used when the host resets the session: a burst still inside the
        debounce window must not fire a turn into the freshly-reset
        conversation. Equivalent to ending the cycle with an empty batch.
        """
        self._after_think = False
        self._cap_start_ts = None
        self._timed_out = False
        if self._pending:
            self._pending = []
            self._set_typing(False)
            self._safe_hook("on_cycle_end", self._build_context())
            self._policy.reset()
        self._sync_pending_flag()

    # ── Status ──────────────────────────────────────────────────────

    def status_info(self) -> dict[str, float | int | bool | str]:
        """Status payload (method, typing state, polarity, remaining wait)."""
        status: dict[str, float | int | bool | str] = {
            "debounce_method": self.method_name,
            "debounce_method_requested": self._method_requested,
            "debounce_method_known": not self.method_mismatch,
            "debounce_typing": self._typing,
            "debounce_emotion": self._last_polarity,
            # Total-collection cap (0 = disabled) + whether it last fired.
            "debounce_max_wait_seconds": round(self._max_wait, 1),
            "debounce_timeout": self._timed_out,
            **self._policy_status(),
        }

        if not self._pending:
            status.update({
                "debounce_pending": False,
                "debounce_pending_count": 0,
                "debounce_wait_seconds": 0.0,
                "debounce_remaining": 0.0,
                "debounce_elapsed": 0.0,
                "debounce_cap_remaining": 0.0,
                "debounce_reason": "",
            })
            return status

        wait = self._current_wait()
        remaining = max(0.0, wait - (time.monotonic() - self._last_event_ts))
        elapsed = self._cap_elapsed()
        cap_remaining = max(0.0, self._max_wait - elapsed) if self._max_wait > 0 else 0.0
        status.update({
            "debounce_pending": True,
            "debounce_pending_count": len(self._pending),
            "debounce_wait_seconds": round(wait, 1),
            "debounce_remaining": round(remaining, 1),
            "debounce_elapsed": round(elapsed, 1),
            "debounce_cap_remaining": round(cap_remaining, 1),
            "debounce_reason": self._current_result.reason,
        })
        return status

    async def flush(self) -> None:
        """Hand the pending batch to the flush callback as a single turn."""
        self._after_think = False
        self._set_typing(False)
        if not self._pending or self._flush_cb is None:
            self._sync_pending_flag()
            return

        batch = list(self._pending)
        self._pending = []
        try:
            await self._flush_cb(batch)
        except Exception:
            # The batch left the queue, so a failure here loses those
            # messages — make that loud rather than silently dropping them.
            logger.exception(
                "[debouncer] flush callback failed — %d pending message(s) dropped: %r",
                len(batch),
                batch,
            )
        finally:
            self._safe_hook("on_cycle_end", self._build_context())
            self._sync_pending_flag()

    # ── Internals ───────────────────────────────────────────────────

    def _build_context(self) -> PolicyContext:
        """Snapshot the state a policy is allowed to see."""
        now = time.monotonic()
        emotion = self._get_current_emotion()
        reading = classify_emotion(emotion)
        self._last_polarity = reading.polarity
        return PolicyContext(
            pending_messages=tuple(self._pending),
            emotion=emotion,
            emotion_reading=reading,
            is_thinking=bool(self._is_thinking()),
            after_think=self._after_think,
            typing=self._typing,
            idle_seconds=max(0.0, now - self._last_event_ts),
            now=now,
        )

    def _refresh_decision(self, *, fire_actions: bool = False) -> DebouncePolicyResult:
        """Re-run the method and cache its decision.

        Called on every tick, message and typing signal, so it stays side
        effect free unless *fire_actions* is set — immediate fillers are only
        emitted once per queued message, never once per tick.
        """
        try:
            result = self._policy.evaluate(self._build_context())
        except Exception:
            logger.exception(
                "[debouncer] method %r failed to evaluate — using safe fallback",
                self.method_name,
            )
            result = self._fallback_result()

        if not isinstance(result, DebouncePolicyResult):
            logger.error(
                "[debouncer] method %r returned %s instead of DebouncePolicyResult — using safe fallback",
                self.method_name,
                type(result).__name__,
            )
            result = self._fallback_result()

        self._current_result = result
        if fire_actions:
            self._handle_immediate_action(result)
        return result

    def _fallback_result(self) -> DebouncePolicyResult:
        """Degrade to a short, safe window if a method misbehaves."""
        return DebouncePolicyResult(
            wait_seconds=1.0,
            after_think_seconds=1.0,
            reason="policy-error-fallback",
        )

    def _safe_hook(self, name: str, *args: Any) -> None:
        """Run an optional policy hook, never letting a method break the loop."""
        hook = getattr(self._policy, name, None)
        if hook is None:
            return
        try:
            hook(*args)
        except Exception:
            logger.exception("[debouncer] method %r %s failed", self.method_name, name)

    def _policy_status(self) -> dict[str, Any]:
        """Extra status fields contributed by the active method."""
        try:
            extra = self._policy.status()
        except Exception:
            logger.exception("[debouncer] method %r status failed", self.method_name)
            return {}
        return dict(extra) if extra else {}

    def _current_wait(self) -> float:
        """Effective window: the after-think variant when applicable."""
        result = self._current_result
        raw = result.after_think_seconds if self._after_think else result.wait_seconds
        try:
            wait = float(raw)
        except (TypeError, ValueError):
            wait = 1.0
        return max(_MIN_WAIT, wait)

    def _cap_elapsed(self) -> float:
        """Seconds since the LAST new message (0 when nothing is queued).

        Typing refreshes move ``_last_event_ts`` only, so they never reset this
        — that is what makes the cap a hard upper bound.
        """
        if self._cap_start_ts is None:
            return 0.0
        return max(0.0, time.monotonic() - self._cap_start_ts)

    def _cap_reached(self) -> bool:
        """True once the cap has elapsed since the last message (typing ignored)."""
        if self._max_wait <= 0 or self._cap_start_ts is None or not self._pending:
            return False
        return self._cap_elapsed() >= self._max_wait

    def _handle_immediate_action(self, result: DebouncePolicyResult) -> None:
        """Run a method's ``micro_response`` / ``silence`` decision, if any."""
        if not result.has_immediate_action or self._on_immediate_action is None:
            return
        try:
            outcome = self._on_immediate_action(
                result.immediate_action,
                result.micro_response_text,
            )
        except Exception:
            logger.exception("[debouncer] immediate action %r failed", result.immediate_action)
            return

        if asyncio.iscoroutine(outcome):
            try:
                task = asyncio.ensure_future(outcome)
            except RuntimeError:
                # No running loop (sync caller) — never leak the coroutine.
                outcome.close()
                logger.debug("[debouncer] no running loop for immediate action %r", result.immediate_action)
                return
            # asyncio holds only weak references, so keep a strong one.
            self._action_tasks.add(task)
            task.add_done_callback(self._action_tasks.discard)

    def _set_typing(self, value: bool) -> None:
        if self._typing == value:
            return
        self._typing = value
        self._sync_typing_flag()

    def _sync_pending_flag(self) -> None:
        if self._set_input_pending is not None:
            self._set_input_pending(bool(self._pending))

    def _sync_typing_flag(self) -> None:
        """Tell the host whether to stand its idle timer down while typing.

        Only methods that ask for it (`suppresses_idle_timer`) opt into this.
        Every other method — and the default — leaves the host's idle /
        adaptive-wait clock completely untouched by typing, so the client keeps
        its own schedule while the user composes.
        """
        if self._set_typing_active is None:
            return
        suppress = False
        try:
            suppress = bool(self._policy.suppresses_idle_timer())
        except Exception:
            logger.exception(
                "[debouncer] method %r suppresses_idle_timer failed", self.method_name
            )
        self._set_typing_active(bool(self._typing and suppress))

    async def _run(self) -> None:
        """Poll loop: flush once the active method's window has elapsed."""
        while not self._closed:
            if not self._pending:
                await asyncio.sleep(0.1)
                continue

            if self._is_thinking():
                # The client is still generating — keep collecting.
                await asyncio.sleep(0.2)
                continue

            # Hard total-wait cap: however many times the window was refreshed
            # (new messages / typing), once the burst has been open this long
            # the batch is released regardless.
            if self._cap_reached():
                self._timed_out = True
                logger.info(
                    "[debouncer] max_wait_seconds=%.1fs reached — flushing %d message(s)",
                    self._max_wait,
                    len(self._pending),
                )
                await self.flush()
                self._refresh_decision()
                continue

            wait = self._current_wait()
            elapsed = time.monotonic() - self._last_event_ts

            if elapsed >= wait:
                await self.flush()
            else:
                await asyncio.sleep(min(0.1, max(0.0, wait - elapsed)))

            # Re-evaluate every tick so methods that depend on live state
            # (emotion, typing, after-think) always see fresh input.
            self._refresh_decision()