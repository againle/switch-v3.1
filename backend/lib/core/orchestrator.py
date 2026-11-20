"""
Orchestrator for CogniLLM deployment with dual triggering modes:

(1) User input → send_message(EVAL=False)
(2) Time interval → if no user message for threshold duration,
    sends a message describing elapsed silence to CogniLLM

Thread-safe: when a response is being generated and a new user message
arrives, the message waits for the current call to finish before
processing.

Interrupted generations: a reply that new trainee input overtook is
*withheld* rather than delivered — answering a question the trainee has
already moved past reads worse than a brief silence. The pending input is
then processed as its own turn, and that turn carries a system note
(see `_format_interruption_message`) saying the client was cut off
mid-reply, so it can react to the interruption instead of silently
absorbing the extra message. The withheld turn still happened, so it is
recorded in the transcript with `interrupted: true`.

Timer interval is adaptive (AdaWaitController) - `_time_threshold` gets
recomputed after every response based on the client's emotional state
and the trainee's typing velocity, rather than staying fixed.

Idle-tick detection does NOT rely on string-matching message content -
`is_idle_trigger` (bool) and `idle_seconds` (float) are computed here,
where the real wall-clock timing lives, and passed explicitly down
through cognillm.py -> ai.py -> policy.py as parameters.

Leave handling: when CogniLLM's response has action.type == "leave"
(see ActionPolicy.should_leave in lib/action/policy.py), the orchestrator:
  - stops the timer loop (no more idle ticks prompting a client who left)
  - refuses further handle_user_input() calls
  - exposes `ended_by_leave` so a consumer (TUI, HTML page, whatever) can
    show something different from a plain "/quit"

Debounce: the orchestrator owns a MessageDebouncer (see lib/core/debouncer).
Consumers call `add_user_input()` instead of wiring their own debouncer —
rapid user messages are batched into a single CogniLLM turn, the idle timer
is suppressed while input is pending, and the after-think wait is driven
internally from `_mark_idle`. The turn-taking *method* (how long to wait,
whether to emit a filler) is pluggable and chosen by name in
config.yaml (`debounce.method`); `notify_typing()` feeds frontend typing
signals to whichever method is active.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .adawait.config import AdaWaitPreset
from .adawait.controller import AdaWaitController
from .cognillm import CogniLLM
from .debouncer import MessageDebouncer

# ── Types ─────────────────────────────────────────────────────────
# A parsed CogniLLM response dict.  Top-level keys include
# automatic_thoughts, emotions, behaviors, openness, chain_of_thought,
# plus an ``action`` sub-dict: {"type": "speak"/"silence"/"behaviour"/
# "leave", "message": "...", ...}.
ResponseDict = dict[str, Any]


def _format_idle_message(idle_seconds: float) -> str:
    """
    What gets sent to CogniLLM when the trainee has gone quiet for
    `time_threshold` seconds. Deliberately NOT framed as an
    interruption - that reads to the model like an external event (a
    phone ringing, someone walking in), not what's actually happening:
    the trainee is just taking a while to respond. Framing it as
    elapsed silence keeps call 1 from inventing a disruption that
    didn't occur.
    """
    seconds = max(0, round(idle_seconds))
    return f"[[System: {seconds} seconds have passed in silence. The trainee has not said anything.]]"


def _format_interruption_message() -> str:
    """
    System note for a reply that new trainee input cut off.

    Mirror of :func:`_format_idle_message`: an explicit system marker placed
    above the trainee's own words, so the client can *react* to being
    interrupted (annoyance, urgency, relief, ...) instead of silently
    absorbing an extra message it never got to answer.

    Prepended to the trainee's message when the previous turn's reply had to
    be withheld because this input arrived while it was still being
    generated.
    """
    return (
        "[[System: The trainee kept typing while you were still composing your "
        "previous reply. That reply was never delivered to them - they have "
        "not read or heard it. Your turn was interrupted. Respond to the "
        "message below now, and react to the interruption in character.]]"
    )


# ── Config ────────────────────────────────────────────────────────


@dataclass
class CogniLLMConfig:
    """Configuration schema for creating a CogniLLM instance.

    This is the single config object the Orchestrator needs — it
    constructs CogniLLM internally, so consumers never touch the
    CogniLLM class directly.
    """

    # ── Required ──────────────────────────────────────────────────
    profile_data: dict
    endpoint: str
    deployment: str
    api_key: str

    # ── Optional with sensible defaults ───────────────────────────
    api_version: str = "2024-08-01-preview"
    temperature: float = 0.5
    max_tokens: int = 1000
    summary_enabled: bool = True
    summary_window_size: int = 5
    summary_model: str = "openai"
    summary_start_round: int = 50
    skill_gate_threshold: float = 6.8

    # ── CONTEXT component ─────────────────────────────────────────
    # How the conversation context is supplied:
    #   "summary" — accumulate + summarize older windows (uses
    #               `summary_*` above; toggle with `summary_enabled`).
    #   "memory"  — CONTEXT is read on every turn from the standalone
    #               `neo4j-db` graph through `memory_source`, a direct
    #               function call (no HTTP hop). The summary method is
    #               bypassed, and each finished turn is written back to the
    #               graph by a background task.
    context_type: str = "summary"
    # The context manager that owns the graph. Duck-typed: it only needs
    # `fetch_memory_context(...)` and `record_memory_turn(...)`.
    memory_source: Any = None
    # memgraph profile name (the source maps it to a graph persona) and the
    # graph session id this conversation is stored under.
    memory_profile_name: str = ""
    memory_session_id: str = ""

    # ── MI stage (Motivational Interviewing) ──────────────────────
    # `mi_stage_enabled` is the master switch; `inject_stage_info` controls
    # whether STAGE_INFO is injected into each user prompt;
    # `evaluate_stage_progression` (only used while stage_info is injected)
    # controls whether the stage is evaluated/advanced.
    mi_stage_enabled: bool = True
    inject_stage_info: bool = True
    evaluate_stage_progression: bool = True
    # Pinned stage injected while inject_stage_info=True and
    # evaluate_stage_progression=False ("" = current stage, no pinning).
    injected_stage: str = ""

    # ── Debounce (user-message batching) ─────────────────────────
    # The orchestrator owns a MessageDebouncer when enabled: rapid user
    # messages are batched into a single CogniLLM turn so the client
    # "thinks" once instead of once per message.
    # `debounce_method` selects the pluggable turn-taking method by name
    # ("fixed_rules" = the original rule set, "typing_aware" = typing- and
    # emotion-aware, ...) and `debounce_method_settings` carries that
    # method's block from config.yaml. See lib/core/debouncer/README.md.
    debounce_enabled: bool = True
    debounce_method: str = "fixed_rules"
    debounce_method_settings: dict[str, Any] = field(default_factory=dict)
    # Hard ceiling (seconds) on the TOTAL collection time for one burst,
    # shared by every method. 0 = no cap.
    debounce_max_wait_seconds: float = 30.0
    # Optional ``(method_name) -> settings`` resolver. Lets an unknown
    # method fall back to a method that is configured properly instead of
    # inheriting the settings block meant for the missing one.
    debounce_settings_for: Callable[[str], dict[str, Any]] | None = None


class Orchestrator:
    """
    Orchestrates a CogniLLM instance with dual triggering:

    * **User-driven**: calls ``send_message(user_msg, EVAL=False)``.
    * **Timer-driven**: after *time_threshold* seconds of silence,
      sends a message describing the elapsed silence. `time_threshold`
      itself is adaptive.

    All calls to ``send_message`` are serialised through an async lock so
    that a time-interrupt in flight is never interrupted; a concurrent
    user message simply queues behind it.

    Once the client leaves (``action.type == "leave"``), the timer loop
    stops and further ``handle_user_input`` calls are refused - see
    ``ended_by_leave``.
    """

    # ────────────────────────────────────────────────────────────────
    def __init__(
        self,
        config: CogniLLMConfig,
        rate: float = 0.5,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            config: A ``CogniLLMConfig`` with all LLM connection and
                profile settings (including the CONTEXT type).  The
                Orchestrator builds CogniLLM internally from this.
            rate: Adaptive-wait rate.  2.0 = very patient
                (long waits), 0.5 = moderate, 0.25 = eager.
            verbose: If True, print E/O/V debug scores from the
                adaptive wait controller.
        """
        self._cognillm: CogniLLM = CogniLLM(
            profile_data=config.profile_data,
            endpoint=config.endpoint,
            deployment=config.deployment,
            api_key=config.api_key,
            api_version=config.api_version,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            summary_enabled=config.summary_enabled,
            summary_window_size=config.summary_window_size,
            summary_model=config.summary_model,
            summary_start_round=config.summary_start_round,
            skill_gate_threshold=config.skill_gate_threshold,
            context_type=config.context_type,
            memory_source=config.memory_source,
            memory_profile_name=config.memory_profile_name,
            memory_session_id=config.memory_session_id,
            mi_stage_enabled=config.mi_stage_enabled,
            inject_stage_info=config.inject_stage_info,
            evaluate_stage_progression=config.evaluate_stage_progression,
            injected_stage=config.injected_stage,
        )

        # ── Adaptive wait controller ────────────────────────────────
        preset = AdaWaitPreset(rate=rate)
        adawait_cfg = preset.to_config(verbose=verbose)
        self._adawait: AdaWaitController = AdaWaitController(adawait_cfg)
        self._rate: float = rate

        # Velocity (typing speed) tracking
        self._last_user_msg_ts: float = 0.0
        self._last_user_msg_len: int = 0

        # Initial threshold — will be overridden by adaptive controller
        # after the first response. Kept separately so a session reset can
        # restore the exact starting point.
        self._initial_time_threshold: float = adawait_cfg.tau_max
        self._time_threshold: float = self._initial_time_threshold

        # ── async primitives ────────────────────────────────────────
        self._lock: asyncio.Lock = asyncio.Lock()
        self._running: bool = False
        self._timer_task: asyncio.Task[Any] | None = None

        # Track the last moment a response *completed* (not started).
        # The timer only counts idle time between completed calls.
        self._last_response_ts: float = time.monotonic()

        # Whether an API call is currently in flight.
        self._api_in_flight: bool = False

        # Set once the client's own action.type == "leave" fires.
        # Distinct from `_running` (which also goes False on a plain
        # /quit) so consumers can tell the two apart if they want to.
        self._ended_by_leave: bool = False

        # Pause / resume — stops timer ticks but still allows user input.
        self._paused: bool = False
        self._pause_start_ts: float = 0.0
        self._paused_age: float = 0.0

        # Debounce / input-pending support — while True the idle-timer
        # loop is suppressed so a pending (debounced) user message can't
        # trigger an idle "silence" prompt while we are still collecting
        # the user's burst of messages.
        self._input_pending: bool = False

        # Interrupted-generation support.
        # `_last_turn_interrupted` — the turn that just finished was overtaken
        # by new trainee input, so its reply is withheld from the UI.
        # `_interrupted_generation` — that has happened since the last flush,
        # so the *next* turn must carry the interruption system note (see
        # `_flush_debounced_batch`).
        self._last_turn_interrupted: bool = False
        self._interrupted_generation: bool = False
        # Same event, tracked for the memory write-back: `_pending_interruption_note`
        # is raised by the flush that consumes `_interrupted_generation`, and
        # `_turn_follows_interruption` carries it into the turn being generated,
        # so the graph can record that the previous reply never landed.
        self._pending_interruption_note: bool = False
        self._turn_follows_interruption: bool = False

        # Frontend typing signal — raised only by methods that opt in via
        # `suppress_idle_timer_while_typing`. While True the idle-timer loop
        # is suppressed. By default typing never sets this: the debounce
        # window reacts to typing, but the client's adaptive-wait interval
        # keeps running on its own schedule.
        self._typing_active: bool = False

        # Frontend chunking/display state — while True the idle-timer
        # loop is suppressed because the client is still "typing out" its
        # chunked reply, so the silence clock is not counted during the
        # display (otherwise the wait timer would pressure the user to
        # reply while the client is still talking).
        self._displaying: bool = False

        # Optional async callback invoked with every assistant response dict.
        self._on_response: Callable[..., Awaitable[None]] | None = None

        # Optional sync callback for non-async consumers.
        self._on_response_sync: Callable[[ResponseDict], None] | None = None

        # Optional lifecycle / diagnostic callbacks (purely for consumers).
        self._on_trigger: Callable[[str, str], None] | None = None
        self._on_error: Callable[[str], None] | None = None

        # ── CONTEXT ────────────────────────────────────────────────
        # `context_type` selects "summary" (accumulate + summarize old
        # windows) or "memory" (read from the standalone neo4j-db graph).
        # For "memory": the *read* happens in the generation layer
        # (CogniLLM → ContextManager → graph) on the turn's critical path; the
        # *write* of the finished turn is dispatched from here as a background
        # task (see `_schedule_memory_update`), so graph/Ollama latency never
        # delays the trainee's reply.
        self._memory_source: Any = config.memory_source
        self._memory_profile_name: str = config.memory_profile_name
        self._memory_session_id: str = config.memory_session_id
        # Strong references to in-flight write-backs (so the tasks are not
        # garbage-collected mid-flight); awaited on `stop()`.
        self._memory_tasks: set[asyncio.Task[Any]] = set()

        # ── Session transcript (Save-PDF report) ───────────────────
        # Structured record of every completed turn. Captured here rather
        # than rebuilt from the chat history because the assistant content
        # there is a JSON *string* with chain_of_thought stripped, and
        # because the Tier-0 leave verdict never enters the history at all.
        self._transcript: list[dict[str, Any]] = []
        self._session_started_at: str = datetime.now().isoformat(timespec="seconds")

        # ── Message debouncer ──────────────────────────────────────
        # Owned here (rather than by the WebSocket handler) so the
        # session / app layers never have to wire it up. Rapid user
        # messages are batched into a single CogniLLM turn, and the
        # debouncer's after-think wait is driven straight from
        # `_mark_idle` below. While input is pending the debouncer also
        # suppresses idle-timer ticks via `set_input_pending`, and — for
        # methods that opt in — while the trainee is typing via
        # `set_typing_active`. Which *method* decides the wait comes from
        # config.yaml (see lib/core/debouncer/README.md).

        # Last emotion reported by the client, tracked directly from each
        # parsed response so `last_emotion` (and therefore the debouncer's
        # policy) reflects the CURRENT emotional state. Reading it back
        # from the conversation history is unreliable — assistant content
        # is stored there as a JSON string, not a dict.
        self._last_emotion: Any = "neutral"
        self._debounce_method_requested: str = str(config.debounce_method or "").strip().lower()
        self._debounce_method: str = config.debounce_method
        self._debouncer: MessageDebouncer | None = None
        if config.debounce_enabled:
            self._debouncer = MessageDebouncer(
                method=config.debounce_method,
                settings=config.debounce_method_settings,
                settings_for=config.debounce_settings_for,
                max_wait_seconds=config.debounce_max_wait_seconds,
                is_thinking=lambda: self._api_in_flight,
                set_input_pending=self.set_input_pending,
                set_typing_active=self.set_typing_active,
                get_current_emotion=lambda: self.last_emotion,
                on_immediate_action=self.handle_immediate_debounce_action,
            )
            self._debouncer.set_flush_callback(self._flush_debounced_batch)
            # An unknown method name degrades to the default, so report what
            # is actually live rather than what was asked for — and say so
            # loudly, because silently pacing with another method is a
            # confusing failure to debug.
            self._debounce_method = self._debouncer.method_name
            self._debounce_method_requested = self._debouncer.method_requested
            if self._debouncer.method_mismatch:
                print(
                    f"[orchestrator] WARNING: debounce.method={self._debounce_method_requested!r} "
                    f"is not a known method — using {self._debounce_method!r} instead. "
                    f"Fix backend/config.yaml (debounce.method).",
                    flush=True,
                )

    # ── Public API ──────────────────────────────────────────────────

    @property
    def time_threshold(self) -> float:
        """Seconds of silence before a time-interrupt fires."""
        return self._time_threshold

    @time_threshold.setter
    def time_threshold(self, value: float) -> None:
        if value <= 0:
            raise ValueError("time_threshold must be > 0")
        self._time_threshold = value

    @property
    def last_input_age(self) -> float:
        """Seconds elapsed since the last completed response.
        Returns 0 while an API call is in flight; frozen value when paused."""
        if self._api_in_flight:
            return 0.0
        if self._paused:
            return self._paused_age
        return time.monotonic() - self._last_response_ts

    @property
    def api_in_flight(self) -> bool:
        """True while an API call to CogniLLM is in progress."""
        return self._api_in_flight

    @property
    def adawait_state(self) -> dict[str, float | None]:
        """Current (E, O, V) state from the adaptive wait controller."""
        return self._adawait.state

    @property
    def transcript(self) -> list[dict[str, Any]]:
        """Structured record of every completed turn (for reporting)."""
        return self._transcript

    @property
    def session_started_at(self) -> str:
        """ISO timestamp of when this orchestrator (i.e. session) was created."""
        return self._session_started_at

    @property
    def ended_by_leave(self) -> bool:
        """True once the client has ended the session via a Leave action."""
        return self._ended_by_leave

    @property
    def is_paused(self) -> bool:
        """True when the timer loop is paused (no idle ticks)."""
        return self._paused

    def pause(self) -> None:
        """Pause the timer loop — no idle ticks fire, but user input still works."""
        if not self._paused:
            self._pause_start_ts = time.monotonic()
            self._paused_age = time.monotonic() - self._last_response_ts
        self._paused = True

    def resume(self) -> None:
        """Resume the timer loop — idle timer continues from where it was paused."""
        if self._paused:
            paused_duration = time.monotonic() - self._pause_start_ts
            self._last_response_ts += paused_duration
        self._paused = False

    def set_input_pending(self, value: bool) -> None:
        """Stop/allow idle-timer ticks while user input is pending.

        The message debouncer sets this to True while it is still
        collecting the user's burst of messages, so the idle timer can't
        fire a silence prompt in the middle of that window.  Setting it
        to True also restarts the silence clock (``_last_response_ts``),
        so the moment a new message arrives the "timeout" stops and the
        timer cannot fire before this input is processed.
        """
        value = bool(value)
        if value:
            # A new user message just arrived — restart the idle clock so
            # the timer loop can't fire a silence prompt while we are
            # waiting (debouncing) or about to start thinking.
            self._last_response_ts = time.monotonic()
        self._input_pending = value

    def set_typing_active(self, value: bool) -> None:
        """Pause/resume the idle timer while the trainee is typing (opt-in).

        Raised only by debounce methods that ask for it
        (``DebouncePolicy.suppresses_idle_timer()``), because typing touches
        two different clocks:

        * the debounce clock (how long to keep collecting messages) — typed
          content always refreshes that one, inside the debouncer;
        * the idle / adaptive-wait clock (``_last_response_ts`` vs
          ``time_threshold``) — *this* method, and only for methods that opt
          in. By default typing leaves it alone, so the interval keeps running
          and the client can speak while the trainee composes.
        """
        value = bool(value)
        if value:
            # Opted in: the trainee is composing, so restart the silence
            # clock and hold the timer until they stop.
            self._last_response_ts = time.monotonic()
        self._typing_active = value

    def set_displaying(self, value: bool) -> None:
        """Pause/resume the idle timer while the frontend types out a reply.

        The frontend calls this with True when it starts chunking a long
        reply and with False when the last chunk lands.  The silence
        clock restarts on both transitions, so idle time is never counted
        while the client is still "talking".
        """
        value = bool(value)
        # Restart the clock on both transitions — freeze it while
        # displaying, and restart fresh once display finishes.
        self._last_response_ts = time.monotonic()
        self._displaying = value

    @property
    def is_displaying(self) -> bool:
        """True while the frontend is typing out a chunked reply."""
        return self._displaying

    async def start(
        self,
        on_response: Callable[[ResponseDict], Awaitable[None]] | Callable[[ResponseDict], None] | None = None,
        on_trigger: Callable[[str, str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        """
        Start the orchestrator's background timer loop.

        Args:
            on_response: Optional callback invoked whenever the
                assistant produces a full response dict.
                May be sync ``(dict) -> None`` or async ``(dict) -> Awaitable[None]``.
            on_trigger: Optional callback ``(tag, message)`` fired when
                any trigger (user, timer, or leave) dispatches. `tag`
                is "user", "timer", or "leave".
            on_error: Optional callback ``(error_message)`` fired when
                ``send_message`` raises an exception, or when input
                arrives after the session has already ended.

        Callbacks are only overwritten when a non-None value is passed,
        so callbacks registered earlier (see ``register_callbacks``) are
        preserved. The message debouncer's after-think wait is driven
        internally — no ``on_idle`` wiring is needed.
        """
        self.register_callbacks(
            on_response=on_response,
            on_trigger=on_trigger,
            on_error=on_error,
        )

        self._running = True
        self._ended_by_leave = False  # allow session reuse if the same instance is start()-ed again
        self._last_response_ts = time.monotonic()
        self._api_in_flight = False
        if self._debouncer is not None:
            self._debouncer.start()
        self._timer_task = asyncio.create_task(self._timer_loop())

    def register_callbacks(
        self,
        on_response: Callable[..., Awaitable[None]] | None = None,
        on_trigger: Callable[[str, str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        """Register lifecycle callbacks without starting the timer loop.

        Unlike ``start()``, these take effect even before the session is
        running, so a Leave fired by a debounced message still reaches the
        consumer. Only non-None values overwrite the previously-registered
        callbacks.
        """
        if on_response is not None:
            if asyncio.iscoroutinefunction(on_response):
                self._on_response = on_response
                self._on_response_sync = None
            else:
                self._on_response_sync = on_response
                self._on_response = None
        if on_trigger is not None:
            self._on_trigger = on_trigger
        if on_error is not None:
            self._on_error = on_error

    def get_conversation_history(self) -> list[Any]:
        return self._cognillm.get_conversation_history()

    def set_context_type(self, context_type: str) -> None:
        """Switch the CONTEXT type at runtime ("summary" | "memory")."""
        self._cognillm.set_context_type(context_type)

    def reset_conversation(self) -> None:
        self._cognillm.reset_conversation()

    def reset_session(self) -> None:
        """Reset the conversation **and** the pacing state that depends on it.

        ``reset_conversation`` only clears the chat history. This is what the
        UI's Reset button calls: the idle / adaptive-wait clock, the AdaWait
        state, the typing-velocity sample, the client's emotion, the leave
        backstop and the debounce queue all belong to the conversation being
        discarded — leaving them running is why the wait timer kept counting
        down a deadline that belonged to the old conversation.

        The transcript is deliberately *not* cleared: it is the session log
        the Save-PDF report is built from, and a reset does not end the
        session.
        """
        self._cognillm.reset_conversation()

        # Restart the idle / adaptive-wait clock so the timer visibly starts
        # over instead of carrying the old age across the reset.
        now = time.monotonic()
        self._last_response_ts = now
        self._pause_start_ts = now
        self._paused_age = 0.0
        self._adawait.reset()
        self._time_threshold = self._initial_time_threshold

        # The velocity sample and the tracked emotion describe the old
        # conversation too.
        self._last_user_msg_ts = 0.0
        self._last_user_msg_len = 0
        self._last_emotion = "neutral"

        # A burst that is still being debounced must not fire a turn into the
        # freshly-reset history.
        if self._debouncer is not None:
            self._debouncer.clear_pending()

        # An interruption note only makes sense for the turn that queues next;
        # it must not leak into a fresh conversation.
        self._interrupted_generation = False
        self._last_turn_interrupted = False
        self._pending_interruption_note = False
        self._turn_follows_interruption = False

    async def stop(self) -> None:
        """Gracefully stop the orchestrator, timer loop and debouncer."""
        self._running = False
        if self._debouncer is not None:
            await self._debouncer.stop()
        if self._timer_task is not None:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
        # Let in-flight memory write-backs finish so no turn is left
        # half-written in the graph. They are off the critical path, so this
        # only delays session teardown, never a reply.
        if self._memory_tasks:
            await asyncio.gather(*list(self._memory_tasks), return_exceptions=True)

    async def handle_user_input(self, message: str) -> ResponseDict | None:
        """
        Process a single user message through CogniLLM (EVAL=False),
        bypassing the debouncer, and return the response to the caller.

        Serialised behind any in-flight time-interrupt.

        Refuses input once the client has left (``ended_by_leave``).

        Prefer ``add_user_input`` for normal chat — this is the direct,
        non-batched path used by commands and the debouncer's flush.
        """
        return await self._process_user_message(message, emit=False)

    async def _process_user_message(
        self,
        message: str,
        emit: bool = False,
    ) -> ResponseDict | None:
        """Core user-turn processing shared by every entry point.

        ``emit`` controls whether the resulting response is pushed
        through the response callback with ``emit=True``. That is used
        by the debouncer's flush and the non-debounced ``add_user_input``
        path, where nobody awaits the return value; ``handle_user_input``
        passes ``emit=False`` and returns the response to its caller.
        """
        if self._ended_by_leave:
            if self._on_error is not None:
                self._on_error("The client has ended this session - no further input is being processed.")
            return None

        now = time.monotonic()
        if self._last_user_msg_ts > 0:
            delta = now - self._last_user_msg_ts
            cps = len(message) / delta if delta > 0 else 0.0
            max_cps = self._adawait._config.V_max_cps
            V_raw = min(1.0, cps / max_cps) if max_cps > 0 else 0.5
        else:
            V_raw = None
        self._last_user_msg_ts = now
        self._last_user_msg_len = len(message)

        async with self._lock:
            # Re-check after acquiring the lock: a Leave may have fired
            # while this turn was queued behind an in-flight call.
            if self._ended_by_leave:
                return None

            # ── CogniLLM call ──────────────────────────────────
            self._api_in_flight = True
            response = await self._call_cognillm(
                message, EVAL=False, tag="user",
                is_idle_trigger=False, idle_seconds=0.0,
            )

            # ── Phase 3: Post-response ─────────────────────────
            self._last_response_ts = time.monotonic()
            await self._mark_idle()

            leave_message: str | None = None
            if isinstance(response, dict):
                action = response.get("action", {})
                if isinstance(action, dict) and action.get("type") == "leave":
                    self._ended_by_leave = True
                    self._running = False
                    leave_message = action.get("message", "Client ended the conversation.")
                # Track the client's latest emotional state — this is
                # what `last_emotion` (and the debouncer's policy) sees.
                if response.get("emotions"):
                    self._last_emotion = response["emotions"]

            if response:
                try:
                    self._time_threshold = self._adawait.compute_wait(response, V_raw=V_raw)
                except Exception:
                    pass
                # A reply that new input overtook is withheld from the UI (see
                # _reply_was_interrupted). It still ran, so the AdaWait/emotion
                # update above stands; the follow-up turn is what explains why
                # the client got cut off.
                deliver = not self._last_turn_interrupted
                await self._fire_callback(response, emit=emit and deliver)

            # Dispatch the Leave trigger after the response callback so
            # consumers (e.g. the app's socket-close handler) run only
            # once the session-end event has been pushed to the UI.
            if leave_message is not None:
                self._dispatch_trigger("leave", leave_message)

            # A withheld reply is not handed back to the caller either —
            # handle_user_input's caller would push it to the UI itself.
            if response is not None and self._last_turn_interrupted:
                return None
            return response

    async def _call_cognillm(
        self,
        message: str,
        EVAL: bool = False,
        tag: str = "user",
        is_idle_trigger: bool = False,
        idle_seconds: float = 0.0,
    ) -> ResponseDict | None:
        """Call CogniLLM.send_message in a thread executor.

        Does NOT toggle ``_api_in_flight`` — the caller owns that.
        """
        self._last_turn_interrupted = False
        # Claim the flag the flush raised for THIS turn (if any), so the memory
        # write-back can annotate the trainee utterance in the graph.
        self._turn_follows_interruption = self._pending_interruption_note
        self._pending_interruption_note = False
        self._dispatch_trigger(tag, message)

        # DEBUG — the trainee turn exactly as it is handed to the core
        # (CogniLLM): a normal message is the trainee's own words; on a timer
        # tick it is the elapsed-silence system prompt; when the previous
        # generation was interrupted it is the interruption note followed by
        # the trainee's messages. Delete this block to silence it.
        print(
            f"\n===== USER MESSAGE -> CORE "
            f"(tag={tag}{', idle-trigger' if is_idle_trigger else ''}) =====\n"
            f"{message}\n"
            f"===== end user message =====",
            flush=True,
        )

        loop = asyncio.get_running_loop()
        try:
            call = functools.partial(
                self._cognillm.send_message,
                message,
                EVAL,
                is_idle_trigger=is_idle_trigger,
                idle_seconds=idle_seconds,
            )
            response, tokens = await loop.run_in_executor(None, call)
        except Exception as exc:
            print(f"[orchestrator] CogniLLM call failed: {exc}", flush=True)
            if self._on_error is not None:
                self._on_error(str(exc))
            return None

        # Did the trainee speak while this turn was generating? Computed here
        # because both the emission decision (callers) and the report need it.
        self._last_turn_interrupted = self._reply_was_interrupted(response)

        # The CONTEXT block this turn was actually generated from (memory mode
        # only; None on summary/unreachable graph). Copied into the transcript
        # so the session report can show what the client was given.
        try:
            context_used = self._cognillm.get_last_context()
        except Exception:  # noqa: BLE001 — reporting must never break a turn
            context_used = None
        if self._last_turn_interrupted:
            print(
                "[orchestrator] reply withheld — trainee sent new input while it was generating",
                flush=True,
            )
            self._interrupted_generation = True

        self._record_turn(
            message=message,
            tag=tag,
            is_idle_trigger=is_idle_trigger,
            idle_seconds=idle_seconds,
            response=response,
            tokens=tokens,
            interrupted=self._last_turn_interrupted,
            context=context_used,
        )

        # Fold the finished turn into the memory graph. Dispatched here
        # (before returning) but executed as a background task, so it overlaps
        # with the debounce window / idle timer / next generation instead of
        # delaying this reply.
        self._schedule_memory_update(message, response, tag)
        return response

    def _reply_was_interrupted(self, response: dict[str, Any] | None) -> bool:
        """Whether *response* must be withheld because the trainee spoke again.

        A reply is withheld when new input arrived while it was being
        generated: answering a message the trainee has already moved past is
        worse than a brief silence, and the input itself deserves a reply that
        takes it into account.

        The signal is the debounce queue. The debouncer refuses to flush while
        a call is in flight (see ``MessageDebouncer._run``), so anything still
        queued when a call returns arrived *during* it.

        The one exception is a Leave: that is a terminal decision about the
        session and must not be undone by the trainee simply typing — the same
        rule ai.py applies when a generation call fails after a LEAVE.
        """
        if not isinstance(response, dict):
            return False
        action = response.get("action")
        if isinstance(action, dict) and action.get("type") == "leave":
            return False
        return self._debouncer is not None and self._debouncer.has_pending

    def _record_turn(
        self,
        *,
        message: str,
        tag: str,
        is_idle_trigger: bool,
        idle_seconds: float,
        response: dict[str, Any] | None,
        tokens: int | None,
        interrupted: bool = False,
        context: str | None = None,
    ) -> None:
        """Append one completed turn to the session transcript.

        The cognitive state and action come from *response*; the Tier-0
        leave verdict comes from the CogniLLM client, which is the only
        place that computes it. Nothing here is fed back to the model.

        *interrupted* marks a turn whose reply was withheld because the
        trainee sent new input while it was generating — it happened, so it
        is recorded, but the report should show it was never delivered.

        *context* is the CONTEXT block the turn was generated from (the
        ``memory`` context type); ``None`` when the session runs on
        ``summary`` or the graph was unreachable.
        """
        if not isinstance(response, dict):
            return

        meta: dict[str, Any] = {}
        try:
            meta = self._cognillm.get_last_turn_meta()
        except Exception:  # noqa: BLE001 — reporting must never break a turn
            meta = {}

        self._transcript.append({
            "index": len(self._transcript) + 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trigger": tag,
            "is_idle_trigger": bool(is_idle_trigger),
            "idle_seconds": round(float(idle_seconds or 0.0), 1),
            "trainee_message": message,
            "context": context,
            "interrupted": bool(interrupted),
            "cognitive_state": {k: v for k, v in response.items() if k != "action"},
            "leave_decision": {
                "should_leave": bool(meta.get("should_leave")),
                "reason": meta.get("leave_reason"),
            },
            "decision": {
                "tier1": meta.get("tier1"),
                "pacing": meta.get("pacing"),
                "content_mode": meta.get("content_mode"),
                "generation_call": meta.get("generation_call"),
            },
            "action": response.get("action"),
            "tokens": tokens,
        })

    # ── Memory CONTEXT write-back ───────────────────────────────

    def _schedule_memory_update(
        self,
        message: str,
        response: dict[str, Any] | None,
        tag: str,
    ) -> None:
        """Queue the finished turn for the graph, off the critical path.

        Only active while ``context_type`` is ``"memory"`` and a source is
        wired. The write runs in a background task on a thread executor, so it
        runs *in parallel with the orchestrator* — the trainee never waits on
        bolt or Ollama latency for the memory update.

        A turn is skipped when there is nothing worth remembering: a timer tick
        with no client output (the client stayed silent), or a turn with neither
        trainee words nor client output.

        A withheld (interrupted) reply is NOT skipped — it happened, and people
        recall things they never said — it is written with an in-band directive
        recording that the social worker never saw it.
        """
        if self._memory_source is None or not self._memory_session_id:
            return

        trainee_text = self._memory_trainee_text(message, tag)
        state = response if isinstance(response, dict) else None
        has_output = bool(state and self._memory_client_text(state).strip())

        if not trainee_text and not has_output:
            return

        task = asyncio.create_task(
            self._run_memory_update(
                trainee_text,
                state,
                interrupted=self._last_turn_interrupted,
                follows_interruption=self._turn_follows_interruption,
            )
        )
        # Strong reference so the task cannot be garbage-collected mid-flight;
        # removed once it settles.
        self._memory_tasks.add(task)
        task.add_done_callback(self._memory_tasks.discard)

    async def _run_memory_update(
        self,
        trainee_text: str,
        state: dict[str, Any] | None,
        interrupted: bool = False,
        follows_interruption: bool = False,
    ) -> None:
        """Execute one graph write-back in the executor. Never raises."""
        loop = asyncio.get_running_loop()
        call = functools.partial(
            self._memory_source.record_memory_turn,
            session_id=self._memory_session_id,
            profile_name=self._memory_profile_name,
            trainee_message=trainee_text,
            state=state,
            interrupted=interrupted,
            follows_interruption=follows_interruption,
        )
        try:
            await loop.run_in_executor(None, call)
        except Exception as exc:  # noqa: BLE001 — memory writes are best-effort
            print(f"[orchestrator] memory write-back failed: {exc}", flush=True)

    @staticmethod
    def _memory_client_text(state: dict[str, Any]) -> str:
        """The client's observable output of a turn ("" when it stayed silent)."""
        action = state.get("action")
        action = action if isinstance(action, dict) else {}
        return str(action.get("message") or action.get("behaviour") or "")

    @staticmethod
    def _memory_trainee_text(message: str, tag: str) -> str:
        """The trainee side of a turn exactly as it should be stored.

        Timer ticks carry no trainee words at all — the client spoke into
        silence — so they are stored with an empty trainee side, and the write
        path then skips belief scoring for them. An interrupted turn carries
        the ``[[System: …]]`` note prepended to the real message; that note is
        stripped so it is never embedded or belief-matched as if the trainee
        had said it.
        """
        if tag != "user":
            return ""
        text = message or ""
        if text.startswith("[[System:"):
            end = text.find("]]")
            if end != -1:
                text = text[end + 2:]
        return text.strip()

    def _dispatch_trigger(self, tag: str, message: str) -> None:
        """Fire the trigger callback, handling both sync and async handlers.

        Async handlers are scheduled as background tasks so an un-awaited
        coroutine is never leaked. Used for every trigger (user, timer,
        leave).
        """
        if self._on_trigger is None:
            return
        try:
            result = self._on_trigger(tag, message)
        except Exception:
            return
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)

    # ── Debouncer integration ───────────────────────────────────
    # The MessageDebouncer is owned here and wired to the orchestrator's
    # own state (api_in_flight, input-pending, emotion, immediate
    # actions), so consumers only see a simple "add input" entry point.

    @property
    def has_pending_input(self) -> bool:
        """True while the debouncer still holds unflushed user messages."""
        return bool(self._debouncer is not None and self._debouncer.has_pending)

    def debounce_status(self) -> dict[str, float | int | bool | str]:
        """Status payload for the debouncer (used by session status)."""
        if self._debouncer is None:
            return {
                "debounce_pending": False,
                "debounce_pending_count": 0,
                "debounce_wait_seconds": 0.0,
                "debounce_remaining": 0.0,
                "debounce_elapsed": 0.0,
                "debounce_cap_remaining": 0.0,
                "debounce_max_wait_seconds": 0.0,
                "debounce_timeout": False,
                "debounce_method": self._debounce_method,
                "debounce_method_requested": self._debounce_method_requested,
                "debounce_method_known": True,
                "debounce_typing": False,
                "debounce_emotion": "neutral",
                "debounce_reason": "",
            }
        return self._debouncer.status_info()

    def notify_typing(self, active: bool) -> None:
        """Forward a frontend typing signal to the active debounce method.

        ``active=True`` means the trainee just typed new content — that is
        what the ``typing_aware`` method uses to refresh its wait window.
        ``active=False`` (stopped typing, box cleared, message sent) is
        recorded but never refreshes the window, so a trainee who walks away
        mid-sentence cannot freeze the turn forever.
        """
        if self._debouncer is not None:
            self._debouncer.notify_typing(active)

    async def add_user_input(self, text: str) -> None:
        """Queue a user message through the debouncer.

        The message is batched with any recent input and flushed as a
        single CogniLLM turn once the debounce window elapses. The
        resulting response is pushed through the registered response
        callback (``emit=True``), so callers do not await a reply here.

        If debouncing is disabled, the message is processed immediately
        (also pushed through the response callback).
        """
        if self._debouncer is None:
            await self._process_user_message(text, emit=True)
            return
        # Idempotent — covers the case where a message arrives before
        # start() (e.g. a message arriving before the session starts).
        self._debouncer.start()
        self._debouncer.add(text)

    async def _flush_debounced_batch(self, batch: list[str]) -> None:
        """Process a debounced batch of user messages as a single turn.

        Called by the internal ``MessageDebouncer`` once its wait window
        elapses. Messages are combined into one user turn and pushed
        through the response callback (``emit=True``) so consumers that
        use ``add_user_input`` see the reply without awaiting it.
        """
        if self._ended_by_leave or not batch:
            return
        text = "\n".join(batch)
        if self._interrupted_generation:
            # The previous reply was withheld because this input arrived while
            # it was generating. Say so in-band, exactly like the silence
            # prompt, so the client can react to being interrupted rather than
            # silently absorbing the extra message. Consumed once, so the note
            # applies to the turn it actually describes.
            self._interrupted_generation = False
            # Tell the memory write-back that THIS trainee message is the one
            # that interrupted the previous reply.
            self._pending_interruption_note = True
            text = f"{_format_interruption_message()}\n{text}"
        await self._process_user_message(text, emit=True)

    @property
    def last_emotion(self) -> Any:
        """Latest emotion reported by the client.

        Prefers the emotion tracked directly from the last assistant
        response (``_last_emotion``), which is the reliable source. Falls
        back to parsing the most recent assistant message in the
        conversation history — its ``content`` is stored as a JSON string,
        not a dict, so it must be parsed.

        Note: debounce methods poll this on every typing signal and every
        tick, so this property must stay side-effect free — no logging.
        """
        if self._last_emotion not in (None, "", "neutral"):
            return self._last_emotion

        history = self._cognillm.get_conversation_history()
        if history:
            last_turn = history[-1]
            last_content = last_turn.get("content") if isinstance(last_turn, dict) else None
            if isinstance(last_content, dict):
                return last_content.get("emotions") or last_content.get("emotion") or "neutral"
            if isinstance(last_content, str):
                try:
                    import json as _json
                    data = _json.loads(last_content)
                except (ValueError, TypeError):
                    data = None
                if isinstance(data, dict):
                    return data.get("emotions") or data.get("emotion") or "neutral"
        return self._last_emotion if self._last_emotion is not None else "neutral"

    async def handle_immediate_debounce_action(
        self,
        action_type: str,
        micro_text: str | None = None,
    ) -> None:
        """Emits micro-responses or updates state when requested by DebouncePolicy."""
        if action_type == "micro_response" and micro_text:
            synthetic_response: ResponseDict = {
                "automatic_thoughts": "[Interim filler output while user continues typing...]",
                "emotions": self.last_emotion,
                "action": {
                    "type": "speak",
                    "message": micro_text,
                },
            }
            await self._fire_callback(synthetic_response, emit=True)
        elif action_type == "silence":
            # Optionally freeze or log silence tick without advancing full turn
            pass

    # ── Internals (timer path) ──────────────────────────────────────

    async def _timer_loop(self) -> None:
        """Background coroutine that fires time-interrupts on silence.

        The timer only counts idle time *between* completed API calls.
        While an API call is in flight ``last_input_age`` returns 0 so
        this loop sleeps without acting. Exits naturally (within one
        0.5s poll) once `_running` goes False - including when it goes
        False from a Leave fired by this very loop's own tick, see the
        comment in `_send_and_emit`.
        """
        while self._running:
            await asyncio.sleep(0.5)  # check every 500 ms

            if self._api_in_flight or self._paused or self._input_pending or self._displaying or self._typing_active:
                continue

            age = time.monotonic() - self._last_response_ts
            if age < self._time_threshold:
                continue

            # Threshold exceeded — attempt to claim the lock.
            async with self._lock:
                # Double-check *after* acquiring the lock.
                if not self._running or self._ended_by_leave:
                    break
                if self._api_in_flight:
                    continue
                if time.monotonic() - self._last_response_ts < self._time_threshold:
                    continue
                
                await self._send_and_emit(
                    _format_idle_message(age),
                    EVAL=False,
                    tag="timer",
                    V_raw=None,  # timeout — no user input, V stays frozen
                    is_idle_trigger=True,
                    idle_seconds=age,
                )

    async def _send_and_emit(
        self,
        message: str,
        EVAL: bool,
        tag: str,
        V_raw: float | None = None,
        is_idle_trigger: bool = False,
        idle_seconds: float = 0.0,
    ) -> ResponseDict | None:
        """
        Timer-path entry point.  For user messages prefer
        ``handle_user_input``.
        """
        # If a Leave fired while this tick was queued (waiting on the
        # lock), do not process another round.
        if self._ended_by_leave or not self._running:
            return None

        self._api_in_flight = True
        response = await self._call_cognillm(
            message, EVAL=EVAL, tag=tag,
            is_idle_trigger=is_idle_trigger,
            idle_seconds=idle_seconds,
        )
        self._last_response_ts = time.monotonic()
        await self._mark_idle()

        leave_message: str | None = None
        if isinstance(response, dict):
            action = response.get("action", {})
            if isinstance(action, dict) and action.get("type") == "leave":
                self._ended_by_leave = True
                self._running = False
                leave_message = action.get("message", "Client ended the conversation.")
            # Track the client's latest emotional state (timer-driven
            # responses carry the client's emotion too).
            if response.get("emotions"):
                self._last_emotion = response["emotions"]

        if response:
            try:
                self._time_threshold = self._adawait.compute_wait(response, V_raw=V_raw)
            except Exception:
                pass
            # Timer-driven replies are withheld on the same rule as user ones:
            # if the trainee spoke while the client was composing its silence
            # prompt, that prompt is obsolete.
            deliver = not self._last_turn_interrupted
            await self._fire_callback(response, emit=deliver)

        # Dispatch the Leave trigger after the response callback (see
        # _process_user_message for the rationale).
        if leave_message is not None:
            self._dispatch_trigger("leave", leave_message)
        return response

    async def _fire_callback(self, response: ResponseDict, emit: bool = False) -> None:
        """Invoke the registered response callback (sync or async).

        ``emit=True`` marks a response that came from a non-user path
        (the idle timer), which the session must push to the UI itself —
        user-initiated responses are emitted by ``handle_text`` instead.
        """
        if self._on_response is not None:
            await self._on_response(response, emit=emit)
        elif self._on_response_sync is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._on_response_sync, response)

    async def _mark_idle(self) -> None:
        """Mark an API call as finished and notify the debouncer.

        Sets ``_api_in_flight`` to False and lets the internal debouncer
        start its after-think wait. Fired whether or not a response dict
        was produced.
        """
        self._api_in_flight = False
        if self._debouncer is not None:
            self._debouncer.notify_think_completed()


