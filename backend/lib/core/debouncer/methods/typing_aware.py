"""``typing_aware`` — wait, watch the user type, scale the wait by emotion.

The method behind the second requirement. Behaviour:

1. After the user sends a message the debouncer waits ``wait_seconds``.
2. While the user keeps **typing** in the frontend input box, every typing
   signal refreshes the window, so a long message can be composed without the
   client cutting in. A *stop typing* signal deliberately does **not** refresh
   it — otherwise walking away from the keyboard would freeze the turn
   forever. The measure is therefore "did the user type something new?", not
   "is the box non-empty?".
3. The window is scaled by the client's emotion, read from the ``emotions``
   list of its latest response and bucketed into four types (see
   ``..emotions``). Longest wait → shortest:

       ``negative`` (withdrawn, disengaged)  → longest wait: give them room,
                                                do not pile on.
       ``neutral``  (calm, reflective)       → long wait: they are thinking,
                                                not waiting for us.
       ``positive`` (happy, satisfied)       → medium wait: engaged and
                                                patient, but keen.
       ``anxious``  (nervous, irritated)     → shortest wait: wound up and
                                                eager to be heard now.
4. Messages that arrived while the client was already generating use
   ``after_think_seconds`` instead, like every other method.
5. **Activated-client interruption.** A wound-up client does not always wait
   politely: with ``interrupt_typing_probability`` a turn is declared
   *interrupting*, and for that turn typing signals stop refreshing the window
   — the client has noticed the trainee is typing and answers anyway. The roll
   happens once per turn (:meth:`on_cycle_start`), only for the polarities
   listed in ``interrupt_polarities``.

Two clocks are deliberately kept apart:

* the **debounce** clock — how long to keep collecting messages. Typing
  refreshes it (that is the whole point of this method), and it is only held
  while messages are actually queued.
* the **idle / adaptive-wait** clock — how long the client has been left in
  silence (``Orchestrator.time_threshold``, driven by AdaWait). Typing does
  **not** touch it by default, so the interval keeps running while the trainee
  types; the client speaks on its own schedule. Set
  ``suppress_idle_timer_while_typing: true`` to hold it instead.

Settings (``debounce.methods.typing_aware``):
``wait_seconds``, ``negative_wait_seconds``, ``neutral_wait_seconds``,
``positive_wait_seconds``, ``anxious_wait_seconds``, ``after_think_seconds``,
``typing_refresh``, ``typing_grace_seconds``, ``emotion_strategy``,
``interrupt_typing_probability``, ``interrupt_polarities``,
``suppress_idle_timer_while_typing``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from ..emotions import EmotionPolarity
from ..registry import register_policy
from ..schema import DebouncePolicy, DebouncePolicyResult, PolicyContext

#: Which setting holds the wait for each emotion bucket.
_WAIT_KEYS: dict[EmotionPolarity, str] = {
    "negative": "negative_wait_seconds",
    "anxious": "anxious_wait_seconds",
    "positive": "positive_wait_seconds",
    "neutral": "neutral_wait_seconds",
}

#: Shortest wait we ever hand the debouncer, so a config typo cannot turn
#: into a busy flush loop.
_MIN_WAIT = 0.05


@register_policy("typing_aware")
class TypingAwarePolicy(DebouncePolicy):
    """Fixed window + typing refresh + emotion-scaled waits."""

    #: Typing only touches the *debounce* clock (how long to keep collecting),
    #: never the host's idle / adaptive-wait clock: the client keeps its own
    #: fixed schedule and may speak while the trainee is still composing.
    #: Set ``suppress_idle_timer_while_typing: true`` to hold the idle timer
    #: instead.
    suppresses_idle_timer_while_typing = False

    DEFAULTS: Mapping[str, Any] = {
        # Base window, also the fallback for an unmapped polarity.
        "wait_seconds": 4.0,
        # ── The four emotion buckets, longest wait → shortest ─────────
        # (1) negative: withdrawn / disengaged — give them room.
        "negative_wait_seconds": 8.0,
        # (4) neutral: calm and thinking it over — long, but not endless.
        "neutral_wait_seconds": 6.0,
        # (3) positive: engaged and patient, though keen — medium.
        "positive_wait_seconds": 4.0,
        # (2) anxious: wound up and waiting eagerly — answer quickly.
        "anxious_wait_seconds": 1.5,
        "after_think_seconds": 5.0,
        # A new keystroke refreshes the window; "stopped typing" never does.
        "typing_refresh": True,
        # Optional extra floor while typing is live (0 = refresh-only, i.e.
        # exactly the "refresh the timer" behaviour).
        "typing_grace_seconds": 0.0,
        # "any" (default) | "majority" — see emotions.classify_emotion.
        "emotion_strategy": "any",
        # ── Activated-client interruption ───────────────────────────
        # Chance that a turn becomes "interrupting": the client notices the
        # trainee is typing and answers anyway. 0.0 = never interrupt (pure
        # typing-aware waiting), 1.0 = always interrupt for those polarities.
        "interrupt_typing_probability": 0.0,
        # Which emotion readings may produce an interrupting turn. An anxious
        # client is eager to be heard, so it cuts in as readily as a
        # distressed withdrawer.
        "interrupt_polarities": ["negative", "anxious"],
        # ── Idle / adaptive-wait timer ──────────────────────────
        # False (default): typing does NOT reset or pause the client's
        # silence timer — it keeps ticking on its own interval. True: hold
        # the timer (and restart its clock) while the user is typing.
        "suppress_idle_timer_while_typing": False,
    }

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        super().__init__(settings)
        # Per-turn: rolled once in on_cycle_start, cleared in on_cycle_end.
        self._interrupting: bool = False

    # ── Per-turn lifecycle ──────────────────────────────────────────

    def on_cycle_start(self, ctx: PolicyContext) -> None:
        """Decide, once per client turn, whether this client will interrupt.

        Deliberately *not* done in :meth:`evaluate`: that runs on every tick,
        so a draw there would be re-rolled several times a second and the
        turn would flip between interrupting and not. One draw per turn also
        matches what it models — a decision the client makes once, then acts
        on, rather than a fresh impulse on every keystroke.
        """
        self._interrupting = False

        chance = self.number("interrupt_typing_probability", 0.0)
        if chance <= 0.0:
            return

        # The reading at cycle start is the client's state going into this
        # turn (its emotions from the previous response).
        if self._polarity(ctx) not in self._interrupt_polarities():
            return

        if chance >= 1.0 or random.random() < chance:
            self._interrupting = True

    def on_cycle_end(self, ctx: PolicyContext) -> None:
        self._interrupting = False

    def reset(self) -> None:
        self._interrupting = False

    # ── Port hooks ──────────────────────────────────────────────────

    def evaluate(self, ctx: PolicyContext) -> DebouncePolicyResult:
        after_think = self.number("after_think_seconds", 5.0)

        if not ctx.pending_messages:
            return DebouncePolicyResult(
                wait_seconds=self._wait_for("neutral"),
                after_think_seconds=after_think,
                reason="idle",
            )

        polarity = self._polarity(ctx)
        wait = self._wait_for(polarity)
        reason = f"polarity={polarity}"
        if self._interrupting:
            reason += " interrupt"

        grace = self.number("typing_grace_seconds", 0.0)
        if ctx.typing and grace > wait and not self._interrupting:
            # Keep a floor open while the user is visibly composing — but not
            # on an interrupting turn, where the client is not going to wait
            # for them to finish anyway.
            wait = grace
            reason += "+typing-grace"

        if ctx.after_think:
            reason += " after-think"

        return DebouncePolicyResult(
            wait_seconds=wait,
            after_think_seconds=after_think,
            reason=reason,
        )

    def on_typing(self, active: bool, ctx: PolicyContext) -> bool:
        """Refresh the window only on *new* typed content.

        ``active=False`` (the user stopped typing, cleared the box, or sent the
        message) must never refresh — otherwise the turn would never be
        released.

        On an interrupting turn (see :meth:`on_cycle_start`) ``active=True``
        does not refresh either: that is the interruption — the client can see
        the trainee typing and speaks into the pause anyway.
        """
        if not active:
            return False
        if self._interrupting:
            return False
        return self.flag("typing_refresh", True)

    def status(self) -> Mapping[str, Any]:
        return {"debounce_interrupting": self._interrupting}

    # ── Internals ───────────────────────────────────────────────────

    def _interrupt_polarities(self) -> frozenset[str]:
        raw = self.setting("interrupt_polarities", ["negative", "anxious"])
        if isinstance(raw, str):
            parts = raw.replace(";", ",").split(",")
        elif isinstance(raw, (list, tuple, set, frozenset)):
            parts = list(raw)
        else:
            parts = []
        values = {str(part).strip().lower() for part in parts if str(part).strip()}
        return frozenset(values) or frozenset({"negative", "anxious"})

    def _polarity(self, ctx: PolicyContext) -> EmotionPolarity:
        strategy = str(self.setting("emotion_strategy", "any")).strip().lower()
        if strategy not in ("any", "majority"):
            strategy = "any"
        if strategy == "any":
            # The debouncer already classified with the default strategy.
            return ctx.emotion_reading.polarity
        return self.reading(ctx, strategy=strategy).polarity

    def _wait_for(self, polarity: EmotionPolarity) -> float:
        """Configured wait for *polarity*, with a sane fallback chain."""
        base = self.number("wait_seconds", 4.0)
        neutral = self.number("neutral_wait_seconds", base)
        raw = self.settings.get(_WAIT_KEYS.get(polarity, "neutral_wait_seconds"))
        if raw is None:
            # An unmapped polarity behaves like neutral.
            raw = neutral
        try:
            wait = float(raw)
        except (TypeError, ValueError):
            wait = neutral
        return max(_MIN_WAIT, wait)


__all__ = ["TypingAwarePolicy"]
