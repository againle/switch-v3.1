"""
MVP action policy.


Two entry points, matching the two-phase check in the design doc:

    should_leave(...)  -> Tier 0. Called by ai.py, AFTER call 1, using
                          the same cognitive_state the Tier 1 decision
                          needs - emotions/automatic_thoughts only exist
                          post-call-1, so Leave can't be a free
                          pre-check. NOT stage-gated - decoupled
                          entirely from MI stage/stage_manager per
                          explicit request; stages aren't in scope here.

    decide(...)        -> Tier 1 (+ Tier 2 if SPEAK). Called by ai.py
                          right after should_leave returns False, same
                          cognitive_state.
"""

from __future__ import annotations

from .schema import ContentMode, Decision, Pacing, Tier1Action

TIME_TRIGGER_SENTINEL = "[[System: Time interruption.]]"  # unused for matching now, kept for reference only

#: ``action_intent.type`` values call 1 may commit to. When the model emits
#: one of these the policy honours it - on ANY turn, real input or idle
#: timeout (mirrors how ``leave_session`` is a model verdict consumed by
#: ``should_leave``); anything else - including a missing/malformed field -
#: keeps the original rule-based fallback, so an older prompt that never
#: emits ``action_intent`` behaves exactly as before.
_VALID_INTENTS: dict[str, Tier1Action] = {
    "speak": Tier1Action.SPEAK,
    "typing_reflection": Tier1Action.TYPING_REFLECTION,
    "silence": Tier1Action.SILENCE,
    "behaviour": Tier1Action.BEHAVIOUR,
}


class ActionPolicy:
    """
    One instance lives on the Client (ai.py) for the life of a session,
    which lets it hold small pieces of running state: a consecutive-
    idle-tick counter (Behaviour timing) and a cumulative-idle-seconds
    counter (Leave's silence backstop).

    `is_idle_trigger` and `idle_seconds` are passed in explicitly by
    the caller (ai.py, from cognillm.py, from Orchestrator's real
    wall-clock timing) rather than inferred from message content.
    """

    def __init__(
    self,
    behaviour_after_n_idle_ticks: int = 1,
    speak_after_n_idle_ticks: int = 3,
    leave_after_idle_seconds: float = 90.0,
) -> None:
        self.behaviour_after_n_idle_ticks = behaviour_after_n_idle_ticks
        self.speak_after_n_idle_ticks = speak_after_n_idle_ticks
        self.leave_after_idle_seconds = leave_after_idle_seconds
        self._consecutive_idle_ticks = 0
        self._cumulative_idle_seconds = 0.0

    def reset(self) -> None:
        """Clear the running turn counters (used when the session resets).

        ``_cumulative_idle_seconds`` is the Leave backstop, so leaving it
        populated after a reset would let the *old* conversation's silence end
        the new one on its first idle tick.
        """
        self._consecutive_idle_ticks = 0
        self._cumulative_idle_seconds = 0.0

    # ── Tier 0 ──────────────────────────────────────────────────────

    def should_leave(self, *, cognitive_state: dict) -> tuple[bool, str | None]:
        """
        PRIMARY signal: the model's own leave_session verdict

        BACKSTOP signal: cumulative silence. Kept rule-based on
        purpose - "no message for 90+ seconds" is a plain fact, not
        something requiring the model's judgment, so there's nothing
        gained by routing it through a prompt.

        Gracefully returns (False, None) if `leave_session` is absent
        from cognitive_state (i.e. the prompt hasn't been updated yet)
        rather than raising - see module docstring.
        """
        leave_assessment = cognitive_state.get("leave_session")
        if isinstance(leave_assessment, dict) and leave_assessment.get("should_leave"):
            reason = leave_assessment.get("reason") or "model judged the client would end the session here"
            return True, reason

        if self._cumulative_idle_seconds >= self.leave_after_idle_seconds:
            return True, f"{self._cumulative_idle_seconds:.0f}s of continued silence"

        return False, None

    # ── Tier 1 + Tier 2 ─────────────────────────────────────────────

    def decide(
        self,
        *,
        user_message: str,
        cognitive_state: dict,
        is_idle_trigger: bool = False,
        idle_seconds: float = 0.0,
    ) -> Decision:
        # Counters update first either way: `_cumulative_idle_seconds` feeds
        # Leave's silence backstop, and `_consecutive_idle_ticks` is the
        # fallback pacing used only when the model names no action.
        if is_idle_trigger:
            self._consecutive_idle_ticks += 1
            self._cumulative_idle_seconds += idle_seconds
        else:
            # Any real user message resets both counters - the trainee broke
            # the silence, so neither "ticks since last behaviour" nor
            # "cumulative silence" should carry over.
            self._consecutive_idle_ticks = 0
            self._cumulative_idle_seconds = 0.0

        # The model decides WHICH action to take - on REAL input AND on an idle
        # timeout. action_intent.type comes from call 1, the same model-verdict
        # pattern as leave_session, so the timeout case is no longer forced by
        # the tick rules below.
        intent = self._requested_action(cognitive_state)
        if intent is Tier1Action.TYPING_REFLECTION:
            return Decision(tier1=Tier1Action.TYPING_REFLECTION)
        if intent is Tier1Action.SILENCE:
            return Decision(tier1=Tier1Action.SILENCE)
        if intent is Tier1Action.BEHAVIOUR:
            return Decision(tier1=Tier1Action.BEHAVIOUR)
        if intent is Tier1Action.SPEAK:
            if is_idle_trigger:
                # The model chose to speak into the silence (double-text,
                # chase, vent, needle, ...), so this is a full turn: let Tier 2
                # read the character's state instead of forcing the old fixed
                # "short nudge", and restart the idle cycle for the fallback.
                self._consecutive_idle_ticks = 0
            pacing = self._pick_pacing(cognitive_state, user_message)
            content_mode = self._pick_content_mode(cognitive_state, user_message)
            return Decision(tier1=Tier1Action.SPEAK, pacing=pacing, content_mode=content_mode)

        # No usable action_intent (older prompt / malformed field): fall back to
        # the original deterministic rules so nothing regresses.
        if is_idle_trigger:
            if self._consecutive_idle_ticks >= self.speak_after_n_idle_ticks:
                self._consecutive_idle_ticks = 0
                return Decision(
                    tier1=Tier1Action.SPEAK,
                    pacing=Pacing.SHORT,
                    content_mode=ContentMode.QUESTION,
                )
            if self._consecutive_idle_ticks >= self.behaviour_after_n_idle_ticks:
                return Decision(tier1=Tier1Action.BEHAVIOUR)
            return Decision(tier1=Tier1Action.SILENCE)

        pacing = self._pick_pacing(cognitive_state, user_message)
        content_mode = self._pick_content_mode(cognitive_state, user_message)
        return Decision(tier1=Tier1Action.SPEAK, pacing=pacing, content_mode=content_mode)

    @staticmethod
    def _requested_action(cognitive_state: dict) -> Tier1Action | None:
        """The action the model asked for in call 1, or None.

        Reads ``action_intent`` -> ``{"type": "...", "reason": "..."}``.
        Lenient by design (a bad value is not an error): an unknown type or a
        non-dict field yields None and lets the rule-based fallback run.
        """
        intent = cognitive_state.get("action_intent")
        if not isinstance(intent, dict):
            return None
        kind = str(intent.get("type", "")).strip().lower()
        return _VALID_INTENTS.get(kind)

    # ── Tier 2 heuristics (placeholders - cheap, not learned) ────────

    def _pick_pacing(self, cognitive_state: dict, user_message: str) -> Pacing:
        """
        Short/clipped when openness reads low (mirrors Table I's
        pre-contemplation "give minimal or clipped replies") or when
        the user's own message is short. Both are string-matching over
        the openness field's free text, which is fragile - fine for
        MVP, but a real classifier over openness would be sturdier.
        """
        openness_text = str(cognitive_state.get("openness", "")).lower()
        low_openness_markers = ("extremely low", "very low", "resist")
        if any(marker in openness_text for marker in low_openness_markers):
            return Pacing.SHORT
        if len(user_message.split()) <= 6:
            return Pacing.SHORT
        return Pacing.LONG

    def _pick_content_mode(self, cognitive_state: dict, user_message: str) -> ContentMode:
        """
        Keyword-scan automatic_thoughts for defensiveness/denial
        markers and counter with a QUESTION; otherwise NEUTRAL.

        TOPIC_SHIFT is intentionally never returned yet - that's the
        design doc's "topic graph," which doesn't exist as a module
        yet (doc-1's step 4, deliberately after the base loop is
        validated).
        """
        thoughts = str(cognitive_state.get("automatic_thoughts", "")).lower()
        defensive_markers = ("deny", "refuse", "not my", "fine", "nothing's wrong", "overreact")
        if any(marker in thoughts for marker in defensive_markers):
            return ContentMode.QUESTION
        return ContentMode.NEUTRAL