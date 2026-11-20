"""
Action schema for CogniLLM response generation.

This mirrors the tier structure from the design discussion:
    Tier 0 (rare, threshold-gated): Leave
    Tier 1 (per-tick, the actual policy):      Silence | Behaviour | Speak | TypingReflection
    Tier 2 (only if Speak, two independent small choices):         Pacing x ContentMode

Deliberately zero imports from ai.py, cognillm.py, or the OpenAI SDK -
this file is just data + validation, so nothing importing it risks a
circular import, and it's trivially unit-testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier1Action(str, Enum):
    """
    The per-tick decision. Exactly one of these is "the action" for a
    given send_message() call.

    LEAVE is listed here as a value (not a separate type) purely so the
    "action" JSON field always has one consistent shape to parse on the
    consuming end. Its Tier-0-ness is about WHEN and HOW it's decided
    (a cheap threshold check on cumulative/turn-level state, run before
    Tier 1 and requiring no extra Azure call of its own) - not a
    different data type. See ActionPolicy.should_leave in policy.py,
    called from within ai.py's send_message right after call 1.
    """

    LEAVE = "leave"
    SILENCE = "silence"
    BEHAVIOUR = "behaviour"
    TYPING_REFLECTION = "typing_reflection"
    SPEAK = "speak"


class Pacing(str, Enum):
    LONG = "long"
    SHORT = "short"


class ContentMode(str, Enum):
    NEUTRAL = "neutral"
    QUESTION = "question"
    TOPIC_SHIFT = "topic_shift"


@dataclass
class Decision:
    """
    The full output of one policy evaluation. Only `tier1` is always
    populated - the rest are conditionally required, enforced in
    __post_init__ so a malformed Decision fails immediately at
    construction time instead of silently producing a bad "action" dict
    three function calls later.
    """

    tier1: Tier1Action
    pacing: Pacing | None = None
    content_mode: ContentMode | None = None
    behaviour_text: str | None = None
    leave_reason: str | None = None
    # TYPING_REFLECTION only: the abandoned drafts (in order) the client
    # typed then deleted before sending. PRIVATE - the browser must never
    # receive the text, only a count/length summary used to replay the
    # "…"→vanish→"…" hesitation (see session._redact_private).
    drafts: list[str] | None = None

    def __post_init__(self) -> None:
        if self.tier1 == Tier1Action.SPEAK and (self.pacing is None or self.content_mode is None):
            raise ValueError("SPEAK decisions require both pacing and content_mode (Tier 2)")

    def needs_generation_call(self) -> bool:
        """
        Whether this decision requires a second Azure call to produce
        content.

        SILENCE never does - doing nothing IS the output.
        BEHAVIOUR, TYPING_REFLECTION, SPEAK, and LEAVE all do - LEAVE
            included, since an in-character exit line lands far better
            than a cold system message. The *decision* to leave is still
            a cheap programmatic check (see ActionPolicy.should_leave) -
            only the parting line itself is generated.
        """
        return self.tier1 in (
            Tier1Action.BEHAVIOUR,
            Tier1Action.TYPING_REFLECTION,
            Tier1Action.SPEAK,
            Tier1Action.LEAVE,
        )

    def to_action_dict(
        self,
        message: str | None = None,
        behaviour: str | None = None,
        hidden: str | None = None,
        drafts: list[str] | None = None,
    ) -> dict:
        """
        Builds the exact dict that goes into the response JSON's
        "action" key. All arguments come from the second Azure call, when
        applicable (None/empty otherwise).

        SPEAK / LEAVE keep a ``message`` (the chat text the trainee reads).

        BEHAVIOUR deliberately has **no** ``message`` key: a behaviour is an
        action, not a chat bubble. Its observable cue lives in ``behaviour``
        (may be ``""`` when the whole action is private) and its private
        inner voice / unobservable action in ``hidden`` (never shown).

        TYPING_REFLECTION keeps a ``message`` (like SPEAK) plus a private
        ``drafts`` list - the earlier revisions the client typed, deleted and
        replaced. The UI shows only the hesitation ("…" appears, vanishes,
        appears again) and then the final ``message``; draft text is never
        rendered and is redacted before it reaches the browser.

        NOTE: this changes the existing stub shape from
        {"type": "response", "message": "..."} to {"type": "speak", ...}
        etc. - i.e. "response" becomes "speak" to match the tier name.
        Flagging this explicitly: if anything downstream (frontend,
        logs) already matches on the literal string "response", this
        rename will break it silently. Easy to alias back if needed.
        """
        d: dict = {"type": self.tier1.value}

        if self.tier1 == Tier1Action.SPEAK:
            d["message"] = message or ""
            d["pacing"] = self.pacing.value
            d["content_mode"] = self.content_mode.value

        elif self.tier1 == Tier1Action.TYPING_REFLECTION:
            # Final text actually sent. May be "" when the last revision was
            # also deleted (typing indicator shown, nothing sent) - the
            # distinction from SILENCE is the visible edit cycles in drafts.
            d["message"] = message or ""
            # PRIVATE: never rendered; the UI uses only count/length.
            d["drafts"] = drafts or []

        elif self.tier1 == Tier1Action.BEHAVIOUR:
            # No "message" key on purpose: a behaviour is not a chat bubble.
            d["behaviour"] = behaviour or self.behaviour_text or ""
            # Always present (even when empty) so a consumer can tell "this
            # cue is entirely hidden" from "this action has no hidden part".
            d["hidden"] = hidden or ""

        elif self.tier1 == Tier1Action.LEAVE:
            d["message"] = message or ""
            d["reason"] = self.leave_reason or "unspecified"

        # SILENCE: type only, deliberately no "message" key. Adding
        # "message": "" would invite a frontend to render an empty chat
        # bubble: the absence of the key is the signal to render nothing.

        return d