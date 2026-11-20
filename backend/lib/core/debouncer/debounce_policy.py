from __future__ import annotations

import random
from typing import Sequence

from .emotions import NEGATIVE_EMOTIONS, is_negative  # noqa: F401  (re-exported)
from .schema import DebouncePolicyResult

__all__ = [
    "ANNOYED_MICRO_RESPONSES",
    "NEUTRAL_MICRO_RESPONSES",
    "DebouncePolicyResult",
    "DynamicDebouncePolicy",
]

NEUTRAL_MICRO_RESPONSES = [
    ["Mm-hmm...", "OK...", "I see.", "Let me think...", "Interesting..."],
    ["mm", "uh", "hmm", "yeah", "oh"],
    ["K", "And?", "So?", "Yeah", "Hmm"]
]

ANNOYED_MICRO_RESPONSES = [
    ["Ugh...", "??", "Really?", "What?", "Hmm..."],
    ["...", "Huh?", "Then?", "Hmm...", "OK?"],
    ["Whatever..", "Huh?", "And?", "Hmm...", "OK?"]
]


class DynamicDebouncePolicy:
    """Analyzes incoming message sequences, cumulative text length, and current
    emotional state to dynamically determine debounce timing and micro-responses.

    This is the *engine* behind the ``fixed_rules`` method. It is kept as a
    standalone collaborator (rather than folded into the policy class) so the
    original rule behaviour stays byte-for-byte reproducible. New methods
    should implement :class:`~..schema.DebouncePolicy` directly.
    """

    def __init__(
        self,
        short_wait: float = 1.2,
        long_wait: float = 10.0,
        default_after_think: float = 1.0,
        cum_char_threshold: int = 120,
        debug: bool = True,
        micro_response_probability: float = 0.2,
        enable_micro_responses: bool = True,
    ) -> None:
        self.short_wait = short_wait
        self.long_wait = long_wait
        self.default_after_think = default_after_think
        self.cum_char_threshold = cum_char_threshold
        self.debug = debug
        self.micro_response_probability = min(1.0, max(0.0, micro_response_probability))
        self.enable_micro_responses = enable_micro_responses
        self.ANNOYED_MICRO = random.choice(ANNOYED_MICRO_RESPONSES)
        self.NEUTRAL_MICRO = random.choice(NEUTRAL_MICRO_RESPONSES)

    def __call__(
        self,
        pending_messages: Sequence[str],
        current_emotion: str | dict | None = None,
    ) -> DebouncePolicyResult:
        if not pending_messages:
            if self.debug:
                print("[DebouncePolicy] [DEBUG] Empty pending queue -> Returning short wait.")
            return DebouncePolicyResult(
                wait_seconds=self.short_wait,
                after_think_seconds=self.default_after_think,
            )

        last_msg = pending_messages[-1].strip()
        combined_text = " ".join(msg.strip() for msg in pending_messages)
        total_chars = len(combined_text)

        # Parse normalized emotion string
        is_negative = self._is_negative_emotion(current_emotion)

        if self.debug:
            print("\n" + "=" * 50)
            print("[DebouncePolicy] [EVALUATING INPUT]")
            print(f"  • Pending Msgs Count : {len(pending_messages)}")
            print(f"  • Last Message       : '{last_msg}'")
            print(f"  • Total Chars        : {total_chars} / Threshold: {self.cum_char_threshold}")
            print(f"  • Emotion Input      : {current_emotion}")
            print(f"  • Emotion Detect     : '{'NEGATIVE' if is_negative else 'NEUTRAL/POSITIVE'}'")

        # ── Rule 1: User explicitly yields or invites client to talk ──────────
        intent_to_talk = self._is_intent_to_talk(last_msg)
        if self.debug:
            print(f"  • Rule 1 Check (Intent to talk / explicit handoff): {intent_to_talk}")

        if intent_to_talk:
            res = DebouncePolicyResult(
                wait_seconds=self.short_wait,
                after_think_seconds=self.default_after_think,
                immediate_action="none",
            )
            if self.debug:
                print(f"[DebouncePolicy] -> MATCH Rule 1! Returning short wait ({res.wait_seconds}s).")
                print("=" * 50 + "\n")
            return res

        # ── Rule 2: Cumulative text is long & incomplete (chunked input) ──────
        is_incomplete = self._is_incomplete_sentence(last_msg)
        if self.debug:
            print(f"  • Rule 2 Check (Incomplete sentence): {is_incomplete}")

        if total_chars >= self.cum_char_threshold and is_incomplete:
            # Micro-respond with probability `micro_response_probability`,
            # stay silent otherwise (default: 20% filler / 80% silence).
            use_micro = (
                self.enable_micro_responses
                and random.random() < self.micro_response_probability
            )
            if not use_micro:
                res = DebouncePolicyResult(
                    wait_seconds=self.long_wait,
                    after_think_seconds=self.default_after_think,
                    immediate_action="silence",
                )
                if self.debug:
                    print(f"[DebouncePolicy] -> MATCH Rule 2! Selected: SILENCE ({self.long_wait}s wait).")
            else:
                if is_negative:
                    micro_text = random.choice(self.ANNOYED_MICRO) # it samples from the pre-selected annoyed micro-responses
                    res = DebouncePolicyResult(
                        wait_seconds=self.long_wait,
                        after_think_seconds=self.default_after_think,
                        immediate_action="micro_response",
                        micro_response_text=micro_text,
                    )
                    if self.debug:
                        print(f"[DebouncePolicy] -> MATCH Rule 2 (Negative Emotion)! Selected: ANNOYED MICRO-RESPONSE: '{micro_text}'")
                else:
                    micro_text = random.choice(self.NEUTRAL_MICRO)
                    res = DebouncePolicyResult(
                        wait_seconds=self.long_wait,
                        after_think_seconds=self.default_after_think,
                        immediate_action="micro_response",
                        micro_response_text=micro_text,
                    )
                    if self.debug:
                        print(f"[DebouncePolicy] -> MATCH Rule 2 (Neutral/Positive)! Selected: FILLER MICRO-RESPONSE: '{micro_text}'")

            if self.debug:
                print("=" * 50 + "\n")
            return res

        # ── Default Fallback ──────────────────────────────────────────────────
        calculated_wait = self.long_wait # self.short_wait if total_chars < 50 else self.long_wait / 2.0
        res = DebouncePolicyResult(
            wait_seconds=calculated_wait,
            after_think_seconds=self.default_after_think,
            immediate_action="none",
        )
        if self.debug:
            print(f"[DebouncePolicy] -> NO RULE MATCH. Fallback calculated wait: {calculated_wait}s")
            print("=" * 50 + "\n")
        return res

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _is_intent_to_talk(self, text: str) -> bool:
        """Returns True if the message actively hands over the speaking turn."""
        if text.endswith(("?", "!", ".", "```")):
            return True
        yield_phrases = [
            "what do you think", "go ahead", "right?",
            "thoughts?", "tell me", "response"
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in yield_phrases)

    def _is_incomplete_sentence(self, text: str) -> bool:
        """Returns True if message ends mid-clause or without sentence boundary."""
        if text.endswith((",", "...", "-", "and", "or", "because", "but", "if", "so", "that")):
            return True
        return not text.endswith((".", "?", "!", "```"))

    def _is_negative_emotion(self, emotion: str | dict | None) -> bool:
        """
        True when the reported emotion list contains a negative emotion.

        Delegates to :func:`..emotions.is_negative`, which uses the original
        vocabulary so this method's behaviour is unchanged.
        """
        return is_negative(emotion)