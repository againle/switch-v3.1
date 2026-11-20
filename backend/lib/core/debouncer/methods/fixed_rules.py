"""``fixed_rules`` — the original rule-based turn-taking method.

This is the baseline method: the exact rule set that shipped before methods
became pluggable (``DynamicDebouncePolicy``). It is kept first-class so the
system can always be degraded back to it by editing one config line::

    debounce:
      method: fixed_rules

Rules (unchanged):

1. **intent to talk** — the last message ends a sentence or asks something
   (``? ! . ``` `` or a hand-off phrase) → wait ``short_wait``.
2. **long & unfinished** — the batch is at least ``cum_char_threshold`` chars
   and ends mid-clause → wait ``long_wait``, and with probability
   ``micro_response_probability`` emit an annoyed filler (negative emotion)
   or a neutral one.
3. **fallback** — anything else → wait ``long_wait``.

Settings (``debounce.methods.fixed_rules``): ``cum_char_threshold``,
``micro_response_probability`` (0.0–1.0), ``enable_micro_responses``,
``debug``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..debounce_policy import DynamicDebouncePolicy
from ..registry import register_policy
from ..schema import DebouncePolicy, DebouncePolicyResult, PolicyContext


@register_policy("fixed_rules")
class FixedRulePolicy(DebouncePolicy):
    """Adapter that exposes :class:`DynamicDebouncePolicy` through the port."""

    DEFAULTS: Mapping[str, Any] = {
        "short_wait": 1.2,
        "long_wait": 10.0,
        "after_think_seconds": 3.0,
        "cum_char_threshold": 120,
        "micro_response_probability": 0.2,
        "enable_micro_responses": True,
        # Verbose per-evaluation trace to stdout. Off by default: it is very
        # noisy and fires on every tick. One config line turns it on:
        #   debounce: { methods: { fixed_rules: { debug: true } } }
        "debug": False,
    }

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        super().__init__(settings)
        self._engine = DynamicDebouncePolicy(
            short_wait=self.number("short_wait", 1.2),
            long_wait=self.number("long_wait", 10.0),
            default_after_think=self.number("after_think_seconds", 3.0),
            cum_char_threshold=self.integer("cum_char_threshold", 120),
            debug=self.flag("debug", False),
            micro_response_probability=self.number("micro_response_probability", 0.2),
            enable_micro_responses=self.flag("enable_micro_responses", True),
        )

    def evaluate(self, ctx: PolicyContext) -> DebouncePolicyResult:
        result = self._engine(list(ctx.pending_messages), ctx.emotion)
        result.reason = result.reason or f"fixed_rules pending={ctx.count}"
        return result


__all__ = ["FixedRulePolicy"]
