"""Message debouncer package — batches rapid user messages into a single
CogniLLM turn so the client "thinks" once instead of once per message.

Two concerns are deliberately separated:

* ``MessageDebouncer`` owns the queue, the clock and the flush — it knows
  nothing about any particular turn-taking method.
* A ``DebouncePolicy`` (see ``methods/``) owns the *decision*: how long to
  wait, and whether to emit a filler. The method is selected by name in
  ``backend/config.yaml`` (``debounce.method``).

Quick start::

    from backend.lib.core.debouncer import MessageDebouncer

    debouncer = MessageDebouncer(method="typing_aware", settings={...})

Adding a method: read ``README.md`` in this package.
"""

from __future__ import annotations

from .emotions import (
    EmotionPolarity,
    EmotionReading,
    classify_emotion,
    polarity_of,
)
from .message_debouncer import MessageDebouncer
from .registry import (
    DEFAULT_METHOD,
    available_policies,
    build_policy,
    register_policy,
)
from .schema import (
    DebouncePolicy,
    DebouncePolicyResult,
    ImmediateAction,
    PolicyContext,
)

__all__ = [
    "DEFAULT_METHOD",
    "DebouncePolicy",
    "DebouncePolicyResult",
    "EmotionPolarity",
    "EmotionReading",
    "ImmediateAction",
    "MessageDebouncer",
    "PolicyContext",
    "available_policies",
    "build_policy",
    "classify_emotion",
    "polarity_of",
    "register_policy",
]
