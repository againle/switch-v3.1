"""Coarse emotion classification shared by every debounce method.

The client reports its emotional state as a list of short strings
(``["Defensive", "Annoyed"]``, ``["Hopeful", "Anxious"]``, ...). Turn-taking
methods only need a *coarse* read of that list, and there are four buckets:

    negative → deactivated distress: sad, tired, guarded, disengaged. The
               client has withdrawn, so they get the **longest** wait.
    anxious  → activated distress: anxious, nervous, eagerly waiting,
               irritated, angry. The client is wound up and wants to be
               heard now, so they get the **shortest** wait.
    positive → happy, satisfied, engaged. The client is patient but keen, so
               they get a **medium** wait — waiting too long loses the
               momentum they are offering.
    neutral  → calm, thoughtful, reflective. The client is thinking, not
               withdrawn: a **long** wait, like ``negative`` but for the
               opposite reason.

Wait ordering, longest → shortest: ``negative`` ≈ ``neutral`` > ``positive``
> ``anxious``.

A payload may match several buckets (``["Sad", "Annoyed"]``);
:func:`classify_emotion` resolves that with ``strategy`` plus a fixed
priority order — see :data:`_PRIORITY`.

Two vocabularies exist on purpose:

* :data:`NEGATIVE_EMOTIONS` — the *original* rule set, kept byte-identical so
  the legacy ``fixed_rules`` method behaves exactly as it did before methods
  became pluggable. :func:`is_negative` uses only this one; it predates the
  four-bucket split and is a union of the ``negative`` and ``anxious``
  classifier vocabularies.
* :data:`NEGATIVE_VOCABULARY`, :data:`ANXIOUS_VOCABULARY`,
  :data:`POSITIVE_EMOTIONS` and :data:`NEUTRAL_EMOTIONS` — the classifier's
  four buckets. They are kept **disjoint**, so a word never counts for two
  buckets by design; substring matching can still surface cross-bucket hits
  for one word ("hopelessness" contains both "hopeless" and "hope"), which
  is what the priority order is for.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

#: The coarse buckets a method can branch on.
EmotionPolarity: TypeAlias = Literal["negative", "anxious", "positive", "neutral"]

#: How to resolve a payload that matches several buckets.
#:   "any"      → the highest-priority bucket that matched (see _PRIORITY)
#:   "majority" → the bucket with the most matched words, priority on a tie
EmotionStrategy: TypeAlias = Literal["any", "majority"]

#: Which bucket wins when several match. Most time-critical first: an
#: *activated* client (``anxious``) beats a withdrawn one, because answering a
#: wound-up client slightly too early is a cheaper mistake than leaving them
#: hanging. ``neutral`` is last — it is the bucket of last resort.
_PRIORITY: tuple[EmotionPolarity, ...] = ("anxious", "negative", "positive", "neutral")

#: The original rule set. Do not edit: the fixed-rule method's behaviour
#: depends on it (see ``debounce_policy.DynamicDebouncePolicy``). It predates
#: the four-bucket split: :func:`is_negative` reads it as "any distress", so
#: both the ``negative`` and ``anxious`` buckets count.
NEGATIVE_EMOTIONS: frozenset[str] = frozenset({
    "angry", "anxious", "sad", "frustrated", "annoyed",
    "irritated", "defensive", "overwhelmed", "disgusted", "fearful",
    "guarded", "hurt", "wary", "pressured", "stressed", "tense", "upset", "vulnerable",
    "uneasy",
})

#: Bucket 1 — deactivated distress: withdrawn, flat, shut down. Longest wait.
NEGATIVE_VOCABULARY: frozenset[str] = frozenset({
    "sad", "sadness", "unhappy", "down", "depressed", "depression",
    "miserable", "hopeless", "helpless", "despair", "grief", "grieving",
    "lonely", "isolated", "abandoned", "rejected", "betrayed",
    "disappointed", "disappointment", "discouraged", "discouragement",
    "defeated", "resigned", "tired", "exhausted", "drained", "weary",
    "fatigued", "numb", "flat", "withdrawn", "disengaged", "checked-out",
    "apathetic", "unmotivated", "passive", "distant", "closed", "shut-down",
    "overwhelmed", "powerless", "trapped", "worthless", "misunderstood",
    "invalidated", "ashamed", "guilty", "shame", "embarrassed",
    "humiliated", "disgusted", "defensive", "defensiveness", "guarded",
    "wary", "distrustful", "suspicious", "dismissive", "vulnerable",
    "hurt", "upset", "conflicted", "hesitant", "uncertain", "insecure",
})

#: Bucket 2 — activated distress: wound up, impatient, agitated. Shortest wait.
ANXIOUS_VOCABULARY: frozenset[str] = frozenset({
    "angry", "anger", "mad", "furious", "fury", "rage", "enraged",
    "irritated", "irritation", "annoyed", "annoyance", "frustrated",
    "frustration", "resentful", "resentment", "bitter", "impatient",
    "impatience", "agitated", "agitation", "anxious", "anxiety",
    "nervous", "nervousness", "worried", "worry", "concerned",
    "apprehensive", "uneasy", "tense", "tension", "stressed", "stress",
    "pressured", "pressure", "on-edge", "restless", "eager", "eagerly",
    "anticipatory", "jittery", "alarmed", "fearful", "fear", "scared",
    "afraid", "panicked", "panicky", "dread",
})

#: Bucket 3 — engaged and content. Medium wait.
POSITIVE_EMOTIONS: frozenset[str] = frozenset({
    "happy", "happiness", "joyful", "joy", "content", "contentment",
    "cheerful", "delighted", "delight", "grateful", "gratitude",
    "excited", "excitement", "loved", "hopeful", "hope", "optimistic",
    "optimism", "proud", "pride", "enthusiastic", "enthusiasm",
    "satisfied", "satisfaction", "relieved", "relief", "encouraged",
    "motivated", "engaged", "interested", "curious", "warm", "trusting",
    "reassured", "confident", "confidence", "validated", "understood",
    "appreciative", "appreciated", "amused", "comfortable", "safe",
    "secure", "willing", "receptive", "empowered", "determined", "open",
    "patient", "glad", "pleased", "supported", "connected",
})

#: Bucket 4 — calm, reflective, thinking it over. Long wait.
NEUTRAL_EMOTIONS: frozenset[str] = frozenset({
    "neutral", "calm", "calmness", "relaxed", "composed", "composure",
    "steady", "balanced", "poised", "serene", "tranquil", "still",
    "indifferent", "unaffected", "detached", "reserved", "thoughtful",
    "reflective", "pensive", "contemplative", "analytical", "objective",
    "measured", "pragmatic", "matter-of-fact", "skeptical", "cautious",
    "watchful", "polite", "quiet", "focused", "pondering", "considering",
    "deliberating", "thinking", "deep-in-thought",
})

#: Prefixes that invert a positive word ("uncomfortable", "insecure").
_NEGATION_PREFIXES: tuple[str, ...] = ("un", "dis", "in", "im", "non")

#: Shortest word allowed to match a vocabulary entry *backwards* (the word is
#: a stem of the entry, like "fear" → "fearful"), which avoids stray
#: two-letter matches.
_REVERSE_MATCH_MIN_LEN = 4

_VOCABULARIES: tuple[tuple[EmotionPolarity, frozenset[str]], ...] = (
    ("negative", NEGATIVE_VOCABULARY),
    ("anxious", ANXIOUS_VOCABULARY),
    ("positive", POSITIVE_EMOTIONS),
    ("neutral", NEUTRAL_EMOTIONS),
)


@dataclass(frozen=True)
class EmotionReading:
    """Immutable result of classifying an emotion payload."""

    polarity: EmotionPolarity = "neutral"
    #: Normalised emotion words found in the payload.
    words: tuple[str, ...] = ()
    #: Vocabulary words that matched, per polarity.
    matched: tuple[tuple[str, tuple[str, ...]], ...] = ()

    # ── Accessors ───────────────────────────────────────────────────

    def matched_for(self, polarity: str) -> tuple[str, ...]:
        """Vocabulary words that matched *polarity*."""
        return dict(self.matched).get(polarity, ())

    @property
    def negative(self) -> bool:
        return self.polarity == "negative"

    @property
    def positive(self) -> bool:
        return self.polarity == "positive"

    @property
    def anxious(self) -> bool:
        return self.polarity == "anxious"

    @property
    def neutral(self) -> bool:
        return self.polarity == "neutral"

    @property
    def is_negative(self) -> bool:
        """True for a clearly ``negative`` (withdrawn) reading."""
        return self.polarity == "negative"

    def summary(self) -> str:
        """One-line description for logs / the status payload."""
        if not self.words:
            return "neutral (no emotions reported)"
        hits = ", ".join(
            f"{name}:{'/'.join(words)}" for name, words in self.matched if words
        )
        return f"{self.polarity} ({hits})" if hits else f"{self.polarity} ({', '.join(self.words)})"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form (services the status payload / future UI)."""
        return {
            "polarity": self.polarity,
            "words": list(self.words),
            "matched": {name: list(words) for name, words in self.matched},
        }


# ── Normalisation ────────────────────────────────────────────────────


def normalize_emotions(raw: Any) -> tuple[str, ...]:
    """Flatten any emotion payload into a tuple of lowercase words.

    Accepts the shapes that reach us in practice: ``None``, a single string
    (plain, comma-separated or JSON), a sequence of strings, or a mapping
    such as ``{"emotions": [...]}`` / ``{"emotion": "..."} ``.
    """
    values: list[Any] = []

    if raw is None:
        return ()

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ()
        if text[:1] in "[{":  # serialised list/dict
            try:
                values = _as_list(json.loads(text))
            except (ValueError, TypeError):
                values = [part for part in text.replace(";", ",").split(",")]
        else:
            values = [part for part in text.replace(";", ",").split(",")]
    elif isinstance(raw, Mapping):
        for key in ("emotions", "emotion", "primary_emotions"):
            if key in raw:
                return normalize_emotions(raw[key])
        return ()
    else:
        values = _as_list(raw)

    words: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip().lower()
        if text:
            words.append(text)
    return tuple(words)


def _as_list(raw: Any) -> list[Any]:
    if isinstance(raw, (list, tuple, set, frozenset)):
        return list(raw)
    return [raw]


# ── Matching ─────────────────────────────────────────────────────────


def _containment_hits(word: str, vocabulary: frozenset[str]) -> list[str]:
    """Vocabulary entries embedded in *word*, or stems of them."""
    hits = [entry for entry in vocabulary if entry in word]
    if hits:
        return hits
    if len(word) >= _REVERSE_MATCH_MIN_LEN:
        return [entry for entry in vocabulary if word in entry]
    return []


def _negated_positive(word: str) -> str | None:
    """Return the positive word *word* negates, if any.

    ``"uncomfortable"`` → ``"comfortable"``. Only consulted when the word
    matched nothing exactly, so explicit vocabulary entries always win over
    this heuristic.
    """
    for prefix in _NEGATION_PREFIXES:
        if not word.startswith(prefix) or len(word) <= len(prefix) + 2:
            continue
        stem = word[len(prefix):]
        for positive in POSITIVE_EMOTIONS:
            if stem == positive or stem.startswith(positive):
                return positive
    return None


def classify_emotion(raw: Any, strategy: EmotionStrategy = "any") -> EmotionReading:
    """Classify an emotion payload into an :class:`EmotionReading`.

    Matching is per word, in three passes: an exact vocabulary hit wins;
    otherwise a negated positive word ("uncomfortable") counts as negative;
    otherwise substring matching applies, so "Slightly annoyed" and
    "Extremely Low" still register.

    A payload may match several buckets. ``strategy`` picks the winner:

    * ``"any"`` (default) — the highest-priority bucket that matched, in the
      order of :data:`_PRIORITY` (``anxious`` → ``negative`` → ``positive``
      → ``neutral``).
    * ``"majority"`` — the bucket with the most matched words, falling back to
      the same priority order on a tie.

    Args:
        raw: Whatever the client reported (see :func:`normalize_emotions`).
        strategy: How to resolve a payload that matches several buckets.
    """
    words = normalize_emotions(raw)
    if not words:
        return EmotionReading()

    hits: dict[EmotionPolarity, list[str]] = {name: [] for name in _PRIORITY}

    def record(polarity: EmotionPolarity, value: str) -> None:
        if value not in hits[polarity]:
            hits[polarity].append(value)

    for word in words:
        exact = [
            (polarity, entry)
            for polarity, vocabulary in _VOCABULARIES
            for entry in vocabulary
            if entry == word
        ]
        if exact:
            for polarity, entry in exact:
                record(polarity, entry)
            continue

        negated = _negated_positive(word)
        if negated is not None:
            record("negative", negated)
            continue

        for polarity, vocabulary in _VOCABULARIES:
            for entry in _containment_hits(word, vocabulary):
                record(polarity, entry)

    polarity = _resolve_polarity(hits, strategy)

    return EmotionReading(
        polarity=polarity,
        words=words,
        matched=tuple((name, tuple(values)) for name, values in hits.items() if values),
    )


def _resolve_polarity(
    hits: Mapping[EmotionPolarity, list[str]],
    strategy: EmotionStrategy,
) -> EmotionPolarity:
    """Pick the winning bucket from the per-bucket match counts."""
    counts = {name: len(values) for name, values in hits.items()}

    if strategy == "majority":
        best = max(counts.values(), default=0)
        if best:
            return next(name for name in _PRIORITY if counts[name] == best)
        return "neutral"

    # "any": the first bucket in priority order that matched anything.
    return next((name for name in _PRIORITY if counts[name]), "neutral")


def polarity_of(raw: Any, strategy: EmotionStrategy = "any") -> EmotionPolarity:
    """Shorthand for ``classify_emotion(raw, strategy).polarity``."""
    return classify_emotion(raw, strategy).polarity


def is_negative(raw: Any) -> bool:
    """True when any *legacy* negative word appears in the payload.

    Deliberately narrow: uses :data:`NEGATIVE_EMOTIONS` only, with the same
    substring semantics as the original
    ``DynamicDebouncePolicy._is_negative_emotion``, so the fixed-rule method
    keeps its exact behaviour. New methods should use
    :func:`classify_emotion` instead.
    """
    for word in normalize_emotions(raw):
        for hit in NEGATIVE_EMOTIONS:
            if hit in word:
                return True
            if len(word) >= _REVERSE_MATCH_MIN_LEN and word in hit:
                return True
    return False


def count_matches(raw: Any) -> dict[str, int]:
    """Match counts per polarity — a hook for future, finer-grained methods."""
    reading = classify_emotion(raw)
    return {name: len(words) for name, words in reading.matched}


__all__ = [
    "ANXIOUS_VOCABULARY",
    "EmotionPolarity",
    "EmotionReading",
    "EmotionStrategy",
    "NEGATIVE_EMOTIONS",
    "NEGATIVE_VOCABULARY",
    "NEUTRAL_EMOTIONS",
    "POSITIVE_EMOTIONS",
    "classify_emotion",
    "count_matches",
    "is_negative",
    "normalize_emotions",
    "polarity_of",
]
