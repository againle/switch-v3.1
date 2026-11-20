"""Fixed-vocabulary emotion -> patience scoring for AdaWait.

The client reports its emotion as a list of short strings. AdaWait needs one
number: how long this client will happily wait for the trainee. That number is
the ``E`` patience score in [0, 1] which
:class:`~backend.lib.core.adawait.utils.state.State` smooths and
:class:`~backend.lib.core.adawait.utils.sampler.WaitTimeSampler` turns into the
next interval -- higher ``E`` means a longer wait.

The category comes from the **same** fixed-vocabulary classifier the message
debouncer uses (:func:`backend.lib.core.debouncer.emotions.classify_emotion`),
so both halves of the pacing logic read emotion identically. There is no model
and no embedding here: this replaces the old sentence-transformer scorer, which
was slow to load and produced a *valence* score -- and valence cannot express
the four regimes, because an anxious client is unpleasant but impatient, so
valence alone says "wait long".

Category -> ``E`` (higher = more patient = longer wait)::

    negative (withdrawn, disengaged)   0.90   longest wait
    neutral  (calm, reflective)        0.85   long wait
    positive (happy, satisfied)        0.50   medium wait
    anxious  (nervous, irritated)      0.15   shortest wait

These are the same four regimes, and the same longest-to-shortest ordering,
that the ``typing_aware`` debounce method uses for its own waits.

An empty or unrecognised payload is *not* an emotion: it keeps the historical
neutral default of 0.5 rather than being read as calm.
"""

from __future__ import annotations

from collections.abc import Mapping

from ....debouncer.emotions import EmotionPolarity, classify_emotion

#: Category -> patience score E. See the module docstring for the ordering.
DEFAULT_CATEGORY_SCORES: dict[EmotionPolarity, float] = {
    "negative": 0.90,
    "neutral": 0.85,
    "positive": 0.50,
    "anxious": 0.15,
}

#: Score used when the payload carries no word the classifier recognises.
UNKNOWN_SCORE: float = 0.5


class EmotionScore:
    """Score an emotion list on the client's patience scale ``E`` in [0, 1].

    Higher ``E`` = the client tolerates a longer silence. The category comes
    from the debouncer's fixed-vocabulary classifier, so a client is read the
    same way by the turn-taking logic and by the adaptive-wait timer.
    """

    #: Category -> patience score. Override per instance if a profile needs a
    #: different temperament; the keys are the debouncer's four polarities.
    CATEGORY_SCORES: Mapping[EmotionPolarity, float] = DEFAULT_CATEGORY_SCORES

    def __init__(self, scores: Mapping[str, float] | None = None) -> None:
        merged: dict[str, float] = dict(self.CATEGORY_SCORES)
        if scores:
            for name, value in scores.items():
                key = str(name).strip().lower()
                if key in merged:
                    merged[key] = float(value)
        self._scores: Mapping[str, float] = merged

    # -- Public API -------------------------------------------------

    def score(self, emotion_list: list[str] | None) -> float:
        """Return this client's patience score ``E`` for *emotion_list*.

        Unrecognised or empty input returns :data:`UNKNOWN_SCORE` (0.5) -- a
        missing reading is not the same as a calm client.
        """
        reading = classify_emotion(emotion_list)
        if not reading.matched:
            return UNKNOWN_SCORE
        return self._scores.get(reading.polarity, UNKNOWN_SCORE)

    def explain(self, emotion_list: list[str] | None) -> str:
        """Human-readable ``category -> E`` line for logs / debugging."""
        reading = classify_emotion(emotion_list)
        if not reading.matched:
            return f"no recognised emotion -> E={UNKNOWN_SCORE:.2f}"
        return f"{reading.summary()} -> E={self.score(emotion_list):.2f}"


__all__ = ["DEFAULT_CATEGORY_SCORES", "UNKNOWN_SCORE", "EmotionScore"]
