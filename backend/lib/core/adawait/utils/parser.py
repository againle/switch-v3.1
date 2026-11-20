# adawait/utils/parser.py
# Parse raw Emotion list and Openness text into scores for the adawait package.
from __future__ import annotations

from typing import Dict, List

from .models.emotions import EmotionScore
from .models.openness import OpennessScore


class Parser:
    """Parser for raw Emotion list and Openness text into scores."""

    def __init__(self) -> None:
        self._emotion_scorer = EmotionScore()
        self._openness_scorer = OpennessScore()

    def parse(self, emotion_list: List[str], openness_text: str) -> Dict[str, float]:
        """
        Parse raw Emotion list and Openness text into scores.

        Args:
            emotion_list: A list of emotion strings.
            openness_text: A text description of openness.

        Returns:
            A dict with ``E`` (emotional patience: higher = the client will
            wait longer — see ``models.emotions``) and ``O`` (openness).
        """
        return {
            "E": self._emotion_scorer.score(emotion_list),
            "O": self._openness_scorer.score(openness_text),
        }