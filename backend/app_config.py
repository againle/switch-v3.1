"""Runtime configuration loader for the Memgraph backend.

Reads ``backend/config.yaml`` and exposes typed dataclasses so the rest
of the app (WebSocket handler, session, orchestrator, and the frontend
via ``GET /api/config``) share a single source of truth.

Any missing/invalid key falls back to a sensible default rather than
crashing startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .lib.core.debouncer.registry import DEFAULT_METHOD as DEFAULT_DEBOUNCE_METHOD

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BACKEND_DIR / "config.yaml"


@dataclass
class DebounceConfig:
    """Settings for batching rapid user messages into a single turn.

    ``method`` names the turn-taking policy (see
    ``lib/core/debouncer/methods/``); ``methods`` holds the per-method
    settings blocks from ``config.yaml``. ``after_think_seconds`` is the only
    value shared by every method — everything else lives in the method's own
    block, so a parameter is never configured in two places.
    """

    enabled: bool = True
    method: str = DEFAULT_DEBOUNCE_METHOD
    after_think_seconds: float = 3.0
    # Hard ceiling on the TOTAL collection time for one burst: no matter how
    # many times the window is refreshed (new messages / typing), the batch is
    # flushed once this many seconds have elapsed since the burst began.
    # 0 = no cap.
    max_wait_seconds: float = 30.0
    methods: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DebounceConfig":
        data = data or {}
        raw_methods = data.get("methods") or {}
        methods = {
            str(name): dict(block)
            for name, block in raw_methods.items()
            if isinstance(block, dict)
        }
        return cls(
            enabled=bool(data.get("enabled", True)),
            method=str(data.get("method") or DEFAULT_DEBOUNCE_METHOD).strip().lower(),
            after_think_seconds=float(data.get("after_think_seconds", 3.0)),
            max_wait_seconds=float(data.get("max_wait_seconds", 30.0) or 0.0),
            methods=methods,
        )

    def settings_for(self, method: str) -> dict[str, Any]:
        """Merge the shared default with *method*'s own config block.

        Unknown keys are harmless: each policy only reads the ones it knows.
        """
        merged: dict[str, Any] = {"after_think_seconds": self.after_think_seconds}
        merged.update(self.methods.get(method) or {})
        return merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "after_think_seconds": self.after_think_seconds,
            "max_wait_seconds": self.max_wait_seconds,
            "methods": dict(self.methods),
        }


@dataclass
class ChunkingConfig:
    """Frontend settings for playing a reply back as several bubbles.

    The reply is split at the model's own line breaks (one bubble per line),
    and each bubble is preceded by a simulated "typing" bubble whose length
    is derived from ``*_typing_speed_cps``.
    """

    enabled: bool = True
    show_typing_indicator: bool = True
    max_chars: int = 220
    min_typing_speed_cps: float = 4.0
    max_typing_speed_cps: float = 12.0
    punctuation_pause_ms: int = 160
    max_typing_ms: int = 6000
    bubble_gap_ms: int = 700
    first_bubble_delay_ms: int = 700
    # TYPING_REFLECTION only: pause after a deleted draft's "…" vanishes and
    # before the next attempt begins (milliseconds).
    reflection_vanish_ms: int = 900

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChunkingConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            show_typing_indicator=bool(data.get("show_typing_indicator", True)),
            max_chars=int(data.get("max_chars", 220)),
            min_typing_speed_cps=float(data.get("min_typing_speed_cps", 4.0)),
            max_typing_speed_cps=float(data.get("max_typing_speed_cps", 12.0)),
            punctuation_pause_ms=int(data.get("punctuation_pause_ms", 160)),
            max_typing_ms=int(data.get("max_typing_ms", 6000)),
            bubble_gap_ms=int(data.get("bubble_gap_ms", 700)),
            first_bubble_delay_ms=int(data.get("first_bubble_delay_ms", 700)),
            reflection_vanish_ms=int(data.get("reflection_vanish_ms", 900)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "show_typing_indicator": self.show_typing_indicator,
            "max_chars": self.max_chars,
            "min_typing_speed_cps": self.min_typing_speed_cps,
            "max_typing_speed_cps": self.max_typing_speed_cps,
            "punctuation_pause_ms": self.punctuation_pause_ms,
            "max_typing_ms": self.max_typing_ms,
            "bubble_gap_ms": self.bubble_gap_ms,
            "first_bubble_delay_ms": self.first_bubble_delay_ms,
            "reflection_vanish_ms": self.reflection_vanish_ms,
        }


@dataclass
class ContextSections:
    """Which pieces the ``memory`` CONTEXT block carries.

    ``context.memory.sections`` in ``config.yaml``: ``true`` includes the
    section (and runs its query), ``false`` drops it. The block renders in this
    order — activated beliefs, related past conversation, then recent dialogue
    last, so the freshest turns sit closest to the message being answered.

    With every section off the block is empty, which the generation layer reads
    as "no CONTEXT supplied" and falls back to the raw history.
    """

    activated_beliefs: bool = True
    related_past: bool = True
    recent_dialogue: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ContextSections":
        data = data or {}
        return cls(
            activated_beliefs=bool(data.get("activated_beliefs", True)),
            related_past=bool(data.get("related_past", True)),
            recent_dialogue=bool(data.get("recent_dialogue", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "activated_beliefs": self.activated_beliefs,
            "related_past": self.related_past,
            "recent_dialogue": self.recent_dialogue,
        }


@dataclass
class ContextConfig:
    """Settings for the CONTEXT component (how conversation context is supplied).

    ``type`` selects the strategy (``context_type`` in ``config.yaml``):

    - ``"summary"`` — accumulate the conversation and summarize older windows
      after ``start_round`` rounds (``window_size`` rounds per window). The
      mechanism itself lives in the generation layer
      (``cognillm/lib/action``); ``summary_enabled`` toggles it on/off.
    - ``"memory"`` — read CONTEXT every turn straight from the standalone
      ``neo4j-db`` graph container through
      ``ContextManager.fetch_memory_context`` (a direct in-process call). The
      old summary method is bypassed entirely, and each finished turn is
      written back to the graph by a background task.
    """

    type: str = "summary"
    # ── summary type ────────────────────────────────────────────────
    summary_enabled: bool = True
    summary_window_size: int = 5
    summary_start_round: int = 50
    # ── memory type ─────────────────────────────────────────────────
    # ``memory_enabled`` greys out the read/write path. There is no persona
    # list to configure: the profile folders under ``backend/profiles/`` are the
    # source of truth and each session syncs its own profile into the graph
    # (see ``lib/core/context/profile_sync.py``).
    memory_enabled: bool = True
    # Which sections of the CONTEXT block to render (see ContextSections).
    memory_sections: ContextSections = field(default_factory=ContextSections)
    # Caps for the two recall sections.
    memory_history_limit: int = 8
    memory_related_limit: int = 5

    @classmethod
    def from_dict(cls, context_type: Any = None, data: dict[str, Any] | None = None) -> "ContextConfig":
        data = data or {}
        summary = data.get("summary") or {}
        memory = data.get("memory") or {}

        ctx_type = str(context_type or "summary").strip().lower()
        if ctx_type not in ("summary", "memory"):
            ctx_type = "summary"

        return cls(
            type=ctx_type,
            summary_enabled=bool(summary.get("enabled", True)),
            summary_window_size=int(summary.get("window_size", 5)),
            summary_start_round=int(summary.get("start_round", 50)),
            memory_enabled=bool(memory.get("enabled", False)),
            memory_sections=ContextSections.from_dict(memory.get("sections")),
            memory_history_limit=int(memory.get("history_limit", 8)),
            memory_related_limit=int(memory.get("related_limit", 5)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "summary": {
                "enabled": self.summary_enabled,
                "window_size": self.summary_window_size,
                "start_round": self.summary_start_round,
            },
            "memory": {
                "enabled": self.memory_enabled,
                "sections": self.memory_sections.to_dict(),
                "history_limit": self.memory_history_limit,
                "related_limit": self.memory_related_limit,
            },
        }


@dataclass
class MiStageConfig:
    """Motivational-Interviewing stage control for CogniLLM.

    - ``enabled``              — master switch. False disables the StageManager
      entirely: no stage_info is available and no progression is tracked.
    - ``inject_stage_info``    — inject the current STAGE_INFO into every user
      prompt (``{"stage_info": …, "user_message": …}``). When False the prompt
      carries only ``{"user_message": …}``.
    - ``evaluate_progression`` — whether to run the stage evaluation and
      advance the stage on every turn (forces ``EVAL=True``). False freezes the
      stage: the turn is recorded but nothing is evaluated/advanced.
    - ``injected_stage``       — only used while
      ``inject_stage_info=True`` **and** ``evaluate_progression=False``: pins
      the injected STAGE to a fixed stage (e.g. ``"pre_contemplation"``)
      instead of the manager's current stage. Empty string = no pinning (inject
      the current stage).
    """

    enabled: bool = True
    inject_stage_info: bool = True
    evaluate_progression: bool = True
    injected_stage: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MiStageConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            inject_stage_info=bool(data.get("inject_stage_info", True)),
            evaluate_progression=bool(data.get("evaluate_progression", True)),
            injected_stage=str(data.get("injected_stage", "") or "").strip().lower(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "inject_stage_info": self.inject_stage_info,
            "evaluate_progression": self.evaluate_progression,
            "injected_stage": self.injected_stage,
        }


@dataclass
class AppConfig:
    """Top-level application configuration."""

    rate: float = 0.5
    context: ContextConfig = field(default_factory=ContextConfig)
    mi_stage: MiStageConfig = field(default_factory=MiStageConfig)
    debounce: DebounceConfig = field(default_factory=DebounceConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AppConfig":
        data = data or {}
        return cls(
            rate=float(data.get("rate", 0.5)),
            context=ContextConfig.from_dict(
                data.get("context_type"), data.get("context")
            ),
            mi_stage=MiStageConfig.from_dict(data.get("mi_stage")),
            debounce=DebounceConfig.from_dict(data.get("debounce")),
            chunking=ChunkingConfig.from_dict(data.get("chunking")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "context_type": self.context.type,
            "context": self.context.to_dict(),
            "mi_stage": self.mi_stage.to_dict(),
            "debounce": self.debounce.to_dict(),
            "chunking": self.chunking.to_dict(),
        }


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load the YAML config file, falling back to defaults on any error."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppConfig()
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except Exception:
        return AppConfig()
    return AppConfig.from_dict(data)


# Module-level cached instance — loaded once at import time.
CONFIG: AppConfig = load_config()
