"""
Context management helpers.

This module owns the full workflow around profile loading and the CONTEXT
component. It replaces the old graph-memory ``MemoryOrchestrator``: the
graph-memory code (tree build/lookup/update/visualization) has been removed,
and only the profile helpers plus the reserved external-``memory`` source
remain.

Context types (``context_type`` in ``config.yaml``):

- ``summary`` — accumulate the conversation, summarize older windows after
                N rounds. The implementation is in the generation layer
                (``cognillm/lib/action``); this module only carries the
                profile/config side.
- ``memory``  — read CONTEXT from the standalone ``neo4j-db`` graph container
                and hand it to the generation layer. See
                :meth:`fetch_memory_context` and :mod:`.neo4j_source`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ....logger import get_logger

logger = get_logger(__name__)


class ContextManager:
    """High-level manager for profiles and the CONTEXT component."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = (
            Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[3]
        )
        self.profiles_dir = self.base_dir / "profiles"

    # ── Profiles ────────────────────────────────────────────────────

    def load_profile(self, profile_name: str) -> dict[str, Any]:
        profile_path = self.profiles_dir / profile_name / "profile.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        with open(profile_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def load_stage_config(self, profile_name: str) -> dict[str, Any]:
        config_path = self.profiles_dir / profile_name / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file) or {}
        return {}

    def list_profiles(self) -> list[str]:
        if not self.profiles_dir.exists():
            return []
        return sorted(
            directory.name
            for directory in self.profiles_dir.iterdir()
            if directory.is_dir() and (directory / "profile.yaml").exists()
        )

    # ── External ``memory`` context type (RESERVED) ─────────────────

    # ── ``memory`` context type ─────────────────────────────────────
    # CONTEXT is read straight from the standalone ``neo4j-db`` graph by a
    # direct function call in this process — there is no HTTP/URL hop.

    @staticmethod
    def _graph_source(profile_name: str = ""):
        """Build the graph adapter for *profile_name*."""
        from .neo4j_source import Neo4jContextSource

        return Neo4jContextSource(profile_name)

    @staticmethod
    def _memory_render_kwargs() -> dict[str, Any]:
        """``context.memory`` render options from ``config.yaml``.

        Returns the kwargs for :meth:`Neo4jContextSource.render_context`:
        which sections to emit (``sections``) and the two recall caps. Read
        lazily so the context package never imports the app config at module
        load; an empty dict means "use the adapter's own defaults" (everything
        on).
        """
        try:
            from ....app_config import CONFIG

            ctx = CONFIG.context
            return {
                "sections": ctx.memory_sections.to_dict(),
                "history_limit": int(ctx.memory_history_limit),
                "related_limit": int(ctx.memory_related_limit),
            }
        except Exception:  # noqa: BLE001 — config is optional here
            return {}

    def sync_profile(self, profile_name: str) -> dict[str, Any] | None:
        """Mirror a memgraph profile into the graph, idempotently.

        Reads ``profiles/<profile_name>/profile.yaml`` and upserts the matching
        ``ClientProfile`` (its beliefs) through the graph, so a session started
        for that profile reads and writes beliefs that belong to that same
        client. Called when a session starts.

        Returns the write counts, or ``None`` when the graph is unreachable or
        the profile cannot be read — a sync failure must never block a chat.
        """
        try:
            from .profile_sync import sync_profile as _sync

            source = self._graph_source(profile_name)
            if not source.available():
                logger.warning(
                    "profile sync skipped — neo4j-db unreachable at %s",
                    source.uri() or "<unset NEO4J_URI>",
                )
                return None
            counts = _sync(source, profile_name, self.load_profile(profile_name))
        except Exception as exc:  # noqa: BLE001 — chat must still start
            logger.warning("profile sync failed for %s: %s", profile_name, exc)
            return None

        logger.info(
            "Synced profile %s into neo4j-db (%s beliefs, %s newly embedded)",
            profile_name,
            counts.get("beliefs"),
            counts.get("beliefs_embedded"),
        )
        return counts

    def sync_all_profiles(self) -> dict[str, Any]:
        """Mirror every memgraph profile into the graph and prune stale personas.

        The maintenance counterpart to :meth:`sync_profile` (also available as
        ``python -m backend.lib.core.context.profile_sync``).
        """
        from .profile_sync import sync_all

        return sync_all(self)

    def memory_available(self) -> bool:
        """Whether the ``neo4j-db`` graph container is reachable right now."""
        try:
            return self._graph_source().available()
        except Exception:  # noqa: BLE001
            return False

    def fetch_memory_context(
        self,
        profile_name: str = "",
        user_message: str = "",
        session_id: str = "",
        include_full_schema: bool = False,
    ) -> str | None:
        """Read CONTEXT from the standalone ``neo4j-db`` graph container.

        When ``context_type == "memory"`` this replaces the ``summary``
        mechanism entirely: the block returned here is pushed into the prompt
        for the turn via ``Client.set_context`` (see
        ``cognillm/lib/action/ai.py``). The graph runs in its own Docker
        container (``/neo4j-db``) and is read in-process over bolt at the URI
        configured in ``neo4j_db/.env`` — no HTTP round-trip.

        Args:
            profile_name: memgraph profile name, also the graph ``ClientProfile``
                name (see :meth:`sync_profile`).
            user_message: the message being answered, used to mark which
                beliefs it activates.
            session_id: graph session whose recent turns are replayed.
            include_full_schema: also list every stored belief.

        Returns:
            The rendered CONTEXT block, or ``None`` when the container is
            unreachable or no persona can be resolved. Callers treat ``None``
            as "no external CONTEXT available" and fall back to the raw
            history rather than failing the turn.
        """
        graph_persona = profile_name
        try:
            source = self._graph_source(graph_persona)
            if not source.available():
                logger.warning(
                    "context_type='memory' but the neo4j-db container at %s is unreachable",
                    source.uri() or "<unset NEO4J_URI>",
                )
                return None
            block = source.render_context(
                user_message=user_message,
                profile_name=graph_persona,
                session_id=session_id,
                include_full_schema=include_full_schema,
                **self._memory_render_kwargs(),
            )
        except ValueError as exc:
            logger.warning("context_type='memory' persona lookup failed: %s", exc)
            return None
        except Exception as exc:  # never break the turn loop over missing CONTEXT
            logger.warning("context_type='memory' fetch failed: %s", exc)
            return None

        logger.info(
            "Fetched %d-char CONTEXT block from neo4j-db (%s)",
            len(block or ""),
            source.uri() or "<unset NEO4J_URI>",
        )
        # An empty block (every section switched off in config.yaml) is treated
        # as "no CONTEXT supplied", exactly like an unreachable graph: the
        # generation layer then falls back to the raw history.
        return block or None

    def record_memory_turn(
        self,
        *,
        session_id: str,
        profile_name: str = "",
        trainee_message: str = "",
        state: dict[str, Any] | None = None,
        interrupted: bool = False,
        follows_interruption: bool = False,
    ) -> dict[str, Any] | None:
        """Write one finished turn back into the graph (``memory`` type only).

        Deliberately **blocking**: the caller runs it off the critical path
        (a background task in a thread executor), so the trainee never waits
        on graph latency. Never raises — a failed write is logged and the
        conversation carries on.

        *interrupted* / *follows_interruption* mark a reply that was withheld
        because new input overtook it; the turn is still stored, with an in-band
        directive saying so (see :mod:`.neo4j_source`).
        """
        graph_persona = profile_name
        try:
            source = self._graph_source(graph_persona)
            result = source.record_turn(
                session_id=session_id,
                profile_name=graph_persona,
                trainee_message=trainee_message,
                state=state,
                interrupted=interrupted,
                follows_interruption=follows_interruption,
            )
        except Exception as exc:  # noqa: BLE001 — memory writes are best-effort
            logger.warning("context_type='memory' turn write-back failed: %s", exc)
            return None

        logger.info(
            "Wrote turn %s for graph session %s (triggered=%d)",
            result.get("turn_index"),
            session_id,
            len(result.get("triggered_beliefs") or []),
        )
        return result
