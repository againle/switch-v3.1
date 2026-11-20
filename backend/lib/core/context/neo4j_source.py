"""Context source backed by the standalone ``neo4j-db`` graph container.

This is the bridge between memgraph and the copy of the MemChat backend that
lives in :mod:`._neo4j_db` (directory ``neo4j_db/``). It lets
:class:`~.manager.ContextManager` serve the CONTEXT component when
``context_type == "memory"`` (``backend/config.yaml``): the simulated client's
psychological state is read from the Neo4j graph running in its *own* Docker
container (``/neo4j-db``), and rendered into a text block the generation layer
can inject via ``Client.set_context`` (``cognillm/lib/action/ai.py``).

Design notes
------------
* **Two directions.** :meth:`Neo4jContextSource.render_context` is the read
  path (called once per turn, on the critical path, to build the CONTEXT the
  generation layer uses); :meth:`Neo4jContextSource.record_turn` is the write
  path (called *after* the reply is delivered, off the critical path, to fold
  the finished turn back into the graph).
* **Lazy imports.** The ``neo4j`` driver, ``numpy`` and the HTTP call to Ollama
  (for embeddings) are imported/called only inside methods, so importing the
  ``context`` package never requires a reachable database.
* **Failures never raise past the caller.** :meth:`available` returns a bool,
  persona resolution raises ``ValueError``, and the manager catches it so the
  turn can fall back to the ``summary`` CONTEXT instead of failing.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

#: Maximum number of past turns pulled from the graph into the CONTEXT block.
DEFAULT_HISTORY_LIMIT = 8

#: Maximum number of belief-overlap recall rounds added to the CONTEXT block.
DEFAULT_RELATED_LIMIT = 5

#: Directive appended to a client reply that was WITHHELD because new trainee
#: input arrived while it was still generating. The reply is still stored — a
#: person can say something that goes unheard and still remember it — but the
#: directive stops it reading as "the social worker saw this".
INTERRUPTED_CLIENT_NOTE = (
    "[[system: this message is not seen by the social worker because of "
    "interruption from the social worker]]"
)

#: Directive appended to the trainee message that caused such an interruption,
#: so the turn that follows records that the client's previous reply never
#: landed (and the trainee therefore never reacted to it).
INTERRUPTED_TRAINEE_NOTE = (
    "[[system: the client's previous message was not seen by the social worker "
    "because this message interrupted it]]"
)


def _with_directive(text: str, directive: str) -> str:
    """Append *directive* after *text*, unless it is already there."""
    text = (text or "").strip()
    if not text:
        return directive
    if directive in text:
        return text
    return f"{text}\n{directive}"


#: Matches an in-band ``[[system: …]]`` directive inside a stored utterance.
_DIRECTIVE_RE = re.compile(r"\s*(\[\[system:.*?\]\])", re.IGNORECASE | re.DOTALL)


def _split_directives(text: str) -> tuple[str, list[str]]:
    """Split in-band directives off a stored utterance.

    Returns ``(body, directives)`` — the message without its ``[[system: …]]``
    notes, and the notes themselves in order.
    """
    parts = _DIRECTIVE_RE.split(text or "")
    body = (parts[0] or "").strip()
    directives = [part.strip() for part in parts[1:] if part and part.strip()]
    return body, directives


def _render_turn(turn: dict[str, Any], *, show_round: bool = False) -> list[str]:
    """One dialogue turn as prompt lines (``- trainee: …`` / ``- client: …``).

    The client line carries the cognitive state captured with that reply: the
    emotions it was spoken with, plus the ``AutomaticThought`` stored beside it
    (the client's private reading of what the trainee just said). With
    *show_round* the turn's graph ``turn_index`` becomes a ``Round N:`` header —
    the recalled section interleaves non-adjacent turns, so it needs to say
    where each one came from; the recent-dialogue section is already
    contiguous and in order, so it omits it.

    In-band directives (see :data:`INTERRUPTED_CLIENT_NOTE`) are hoisted onto
    their own line after the message, so the ``[emotions: …]`` tag stays
    attached to what the client actually said rather than to the note.
    """
    lines: list[str] = []

    if show_round:
        index = turn.get("turn_index")
        lines.append(f"Round {index if index is not None else '?'}:")

    trainee, trainee_notes = _split_directives(turn.get("trainee") or "")
    if trainee:
        lines.append(f"- trainee: {trainee}")
    lines.extend(f"  {note}" for note in trainee_notes)

    client, client_notes = _split_directives(turn.get("client") or "")
    if client:
        tags = ""
        emotions = turn.get("emotions") or []
        if emotions:
            tags += f" [emotions: {', '.join(emotions)}]"
        thought = str(turn.get("thought") or "").strip()
        if thought:
            tags += f" [thought: {thought}]"
        lines.append(f"- client: {client}{tags}")
    lines.extend(f"  {note}" for note in client_notes)

    return lines


def _as_list(value: Any) -> list[Any]:
    """Normalise a response field to a list (CogniLLM may emit a str or list)."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item not in (None, "")]
    return [value]


def _cue_list(value: Any) -> list[str]:
    """Normalise a stored ``cues`` value to a clean string list.

    Handles the array stored today, the legacy empty string, and a
    comma-separated string.
    """
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts: list[Any] = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = [value]
    return [str(part).strip() for part in parts if str(part).strip()]


def response_to_graph_state(response: dict[str, Any] | None) -> dict[str, Any]:
    """Map a CogniLLM response onto the graph's per-turn state fields.

    CogniLLM returns ``automatic_thoughts`` / ``emotions`` / ``behaviors`` /
    ``openness`` plus an ``action`` dict; the graph's ``finalize_turn`` expects
    ``automatic_thought`` / ``dialogue_response`` / ``emotions`` /
    ``coping_mode_activated``. This is the only place that translation lives.

    ``coping_mode_activated`` is still populated so the vendored
    ``finalize_turn`` contract stays intact, but coping modes are retired: the
    backend's fuzzy matcher finds no mode to link, so no ``ACTIVATES_MODE`` edge is
    ever drawn.
    """
    response = response or {}
    action = response.get("action")
    action = action if isinstance(action, dict) else {}

    behaviours = _as_list(response.get("behaviors"))
    spoken = action.get("message") or action.get("behaviour") or ""

    return {
        "automatic_thought": str(response.get("automatic_thoughts") or ""),
        "dialogue_response": str(spoken or ""),
        "emotions": [str(emotion) for emotion in _as_list(response.get("emotions"))],
        "coping_mode_activated": str(behaviours[0]) if behaviours else "",
    }


def _load() -> SimpleNamespace:
    """Import the copied backend for this call.

    Kept lazy on purpose: ``numpy`` and the ``neo4j`` driver (and, indirectly,
    the `.env` that points at the container) are only needed when the graph is
    actually read.
    """
    import numpy as np

    from .neo4j_db import config, db, embeddings, pipeline, schema

    return SimpleNamespace(
        np=np, config=config, db=db, embeddings=embeddings, pipeline=pipeline, schema=schema
    )


class Neo4jContextSource:
    """Read-only bridge between memgraph and the ``neo4j-db`` container.

    A single instance is cheap; it holds no connection of its own (the driver
    is a module-level singleton inside ``neo4j_db.db``).
    """

    def __init__(self, profile_name: str = "") -> None:
        self.profile_name = (profile_name or "").strip()

    # ── availability ────────────────────────────────────────────────

    @staticmethod
    def available() -> bool:
        """True when the driver is importable and the container answers."""
        try:
            backend = _load()
            backend.db.get_driver().verify_connectivity()
            return True
        except Exception:
            return False

    @staticmethod
    def uri() -> str:
        """Configured bolt URI (for diagnostics/logging); ``""`` if unavailable."""
        try:
            return _load().config.NEO4J_URI
        except Exception:
            return ""

    @staticmethod
    def close() -> None:
        """Release the shared driver (call on shutdown)."""
        try:
            _load().db.close()
        except Exception:
            pass

    # ── personas ────────────────────────────────────────────────────

    def personas(self) -> list[dict[str, Any]]:
        """Every ``ClientProfile`` seeded in the graph."""
        return _load().pipeline.list_profiles()

    def resolve_persona(self, profile_name: str = "") -> str:
        """Pick which graph persona to read.

        Matches ``profile_name`` case-insensitively. When it is empty (or does
        not match) and the graph holds exactly one persona, that one is used.

        Raises:
            ValueError: if the graph is empty, the name is unknown among
                several personas, or the name is missing among several.
        """
        wanted = (profile_name or self.profile_name).strip()
        names = [p["name"] for p in self.personas()]
        if not names:
            raise ValueError(
                "neo4j-db graph has no ClientProfile — seed it first "
                "(python -m backend.lib.core.context.neo4j_db.seed from the memgraph root)"
            )
        if wanted:
            for name in names:
                if name.lower() == wanted.lower():
                    return name
        if len(names) == 1:
            return names[0]
        known = ", ".join(names)
        if wanted:
            raise ValueError(f"no persona {wanted!r} in graph; known: {known}")
        raise ValueError(f"multiple personas in graph ({known}); profile_name required")

    def persona_state(self, profile_name: str = "") -> dict[str, Any]:
        """The persona's static schema: background and beliefs."""
        name = self.resolve_persona(profile_name)
        backend = _load()
        rows = backend.db.run(
            """
            MATCH (p:ClientProfile {name: $name})
            OPTIONAL MATCH (p)-[:HOLDS_BELIEF]->(bel:Belief)
            RETURN p.age AS age, p.background AS background,
                   collect(DISTINCT {text: bel.text, intensity: bel.intensity, cues: bel.cues}) AS beliefs
            """,
            name=name,
        )
        row = dict(rows[0]) if rows else {}
        beliefs = [b for b in (row.get("beliefs") or []) if b.get("text")]
        return {
            "name": name,
            "age": row.get("age"),
            "background": row.get("background"),
            "beliefs": beliefs,
        }

    def cues_for(self, profile_name: str = "") -> dict[str, list[str]]:
        """Stored retrieval cues per belief text for one persona (read-only)."""
        name = self.resolve_persona(profile_name)
        backend = _load()
        rows = backend.db.run(
            "MATCH (b:Belief {profile_name: $p}) RETURN b.text AS text, b.cues AS cues",
            p=name,
        )
        return {row["text"]: _cue_list(row["cues"]) for row in rows}

    def clear_embeddings(self, profile_name: str = "") -> int:
        """Drop the stored belief vectors for one persona.

        The next ``sync_persona`` then rebuilds them from scratch (it treats a
        missing embedding as stale). Used to repair vectors left behind by an
        older encoder — e.g. after changing the embedding model or the
        ``text + cues`` format — where the text/cues still look unchanged so a
        normal sync would skip them. Returns the number of beliefs cleared.
        """
        name = self.resolve_persona(profile_name)
        backend = _load()
        rows = backend.db.run(
            """
            MATCH (b:Belief {profile_name: $p})
            SET b.embedding = null
            RETURN count(b) AS cleared
            """,
            p=name,
        )
        return rows[0]["cleared"] if rows else 0

    # ── per-message relevance ───────────────────────────────────────

    def match_beliefs(
        self,
        user_message: str,
        profile_name: str = "",
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Score *user_message* against the persona's beliefs (read-only).

        Uses the same weighted max-pooled cosine scoring as the pipeline, but
        writes **no** ``TRIGGERS`` edge and needs no session/turn. ``threshold``
        / ``top_k`` default to the configured ``BELIEF_MATCH_*`` values.
        """
        name = self.resolve_persona(profile_name)
        backend = _load()
        rows = backend.db.run(
            """
            MATCH (:ClientProfile {name: $profile})-[:HOLDS_BELIEF]->(b:Belief)
            WHERE b.embedding IS NOT NULL
            RETURN b.text AS text, b.intensity AS intensity, b.embedding AS embedding
            """,
            profile=name,
        )
        if not rows or not user_message.strip():
            return {"beliefs": []}

        query_vectors = backend.embeddings.embed([user_message])
        belief_matrix = backend.np.array(
            [r["embedding"] for r in rows], dtype=backend.np.float32
        )
        weights = backend.np.array(
            [backend.embeddings.WEIGHT_CURRENT_TRAINEE], dtype=backend.np.float32
        )
        scores = backend.embeddings.score_beliefs(query_vectors, weights, belief_matrix)
        hits = backend.embeddings.select_matches(scores, threshold=threshold, top_k=top_k)

        beliefs = [
            {
                "text": rows[idx]["text"],
                "intensity": rows[idx]["intensity"],
                "similarity": round(float(sim), 4),
            }
            for idx, sim in hits
        ]

        return {"beliefs": beliefs}

    # ── session history ─────────────────────────────────────────────

    def recent_turns(self, session_id: str, limit: int = DEFAULT_HISTORY_LIMIT) -> list[dict[str, Any]]:
        """The last *limit* turns of *session_id*, oldest first (read-only).

        Each turn also carries the ``AutomaticThought`` written beside its
        ``ClientUtterance``, so the CONTEXT can show what the client was
        privately thinking when it said that line, not just how it felt.
        """
        if not session_id:
            return []
        backend = _load()
        rows = backend.db.run(
            """
            MATCH (s:Session {session_id: $sid})-[:HAS_TURN]->(t:DialogueTurn)
            OPTIONAL MATCH (t)-[:CONTAINS_INPUT]->(in:TraineeUtterance)
            OPTIONAL MATCH (t)-[:CONTAINS_OUTPUT]->(out:ClientUtterance)
            WITH t, in, out ORDER BY t.turn_index DESC
            LIMIT $limit
            OPTIONAL MATCH (t)-[:EVOKES_THOUGHT]->(at:AutomaticThought)
            WITH t, in, out, at ORDER BY t.turn_index ASC
            RETURN t.turn_index AS idx, in.text AS trainee, out.text AS client,
                   out.emotions AS emotions, at.text AS thought
            """,
            sid=session_id,
            limit=int(limit),
        )
        return [
            {
                "turn_index": row["idx"],
                "trainee": row["trainee"],
                "client": row["client"],
                "emotions": list(row["emotions"] or []),
                "thought": str(row["thought"] or ""),
            }
            for row in rows
            if row["trainee"] or row["client"]
        ]

    def _open_turn(self, session_id: str) -> dict[str, Any] | None:
        """The turn currently awaiting a client reply (mirrors pipeline._current_turn)."""
        backend = _load()
        rows = backend.db.run(
            """
            MATCH (s:Session {session_id: $sid})-[:HAS_TURN]->(t:DialogueTurn)
            WHERE NOT (t)-[:CONTAINS_OUTPUT]->(:ClientUtterance)
            RETURN t.turn_id AS turn_id, t.turn_index AS turn_index
            ORDER BY t.turn_index ASC
            LIMIT 1
            """,
            sid=session_id,
        )
        return dict(rows[0]) if rows else None

    def _ensure_session(self, session_id: str, persona: str, user_id: str = "trainee") -> None:
        """Create the graph ``Session`` + turn 0, keyed by the *caller's* id.

        Deliberately not ``pipeline.create_session``: that helper mints its own
        ``uuid4`` and returns it, which would leave the graph session keyed by
        an id nobody else knows and make the history unreadable for this chat.
        The Cypher mirrors the helper's, minus the generated id.
        """
        backend = _load()
        now = datetime.now(timezone.utc).isoformat()
        backend.db.run(
            """
            MATCH (p:ClientProfile {name: $profile})
            CREATE (s:Session {
                session_id: $sid, user_id: $uid, timestamp: $ts, profile_name: $profile
            })
            CREATE (t:DialogueTurn {turn_id: $tid, turn_index: 0, timestamp: $ts})
            CREATE (s)-[:HAS_TURN]->(t)
            CREATE (s)-[:WITH_CLIENT]->(p)
            """,
            profile=persona,
            sid=session_id,
            uid=user_id,
            ts=now,
            tid=str(uuid.uuid4()),
        )

    # ── persona sync (write path) ───────────────────────────────────

    def sync_persona(self, persona: dict[str, Any]) -> dict[str, Any]:
        """Upsert one persona into the graph, idempotently.

        Creates/updates the ``ClientProfile`` plus its ``Belief`` nodes and the
        ``HOLDS_BELIEF`` edges, then deletes whatever this profile no longer
        declares. Coping modes were retired, so any ``CopingMode`` node for this
        profile is swept away too. Called at session start so the graph always
        mirrors the memgraph profile being played — see :mod:`.profile_sync` for
        the profile.yaml → ontology translation.

        Retrieval cues (``belief.cues``) are appended to the text before
        embedding — the text plus a ``Related themes: …`` line — so a topical
        trainee message lands closer to the right belief. A persona that
        supplies no cues inherits the ones already in the graph, so a
        session-start sync never erases generated cues. Embeddings are
        (re)computed only for new beliefs or when a cue list changed, so
        re-syncing an unchanged profile costs no embedding calls.

        Args:
            persona: ``{name, age, background, beliefs[{text, intensity, cues}]}``
                where ``cues`` is an optional list of short retrieval hints.

        Returns:
            Counts, for logging: ``{profile, beliefs, beliefs_embedded}``.
        """
        name = str(persona.get("name") or "").strip()
        if not name:
            raise ValueError("persona needs a name")
        backend = _load()
        db = backend.db

        # Constraints are idempotent and the standalone stack no longer seeds
        # its own personas, so this is the only schema bootstrap left.
        backend.schema.init_schema()

        db.run(
            """
            MERGE (p:ClientProfile {name: $name})
            SET p.age = $age, p.background = $background
            """,
            name=name,
            age=persona.get("age"),
            background=str(persona.get("background") or ""),
        )

        # ── beliefs ───────────────────────────────────────────────
        beliefs = [
            belief
            for belief in (persona.get("beliefs") or [])
            if str(belief.get("text") or "").strip()
        ]
        belief_keys = {
            backend.schema.scoped_key(name, str(belief["text"])): belief
            for belief in beliefs
        }

        # Retrieval cues are enrichment, not profile data: a persona that
        # supplies none inherits whatever the graph already holds, so a
        # session-start sync never wipes generated cues.
        stored = {
            row["key"]: {
                "cues": _cue_list(row["cues"]),
                "has_embedding": bool(row["has_embedding"]),
            }
            for row in db.run(
                """
                MATCH (b:Belief {profile_name: $p})
                RETURN b.belief_key AS key, b.cues AS cues,
                       b.embedding IS NOT NULL AS has_embedding
                """,
                p=name,
            )
        }

        resolved: dict[str, dict[str, Any]] = {}
        for key, belief in belief_keys.items():
            previous = stored.get(key) or {}
            resolved[key] = {
                "text": str(belief["text"]),
                "intensity": float(belief.get("intensity") or 5.0),
                "cues": _cue_list(belief.get("cues")) or previous.get("cues") or [],
            }

        # Cues are baked into the vector, so a changed cue list means the stored
        # embedding is stale; embed only those (and brand-new beliefs).
        stale = [
            key
            for key, item in resolved.items()
            if key not in stored
            or stored[key]["cues"] != item["cues"]
            or not stored[key]["has_embedding"]
        ]
        texts_to_embed = [
            backend.embeddings.belief_match_text(
                resolved[key]["text"], ", ".join(resolved[key]["cues"])
            )
            for key in stale
        ]
        vectors = backend.embeddings.embed(texts_to_embed) if stale else None
        vector_for = dict(zip(stale, vectors)) if vectors is not None else {}

        for key, item in resolved.items():
            db.run(
                """
                MATCH (p:ClientProfile {name: $pname})
                MERGE (b:Belief {belief_key: $key})
                SET b.profile_name = $pname,
                    b.text = $text,
                    b.intensity = $intensity,
                    b.cues = $cues,
                    b.embedding = CASE WHEN $embedding IS NULL
                                       THEN b.embedding ELSE $embedding END
                MERGE (p)-[:HOLDS_BELIEF]->(b)
                """,
                pname=name,
                key=key,
                text=item["text"],
                intensity=item["intensity"],
                cues=item["cues"],
                embedding=(vector_for[key].tolist() if key in vector_for else None),
            )
        db.run(
            """
            MATCH (b:Belief {profile_name: $pname})
            WHERE NOT b.belief_key IN $keys
            DETACH DELETE b
            """,
            pname=name,
            keys=list(belief_keys),
        )

        # ── coping modes (retired) ────────────────────────────────
        # No longer mirrored from profile.yaml; sweep away whatever an earlier
        # sync created so the graph holds beliefs + dialogue only. The vendored
        # pipeline degrades cleanly without them: `available_coping_modes` comes
        # back empty and no ACTIVATES_MODE edge is ever drawn.
        db.run(
            "MATCH (cm:CopingMode {profile_name: $pname}) DETACH DELETE cm",
            pname=name,
        )

        return {
            "profile": name,
            "beliefs": len(belief_keys),
            "beliefs_embedded": len(stale),
            "beliefs_with_cues": sum(1 for item in resolved.values() if item["cues"]),
        }

    def prune_profiles(self, keep: list[str]) -> list[str]:
        """Delete graph personas that are not in *keep*. Returns their names.

        Used by :mod:`.profile_sync` after a full sync, so the graph cannot
        keep serving a client that no longer exists as a memgraph profile.
        """
        backend = _load()
        names = [str(name) for name in keep]
        rows = backend.db.run(
            "MATCH (p:ClientProfile) WHERE NOT p.name IN $keep RETURN p.name AS name",
            keep=names,
        )
        removed = [row["name"] for row in rows]
        if not removed:
            return []

        backend.db.run(
            "MATCH (p:ClientProfile) WHERE NOT p.name IN $keep DETACH DELETE p", keep=names
        )
        # Beliefs and coping modes are keyed by `profile_name`, not by an edge,
        # so they need pruning in their own right (this also sweeps up nodes
        # left over from the retired JSON personas, which have no profile_name).
        backend.db.run(
            """
            MATCH (b:Belief)
            WHERE b.profile_name IS NULL OR NOT b.profile_name IN $keep
            DETACH DELETE b
            """,
            keep=names,
        )
        # Coping modes are retired outright: no profile may have any left.
        backend.db.run("MATCH (cm:CopingMode) DETACH DELETE cm")
        return removed

    # ── write path ──────────────────────────────────────────────────

    def record_turn(
        self,
        *,
        session_id: str,
        profile_name: str = "",
        trainee_message: str = "",
        state: dict[str, Any] | None = None,
        user_id: str = "trainee",
        interrupted: bool = False,
        follows_interruption: bool = False,
    ) -> dict[str, Any]:
        """Fold one finished turn back into the graph (WRITE path).

        Meant to run *after* the reply has been delivered and never on the
        request's critical path — the caller (the orchestrator) dispatches it as
        a background task, so graph latency is invisible to the trainee.

        Sequence, reusing the vendored pipeline so the schema stays in one place:

        1. Create the ``Session`` (bound to the persona) on first use.
        2. ``pipeline.ingest`` — ``TraineeUtterance`` + ``TRIGGERS`` edges via
           bge-m3 scoring. Skipped when *trainee_message* is empty, which is how
           a client-initiated (idle-timer) turn is recorded: the client spoke,
           the trainee did not, so there is nothing to score.
        3. ``pipeline.finalize_turn`` — ``AutomaticThought``, ``ClientUtterance``
           (with emotions), then opens the next turn.

        Interruption is recorded in-band rather than by dropping the turn
        (see :data:`INTERRUPTED_CLIENT_NOTE` / :data:`INTERRUPTED_TRAINEE_NOTE`):

        * *interrupted* — this turn's reply was withheld from the social worker,
          so the stored ``ClientUtterance`` carries the "not seen" directive;
        * *follows_interruption* — this trainee message is the one that
          interrupted it, so its ``TraineeUtterance`` carries the matching note.

        The directive is appended AFTER belief scoring, so the bge-m3 match and
        the ``TRIGGERS`` edges still see only the social worker's actual words.

        Raises:
            ValueError: no persona could be resolved, or *session_id* is empty.
            RuntimeError: the session has no open turn.
        """
        if not session_id:
            raise ValueError("record_turn requires a session_id")
        persona = self.resolve_persona(profile_name)
        backend = _load()

        if not backend.db.run(
            "MATCH (s:Session {session_id: $sid}) RETURN s.session_id AS id",
            sid=session_id,
        ):
            self._ensure_session(session_id, persona, user_id=user_id)

        turn = self._open_turn(session_id)
        if turn is None:
            raise RuntimeError(f"session {session_id!r} has no open DialogueTurn")

        graph_state = response_to_graph_state(state)
        if interrupted:
            graph_state["dialogue_response"] = _with_directive(
                graph_state["dialogue_response"], INTERRUPTED_CLIENT_NOTE
            )

        triggered: list[dict[str, Any]] = []
        if trainee_message.strip():
            # Score on the clean words…
            triggered = backend.pipeline.ingest(session_id, turn["turn_id"], trainee_message)
            if follows_interruption:
                # …then annotate the stored utterance, so the directive reaches
                # the CONTEXT without polluting the embedding / belief match.
                backend.db.run(
                    """
                    MATCH (t:DialogueTurn {turn_id: $tid})
                          -[:CONTAINS_INPUT]->(u:TraineeUtterance)
                    SET u.text = $text
                    """,
                    tid=turn["turn_id"],
                    text=_with_directive(trainee_message, INTERRUPTED_TRAINEE_NOTE),
                )

        backend.pipeline.finalize_turn(turn, graph_state, persona)
        return {"turn_index": turn["turn_index"], "triggered_beliefs": triggered}

    # ── belief-overlap recall ───────────────────────────────

    def related_rounds(
        self,
        session_id: str,
        belief_texts: list[str],
        exclude_indices: list[int] | None = None,
        limit: int = DEFAULT_RELATED_LIMIT,
    ) -> list[dict[str, Any]]:
        """Earlier turns of *session_id* that activated one of *belief_texts*.

        The belief-overlap recall: same-session turns whose ``TraineeUtterance``
        has a ``TRIGGERS`` edge to a belief the current message also activates.
        Each hit carries its ``turn_index`` (rendered as ``Round N``), its
        emotions and its ``AutomaticThought`` — which is what makes it "the same
        pattern, last time".

        Turns already covered by the recent-dialogue window are excluded via
        *exclude_indices* so sections 1 and 3 never repeat each other. Read-only.
        """
        if not session_id or not belief_texts:
            return []
        backend = _load()
        rows = backend.db.run(
            """
            MATCH (s:Session {session_id: $sid})-[:HAS_TURN]->(t:DialogueTurn)
            WHERE NOT t.turn_index IN $exclude
            MATCH (t)-[:CONTAINS_INPUT]->(in:TraineeUtterance)-[:TRIGGERS]->(b:Belief)
            WHERE b.text IN $beliefs
            MATCH (t)-[:CONTAINS_OUTPUT]->(out:ClientUtterance)
            WITH DISTINCT t, in, out ORDER BY t.turn_index DESC
            LIMIT $limit
            OPTIONAL MATCH (t)-[:EVOKES_THOUGHT]->(at:AutomaticThought)
            WITH t, in, out, at ORDER BY t.turn_index ASC
            RETURN t.turn_index AS idx, in.text AS trainee, out.text AS client,
                   out.emotions AS emotions, at.text AS thought
            """,
            sid=session_id,
            beliefs=[str(text) for text in belief_texts],
            exclude=[int(idx) for idx in (exclude_indices or [])],
            limit=int(limit),
        )
        return [
            {
                "turn_index": row["idx"],
                "trainee": row["trainee"],
                "client": row["client"],
                "emotions": list(row["emotions"] or []),
                "thought": str(row["thought"] or ""),
            }
            for row in rows
            if row["trainee"] or row["client"]
        ]

    # ── rendering ───────────────────────────────────────────────────

    def render_context(
        self,
        user_message: str = "",
        profile_name: str = "",
        session_id: str = "",
        include_full_schema: bool = False,
        sections: dict[str, bool] | None = None,
        history_limit: int | None = None,
        related_limit: int | None = None,
    ) -> str:
        """Build the injectable CONTEXT block for one turn.

        Sections are emitted in this order:

        1. **Beliefs activated** — the persona beliefs the latest message scores
           closest to (read-only; no ``TRIGGERS`` edge is written for it).
        2. **Related past conversation** — earlier turns that activated one of
           those same beliefs (belief-overlap recall). Each one is introduced by
           a ``Round N:`` header taken from its ``DialogueTurn.turn_index``,
           because these turns are not adjacent and their position matters.
        3. **Recent dialogue** — the last *history_limit* turns of *session_id*,
           read back from the graph. Deliberately LAST: it ends the block right
           before the current user turn, so the freshest exchange sits closest
           to the message being answered.

        There is deliberately no persona header or background line: the system
        prompt already carries the full persona description built from
        ``profile.yaml``, so repeating it here would only burn prompt tokens.

        Args:
            user_message: the message being answered; drives section 1.
            profile_name: graph persona to read (see :meth:`resolve_persona`).
            session_id: graph session whose turns to replay. Empty = no history.
            include_full_schema: also list every stored belief.
            sections: which sections to emit, keyed ``activated_beliefs`` /
                ``related_past`` / ``recent_dialogue`` (``config.yaml`` →
                ``context.memory.sections``). ``None`` or a missing key means
                enabled; a disabled section is neither rendered nor queried.
            history_limit: how many past turns go into section 3.
            related_limit: how many recalled turns go into section 2.

        Returns:
            A plain-text block, or ``""`` when every section is switched off —
            which the generation layer reads as "no CONTEXT supplied".
        """
        flags = sections or {}
        want_beliefs = flags.get("activated_beliefs", True)
        want_related = flags.get("related_past", True)
        want_history = flags.get("recent_dialogue", True)

        history_cap = DEFAULT_HISTORY_LIMIT if history_limit is None else int(history_limit)
        related_cap = DEFAULT_RELATED_LIMIT if related_limit is None else int(related_limit)

        name = self.resolve_persona(profile_name)

        # A disabled section skips its queries too, not just its output.
        matched = (
            self.match_beliefs(user_message, name)
            if want_beliefs and user_message.strip()
            else {"beliefs": []}
        )
        history = self.recent_turns(session_id, limit=history_cap) if want_history else []
        related = (
            self.related_rounds(
                session_id,
                [belief["text"] for belief in matched["beliefs"]],
                exclude_indices=[turn["turn_index"] for turn in history],
                limit=related_cap,
            )
            if want_related
            else []
        )

        lines: list[str] = []

        # 1. Beliefs activated by the message being answered.
        if want_beliefs:
            if matched["beliefs"]:
                lines.append("Beliefs activated by the latest message:")
                for belief in matched["beliefs"]:
                    lines.append(
                        f"- {belief['text']} "
                        f"(similarity {belief['similarity']}, intensity {belief['intensity']})"
                    )
            elif user_message.strip():
                lines.append("Beliefs activated by the latest message: none above threshold.")

        # 2. Same-pattern recall from earlier in the session.
        if related:
            lines.append(
                "Related past conversation (shares an activated belief, oldest first):"
            )
            for turn in related:
                lines.extend(_render_turn(turn, show_round=True))

        # 3. Recent dialogue — last, i.e. closest to the message being answered.
        if want_history:
            if history:
                lines.append("Recent dialogue (oldest first):")
                for turn in history:
                    lines.extend(_render_turn(turn))
            else:
                lines.append("Recent dialogue (oldest first): none yet.")

        if include_full_schema:
            lines.append("Full stored belief schema:")
            for belief in self.persona_state(name)["beliefs"]:
                lines.append(f"- {belief['text']} (intensity {belief['intensity']})")

        return "\n".join(lines)
