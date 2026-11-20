"""End-to-end turn pipeline (spec §3 Steps 1-4)."""

import uuid
from datetime import datetime, timezone

import numpy as np

from . import config, embeddings, graphrag, llm
from .db import run


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_profiles() -> list[dict]:
    rows = run(
        """
        MATCH (p:ClientProfile)
        RETURN p.name AS name, p.age AS age, p.background AS background
        ORDER BY p.name
        """
    )
    return [dict(r) for r in rows]


def load_persona(profile_name: str) -> dict:
    rows = run(
        """
        MATCH (p:ClientProfile {name: $name})
        RETURN p.name AS name, p.age AS age, p.background AS background
        """,
        name=profile_name,
    )
    if not rows:
        raise RuntimeError(
            f"No ClientProfile named {profile_name!r} — that profile has not been "
            f"synced into this graph."
        )
    return {"profile": dict(rows[0])}


def _resolve_profile(profile_name: str | None) -> str:
    profiles = list_profiles()
    if not profiles:
        raise RuntimeError("No ClientProfile in this graph — no profile has been synced yet.")
    names = [p["name"] for p in profiles]
    if profile_name:
        if profile_name not in names:
            known = ", ".join(names)
            raise RuntimeError(f"Unknown client profile {profile_name!r}. Known: {known}.")
        return profile_name
    if len(names) == 1:
        return names[0]
    known = ", ".join(names)
    raise RuntimeError(f"profile_name is required when the graph holds more than one client. Known: {known}.")


def session_profile(session_id: str) -> str:
    rows = run(
        """
        MATCH (s:Session {session_id: $sid})-[:WITH_CLIENT]->(p:ClientProfile)
        RETURN p.name AS name
        """,
        sid=session_id,
    )
    if not rows:
        raise RuntimeError(f"Session {session_id} is not bound to a client profile.")
    return rows[0]["name"]


def create_session(user_id: str = "trainee", profile_name: str | None = None) -> str:
    """Create a Session bound to one ClientProfile, plus DialogueTurn(0)."""
    profile_name = _resolve_profile(profile_name)
    session_id = str(uuid.uuid4())
    run(
        """
        MATCH (p:ClientProfile {name: $profile})
        CREATE (s:Session {session_id: $sid, user_id: $uid, timestamp: $ts, profile_name: $profile})
        CREATE (t:DialogueTurn {turn_id: $tid, turn_index: 0, timestamp: $ts})
        CREATE (s)-[:HAS_TURN]->(t)
        CREATE (s)-[:WITH_CLIENT]->(p)
        """,
        profile=profile_name,
        sid=session_id,
        uid=user_id,
        ts=_now(),
        tid=str(uuid.uuid4()),
    )
    return session_id


def _current_turn(session_id: str) -> dict:
    """The open turn: highest index with no CONTAINS_OUTPUT yet."""
    rows = run(
        """
        MATCH (s:Session {session_id: $sid})-[:HAS_TURN]->(t:DialogueTurn)
        WHERE NOT (t)-[:CONTAINS_OUTPUT]->(:ClientUtterance)
        RETURN t.turn_id AS turn_id, t.turn_index AS turn_index
        ORDER BY t.turn_index ASC
        LIMIT 1
        """,
        sid=session_id,
    )
    if not rows:
        raise RuntimeError(f"Session {session_id} has no open DialogueTurn (unknown session?).")
    return dict(rows[0])


def _recent_window(session_id: str, current_turn_id: str) -> tuple[list[str], np.ndarray]:
    """Current social-worker text is passed separately. This returns prior
    trainee and client utterances from the last p completed turns."""
    rows = run(
        """
        MATCH (s:Session {session_id: $sid})-[:HAS_TURN]->(t:DialogueTurn)
        WHERE t.turn_id <> $tid
          AND (t)-[:CONTAINS_INPUT]->(:TraineeUtterance)
        OPTIONAL MATCH (t)-[:CONTAINS_INPUT]->(in:TraineeUtterance)
        OPTIONAL MATCH (t)-[:CONTAINS_OUTPUT]->(out:ClientUtterance)
        WITH t, in, out ORDER BY t.turn_index DESC
        LIMIT $p
        RETURN in.text AS trainee, out.text AS client
        """,
        sid=session_id,
        tid=current_turn_id,
        p=config.BELIEF_CONTEXT_TURNS,
    )
    texts, weights = [], []
    for r in rows:
        if r["trainee"]:
            texts.append(r["trainee"])
            weights.append(embeddings.WEIGHT_PRIOR_TRAINEE)
        if r["client"]:
            texts.append(r["client"])
            weights.append(embeddings.WEIGHT_PRIOR_CLIENT)
    return texts, np.array(weights, dtype=np.float32)


# --- Step 1: Real-Time Input Processing -------------------------------------

def ingest(session_id: str, turn_id: str, user_text: str) -> list[dict]:
    """Create TraineeUtterance, score the dialogue window against this
    session's persona Beliefs, and write TRIGGERS edges for hits above threshold.

    Returns the list of {text, similarity} beliefs activated this turn.
    """
    run(
        """
        MATCH (t:DialogueTurn {turn_id: $tid})
        CREATE (u:TraineeUtterance {utterance_id: $uid, text: $text, token_count: $tokens})
        CREATE (t)-[:CONTAINS_INPUT]->(u)
        """,
        tid=turn_id,
        uid=str(uuid.uuid4()),
        text=user_text,
        tokens=len(user_text.split()),
    )

    profile_name = session_profile(session_id)
    beliefs = run(
        """
        MATCH (:ClientProfile {name: $profile})-[:HOLDS_BELIEF]->(b:Belief)
        WHERE b.embedding IS NOT NULL
        RETURN b.text AS text, b.embedding AS embedding
        """,
        profile=profile_name,
    )
    if not beliefs:
        return []

    prior_texts, prior_weights = _recent_window(session_id, turn_id)
    query_texts = [user_text, *prior_texts]
    query_weights = np.concatenate(
        [
            np.array([embeddings.WEIGHT_CURRENT_TRAINEE], dtype=np.float32),
            prior_weights,
        ]
    )
    query_vectors = embeddings.embed(query_texts)
    belief_matrix = np.array([r["embedding"] for r in beliefs], dtype=np.float32)
    scores = embeddings.score_beliefs(query_vectors, query_weights, belief_matrix)
    hits = embeddings.select_matches(scores)

    activated: list[dict] = []
    for idx, sim in hits:
        text = beliefs[idx]["text"]
        run(
            """
            MATCH (t:DialogueTurn {turn_id: $tid})-[:CONTAINS_INPUT]->(u:TraineeUtterance)
            MATCH (b:Belief {profile_name: $profile, text: $belief})
            MERGE (u)-[rel:TRIGGERS]->(b)
            SET rel.similarity = $sim, rel.weight = coalesce(rel.weight, 1.0)
            """,
            tid=turn_id,
            profile=profile_name,
            belief=text,
            sim=float(sim),
        )
        activated.append({"text": text, "similarity": round(float(sim), 4)})
    return activated


# --- Step 4: State Finalization & Graph Write --------------------------------

def _match_coping_mode(profile_name: str, requested: str) -> str | None:
    rows = run(
        "MATCH (cm:CopingMode {profile_name: $profile}) RETURN cm.name AS name",
        profile=profile_name,
    )
    names = [r["name"] for r in rows]
    if not names:
        return None
    lowered = requested.strip().lower()
    for n in names:
        if n.lower() == lowered:
            return n
    requested_tokens = set(lowered.split())
    best = max(names, key=lambda n: len(requested_tokens & set(n.lower().split())))
    return best if requested_tokens & set(best.lower().split()) else None


def finalize_turn(turn: dict, state: dict, profile_name: str) -> None:
    """Persist state nodes + dialogue, then open DialogueTurn(N+1)."""
    run(
        """
        MATCH (t:DialogueTurn {turn_id: $tid})
        CREATE (at:AutomaticThought {thought_id: $at_id, text: $thought})
        CREATE (cu:ClientUtterance {
            utterance_id: $cu_id, text: $response, token_count: $tokens, emotions: $emotions
        })
        CREATE (t)-[:EVOKES_THOUGHT]->(at)
        CREATE (t)-[:CONTAINS_OUTPUT]->(cu)
        """,
        tid=turn["turn_id"],
        at_id=str(uuid.uuid4()),
        thought=state["automatic_thought"],
        cu_id=str(uuid.uuid4()),
        response=state["dialogue_response"],
        tokens=len(state["dialogue_response"].split()),
        emotions=state["emotions"],
    )

    mode_name = _match_coping_mode(profile_name, state["coping_mode_activated"])
    if mode_name and mode_name.lower() != "none":
        run(
            """
            MATCH (t:DialogueTurn {turn_id: $tid})-[:CONTAINS_OUTPUT]->(cu:ClientUtterance)
            MATCH (cm:CopingMode {profile_name: $profile, name: $name})
            MERGE (cu)-[:ACTIVATES_MODE]->(cm)
            """,
            tid=turn["turn_id"],
            profile=profile_name,
            name=mode_name,
        )

    run(
        """
        MATCH (t:DialogueTurn {turn_id: $tid})
        CREATE (next:DialogueTurn {turn_id: $next_id, turn_index: $next_idx, timestamp: $ts})
        CREATE (t)-[:FOLLOWED_BY]->(next)
        WITH next
        MATCH (s:Session)-[:HAS_TURN]->(t2:DialogueTurn {turn_id: $tid})
        CREATE (s)-[:HAS_TURN]->(next)
        """,
        tid=turn["turn_id"],
        next_id=str(uuid.uuid4()),
        next_idx=turn["turn_index"] + 1,
        ts=_now(),
    )


# --- Orchestration -----------------------------------------------------------

# def process_turn(session_id: str, input_obj: str) -> dict:
# user_text = input_obj.get("user_message")

def process_turn(session_id: str, user_text: str) -> dict:
    profile_name = session_profile(session_id)
    turn = _current_turn(session_id)
    triggered = ingest(session_id, turn["turn_id"], user_text)  # Step 1
    context = graphrag.retrieve(session_id, turn["turn_id"])    # Step 2
    persona = load_persona(profile_name)
    state = llm.generate(persona, context, user_text)           # Step 3
    finalize_turn(turn, state, profile_name)                    # Step 4

    return {
        "turn_index": turn["turn_index"],
        "dialogue_response": state["dialogue_response"],
        "state": {
            "automatic_thought": state["automatic_thought"],
            "emotions": state["emotions"],
            "coping_mode_activated": state["coping_mode_activated"],
        },
        "triggered_beliefs": triggered,
    }


def get_history(session_id: str) -> list[dict]:
    rows = run(
        """
        MATCH (s:Session {session_id: $sid})-[:HAS_TURN]->(t:DialogueTurn)
        OPTIONAL MATCH (t)-[:CONTAINS_INPUT]->(in:TraineeUtterance)
        OPTIONAL MATCH (t)-[:CONTAINS_OUTPUT]->(out:ClientUtterance)
        OPTIONAL MATCH (t)-[:EVOKES_THOUGHT]->(at:AutomaticThought)
        OPTIONAL MATCH (out)-[:ACTIVATES_MODE]->(cm:CopingMode)
        WITH t, in, out, at, cm ORDER BY t.turn_index ASC
        RETURN t.turn_index AS idx, in.text AS trainee, out.text AS client,
               at.text AS thought, out.emotions AS emotions, cm.name AS coping_mode
        """,
        sid=session_id,
    )
    return [
        {
            "turn_index": r["idx"],
            "trainee": r["trainee"],
            "client": r["client"],
            "state": (
                {
                    "automatic_thought": r["thought"],
                    "emotions": list(r["emotions"] or []),
                    "coping_mode_activated": r["coping_mode"],
                }
                if r["client"] is not None
                else None
            ),
        }
        for r in rows
        if r["trainee"] or r["client"]
    ]
