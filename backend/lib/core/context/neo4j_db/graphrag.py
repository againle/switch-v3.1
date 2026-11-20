"""Step 2: Agentic GraphRAG retrieval.

Builds the serialized context payload for the single-pass generation call:
  1. Sub-graph initialization — last N DialogueTurns with both utterances.
  2. Exploitation — current TraineeUtterance -TRIGGERS-> Belief
     (+FAVORS_COPING-> CopingMode), ranked by similarity then edge weight.
  3. Belief-overlap recall — prior turns that share a triggered Belief, returning
     only that round's emotion adjectives, trainee utterance, and client output.
  4. Memory replay — increments `weight` on traversed TRIGGERS edges.
"""

from . import config
from .db import run


def retrieve(session_id: str, turn_id: str) -> dict:
    turns = run(
        """
        MATCH (s:Session {session_id: $sid})-[:HAS_TURN]->(t:DialogueTurn)
        OPTIONAL MATCH (t)-[:CONTAINS_INPUT]->(in:TraineeUtterance)
        OPTIONAL MATCH (t)-[:CONTAINS_OUTPUT]->(out:ClientUtterance)
        WITH t, in, out ORDER BY t.turn_index DESC
        LIMIT $limit
        RETURN t.turn_index AS idx, in.text AS trainee, out.text AS client
        ORDER BY idx ASC
        """,
        sid=session_id,
        limit=config.CONTEXT_TURNS,
    )
    recent_turns = [
        {"turn_index": r["idx"], "trainee": r["trainee"], "client": r["client"]}
        for r in turns
        if r["trainee"] or r["client"]
    ]

    belief_rows = run(
        """
        MATCH (t:DialogueTurn {turn_id: $tid})-[:CONTAINS_INPUT]->(u:TraineeUtterance)
              -[rel:TRIGGERS]->(b:Belief)
        OPTIONAL MATCH (b)-[:FAVORS_COPING]->(cm:CopingMode)
        RETURN b.text AS belief, b.intensity AS intensity,
               rel.similarity AS similarity, rel.weight AS path_weight,
               collect(DISTINCT cm.name) AS modes
        ORDER BY similarity DESC, path_weight DESC, intensity DESC
        """,
        tid=turn_id,
    )

    triggered, coping_modes = [], []
    seen_modes = set()
    for r in belief_rows:
        triggered.append(
            {
                "text": r["belief"],
                "intensity": r["intensity"],
                "similarity": r["similarity"],
                "path_weight": r["path_weight"],
            }
        )
        for mode in r["modes"]:
            if mode not in seen_modes:
                seen_modes.add(mode)
                coping_modes.append(mode)

    if triggered:
        run(
            """
            MATCH (:DialogueTurn {turn_id: $tid})-[:CONTAINS_INPUT]->(:TraineeUtterance)
                  -[rel:TRIGGERS]->(:Belief)
            SET rel.weight = coalesce(rel.weight, 1.0) + 1.0
            """,
            tid=turn_id,
        )

    same_belief_rows = run(
        """
        MATCH (cur:DialogueTurn {turn_id: $tid})-[:CONTAINS_INPUT]->(:TraineeUtterance)
              -[:TRIGGERS]->(b:Belief)
        MATCH (s:Session {session_id: $sid})-[:HAS_TURN]->(t:DialogueTurn)
        WHERE t.turn_id <> $tid
        MATCH (t)-[:CONTAINS_INPUT]->(in:TraineeUtterance)-[:TRIGGERS]->(b)
        MATCH (t)-[:CONTAINS_OUTPUT]->(out:ClientUtterance)
        WITH DISTINCT t, in, out
        ORDER BY t.turn_index DESC
        LIMIT $limit
        RETURN t.turn_index AS idx, in.text AS trainee, out.text AS client, out.emotions AS emotions
        ORDER BY idx ASC
        """,
        sid=session_id,
        tid=turn_id,
        limit=config.CONTEXT_TURNS,
    )
    same_belief_rounds = [
        {
            "emotions": list(r["emotions"] or []),
            "trainee": r["trainee"],
            "client": r["client"],
        }
        for r in same_belief_rows
    ]

    profile_rows = run(
        """
        MATCH (s:Session {session_id: $sid})-[:WITH_CLIENT]->(p:ClientProfile)
        RETURN p.name AS name
        """,
        sid=session_id,
    )
    profile_name = profile_rows[0]["name"] if profile_rows else None
    all_modes = (
        run(
            """
            MATCH (cm:CopingMode {profile_name: $profile})
            RETURN cm.name AS name, cm.triggers AS triggers, cm.behavior AS behavior
            ORDER BY cm.name
            """,
            profile=profile_name,
        )
        if profile_name
        else []
    )

    return {
        "recent_dialogue": recent_turns,
        "triggered_beliefs": triggered,
        "same_belief_rounds": same_belief_rounds,
        "available_coping_modes": [
            {"name": r["name"], "triggers": r["triggers"], "behavior": r["behavior"]}
            for r in all_modes
        ],
        "suggested_coping_modes": coping_modes,
    }
