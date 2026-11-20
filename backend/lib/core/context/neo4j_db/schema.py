"""Idempotent constraints for the streamlined graph ontology.

Beliefs and coping modes are unique per client, not globally: two profiles
may use the same belief text or mode name and still own separate nodes.
Neo4j Community has no node-key constraint, so the pair is stored as a
single unique property (`belief_key` / `mode_key`).
"""

from .db import run

_KEY_SEP = "\x1f"


def scoped_key(profile_name: str, value: str) -> str:
    """Unique id for a persona-owned belief text or coping-mode name."""
    return f"{profile_name}{_KEY_SEP}{value}"

STATEMENTS = [
    "CREATE CONSTRAINT client_profile_name IF NOT EXISTS FOR (n:ClientProfile) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (n:Session) REQUIRE n.session_id IS UNIQUE",
    "CREATE CONSTRAINT dialogue_turn_id IF NOT EXISTS FOR (n:DialogueTurn) REQUIRE n.turn_id IS UNIQUE",
    # Global uniqueness blocked a second persona from holding its own copy.
    "DROP CONSTRAINT belief_text IF EXISTS",
    "DROP CONSTRAINT coping_mode_name IF EXISTS",
    # Retired labels from the concept-extractor schema.
    "DROP CONSTRAINT core_belief_text IF EXISTS",
    "DROP CONSTRAINT intermediate_belief_text IF EXISTS",
    "DROP CONSTRAINT concept_name IF EXISTS",
    "DROP INDEX concept_embedding IF EXISTS",
]

# Community Edition: uniqueness is one property. The key encodes (profile, text/name).
SCOPED_CONSTRAINTS = [
    "CREATE CONSTRAINT belief_key IF NOT EXISTS FOR (n:Belief) REQUIRE n.belief_key IS UNIQUE",
    "CREATE CONSTRAINT coping_key IF NOT EXISTS FOR (n:CopingMode) REQUIRE n.mode_key IS UNIQUE",
    "CREATE INDEX belief_profile IF NOT EXISTS FOR (n:Belief) ON (n.profile_name)",
    "CREATE INDEX coping_profile IF NOT EXISTS FOR (n:CopingMode) ON (n.profile_name)",
]


def _drop_unscoped_persona_nodes() -> None:
    """Remove Belief and CopingMode nodes from the single-persona schema.

    Those nodes have no profile_name, so they predate per-profile keys.
    DETACH DELETE also drops their TRIGGERS / ACTIVATES_MODE edges.
    """
    run("MATCH (b:Belief) WHERE b.profile_name IS NULL DETACH DELETE b")
    run("MATCH (cm:CopingMode) WHERE cm.profile_name IS NULL DETACH DELETE cm")


def init_schema() -> None:
    for stmt in STATEMENTS:
        try:
            run(stmt)
        except Exception:
            if stmt.strip().upper().startswith("DROP"):
                continue
            raise
    _drop_unscoped_persona_nodes()
    for stmt in SCOPED_CONSTRAINTS:
        run(stmt)


if __name__ == "__main__":
    init_schema()
    print("Schema initialized.")
