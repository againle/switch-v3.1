"""Mirror memgraph profiles into the standalone ``neo4j-db`` graph.

The profile folders under ``backend/profiles/<Name>/`` are the single source of
truth for personas. This module translates one ``profile.yaml`` into the graph
ontology and writes it, so a chat session always finds a ``ClientProfile`` that
mirrors the profile it is playing — same name, same beliefs, same behaviours.

Field mapping (``cognitive_model`` lives under the profile root):

| profile.yaml | graph |
| --- | --- |
| ``name`` (or the folder name) | ``ClientProfile.name`` (unique key) |
| ``short_description`` | ``ClientProfile.background`` |
| age parsed from ``short_description``/``situation`` | ``ClientProfile.age`` |
| ``cognitive_model.core_beliefs`` | ``Belief`` (intensity :data:`CORE_BELIEF_INTENSITY`) |
| ``cognitive_model.intermediate_beliefs`` | ``Belief`` (intensity :data:`INTERMEDIATE_BELIEF_INTENSITY`) |

Coping modes were retired: ``cognitive_model.behaviors`` / ``coping_strategies``
are **not** mirrored into the graph (the ``CopingMode`` label is dead and any
leftover nodes are swept away on sync). Beliefs and dialogue are the whole
model.

**Retrieval cues.** Each ``Belief`` carries a ``cues`` string list — short
keywords/phrases a therapist's message might use when it touches that belief.
Cues are not authored in profile.yaml: ``--cues`` generates them with one
batched LLM call per profile, and they are then baked into the belief's
embedding as ``"<text>\\nRelated themes: <cue, cue, …>"`` (see
``neo4j_db.embeddings.belief_match_text``). That combined vector is what the
trainee's message is compared against, so a topical sentence lands closer to the
right belief. Existing cues are preserved across syncs; only beliefs that have
none (or are selected by ``--force-cues``) are regenerated.

Belief nodes stay profile-scoped (``belief_key`` encodes the profile name), so
two clients may use identical wording without sharing a node.

Usage::

    python -m backend.lib.core.context.profile_sync               # sync, keep cues as-is
    python -m backend.lib.core.context.profile_sync --cues        # + generate missing cues
    python -m backend.lib.core.context.profile_sync --force-cues  # + regenerate every cue
    python -m backend.lib.core.context.profile_sync Anna Grace    # named profiles only
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Intensity given to ``core_beliefs`` — they are the schema-level convictions.
CORE_BELIEF_INTENSITY = 8.0
#: Intensity given to ``intermediate_beliefs`` — the operational conditionals.
INTERMEDIATE_BELIEF_INTENSITY = 6.0

_AGE_RE = re.compile(r"(\d{1,3})\s*[- ]?\s*year[\s-]*old", re.IGNORECASE)


def _clean(value: Any) -> str:
    """Collapse a YAML scalar into a trimmed single-line string."""
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _texts(value: Any) -> list[str]:
    """Normalise a YAML list (or scalar) into clean, deduplicated strings."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = _clean(item)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def extract_age(*sources: Any) -> int | None:
    """Pull an age out of free text like ``"Anna is a 22-year-old female"``."""
    for source in sources:
        match = _AGE_RE.search(_clean(source))
        if match:
            age = int(match.group(1))
            if 0 < age < 130:
                return age
    return None


def build_persona(profile_name: str, profile_data: dict[str, Any] | None) -> dict[str, Any]:
    """Translate a memgraph ``profile.yaml`` dict into the graph persona shape.

    *profile_name* is the profile's identifier (its folder name) and becomes the
    graph ``ClientProfile.name`` — that is what a session looks itself up by, so
    it must match what the session passes down.

    Pure function: no I/O, so the mapping can be inspected/tested on its own.
    """
    data = profile_data or {}
    cognitive = data.get("cognitive_model") or {}
    description = _clean(data.get("short_description"))
    situation = _clean(cognitive.get("situation"))

    beliefs: list[dict[str, Any]] = []
    for text in _texts(cognitive.get("core_beliefs")):
        beliefs.append({"text": text, "intensity": CORE_BELIEF_INTENSITY, "cues": []})
    for text in _texts(cognitive.get("intermediate_beliefs")):
        beliefs.append(
            {"text": text, "intensity": INTERMEDIATE_BELIEF_INTENSITY, "cues": []}
        )

    return {
        "name": profile_name,
        "age": extract_age(description, situation),
        "background": description or situation,
        "beliefs": beliefs,
    }


#: Upper bound on retrieval cues kept per belief.
CUES_PER_BELIEF = 8

_CUE_SYSTEM_PROMPT = (
    "You build retrieval indexes for cognitive-behavioural belief schemas used "
    "in a social-work training simulator.\n\n"
    "For each numbered belief, write 5 to 8 short retrieval cues: words or "
    "1-3 word phrases a therapist's message might contain when it touches that "
    "belief.\n"
    "Mix: (a) the client's own typical wording, (b) concrete situations or "
    "events, (c) emotions, (d) therapy themes such as abandonment, shame, "
    "control or trust.\n\n"
    "Rules: lowercase; 1-3 words; no punctuation; never restate the belief; no "
    "duplicates; do not use the client's name.\n\n"
    'Return ONLY JSON shaped {"cues": [[...], [...], ...]} with exactly one '
    "array per belief, in the same order as the input."
)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model reply."""
    text = text or ""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _request_cues(client: Any, model: str, beliefs: list[dict[str, Any]]) -> list[Any] | None:
    """One model call for *beliefs*; returns the raw cue groups or ``None``.

    ``None`` covers both a transport error and Azure's content filter rejecting
    the prompt — the caller decides whether to retry more granularly.
    """
    listing = "\n".join(
        f"{index}. {belief['text']}" for index, belief in enumerate(beliefs, 1)
    )
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CUE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Beliefs:\n{listing}"},
            ],
            reasoning_effort="minimal",
        )
        raw = completion.choices[0].message.content or ""
    except Exception:  # noqa: BLE001 — caller falls back / gives up
        return None

    groups = (_extract_json(raw) or {}).get("cues") or []
    return groups if len(groups) == len(beliefs) else None


def generate_cues(
    persona: dict[str, Any],
    *,
    client: Any = None,
    deployment: str | None = None,
) -> dict[str, list[str]]:
    """Ask the LLM for retrieval cues for every belief in *persona*.

    One batched call per profile, not one per belief. Returns
    ``{belief_text: [cue, …]}``. Any failure returns ``{}`` so an offline init
    still produces a usable text-only index instead of aborting.
    """
    beliefs = [
        belief
        for belief in (persona.get("beliefs") or [])
        if str(belief.get("text") or "").strip()
    ]
    if not beliefs:
        return {}

    try:
        from .neo4j_db import config, llm

        client = client or llm.get_client()
        model = deployment or config.AZURE_OPENAI_DEPLOYMENT
    except Exception as exc:  # noqa: BLE001 — cues are optional enrichment
        print(f"  [cues] client unavailable for {persona.get('name')}: {exc}")
        return {}

    groups = _request_cues(client, model, beliefs)
    if groups is None:
        # A whole-profile prompt can trip Azure's content filter when the
        # schema contains self-harm or violence wording. Retry belief by belief
        # so one flagged belief does not cost the profile all its cues.
        groups = []
        skipped = 0
        for belief in beliefs:
            single = _request_cues(client, model, [belief])
            if single is None:
                skipped += 1
                groups.append([])
            else:
                groups.append(single[0])
        if skipped:
            print(
                f"  [cues] {persona.get('name')}: batch call rejected; "
                f"{skipped}/{len(beliefs)} beliefs kept text-only"
            )

    result: dict[str, list[str]] = {}
    for belief, group in zip(beliefs, groups):
        seen: set[str] = set()
        cues: list[str] = []
        for raw_cue in group or []:
            cue = " ".join(str(raw_cue).split()).strip(" .,;:").lower()
            if cue and cue not in seen:
                seen.add(cue)
                cues.append(cue)
        if cues:
            result[str(belief["text"])] = cues[:CUES_PER_BELIEF]
    return result


def _cues_for_sync(
    source: Any,
    profile_name: str,
    profile_data: dict[str, Any] | None,
    *,
    with_cues: bool,
    force_cues: bool,
) -> dict[str, list[str]] | None:
    """Generate cues for the beliefs of *profile_name* that lack them.

    Returns ``None`` when there is nothing to generate (cues disabled, or every
    belief already has cues), so the caller can skip the LLM entirely.
    """
    if not with_cues:
        return None
    persona = build_persona(profile_name, profile_data)
    existing = source.cues_for(profile_name)
    todo = [
        belief
        for belief in persona["beliefs"]
        if force_cues or not existing.get(belief["text"])
    ]
    if not todo:
        return None
    target = persona if len(todo) == len(persona["beliefs"]) else {**persona, "beliefs": todo}
    return generate_cues(target) or None


def sync_profile(
    source: Any,
    profile_name: str,
    profile_data: dict[str, Any] | None,
    cues: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build the persona from *profile_data* and upsert it through *source*.

    *cues* (``{belief_text: [cue, …]}``) are merged in; beliefs it does not
    mention keep whatever cues the graph already holds.
    """
    persona = build_persona(profile_name, profile_data)
    if cues:
        for belief in persona["beliefs"]:
            generated = cues.get(belief["text"])
            if generated:
                belief["cues"] = generated
    return source.sync_persona(persona)


def sync_all(
    context_manager: Any = None,
    *,
    with_cues: bool = False,
    force_cues: bool = False,
    reembed: bool = False,
) -> dict[str, Any]:
    """Sync every memgraph profile, then prune graph personas that no longer exist.

    With *with_cues*, retrieval cues are generated for beliefs that have none
    (*force_cues* regenerates every cue); *reembed* drops the stored vectors
    first so every belief is re-encoded. Returns
    ``{"synced": {name: counts}, "removed": [names]}``; a profile that fails
    to sync is reported under its own name instead of aborting the run.
    """
    from .manager import ContextManager
    from .neo4j_source import Neo4jContextSource

    manager = context_manager or ContextManager()
    source = Neo4jContextSource()

    names = manager.list_profiles()
    if not source.available():
        return {"synced": {}, "removed": [], "error": f"graph unreachable ({source.uri()})"}

    synced: dict[str, Any] = {}
    for name in names:
        try:
            data = manager.load_profile(name)
            if reembed:
                source.clear_embeddings(name)
            cues = _cues_for_sync(
                source, name, data, with_cues=with_cues, force_cues=force_cues
            )
            synced[name] = sync_profile(source, name, data, cues=cues)
        except Exception as exc:  # noqa: BLE001 — report, keep going
            synced[name] = {"error": str(exc)}

    return {"synced": synced, "removed": source.prune_profiles(list(names))}


def main(argv: list[str] | None = None) -> int:
    """CLI: sync one, several or all profiles and print a summary."""
    import sys

    from .manager import ContextManager
    from .neo4j_source import Neo4jContextSource

    raw = list(argv if argv is not None else sys.argv[1:])
    force_cues = "--force-cues" in raw
    with_cues = force_cues or "--cues" in raw
    reembed = "--reembed" in raw
    requested = [arg for arg in raw if not arg.startswith("--")]

    manager = ContextManager()
    source = Neo4jContextSource()

    if not source.available():
        print(f"neo4j-db graph unreachable at {source.uri() or '<unset NEO4J_URI>'}")
        return 1

    if not requested:
        result = sync_all(manager, with_cues=with_cues, force_cues=force_cues, reembed=reembed)
        for name, counts in sorted(result["synced"].items()):
            print(f"  {name:12} {counts}")
        print(f"pruned: {result['removed'] or 'none'}")
        return 0

    for name in requested:
        data = manager.load_profile(name)
        if reembed:
            print(f"  {name:12} cleared {source.clear_embeddings(name)} stale vectors")
        cues = _cues_for_sync(
            source, name, data, with_cues=with_cues, force_cues=force_cues
        )
        counts = sync_profile(source, name, data, cues=cues)
        print(f"  {name:12} {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
