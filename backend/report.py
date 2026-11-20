"""Session report — structured JSON plus a formatted PDF.

Both outputs are produced from the structured transcript the orchestrator
captures per turn (``Orchestrator.transcript``), so the report shows everything
the client actually produced: the full cognitive state (``automatic_thoughts``,
``emotions``, ``behaviors``, ``openness``, ``chain_of_thought``, plus any extra
field the model emitted), the Tier-0 ``should_leave`` assessment, the Tier-1/2
decision and the resulting action.

Files written when the user clicks *Save PDF*::

    logs/json/<profile>-<stamp>.json   the data below, verbatim
    logs/pdf/<profile>-<stamp>.pdf     the same data rendered for reading

The PDF is kept on disk as well as downloaded so the browser can fetch it over
HTTP (see the ``/reports`` mount in ``app.py``).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .pdf_writer import ACCENT, MUTED, PdfDocument

__all__ = [
    "JSON_DIR",
    "PDF_DIR",
    "build_pdf_report",
    "ensure_dirs",
    "write_json_report",
]

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
JSON_DIR = LOG_DIR / "json"
PDF_DIR = LOG_DIR / "pdf"

#: Cognitive-state fields rendered in a fixed order; anything else the model
#: returned is appended afterwards so nothing is ever silently dropped.
_COGNITIVE_ORDER: tuple[tuple[str, str], ...] = (
    ("automatic_thoughts", "Automatic thoughts"),
    ("emotions", "Emotions"),
    ("behaviors", "Behaviors"),
    ("openness", "Openness"),
    ("chain_of_thought", "Chain of thought"),
    ("leave_session", "leave_session (model)"),
)


def ensure_dirs() -> tuple[Path, Path]:
    """Create (and return) the JSON and PDF output directories."""
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    return JSON_DIR, PDF_DIR


# ── Value formatting ─────────────────────────────────────────────────


def _as_text(value: Any) -> str:
    """Render any JSON value as report-friendly text."""
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value.strip() or "—"
    if isinstance(value, (list, tuple)):
        if not value:
            return "—"
        if all(isinstance(item, (str, int, float)) for item in value):
            return ", ".join(str(item) for item in value)
        return json.dumps(list(value), ensure_ascii=False, indent=2)
    if isinstance(value, dict):
        if not value:
            return "—"
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _truncate(text: str, limit: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ── JSON ─────────────────────────────────────────────────────────────


def write_json_report(data: dict[str, Any], path: str | Path) -> Path:
    """Write the report data as pretty JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


# ── PDF ──────────────────────────────────────────────────────────────


def _render_meta(doc: PdfDocument, meta: dict[str, Any]) -> None:
    doc.heading("Session", size=12.5, space_before=4, underline=True)
    doc.kv("Generated", _as_text(meta.get("generated_at")))
    doc.kv("Profile", f"{_as_text(meta.get('profile_title'))} ({_as_text(meta.get('profile'))})")
    doc.kv("Context type", _as_text(meta.get("context_type")))
    # The graph session id this chat was written under — the key the context
    # manager retrieves (and the graph browser can look up) the session by.
    doc.kv("Graph session ID", _as_text(meta.get("memory_session_id")))
    doc.kv("Session started", _as_text(meta.get("session_started_at")))
    doc.kv("Turns recorded", _as_text(meta.get("turns")))
    doc.kv("Ended by leave", _as_text(bool(meta.get("ended_by_leave"))))
    doc.kv("Debounce method", _as_text(meta.get("debounce_method")))

    state = meta.get("adawait_state") or {}
    if isinstance(state, dict) and state:
        doc.kv(
            "AdaWait E/O/V",
            " / ".join(
                f"{key}={state.get(key):.2f}" if isinstance(state.get(key), (int, float))
                else f"{key}={_as_text(state.get(key))}"
                for key in ("E", "O", "V")
            ),
        )


def _render_cognitive_state(doc: PdfDocument, cognitive: dict[str, Any]) -> None:
    """Render every cognitive field, known ones first in a stable order."""
    rendered: set[str] = set()
    for key, label in _COGNITIVE_ORDER:
        if key in cognitive:
            rendered.add(key)
            doc.kv(label, _as_text(cognitive.get(key)))

    for key in sorted(k for k in cognitive if k not in rendered):
        doc.kv(key, _as_text(cognitive.get(key)))


def _render_turn(doc: PdfDocument, turn: dict[str, Any]) -> None:
    trigger = "elapsed silence" if turn.get("is_idle_trigger") else "trainee message"
    idle = turn.get("idle_seconds") or 0
    if turn.get("is_idle_trigger") and idle:
        trigger += f" ({idle:g}s)"

    doc.heading(
        f"Turn {turn.get('index', '?')} · {_as_text(turn.get('timestamp'))} · {trigger}",
        size=10.5,
        space_before=10,
        space_after=4,
        underline=True,
    )

    doc.kv("Trainee", _truncate(_as_text(turn.get("trainee_message")), 90))

    # The CONTEXT prompt this turn was generated from (memory context type).
    # Printed in full — wrapped, small, indented — so the report shows exactly
    # what the client was given, not just what it produced.
    context = (turn.get("context") or "").strip()
    if context:
        doc.kv(
            "Context prompt",
            f"{len(context)} chars · {len(context.splitlines())} lines",
        )
        doc.text(context, size=8.0, indent=14.0, space_after=2.0)
    else:
        doc.kv("Context prompt", "— (summary mode, or no context was supplied)")

    cognitive = turn.get("cognitive_state")
    if isinstance(cognitive, dict) and cognitive:
        _render_cognitive_state(doc, cognitive)
    else:
        doc.kv("Cognitive state", "—")

    leave = turn.get("leave_decision")
    if isinstance(leave, dict):
        verdict = "yes — leave" if leave.get("should_leave") else "no — continue"
        reason = leave.get("reason")
        if reason:
            verdict += f"  ({reason})"
        doc.kv("should_leave", verdict)
    else:
        doc.kv("should_leave", "—")

    decision = turn.get("decision")
    if isinstance(decision, dict):
        parts = [
            f"tier1={_as_text(decision.get('tier1'))}",
            f"pacing={_as_text(decision.get('pacing'))}",
            f"content_mode={_as_text(decision.get('content_mode'))}",
            f"generation_call={_as_text(bool(decision.get('generation_call')))}",
        ]
        doc.kv("Decision", " · ".join(parts))

    action = turn.get("action")
    if isinstance(action, dict):
        action_type = _as_text(action.get("type"))
        # Trainee-facing payload is `message` (SPEAK / TYPING_REFLECTION /
        # LEAVE) or `behaviour` (BEHAVIOUR, which has no `message` by design);
        # everything else (hidden, drafts, reason, pacing, …) is shown as
        # extras so the supervisor sees all the client produced.
        primary = action.get("message") or action.get("behaviour")
        extras = [
            f"{key}={_as_text(value)}"
            for key, value in action.items()
            if key not in ("type", "message", "behaviour") and value not in (None, "")
        ]
        line = f"{action_type}: {_as_text(primary)}" if primary else f"{action_type} (no content)"
        if extras:
            line += "   [" + " · ".join(extras) + "]"
        doc.kv("Action", line)
    else:
        doc.kv("Action", "—")

    if turn.get("interrupted"):
        doc.kv(
            "Interrupted",
            "yes — this reply was withheld; the trainee sent new input while it was generating",
        )

    doc.kv("Tokens", _as_text(turn.get("tokens")))


def build_pdf_report(data: dict[str, Any], path: str | Path) -> Path:
    """Render the report data to a PDF at *path*."""
    meta = data.get("report") or {}
    turns = data.get("turns") or []

    profile = _as_text(meta.get("profile_title"))
    doc = PdfDocument(title=f"Session report — {profile}", page_size="A4", margin=48)

    doc.heading("Session Report", size=20, space_before=0, space_after=2)
    doc.text(
        f"{profile} · {_as_text(meta.get('context_type'))} context · "
        f"{len(turns)} turn{'s' if len(turns) != 1 else ''}",
        size=10,
        color=MUTED,
        space_after=6,
    )

    _render_meta(doc, meta)
    doc.rule(space_before=6, space_after=4)

    doc.heading("Conversation and cognitive state", size=12.5, space_before=2, space_after=0)
    if not turns:
        doc.text("No turns were recorded in this session.", size=9.5, color=MUTED, space_after=4)
    else:
        for turn in turns:
            if isinstance(turn, dict):
                _render_turn(doc, turn)

    doc.rule(space_before=8, space_after=2)
    doc.text(
        "Generated by the Memgraph / CogniLLM interactive client.",
        size=8.5,
        color=MUTED,
    )

    return doc.save(path)
