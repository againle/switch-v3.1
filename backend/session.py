from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from .app_config import CONFIG
from .lib.core.orchestrator import CogniLLMConfig, Orchestrator
from .lib.core.context import ContextManager
from .logger import get_logger
from .report import build_pdf_report, ensure_dirs, write_json_report

logger = get_logger(__name__)


EventSink = Callable[[dict[str, Any]], Awaitable[None]]


def get_env_credentials() -> tuple[str, str, str, str]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    return endpoint, deployment, api_key, api_version


@dataclass
class ChatSession:
    profile_name: str
    context_manager: ContextManager = field(default_factory=ContextManager)
    orchestrator: Orchestrator | None = None
    profile_data: dict[str, Any] = field(default_factory=dict)
    stage_config: dict[str, Any] = field(default_factory=dict)
    event_sink: EventSink | None = None
    # CONTEXT type: "summary" (accumulate + summarize older windows) or
    # "memory" (read + write CONTEXT through the context manager). Empty means
    # "use the app config's context_type".
    context_type: str = ""
    # Graph session id used by the "memory" CONTEXT type: every turn of this
    # chat is stored under it in the standalone neo4j-db graph. Stable for the
    # lifetime of the session so the graph history lines up with this chat.
    memory_session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Adaptive-wait rate. Defaults to the app config's `rate` when None.
    rate: float | None = None
    # Lifecycle callbacks, passed through to the orchestrator. Registered
    # via ``setup()`` (works even before ``start()`` so a Leave fired by
    # a debounced message — e.g. before memory init — reaches the app).
    on_trigger: Callable[[str, str], None] | None = None
    on_error: Callable[[str], None] | None = None

    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.context_type:
            self.context_type = CONFIG.context.type
        self.context_type = self.context_type.strip().lower()
        if self.context_type not in ("summary", "memory"):
            self.context_type = "summary"

        self.profile_data = self.context_manager.load_profile(self.profile_name)
        self.stage_config = self.context_manager.load_stage_config(self.profile_name)
        if self.stage_config:
            self.profile_data["stage_config"] = self.stage_config

        endpoint, deployment, api_key, api_version = get_env_credentials()

        ctx = CONFIG.context
        rate = self.rate if self.rate is not None else CONFIG.rate
        self.orchestrator = Orchestrator(
            CogniLLMConfig(
                profile_data=self.profile_data,
                endpoint=endpoint,
                deployment=deployment,
                api_key=api_key,
                api_version=api_version,
                # CONTEXT component — "summary" accumulates + summarizes
                # older windows (rounds from config.yaml); "memory" reads this
                # turn's block from the standalone neo4j-db graph through the
                # context manager (a direct function call, no URL/HTTP hop).
                # `context.memory.enabled=false` greys the path out entirely.
                summary_enabled=ctx.summary_enabled,
                summary_window_size=ctx.summary_window_size,
                summary_start_round=ctx.summary_start_round,
                context_type=self.context_type,
                memory_source=self.context_manager if ctx.memory_enabled else None,
                memory_profile_name=self.profile_name,
                memory_session_id=self.memory_session_id,
                # MI stage control (config.yaml → mi_stage).
                mi_stage_enabled=CONFIG.mi_stage.enabled,
                inject_stage_info=CONFIG.mi_stage.inject_stage_info,
                evaluate_stage_progression=CONFIG.mi_stage.evaluate_progression,
                injected_stage=CONFIG.mi_stage.injected_stage,
                debounce_enabled=CONFIG.debounce.enabled,
                debounce_method=CONFIG.debounce.method,
                debounce_method_settings=CONFIG.debounce.settings_for(CONFIG.debounce.method),
                debounce_max_wait_seconds=CONFIG.debounce.max_wait_seconds,
                debounce_settings_for=CONFIG.debounce.settings_for,
            ),
            rate=rate,
        )
        logger.info(
            "ChatSession created for profile=%s context_type=%s",
            self.profile_name,
            self.context_type,
        )

    def setup(
        self,
        event_sink: EventSink | None = None,
        on_trigger: Callable[[str, str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        """Configure the response path without starting the timer loop.

        Registers the response/trigger/error callbacks on the orchestrator
        so that debounced user messages can push replies to the UI (and a
        Leave can close the socket) even before ``start()``.
        """
        if event_sink is not None:
            self.event_sink = event_sink
        if on_trigger is not None:
            self.on_trigger = on_trigger
        if on_error is not None:
            self.on_error = on_error

        if self.orchestrator is not None:
            self.orchestrator.register_callbacks(
                on_response=self._on_response,
                on_trigger=self.on_trigger,
                on_error=self.on_error,
            )

    async def start(self, event_sink: EventSink | None = None) -> None:
        """Configure the response path and start the orchestrator's timer loop."""
        if event_sink is not None:
            self.event_sink = event_sink
        self.setup(self.event_sink)
        await self.orchestrator.start(on_response=self._on_response)
        logger.debug("Session started — timer loop active for profile=%s", self.profile_name)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.orchestrator.stop()

    def status_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "status",
            "profile": self.profile_name,
            "profile_title": self.profile_data.get("name", self.profile_name),
            "description": self.profile_data.get("short_description", ""),
            # Id of this chat's Session node in the standalone neo4j-db
            # graph (the key the context manager retrieves turns by).
            # Surfaced so the supervisor can copy it from the sidebar.
            "memory_session_id": self.memory_session_id,
            "last_input_age": round(self.orchestrator.last_input_age, 1),
            "time_threshold": round(self.orchestrator.time_threshold, 1),
            "api_in_flight": self.orchestrator.api_in_flight,
            "ended_by_leave": self.orchestrator.ended_by_leave,
            "adawait_state": self.orchestrator.adawait_state,
            "paused": self.orchestrator.is_paused,
            "displaying": self.orchestrator.is_displaying,
            "context_type": self.context_type,
        }
        payload.update(self.orchestrator.debounce_status())
        return payload

    # ── Session report (Save PDF) ────────────────────────────────

    def build_session_report(self) -> dict[str, Any]:
        """Everything the Save-PDF report needs, as plain JSON data."""
        orchestrator = self.orchestrator
        return {
            "report": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "profile": self.profile_name,
                "profile_title": self.profile_data.get("name", self.profile_name),
                "context_type": self.context_type,
                "session_started_at": orchestrator.session_started_at,
                "memory_session_id": self.memory_session_id,
                "turns": len(orchestrator.transcript),
                "ended_by_leave": bool(orchestrator.ended_by_leave),
                "adawait_state": orchestrator.adawait_state,
                "debounce_method": orchestrator.debounce_status().get("debounce_method"),
            },
            "turns": list(orchestrator.transcript),
        }

    def save_session_report(self) -> tuple[Path, Path]:
        """Write this session's report as JSON and PDF.

        Returns:
            tuple[Path, Path]: ``(json_path, pdf_path)``. The JSON lands in
            ``logs/json/`` and the PDF in ``logs/pdf/`` (also served to the
            browser through the ``/reports`` static mount).
        """
        data = self.build_session_report()
        stem = f"{self.profile_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        json_dir, pdf_dir = ensure_dirs()
        json_path = write_json_report(data, json_dir / f"{stem}.json")
        pdf_path = build_pdf_report(data, pdf_dir / f"{stem}.pdf")
        logger.info("Session report written: %s | %s", json_path, pdf_path)
        return json_path, pdf_path

    def set_context_type(self, context_type: str) -> str:
        """Switch the CONTEXT type at runtime.

        "summary" → accumulate the conversation and summarize older
                    windows after N rounds (rounds from ``config.yaml``).
        "memory"  → read (and write back) CONTEXT through the context
                    manager, i.e. the standalone neo4j-db graph.
        """
        context_type = (context_type or "").strip().lower()
        if context_type not in ("summary", "memory"):
            raise ValueError("context type must be 'summary' or 'memory'")

        self.context_type = context_type
        if self.orchestrator is not None:
            # Push the new type down into the generation layer (Client),
            # where the summary mechanism is enabled/disabled.
            self.orchestrator.set_context_type(context_type)
        return context_type

    async def handle_user_message(self, text: str) -> None:
        """Debounced entry point for a user message.

        Routes the message through the orchestrator's internal debouncer,
        which batches bursts of input into a single CogniLLM turn. The
        reply is pushed to the UI via the response callback — nothing is
        returned here.
        """
        await self.orchestrator.add_user_input(text)

    def handle_typing(self, active: bool) -> None:
        """Frontend typing signal (``active`` = the user just typed input).

        Handed straight to the debouncer: the active method decides whether a
        keystroke refreshes its wait window (see the ``typing_aware`` method)
        and may keep the idle timer quiet while the user is composing.
        """
        self.orchestrator.notify_typing(active)

    async def handle_text(self, text: str) -> dict[str, Any]:
        command = text.strip().lower()

        if command == "/reset":
            self.orchestrator.reset_session()
            return {
                "type": "system_message",
                "content": "Conversation reset — client history and wait timer cleared.",
                "reset": True,
                "status": self.status_payload(),
            }

        if command == "/quit":
            return {"type": "session_end", "content": "Session closed by request."}

        if command == "/pause":
            self.orchestrator.pause()
            return {"type": "system_message", "content": "Timer paused."}

        if command == "/resume":
            self.orchestrator.resume()
            return {"type": "system_message", "content": "Timer resumed."}

        if command.startswith("/context"):
            parts = command.split()
            if len(parts) == 1:
                return {
                    "type": "system_message",
                    "content": f"Context type: {self.context_type}. Use /context summary or /context memory.",
                }
            target = parts[1]
            try:
                self.set_context_type(target)
            except ValueError:
                return {"type": "system_message", "content": "Usage: /context summary | /context memory."}
            return {
                "type": "system_message",
                "content": f"Context type switched to {self.context_type}.",
                "status": self.status_payload(),
            }

        if command == "/help":
            return {
                "type": "system_message",
                "content": "Commands: /help, /reset, /context <summary|memory>, /quit.",
            }

        if self.orchestrator.ended_by_leave:
            return {
                "type": "error",
                "content": "The client has already ended this session.",
            }

        response = await self.orchestrator.handle_user_input(text)
        return self._build_assistant_event(response)

    #: Action keys that are private to the supervisor-facing transcript and
    #: must never reach the trainee's browser (even via the raw `response`
    #: passthrough, which a curious trainee could read from the socket).
    _PRIVATE_ACTION_KEYS = ("hidden", "drafts")

    @classmethod
    def _redact_private(cls, response: dict[str, Any] | None) -> dict[str, Any] | None:
        """Copy *response* with the private action channels removed.

        ``action.hidden`` (BEHAVIOUR's unobservable action / inner voice) and
        ``action.drafts`` (TYPING_REFLECTION's deleted drafts) must never reach
        the trainee. The transcript keeps the full action (that is the
        supervisor-facing report), so only the copy sent to the browser is
        redacted.
        """
        if not isinstance(response, dict):
            return response
        action = response.get("action")
        if not isinstance(action, dict) or not any(
            key in action for key in cls._PRIVATE_ACTION_KEYS
        ):
            return response
        redacted = dict(response)
        redacted["action"] = {
            key: value
            for key, value in action.items()
            if key not in cls._PRIVATE_ACTION_KEYS
        }
        return redacted

    def _build_assistant_event(self, response: dict[str, Any] | None) -> dict[str, Any]:
        """Build the assistant_message event for an orchestrator response.

        ``content`` is only ever the text the trainee may read as a chat
        bubble: ``action.message`` for SPEAK / TYPING_REFLECTION / LEAVE, and
        empty for BEHAVIOUR (whose observable cue is ``behaviour``, rendered as
        an action cue rather than a bubble). The private channels never leave
        the backend: BEHAVIOUR's ``hidden`` is dropped, and
        TYPING_REFLECTION's ``drafts`` are reduced to a count/length summary so
        the UI can replay the hesitation without ever seeing the deleted text.
        """
        assistant_text = ""
        action_type = "silence"
        behaviour = ""
        typing: dict[str, Any] | None = None
        if isinstance(response, dict):
            action = response.get("action", {})
            if isinstance(action, dict):
                action_type = str(action.get("type", "silence"))
                assistant_text = str(action.get("message", "") or "").strip()
                behaviour = str(action.get("behaviour", "") or "").strip()
                drafts = action.get("drafts")
                if isinstance(drafts, list):
                    lengths = [len(str(d)) for d in drafts if str(d).strip()]
                    typing = {"cycles": len(lengths), "lengths": lengths}

        event: dict[str, Any] = {
            "type": "assistant_message",
            "content": assistant_text,
            "action_type": action_type,
            "behaviour": behaviour,
            "response": self._redact_private(response),
            "status": self.status_payload(),
        }
        if typing is not None:
            event["typing"] = typing
        if self.orchestrator.ended_by_leave:
            event["session_end"] = True
            event["end_reason"] = "leave"
        return event

    async def _on_response(self, response: dict[str, Any], emit: bool = False) -> None:
        """Called by the orchestrator after every response.

        When ``emit`` is True the response came from the idle-timer path —
        not handle_text — so we push it to the UI here.
        """
        if emit and self.event_sink is not None:
            event = self._build_assistant_event(response)
            if (
                event.get("content")
                or event.get("behaviour")
                or event.get("typing")
                or event.get("session_end")
            ):
                try:
                    await self.event_sink(event)
                except Exception:
                    pass