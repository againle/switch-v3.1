from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .app_config import CONFIG
from .lib.core.context import ContextManager
from .logger import get_logger
from .report import PDF_DIR, ensure_dirs

from .session import ChatSession

logger = get_logger(__name__)


BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
ROOT_DIR = BACKEND_DIR.parent

# Report output directories must exist before the static mount below
# resolves (StaticFiles checks the directory at import time).
ensure_dirs()

app = FastAPI(title="Memgraph Chat", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/reports", StaticFiles(directory=str(PDF_DIR)), name="reports")


@app.get("/api/config")
async def api_config() -> JSONResponse:
    """Serve the runtime config (debounce + chunking) to the frontend."""
    return JSONResponse(CONFIG.to_dict())


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/profiles")
async def profiles() -> JSONResponse:
    manager = ContextManager()
    return JSONResponse({"profiles": manager.list_profiles()})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session: ChatSession | None = None
    status_task: asyncio.Task[Any] | None = None
    send_lock = asyncio.Lock()

    async def send_event(event: dict[str, Any]) -> None:
        async with send_lock:
            try:
                await websocket.send_json(event)
            except RuntimeError:
                pass  # websocket already closed, ignore

    async def status_loop() -> None:
        while True:
            if session is not None:
                await send_event(session.status_payload())
            await asyncio.sleep(0.1)

    async def on_trigger(tag: str, message: str) -> None:
        # The client ended the session via a Leave action. The session-end
        # event was already pushed to the UI by the orchestrator's response
        # callback — just close the socket so the receive loop unwinds.
        if tag == "leave":
            with contextlib.suppress(Exception):
                await websocket.close()

    try:
        manager = ContextManager()
        await send_event({"type": "ready", "profiles": manager.list_profiles()})

        start_message = await websocket.receive_json()
        if start_message.get("type") != "start":
            await send_event({"type": "error", "content": "Expected a start message first."})
            await websocket.close()
            return

        profile_name = str(start_message.get("profile", "")).strip()
        if not profile_name:
            await send_event({"type": "error", "content": "A profile name is required."})
            await websocket.close()
            return

        # The profile folders under backend/profiles/ are the source of truth
        # for personas: mirror THIS session's profile into the graph before it
        # starts, so the CONTEXT it reads — and the turns it writes back —
        # belong to the same client profile (same name, same beliefs). Run off
        # the event loop; a sync failure never blocks the chat.
        if CONFIG.context.type == "memory" and CONFIG.context.memory_enabled:
            await asyncio.to_thread(manager.sync_profile, profile_name)

        # CONTEXT type is chosen by backend/config.yaml (`context_type`).
        session = ChatSession(
            profile_name=profile_name,
            context_manager=manager,
        )
        # Configure the response path up-front (before start). The
        # orchestrator owns the message debouncer, so a Leave produced by
        # a debounced message still closes the socket via on_trigger.
        session.setup(event_sink=send_event, on_trigger=on_trigger)

        await session.start()
        logger.info("Session started for profile=%s context_type=%s", profile_name, session.context_type)
        await send_event({
            "type": "session_started",
            "profile": profile_name,
            "profile_title": session.profile_data.get("name", profile_name),
            "description": session.profile_data.get("short_description", ""),
            "status": session.status_payload(),
            "context_type": session.context_type,
        })
        status_task = asyncio.create_task(status_loop())

        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type == "user_message":
                text = str(message.get("content", "")).strip()
                if not text:
                    continue
                # Debounced inside the orchestrator — bursts of messages
                # are batched into a single CogniLLM turn.
                await session.handle_user_message(text)
                continue

            if message_type == "typing":
                # Live typing signal from the input box. The active debounce
                # method decides what to do with it: `typing_aware` refreshes
                # its wait window on `active=True` and keeps the idle timer
                # quiet while the user is composing. `active=False` means the
                # user stopped typing / cleared the box / pressed send and
                # never refreshes the window.
                session.handle_typing(bool(message.get("active", True)))
                continue

            if message_type == "command":
                command = str(message.get("content", "")).strip()
                if not command:
                    continue
                event = await session.handle_text(command)
                await send_event(event)
                if event.get("session_end"):
                    # e.g. /quit — close the socket and disconnect.
                    with contextlib.suppress(Exception):
                        await websocket.close()
                    return
                continue

            if message_type == "chunking_start":
                # The frontend started typing out a chunked reply — pause
                # the idle timer so it doesn't count during the display.
                if session.orchestrator is not None:
                    session.orchestrator.set_displaying(True)
                continue

            if message_type == "chunking_end":
                # The last chunk landed — resume the idle timer (clock
                # restarts fresh, so display time isn't counted).
                if session.orchestrator is not None:
                    session.orchestrator.set_displaying(False)
                continue

            if message_type == "save_pdf":
                # Build the session report — full cognitive state,
                # should_leave verdict and actions per turn — as JSON
                # (logs/json/) and PDF (logs/pdf/, served from /reports so
                # the browser can download it).
                if session is None:
                    await send_event({"type": "error", "content": "No active session to export."})
                    continue
                try:
                    json_path, pdf_path = await asyncio.to_thread(session.save_session_report)
                except Exception as exc:  # noqa: BLE001 — a failed export must not kill the session
                    logger.exception("Session report generation failed")
                    await send_event({"type": "error", "content": f"Could not build the report: {exc}"})
                    continue
                await send_event({
                    "type": "pdf_ready",
                    "url": f"/reports/{pdf_path.name}",
                    "file": pdf_path.name,
                    "json_file": str(json_path.relative_to(ROOT_DIR)),
                    "turns": len(session.orchestrator.transcript),
                })
                continue

            if message_type == "pause":
                session.orchestrator.pause()
                await send_event(session.status_payload())
                continue

            if message_type == "resume":
                session.orchestrator.resume()
                await send_event(session.status_payload())
                continue

            await send_event({"type": "error", "content": f"Unknown message type: {message_type}"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for profile=%s", session.profile_name if session else "unknown")
    finally:
        if status_task is not None:
            status_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await status_task
        if session is not None:
            await session.close()