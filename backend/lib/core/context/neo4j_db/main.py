from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import pipeline

app = FastAPI(title="MemChat — GraphRAG Client Simulator")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)


class SessionRequest(BaseModel):
    user_id: str = "trainee"
    profile_name: str | None = None


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/profiles")
def profiles():
    return {"profiles": pipeline.list_profiles()}


@app.post("/api/session")
def create_session(req: SessionRequest | None = None):
    try:
        user_id = req.user_id if req else "trainee"
        profile_name = ((req.profile_name or "").strip() or None) if req else None
        session_id = pipeline.create_session(user_id, profile_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"session_id": session_id, "profile_name": pipeline.session_profile(session_id)}


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        return pipeline.process_turn(req.session_id, req.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/session/{session_id}/turns")
def get_turns(session_id: str):
    return {"session_id": session_id, "turns": pipeline.get_history(session_id)}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
