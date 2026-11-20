"""Standalone maintenance page for the Memgraph / CogniLLM demo.

Run this *instead of* the demo while the app is offline, so anyone opening the
demo address sees an on-brand notice rather than a connection error::

    python maintenance.py                           # http://0.0.0.0:8989
    MAINTENANCE_PORT=8111 python maintenance.py     # take over the demo's port

Deliberately imports nothing from ``backend`` and reads no application config:
a maintenance page must not be able to fail because of the app it stands in
for.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
PAGE_PATH = FRONTEND_DIR / "maintenance.html"

#: Port the notice is served on.
DEFAULT_PORT = 8989

#: Status code for the notice. 503 + ``Retry-After`` is the honest answer
#: ("temporarily unavailable"), and browsers render the body normally. Set
#: ``MAINTENANCE_STATUS=200`` if anything in front of this rewrites 503s.
DEFAULT_STATUS = 503

STATUS_CODE = int(os.getenv("MAINTENANCE_STATUS", str(DEFAULT_STATUS)))

#: Used only if the page file is missing, so this can never serve a 500.
FALLBACK_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Maintenance</title></head>
<body style="font-family:Segoe UI,Helvetica,Arial,sans-serif;text-align:center;padding:15vh 24px">
<h1 style="color:#075e54;font-size:24px;margin:0 0 12px">Maintenance in progress</h1>
<p style="color:#667781;font-size:15px;margin:0">The Interactive Client demo is
temporarily unavailable. Please check back shortly.</p>
</body></html>
"""

app = FastAPI(
    title="Memgraph Chat - maintenance",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _notice() -> HTMLResponse:
    """The maintenance notice, served uncached."""
    try:
        html = PAGE_PATH.read_text(encoding="utf-8")
    except OSError:
        html = FALLBACK_HTML
    return HTMLResponse(
        content=html,
        status_code=STATUS_CODE,
        # no-store matters: a cached notice would keep showing after the demo
        # is back up.
        headers={"Cache-Control": "no-store, max-age=0", "Retry-After": "120"},
    )


@app.get("/")
async def index() -> HTMLResponse:
    """The notice."""
    return _notice()


@app.get("/{path:path}")
async def catch_all(path: str) -> HTMLResponse:
    """Every other URL gets the same notice.

    A stale bookmark, a deep link, or a request for a missing asset would
    otherwise land on a bare 404, which reads as "the site is broken" rather
    than "the demo is paused".
    """
    return _notice()


if __name__ == "__main__":
    port = int(os.getenv("MAINTENANCE_PORT", str(DEFAULT_PORT)))
    print(f"[maintenance] serving the demo notice on http://0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
