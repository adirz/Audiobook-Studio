"""Audiobook Studio — main application entry point."""

import os
import signal
import threading

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import STATIC_DIR, PROJECTS_DIR
from app.api import projects, pipeline, audio, settings
from app.database import ProjectDB

app = FastAPI(
    title="Audiobook Studio",
    description="Turn manuscripts into audiobooks with TTS",
    version="0.1.0",
)

# Mount API routers
app.include_router(projects.router)
app.include_router(pipeline.router)
app.include_router(audio.router)
app.include_router(settings.router)

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def reset_interrupted_generations():
    """Sweep all project DBs for 'generating'/'pending' rows left over
    from a previous run that was force-killed.

    At startup no generation task can be running yet, so any row in those
    states is unambiguously stale. Resetting them to 'error' lets the UI
    surface the affected chunks for re-generation instead of treating
    them as in-flight forever.
    """
    if not PROJECTS_DIR.exists():
        return
    for project_dir in PROJECTS_DIR.iterdir():
        db_path = project_dir / "project.db"
        if not db_path.exists():
            continue
        try:
            db = ProjectDB(db_path)
            n = db.reset_stale_generations()
            if n:
                print(f"[startup] reset {n} interrupted generation row(s) in {project_dir.name}")
            db.close()
        except Exception as e:
            print(f"[startup] could not sweep {project_dir.name}: {e}")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/workspace")
async def workspace():
    return FileResponse(str(STATIC_DIR / "workspace.html"))


@app.post("/api/system/shutdown")
async def shutdown():
    """Gracefully shut down the server, unloading GPU models before exit."""
    def _stop():
        try:
            from app.handlers.registry import unload_all
            unload_all()
        except Exception:
            pass
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Timer(0.3, _stop).start()
    return {"status": "shutting down"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8899, reload=True)
