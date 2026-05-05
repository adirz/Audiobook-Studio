"""Audiobook Studio — main application entry point."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import STATIC_DIR
from app.api import projects, pipeline, audio, settings

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


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/workspace")
async def workspace():
    return FileResponse(str(STATIC_DIR / "workspace.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8899, reload=True)
