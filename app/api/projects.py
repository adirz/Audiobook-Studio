"""API routes for project CRUD and lifecycle."""

import os
import re
import shutil
import json
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File
from app.config import PROJECTS_DIR, get_settings
from app.database import ProjectDB
from app.models import ProjectCreate, ProjectInfo

router = APIRouter(prefix="/api/projects", tags=["projects"])

# Active project DB connections
_open_dbs: dict[str, ProjectDB] = {}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50]


def get_project_db(slug: str) -> ProjectDB:
    """Get or open a project's database."""
    if slug in _open_dbs:
        return _open_dbs[slug]

    project_dir = PROJECTS_DIR / slug
    if not project_dir.exists():
        raise HTTPException(404, f"Project '{slug}' not found")

    db_path = project_dir / "project.db"
    db = ProjectDB(db_path)
    _open_dbs[slug] = db
    return db


def get_project_dir(slug: str) -> Path:
    d = PROJECTS_DIR / slug
    if not d.exists():
        raise HTTPException(404, f"Project '{slug}' not found")
    return d


@router.get("/")
def list_projects() -> list[ProjectInfo]:
    """List all projects."""
    projects = []
    if not PROJECTS_DIR.exists():
        return projects

    for d in sorted(PROJECTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        db_path = d / "project.db"
        if not db_path.exists():
            continue

        db = get_project_db(d.name)
        projects.append(ProjectInfo(
            name=db.get_meta("project_name") or d.name,
            slug=d.name,
            description=db.get_meta("description") or "",
            current_step=db.get_meta("current_step") or "new",
            source_file=db.get_meta("source_file") or "",
            created_at=db.get_meta("created_at") or "",
            chapter_count=len(db.get_chapters()),
            chunk_count=len(db.get_chunks()),
            voice=db.get_meta("voice") or "",
        ))

    return projects


@router.post("/")
def create_project(data: ProjectCreate) -> ProjectInfo:
    """Create a new project."""
    slug = slugify(data.name)
    project_dir = PROJECTS_DIR / slug

    if project_dir.exists():
        raise HTTPException(409, f"Project '{slug}' already exists")

    project_dir.mkdir(parents=True)
    (project_dir / "source").mkdir()
    (project_dir / "audio").mkdir()
    (project_dir / "test_clips").mkdir()
    (project_dir / "export").mkdir()

    db = ProjectDB(project_dir / "project.db")
    db.set_meta("project_name", data.name)
    db.set_meta("description", data.description)
    db.set_meta("created_at", datetime.now().isoformat())
    db.set_meta("current_step", "new")

    _open_dbs[slug] = db

    return ProjectInfo(
        name=data.name,
        slug=slug,
        description=data.description,
        current_step="new",
        created_at=db.get_meta("created_at"),
    )


@router.get("/{slug}")
def get_project(slug: str) -> ProjectInfo:
    db = get_project_db(slug)
    return ProjectInfo(
        name=db.get_meta("project_name") or slug,
        slug=slug,
        description=db.get_meta("description") or "",
        current_step=db.get_meta("current_step") or "new",
        source_file=db.get_meta("source_file") or "",
        created_at=db.get_meta("created_at") or "",
        chapter_count=len(db.get_chapters()),
        chunk_count=len(db.get_chunks()),
        voice=db.get_meta("voice") or "",
    )


@router.delete("/{slug}")
def delete_project(slug: str):
    if slug in _open_dbs:
        _open_dbs[slug].close()
        del _open_dbs[slug]

    project_dir = PROJECTS_DIR / slug
    if project_dir.exists():
        shutil.rmtree(project_dir)

    return {"status": "deleted"}


@router.post("/{slug}/upload")
async def upload_manuscript(slug: str, file: UploadFile = File(...)):
    """Upload a manuscript file to the project."""
    project_dir = get_project_dir(slug)
    db = get_project_db(slug)

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    supported = [".docx"]  # Extend via extractor handler registry
    if ext not in supported:
        raise HTTPException(400, f"Unsupported file type: {ext}. Supported: {supported}")

    # Save file
    source_dir = project_dir / "source"
    source_dir.mkdir(exist_ok=True)
    dest = source_dir / file.filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    db.set_meta("source_file", str(dest))
    db.set_meta("current_step", "source_selected")

    return {"status": "uploaded", "path": str(dest), "filename": file.filename}
