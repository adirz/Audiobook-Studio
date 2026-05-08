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


@router.get("/{slug}/backup")
def download_backup(slug: str):
    """Package the project as a ZIP archive for download."""
    import tempfile, zipfile
    from starlette.background import BackgroundTask

    project_dir = get_project_dir(slug)
    db = get_project_db(slug)

    try:
        db.conn.execute("PRAGMA wal_checkpoint(FULL)")
    except Exception:
        pass

    fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)

    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(project_dir.rglob("*")):
            if path.is_file() and path.suffix not in (".wal", ".shm"):
                arcname = str(path.relative_to(project_dir.parent))
                zf.write(str(path), arcname)

    def cleanup():
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    from fastapi.responses import FileResponse
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=f"{slug}_backup.zip",
        background=BackgroundTask(cleanup),
    )


@router.post("/{slug}/copy")
def copy_project(slug: str, payload: dict):
    """Create a server-side copy of the project under a new name."""
    new_name = (payload.get("new_name") or "").strip()
    if not new_name:
        raise HTTPException(400, "new_name is required")

    project_dir = get_project_dir(slug)
    new_slug = slugify(new_name)
    new_dir = PROJECTS_DIR / new_slug

    if new_dir.exists():
        raise HTTPException(409, f"A project named '{new_slug}' already exists")

    db = get_project_db(slug)
    try:
        db.conn.execute("PRAGMA wal_checkpoint(FULL)")
    except Exception:
        pass

    shutil.copytree(str(project_dir), str(new_dir))

    new_db = ProjectDB(new_dir / "project.db")
    new_db.set_meta("project_name", new_name)
    new_db.set_meta("created_at", datetime.now().isoformat())
    _open_dbs[new_slug] = new_db

    return ProjectInfo(
        name=new_name,
        slug=new_slug,
        description=new_db.get_meta("description") or "",
        current_step=new_db.get_meta("current_step") or "new",
        created_at=new_db.get_meta("created_at"),
        chapter_count=len(new_db.get_chapters()),
        chunk_count=len(new_db.get_chunks()),
        voice=new_db.get_meta("voice") or "",
    )


@router.get("/{slug}/usage")
def get_project_usage(slug: str):
    """Get resource usage estimates for a single project."""
    project_dir = get_project_dir(slug)
    db = get_project_db(slug)

    def dir_size(path: Path) -> int:
        total = 0
        if path.exists():
            for f in path.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except Exception:
                        pass
        return total

    audio_bytes = dir_size(project_dir / "audio")
    export_bytes = dir_size(project_dir / "export")
    test_clips_bytes = dir_size(project_dir / "test_clips")
    db_path = project_dir / "project.db"
    db_bytes = db_path.stat().st_size if db_path.exists() else 0

    gen = db.fetchone(
        """SELECT
           COUNT(*) as total_attempts,
           SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) as ok_generations,
           COALESCE(SUM(CASE WHEN status='ok' THEN duration_sec ELSE 0 END), 0) as total_audio_sec,
           COALESCE(SUM(CASE WHEN status='ok' THEN gen_time_sec ELSE 0 END), 0) as total_gen_time_sec
           FROM generations"""
    ) or {}

    qa = db.fetchone("SELECT COUNT(*) as total FROM qa_results") or {}

    return {
        "disk": {
            "audio_bytes": audio_bytes,
            "export_bytes": export_bytes,
            "test_clips_bytes": test_clips_bytes,
            "db_bytes": db_bytes,
            "total_bytes": audio_bytes + export_bytes + test_clips_bytes + db_bytes,
        },
        "audio": {
            "total_generated_sec": float(gen.get("total_audio_sec") or 0),
            "ok_generations": int(gen.get("ok_generations") or 0),
            "total_attempts": int(gen.get("total_attempts") or 0),
        },
        "compute": {
            "total_gen_time_sec": float(gen.get("total_gen_time_sec") or 0),
            "total_qa_runs": int(qa.get("total") or 0),
        },
    }


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
