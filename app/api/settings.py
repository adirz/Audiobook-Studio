"""API routes for application settings."""

from fastapi import APIRouter
from app.config import get_settings, save_settings, AppSettings, EngineSettings, ResourceSettings, AudioSettings
from app.handlers.registry import get_tts, get_stt, get_llm, TTS_ENGINES, STT_ENGINES, LLM_ENGINES
from app.models import SettingsUpdate
from dataclasses import asdict

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/")
def get_current_settings():
    """Get all current settings."""
    settings = get_settings()
    return asdict(settings)


@router.post("/")
def update_settings(update: SettingsUpdate):
    """Update settings."""
    settings = get_settings()

    if update.engine is not None:
        for k, v in update.engine.items():
            if hasattr(settings.engine, k):
                setattr(settings.engine, k, v)

    if update.resource is not None:
        for k, v in update.resource.items():
            if hasattr(settings.resource, k):
                setattr(settings.resource, k, v)

    if update.audio is not None:
        for k, v in update.audio.items():
            if hasattr(settings.audio, k):
                setattr(settings.audio, k, v)

    save_settings(settings)
    return {"status": "saved"}


@router.get("/engines")
def list_engines():
    """List available engines and their configuration schemas."""
    result = {
        "tts": {},
        "stt": {},
        "llm": {},
    }

    for name, cls in TTS_ENGINES.items():
        handler = cls()
        result["tts"][name] = {
            "name": handler.get_name(),
            "config_schema": handler.get_config_schema(),
            "voices": [v.model_dump() for v in handler.get_available_voices()],
            "tags": [t.model_dump() for t in handler.get_supported_tags()],
            "supports_ipa": handler.supports_ipa(),
            "resources": handler.get_resource_requirements().model_dump(),
        }

    for name, cls in STT_ENGINES.items():
        handler = cls()
        result["stt"][name] = {
            "name": handler.get_name(),
            "config_schema": handler.get_config_schema(),
            "resources": handler.get_resource_requirements().model_dump(),
        }

    for name, cls in LLM_ENGINES.items():
        handler = cls()
        result["llm"][name] = {
            "name": handler.get_name(),
            "config_schema": handler.get_config_schema(),
        }

    return result


@router.get("/tts/voices")
def get_voices():
    """Get voices for the currently active TTS engine."""
    settings = get_settings()
    tts = get_tts(settings.engine.tts_engine)
    return [v.model_dump() for v in tts.get_available_voices()]


@router.get("/tts/tags")
def get_tags():
    """Get supported tags for the currently active TTS engine."""
    settings = get_settings()
    tts = get_tts(settings.engine.tts_engine)
    return [t.model_dump() for t in tts.get_supported_tags()]


@router.get("/usage")
def get_aggregate_usage():
    """Aggregate resource usage across all projects."""
    from app.config import PROJECTS_DIR
    from app.database import ProjectDB

    def dir_size(path) -> int:
        total = 0
        if path.exists():
            for f in path.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except Exception:
                        pass
        return total

    totals = {
        "audio_bytes": 0, "export_bytes": 0,
        "test_clips_bytes": 0, "db_bytes": 0,
        "total_audio_sec": 0.0, "ok_generations": 0,
        "total_attempts": 0, "total_gen_time_sec": 0.0,
        "total_qa_runs": 0, "project_count": 0,
    }

    if not PROJECTS_DIR.exists():
        return _format_usage(totals)

    for d in PROJECTS_DIR.iterdir():
        if not d.is_dir():
            continue
        db_path = d / "project.db"
        if not db_path.exists():
            continue

        totals["project_count"] += 1
        totals["audio_bytes"] += dir_size(d / "audio")
        totals["export_bytes"] += dir_size(d / "export")
        totals["test_clips_bytes"] += dir_size(d / "test_clips")
        totals["db_bytes"] += db_path.stat().st_size

        try:
            db = ProjectDB(db_path)
            gen = db.fetchone(
                """SELECT
                   COUNT(*) as total_attempts,
                   SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) as ok_gen,
                   COALESCE(SUM(CASE WHEN status='ok' THEN duration_sec ELSE 0 END), 0) as audio_sec,
                   COALESCE(SUM(CASE WHEN status='ok' THEN gen_time_sec ELSE 0 END), 0) as gen_time_sec
                   FROM generations"""
            ) or {}
            qa = db.fetchone("SELECT COUNT(*) as total FROM qa_results") or {}
            totals["total_attempts"] += int(gen.get("total_attempts") or 0)
            totals["ok_generations"] += int(gen.get("ok_gen") or 0)
            totals["total_audio_sec"] += float(gen.get("audio_sec") or 0)
            totals["total_gen_time_sec"] += float(gen.get("gen_time_sec") or 0)
            totals["total_qa_runs"] += int(qa.get("total") or 0)
        except Exception:
            pass

    return _format_usage(totals)


def _format_usage(t: dict) -> dict:
    total_disk = t["audio_bytes"] + t["export_bytes"] + t["test_clips_bytes"] + t["db_bytes"]
    return {
        "project_count": t.get("project_count", 0),
        "disk": {
            "audio_bytes": t["audio_bytes"],
            "export_bytes": t["export_bytes"],
            "test_clips_bytes": t["test_clips_bytes"],
            "db_bytes": t["db_bytes"],
            "total_bytes": total_disk,
        },
        "audio": {
            "total_generated_sec": t["total_audio_sec"],
            "ok_generations": t["ok_generations"],
            "total_attempts": t["total_attempts"],
        },
        "compute": {
            "total_gen_time_sec": t["total_gen_time_sec"],
            "total_qa_runs": t["total_qa_runs"],
        },
    }
