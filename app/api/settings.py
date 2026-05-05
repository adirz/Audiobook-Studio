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
