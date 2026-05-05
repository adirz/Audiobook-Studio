"""Global application settings and paths."""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
PROJECTS_DIR = ROOT_DIR / "projects"
STATIC_DIR = APP_DIR / "static"
SETTINGS_FILE = ROOT_DIR / "settings.json"

PROJECTS_DIR.mkdir(exist_ok=True)


@dataclass
class EngineSettings:
    """Which engines are active and their configuration."""
    tts_engine: str = "orpheus"
    tts_config: dict = field(default_factory=lambda: {
        "model_path": "~/orpheus-model",
        "max_model_len": 8196,
        "default_voice": "tara",
        "repetition_penalty": 1.1,
        "temperature": 0.7,
    })

    stt_engine: str = "whisper"
    stt_config: dict = field(default_factory=lambda: {
        "model_size": "medium",
    })

    llm_engine: str = "none"  # "none", "anthropic", "openai", "local"
    llm_config: dict = field(default_factory=lambda: {
        "api_key": "",
        "model": "",
        "base_url": "",
    })

    extractor_engine: str = "docx"
    extractor_config: dict = field(default_factory=dict)


@dataclass
class ResourceSettings:
    """Hardware resource constraints."""
    max_vram_gb: float = 24.0
    max_tokens: int = 8196
    pron_test_buffer_size: int = 5  # pre-generate N test clips ahead
    chunk_test_buffer_size: int = 3


@dataclass
class AudioSettings:
    """Audio processing defaults."""
    target_lufs: float = -16.0  # ACX audiobook standard
    chunk_silence_ms: int = 300
    scene_break_silence_ms: int = 1500
    chapter_break_silence_ms: int = 3000
    crossfade_ms: int = 50
    sample_rate: int = 24000
    export_format: str = "wav"  # wav, mp3, m4b


@dataclass
class AppSettings:
    engine: EngineSettings = field(default_factory=EngineSettings)
    resource: ResourceSettings = field(default_factory=ResourceSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)

    def save(self):
        with open(SETTINGS_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "AppSettings":
        if SETTINGS_FILE.exists():
            try:
                data = json.load(open(SETTINGS_FILE))
                settings = cls(
                    engine=EngineSettings(**data.get("engine", {})),
                    resource=ResourceSettings(**data.get("resource", {})),
                    audio=AudioSettings(**data.get("audio", {})),
                )
                # Migration: stale "./orpheus-model" default breaks vLLM
                # (it gets treated as a HuggingFace repo ID). Clear it so
                # the user is forced to set a proper absolute path.
                if settings.engine.tts_config.get("model_path") == "./orpheus-model":
                    settings.engine.tts_config["model_path"] = "~/orpheus-model"
                    settings.save()
                return settings
            except Exception:
                pass
        settings = cls()
        settings.save()
        return settings


# Singleton
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings.load()
    return _settings


def save_settings(settings: AppSettings):
    global _settings
    _settings = settings
    settings.save()
