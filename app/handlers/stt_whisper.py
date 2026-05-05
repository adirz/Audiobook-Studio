"""Whisper STT handler for QA transcription."""

from app.handlers.stt_base import STTHandler
from app.models import ResourceReq


VRAM_BY_SIZE = {
    "tiny": 1.0, "base": 1.0, "small": 2.0,
    "medium": 5.0, "large": 10.0, "large-v2": 10.0, "large-v3": 10.0,
}


class WhisperSTTHandler(STTHandler):

    def __init__(self):
        self._model = None
        self._config = {}

    def get_name(self) -> str:
        return "OpenAI Whisper (Local)"

    def get_config_schema(self) -> dict:
        return {
            "model_size": {
                "type": "select",
                "label": "Model size",
                "options": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                "default": "medium",
            },
        }

    def get_resource_requirements(self) -> ResourceReq:
        size = self._config.get("model_size", "medium")
        return ResourceReq(
            vram_gb=VRAM_BY_SIZE.get(size, 5.0),
            description=f"Whisper {size}",
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, config: dict):
        if self._model is not None:
            return
        import whisper
        self._config = config
        size = config.get("model_size", "medium")
        self._model = whisper.load_model(size)

    def unload_model(self):
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        if not self._model:
            raise RuntimeError("Whisper model not loaded.")
        result = self._model.transcribe(audio_path, language=language)
        return result["text"]
