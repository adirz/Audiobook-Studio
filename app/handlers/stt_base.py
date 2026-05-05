"""Abstract base class for Speech-to-Text engines."""

from abc import ABC, abstractmethod
from app.models import ResourceReq


class STTHandler(ABC):
    """Interface for STT engines used in QA comparison."""

    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def get_config_schema(self) -> dict: ...

    @abstractmethod
    def get_resource_requirements(self) -> ResourceReq: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def load_model(self, config: dict): ...

    @abstractmethod
    def unload_model(self): ...

    @abstractmethod
    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """Transcribe an audio file to text."""
