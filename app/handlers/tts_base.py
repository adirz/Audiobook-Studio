"""Abstract base class for TTS engines."""

from abc import ABC, abstractmethod
from typing import Iterator
from app.models import Voice, Tag, ResourceReq


class TTSHandler(ABC):
    """Interface that all TTS engine implementations must satisfy.

    The application queries this handler for capabilities (voices, tags, IPA)
    and adapts the UI accordingly. This keeps engine-specific details out of
    the pipeline logic.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable engine name."""

    @abstractmethod
    def get_available_voices(self) -> list[Voice]:
        """Return all voices this engine can use."""

    @abstractmethod
    def get_supported_tags(self) -> list[Tag]:
        """Return emotion/expression tags this engine supports.
        Empty list means no tagging support — the tagging step
        will be skipped in the pipeline.
        """

    @abstractmethod
    def supports_ipa(self) -> bool:
        """Whether the engine accepts IPA phonetic input."""

    @abstractmethod
    def get_config_schema(self) -> dict:
        """Return a JSON-schema-like dict describing the settings
        this engine needs. Used to auto-generate the settings UI.
        Example: {"model_path": {"type": "path", "label": "Model directory"}, ...}
        """

    @abstractmethod
    def get_resource_requirements(self) -> ResourceReq:
        """Approximate VRAM/RAM this engine needs when loaded."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently in memory."""

    @abstractmethod
    def load_model(self, config: dict):
        """Load the model into memory. Called before any generation."""

    @abstractmethod
    def unload_model(self):
        """Free GPU/RAM. Called when switching tasks or engines."""

    @abstractmethod
    def generate(self, text: str, voice: str, params: dict | None = None) -> bytes:
        """Generate audio for the given text.

        Args:
            text: Input text, may include engine-specific tags.
            voice: Voice ID from get_available_voices().
            params: Engine-specific overrides (temperature, etc.)

        Returns:
            Raw PCM audio bytes (mono, 16-bit, at the engine's sample rate).
        """

    def generate_stream(self, text: str, voice: str,
                        params: dict | None = None) -> Iterator[bytes]:
        """Streaming generation — yields audio chunks as they are produced.
        Default implementation falls back to non-streaming.
        """
        yield self.generate(text, voice, params)

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Audio sample rate in Hz (e.g. 24000 for Orpheus)."""

    def get_phonetic_suggestions(self, word: str) -> list[dict]:
        """Engine-specific phonetic suggestions for a word.
        Override if the engine has special knowledge about pronunciation.
        Returns list of {"phonetic": str, "rule": str}.
        """
        return []
