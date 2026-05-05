"""Abstract base class for LLM engines (used for emotion tagging, suggestions)."""

from abc import ABC, abstractmethod
from app.models import ResourceReq


class LLMHandler(ABC):

    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def get_config_schema(self) -> dict: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this LLM is configured and ready (has API key, etc.)."""

    @abstractmethod
    def complete(self, system: str, prompt: str,
                 max_tokens: int = 2000) -> str:
        """Single-turn completion."""

    def chat(self, system: str, messages: list[dict],
             max_tokens: int = 2000) -> str:
        """Multi-turn chat. Default: collapse to single completion."""
        combined = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in messages
        )
        return self.complete(system, combined, max_tokens)


class NoLLMHandler(LLMHandler):
    """Placeholder when no LLM is configured. Tagging step is skipped."""

    def get_name(self) -> str:
        return "None"

    def get_config_schema(self) -> dict:
        return {}

    def is_available(self) -> bool:
        return False

    def complete(self, system: str, prompt: str,
                 max_tokens: int = 2000) -> str:
        raise RuntimeError("No LLM configured.")
