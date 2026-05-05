"""Orpheus TTS engine handler."""

import io
import wave
import struct
from typing import Iterator
import threading

from app.handlers.tts_base import TTSHandler
from app.models import Voice, Tag, ResourceReq


class OrpheusTTSHandler(TTSHandler):
    """Handler for the Orpheus local TTS model via vLLM."""

    VOICES = [
        Voice(id="tara", name="Tara", description="Female, warm and clear"),
        Voice(id="leah", name="Leah", description="Female, gentle"),
        Voice(id="jess", name="Jess", description="Female, bright"),
        Voice(id="mia", name="Mia", description="Female, soft"),
        Voice(id="zoe", name="Zoe", description="Female, expressive"),
        Voice(id="zac", name="Zac", description="Male, steady narrator"),
        Voice(id="leo", name="Leo", description="Male, warm"),
        Voice(id="dan", name="Dan", description="Male, deep"),
    ]

    TAGS = [
        Tag(tag="<laugh>", description="Laughter", example="That's hilarious <laugh>"),
        Tag(tag="<chuckle>", description="Soft chuckle", example="Well well <chuckle>"),
        Tag(tag="<sigh>", description="Sigh", example="<sigh> If only things were different"),
        Tag(tag="<cough>", description="Cough", example="<cough> Excuse me"),
        Tag(tag="<sniffle>", description="Sniffle", example="I miss him <sniffle>"),
        Tag(tag="<groan>", description="Groan", example="<groan> Not again"),
        Tag(tag="<yawn>", description="Yawn", example="<yawn> I'm so tired"),
        Tag(tag="<gasp>", description="Gasp", example="<gasp> You scared me!"),
    ]

    def __init__(self):
        self._model = None
        self._config = {}
        # Serialize generation calls to avoid upstream engines that require
        # unique request IDs per in-flight request (prevents "Request req-001 already exists.")
        self._gen_lock = threading.Lock()

    def get_name(self) -> str:
        return "Orpheus TTS (Local)"

    def get_available_voices(self) -> list[Voice]:
        return self.VOICES

    def get_supported_tags(self) -> list[Tag]:
        return self.TAGS

    def supports_ipa(self) -> bool:
        return False

    def get_config_schema(self) -> dict:
        return {
            "model_path": {
                "type": "path",
                "label": "Model directory (absolute path)",
                "default": "~/orpheus-model",
                "required": True,
            },
            "orpheus_pypi_path": {
                "type": "path",
                "label": "Orpheus pypi path (for patched local copy)",
                "default": "orpheus_tts_pypi",
                "required": False,
            },
            "max_model_len": {
                "type": "integer",
                "label": "Max model context length",
                "default": 8196,
                "min": 1024,
                "max": 16384,
            },
            "default_voice": {
                "type": "select",
                "label": "Default voice",
                "options": [v.id for v in self.VOICES],
                "default": "tara",
            },
            "repetition_penalty": {
                "type": "float",
                "label": "Repetition penalty",
                "default": 1.1,
                "min": 1.0,
                "max": 2.0,
                "step": 0.05,
            },
            "temperature": {
                "type": "float",
                "label": "Temperature",
                "default": 0.7,
                "min": 0.1,
                "max": 1.5,
                "step": 0.05,
            },
            "max_tokens": {
                "type": "integer",
                "label": "Max generation tokens",
                "default": 8000,
                "min": 1000,
                "max": 16000,
            },
        }

    def get_resource_requirements(self) -> ResourceReq:
        return ResourceReq(
            vram_gb=12.0,
            ram_gb=4.0,
            description="Orpheus 3B parameter model via vLLM",
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, config: dict):
        if self._model is not None:
            return  # already loaded

        # Lazy import — only pull in heavy deps when actually needed
        import sys
        import os
        # Support local orpheus_tts_pypi directory override
        orpheus_local = config.get("orpheus_pypi_path", "orpheus_tts_pypi")
        sys.path.insert(0, orpheus_local)

        from orpheus_tts import OrpheusModel

        self._config = config
        model_path = config.get("model_path", "./orpheus-model")

        # Resolve to absolute path — relative paths get misinterpreted
        # as HuggingFace repo IDs
        model_path = os.path.abspath(os.path.expanduser(model_path))

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Orpheus model directory not found: {model_path}\n"
                f"Set the correct absolute path in Settings → TTS Engine → Model directory.\n"
                f"Example: /home/youruser/models/orpheus-3b"
            )

        # The underlying OrpheusModel sets up a vLLM engine with default
        # AsyncEngineArgs. Many users need to pass engine args (for example
        # `max_model_len`) to avoid KV-cache / GPU allocation errors. We
        # monkeypatch the model's _setup_engine temporarily so we can inject
        # engine arguments from `config` without requiring users to edit
        # site-packages.
        try:
            from orpheus_tts import OrpheusModel as _OrpheusModel
        except Exception:
            # Fall back to the direct import used above; let the original
            # import/import-time errors surface to the caller.
            from orpheus_tts import OrpheusModel as _OrpheusModel

        orig_setup = getattr(_OrpheusModel, "_setup_engine", None)
        cfg = config or {}

        def _setup_engine_with_config(self_inner):
            try:
                from vllm import AsyncLLMEngine, AsyncEngineArgs
            except Exception:
                # If vllm isn't importable for some reason, fall back to
                # the original setup if available.
                return orig_setup(self_inner) if orig_setup else None

            engine_kwargs = {"model": self_inner.model_name, "dtype": getattr(self_inner, "dtype", None)}
            # Inject user-configured values where provided
            if isinstance(cfg, dict):
                if cfg.get("max_model_len"):
                    try:
                        engine_kwargs["max_model_len"] = int(cfg.get("max_model_len"))
                    except Exception:
                        pass
                if cfg.get("gpu_memory_utilization") is not None:
                    try:
                        engine_kwargs["gpu_memory_utilization"] = float(cfg.get("gpu_memory_utilization"))
                    except Exception:
                        pass

            engine_args = AsyncEngineArgs(**engine_kwargs)
            return AsyncLLMEngine.from_engine_args(engine_args)

        # Apply patch only for the instant of constructing the model, then
        # restore the original method to avoid global side-effects.
        _OrpheusModel._setup_engine = _setup_engine_with_config
        try:
            self._model = _OrpheusModel(model_name=model_path)
        finally:
            if orig_setup is not None:
                _OrpheusModel._setup_engine = orig_setup

    def unload_model(self):
        if self._model is not None:
            # Best-effort cleanup — vLLM doesn't have a clean unload
            del self._model
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def get_sample_rate(self) -> int:
        return 24000

    def generate(self, text: str, voice: str, params: dict | None = None) -> bytes:
        if not self._model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = params or {}
        rep_penalty = params.get(
            "repetition_penalty",
            self._config.get("repetition_penalty", 1.1),
        )
        temperature = params.get(
            "temperature",
            self._config.get("temperature", 0.7),
        )
        max_tokens = params.get(
            "max_tokens",
            self._config.get("max_tokens", 8000),
        )

        prompt = text

        # Serialize calls to the underlying model to avoid duplicate request
        # id collisions in the underlying engine implementation.
        with self._gen_lock:
            tokens = self._model.generate_speech(
                prompt=prompt,
                voice=voice,
                max_tokens=max_tokens,
                repetition_penalty=rep_penalty,
                temperature=temperature,
            )

            # Collect all audio chunks into a single PCM buffer
            pcm_buffer = bytearray()
            for audio_chunk in tokens:
                pcm_buffer.extend(audio_chunk)

        return bytes(pcm_buffer)

    def generate_stream(self, text: str, voice: str,
                        params: dict | None = None) -> Iterator[bytes]:
        if not self._model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = params or {}
        rep_penalty = params.get(
            "repetition_penalty",
            self._config.get("repetition_penalty", 1.1),
        )
        temperature = params.get(
            "temperature",
            self._config.get("temperature", 0.7),
        )
        max_tokens = params.get(
            "max_tokens",
            self._config.get("max_tokens", 8000),
        )

        prompt = text

        # For streaming, also serialize the entire generation so multiple
        # concurrent requests don't collide in the underlying engine.
        with self._gen_lock:
            tokens = self._model.generate_speech(
                prompt=prompt,
                voice=voice,
                max_tokens=max_tokens,
                repetition_penalty=rep_penalty,
                temperature=temperature,
            )

            for audio_chunk in tokens:
                yield audio_chunk
