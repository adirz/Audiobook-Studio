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
        # Load the WAV directly instead of letting whisper call
        # whisper.load_audio() — that helper shells out to ffmpeg, which
        # may not be installed. Our generator only writes 16-bit mono
        # WAVs so stdlib `wave` is enough.
        audio = self._load_wav_as_whisper_input(audio_path)
        result = self._model.transcribe(audio, language=language)
        return result["text"]

    @staticmethod
    def _load_wav_as_whisper_input(path: str):
        """Read a WAV file and return float32 mono samples at 16 kHz.

        Whisper expects 16 kHz mono float32 in [-1, 1]. We produce mono
        16-bit WAVs at the TTS engine's sample rate (typically 24 kHz),
        so we just need to convert types and resample.
        """
        import wave
        import numpy as np

        with wave.open(path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())

        if sample_width == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sample_width == 1:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)

        target_sr = 16000
        if sample_rate != target_sr:
            try:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sample_rate, target_sr)
                arr = resample_poly(arr, target_sr // g, sample_rate // g).astype(np.float32)
            except ImportError:
                # scipy is normally a whisper transitive dep, but fall
                # back to a simple linear-interp resample if it isn't.
                n_in = len(arr)
                n_out = max(1, int(round(n_in * target_sr / sample_rate)))
                xp = np.arange(n_in, dtype=np.float64)
                x = np.linspace(0, n_in - 1, n_out)
                arr = np.interp(x, xp, arr).astype(np.float32)

        return np.ascontiguousarray(arr, dtype=np.float32)
