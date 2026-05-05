"""Central registry for all engine handlers.

Holds references to the active TTS, STT, LLM, and extractor handlers.
The application accesses engines through this module rather than importing
handler implementations directly.
"""

from app.handlers.tts_base import TTSHandler
from app.handlers.tts_orpheus import OrpheusTTSHandler
from app.handlers.stt_base import STTHandler
from app.handlers.stt_whisper import WhisperSTTHandler
from app.handlers.llm_base import LLMHandler, NoLLMHandler
from app.handlers.extractor import ExtractorHandler, DocxExtractorHandler

# Available engine classes keyed by config name
TTS_ENGINES: dict[str, type[TTSHandler]] = {
    "orpheus": OrpheusTTSHandler,
}

STT_ENGINES: dict[str, type[STTHandler]] = {
    "whisper": WhisperSTTHandler,
}

LLM_ENGINES: dict[str, type[LLMHandler]] = {
    "none": NoLLMHandler,
}

EXTRACTOR_ENGINES: dict[str, type[ExtractorHandler]] = {
    "docx": DocxExtractorHandler,
}

# Active instances (singletons)
_tts: TTSHandler | None = None
_stt: STTHandler | None = None
_llm: LLMHandler | None = None
_extractor: ExtractorHandler | None = None


def get_tts(engine_name: str | None = None) -> TTSHandler:
    global _tts
    if engine_name and (_tts is None or type(_tts) != TTS_ENGINES.get(engine_name)):
        if _tts is not None and _tts.is_loaded():
            _tts.unload_model()
        _tts = TTS_ENGINES[engine_name]()
    if _tts is None:
        _tts = OrpheusTTSHandler()
    return _tts


def get_stt(engine_name: str | None = None) -> STTHandler:
    global _stt
    if engine_name and (_stt is None or type(_stt) != STT_ENGINES.get(engine_name)):
        if _stt is not None and _stt.is_loaded():
            _stt.unload_model()
        _stt = STT_ENGINES[engine_name]()
    if _stt is None:
        _stt = WhisperSTTHandler()
    return _stt


def get_llm(engine_name: str | None = None) -> LLMHandler:
    global _llm
    if engine_name:
        cls = LLM_ENGINES.get(engine_name, NoLLMHandler)
        if _llm is None or type(_llm) != cls:
            _llm = cls()
    if _llm is None:
        _llm = NoLLMHandler()
    return _llm


def get_extractor(engine_name: str | None = None) -> ExtractorHandler:
    global _extractor
    if engine_name:
        cls = EXTRACTOR_ENGINES.get(engine_name, DocxExtractorHandler)
        _extractor = cls()
    if _extractor is None:
        _extractor = DocxExtractorHandler()
    return _extractor


# For resource manager: unload everything
def unload_all():
    global _tts, _stt
    if _tts and _tts.is_loaded():
        _tts.unload_model()
    if _stt and _stt.is_loaded():
        _stt.unload_model()
