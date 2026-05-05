"""Pydantic models for API requests/responses and shared data structures."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# -- Enums --

class PipelineStep(str, Enum):
    NEW = "new"
    SOURCE_SELECTED = "source_selected"
    RANGE_SELECTED = "range_selected"
    EXTRACTED = "extracted"
    SYMBOLS_REVIEWED = "symbols_reviewed"
    CLEANED = "cleaned"
    CHUNKED = "chunked"
    WORDS_SCANNED = "words_scanned"
    PRON_BUILT = "pron_built"
    PRON_TESTED = "pron_tested"
    TAGGED = "tagged"
    TEST_GENERATED = "test_generated"
    GENERATED = "generated"
    QA_DONE = "qa_done"
    REVIEWED = "reviewed"
    EXPORTED = "exported"


class FlagType(str, Enum):
    GARBLED = "garbled"
    MISSING = "missing"
    REPETITION = "repetition"
    ADDED_WORDS = "added_words"
    PRONUNCIATION = "pronunciation"
    OTHER = "other"


# -- Engine capability descriptors --

class Voice(BaseModel):
    id: str
    name: str
    description: str = ""
    sample_url: str = ""


class Tag(BaseModel):
    tag: str              # e.g. "<laugh>"
    description: str = ""
    example: str = ""


class ResourceReq(BaseModel):
    vram_gb: float = 0
    ram_gb: float = 0
    description: str = ""


# -- API request/response models --

class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectInfo(BaseModel):
    name: str
    slug: str
    description: str = ""
    current_step: str = "new"
    source_file: str = ""
    created_at: str = ""
    chapter_count: int = 0
    chunk_count: int = 0
    voice: str = ""


class RangeSelection(BaseModel):
    start_chapter_idx: int
    end_chapter_idx: int


class SymbolDecision(BaseModel):
    symbol_id: int
    is_scene_break: bool


class PronUpdate(BaseModel):
    entry_id: int
    phonetic: Optional[str] = None
    status: Optional[str] = None


class PronTestRequest(BaseModel):
    entry_id: int
    phonetic: str
    context_override: Optional[str] = None


class ChunkFlag(BaseModel):
    chunk_id: str
    flag_type: FlagType
    word_range: Optional[str] = None
    notes: str = ""


class LocationOverride(BaseModel):
    word: str
    phonetic: str
    chunk_id: str
    word_offset: Optional[int] = None
    notes: str = ""


class GenerateRequest(BaseModel):
    chunk_ids: list[str] = []      # empty = all pending
    voice: str = ""
    max_retries: int = 3


class QAThresholds(BaseModel):
    min_similarity: float = 0.85
    auto_pass: float = 0.95


class GenQATestRequest(BaseModel):
    chunk_ids: list[str] = []      # specific chunks to test; empty = random/sample
    voice: str = ""
    max_cycles: int = 3           # how many generate->QA cycles to run
    sample_size: int = 10         # if chunk_ids empty, take this many chunks
    max_retries: int = 3          # per-chunk TTS retries during generation


class ExportRequest(BaseModel):
    scope: str = "full"            # full, chapter, scene
    chapter_ids: list[int] = []
    format: str = "wav"            # wav, mp3, m4b


class TaggingRequest(BaseModel):
    system_prompt: str
    test_chunk_ids: list[str] = []  # if empty, pick random N


class TaggingChatRequest(BaseModel):
    message: str
    system_prompt: str = ""


class TaggingApplyRequest(BaseModel):
    system_prompt: str = ""


class TaggingSaveRequest(BaseModel):
    system_prompt: str


class SettingsUpdate(BaseModel):
    engine: Optional[dict] = None
    resource: Optional[dict] = None
    audio: Optional[dict] = None


# -- Phonetic suggestion helpers --

class PhoneticSuggestion(BaseModel):
    original_segment: str
    suggested_replacement: str
    rule: str  # description of why, e.g. "'c' before 'e' → 'k' or 's'"
