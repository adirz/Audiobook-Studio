"""API routes for pipeline step execution and status."""

import json
import random
import threading
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.projects import get_project_db, get_project_dir
from app.handlers.registry import get_tts, get_stt, get_llm, get_extractor
from app.config import get_settings
from app.models import (
    RangeSelection, SymbolDecision, GenerateRequest,
    QAThresholds, ExportRequest, TaggingRequest,
    TaggingChatRequest, TaggingApplyRequest, TaggingSaveRequest,
    PronUpdate, PronTestRequest, GenQATestRequest,
    ChapterNarrationUpdate, ChunkVoiceUpdate, BulkChunkVoiceUpdate,
)
from app.pipeline import analyze, clean, chunk, tagging, generate, qa, merge
from app.pipeline.pron_buffer import get_or_create_buffer

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# In-memory progress tracking for long-running tasks
_task_progress: dict[str, dict] = {}
_task_controls: dict[str, dict] = {}
# Dedicated worker threads keyed by task_id. Long-running jobs run here
# so they don't compete with FastAPI's request-handling threadpool.
_task_threads: dict[str, threading.Thread] = {}


def _run_in_dedicated_thread(task_id: str, func) -> bool:
    """Run a long-running job in its own daemon thread.

    Returns False if a thread for this task is already running (caller
    should treat that as "task already in progress"). Returns True if a
    new thread was started.

    Using a dedicated thread (instead of FastAPI's BackgroundTasks, which
    runs sync functions in the shared request threadpool) keeps the HTTP
    server responsive even during multi-hour TTS / QA / tagging runs.
    """
    existing = _task_threads.get(task_id)
    if existing and existing.is_alive():
        return False

    def wrapper():
        try:
            func()
        except Exception as e:
            traceback.print_exc()
            _task_progress[task_id] = {
                "status": "error",
                "message": f"Task crashed: {e}",
            }

    thread = threading.Thread(
        target=wrapper, name=f"task-{task_id}", daemon=True
    )
    _task_threads[task_id] = thread
    thread.start()
    return True


# ─── Step 2: Heading scan & range selection ────────────────────────────

@router.get("/{slug}/headings")
def scan_headings(slug: str):
    """Scan the manuscript for headings (for range selection UI)."""
    db = get_project_db(slug)
    source = db.get_meta("source_file")
    if not source:
        raise HTTPException(400, "No manuscript uploaded")

    extractor = get_extractor()
    headings = extractor.scan_headings(source)
    return [
        {
            "index": h.index,
            "title": h.title,
            "style": h.style,
            "preview": h.preview,
        }
        for h in headings
    ]


@router.get("/{slug}/preview")
def text_preview(slug: str, start: int = 0, count: int = 50):
    """Get raw paragraph text for scrollable preview."""
    db = get_project_db(slug)
    source = db.get_meta("source_file")
    if not source:
        raise HTTPException(400, "No manuscript uploaded")

    extractor = get_extractor()
    paras = extractor.get_full_text_preview(source, start, count)
    return {"paragraphs": paras, "start": start}


# ─── Step 3: Extract ──────────────────────────────────────────────────

@router.post("/{slug}/extract")
def extract_manuscript(slug: str, selection: RangeSelection):
    """Extract chapters from the manuscript."""
    db = get_project_db(slug)
    source = db.get_meta("source_file")
    if not source:
        raise HTTPException(400, "No manuscript uploaded")

    extractor = get_extractor()
    chapters = extractor.extract_chapters(
        source,
        start_idx=selection.start_chapter_idx,
        end_idx=selection.end_chapter_idx,
    )

    if not chapters:
        raise HTTPException(400, "No chapters found in the selected range")

    db.insert_chapters([
        {"idx": ch.idx, "title": ch.title, "raw_text": ch.raw_text}
        for ch in chapters
    ])

    db.set_meta("current_step", "extracted")
    return {"chapters": len(chapters)}


# ─── Step 4: Symbol detection & review ─────────────────────────────────

@router.get("/{slug}/symbols")
def get_symbols(slug: str):
    """Get all detected non-standard symbols."""
    db = get_project_db(slug)

    # Run detection if not done yet
    existing = db.get_symbols()
    if not existing:
        analyze.find_nonstandard_symbols(db)
        # Re-read from DB to get proper IDs
        existing = db.get_symbols()

    return existing


@router.post("/{slug}/symbols")
def decide_symbols(slug: str, decisions: list[SymbolDecision]):
    """User decides which symbols are scene breaks."""
    db = get_project_db(slug)
    for d in decisions:
        db.update_symbol(d.symbol_id, d.is_scene_break)
    db.set_meta("current_step", "symbols_reviewed")
    return {"updated": len(decisions)}


# ─── Step 5: Clean & chunk ─────────────────────────────────────────────

@router.post("/{slug}/clean")
def clean_text(slug: str):
    """Clean all chapter text."""
    db = get_project_db(slug)
    clean.clean_all_chapters(db)
    return {"status": "cleaned"}


@router.post("/{slug}/chunk")
def chunk_text(slug: str, max_words: int = 150, min_words: int = 30,
               narrate_titles: bool = False, chapter_prefix: bool = True):
    """Chunk all chapters, optionally inserting a narrated title chunk per chapter."""
    db = get_project_db(slug)
    count = chunk.chunk_all_chapters(db, max_words, min_words, narrate_titles, chapter_prefix)
    return {"chunks_created": count}


# ─── Chapter narration titles (for the title-chunk editor) ────────────

@router.get("/{slug}/chapters")
def list_chapters_with_narration(slug: str, chapter_prefix: bool = True):
    """List chapters with their auto-default and explicit narration title.

    Returned per chapter:
      - id, idx, title (the original heading)
      - narration_title (the explicit override, or null if unset)
      - auto_title (what the engine would use if narration_title is null
        and global narrate_titles is on — useful for placeholders in UI)
      - effective_title (what will actually be narrated given the policy)
    """
    db = get_project_db(slug)
    chapters = db.get_chapters()
    out = []
    for ch in chapters:
        auto = chunk.auto_narration_title(ch.get("title", ""), ch["idx"], chapter_prefix)
        nt = ch.get("narration_title")
        if nt is None:
            effective = auto
        else:
            effective = nt  # may be empty string → no narration
        out.append({
            "id": ch["id"],
            "idx": ch["idx"],
            "title": ch["title"],
            "narration_title": nt,
            "auto_title": auto,
            "effective_title": effective,
        })
    return out


@router.patch("/{slug}/chapters/{chapter_id}/narration")
def update_chapter_narration(slug: str, chapter_id: int,
                             req: ChapterNarrationUpdate):
    """Update a chapter's narration title.

    Pass ``narration_title`` as:
      - omitted/null → revert to auto default
      - "" (empty)   → don't narrate this chapter's title
      - any text     → use as the title-chunk text verbatim
    """
    db = get_project_db(slug)
    db.update_chapter_narration_title(chapter_id, req.narration_title)
    return {"status": "updated"}


# ─── Per-chunk voice override ─────────────────────────────────────────

@router.patch("/{slug}/chunks/{chunk_id}/voice")
def update_chunk_voice(slug: str, chunk_id: str, req: ChunkVoiceUpdate):
    """Set or clear a per-chunk voice override.

    Pass ``voice`` as null/empty to revert the chunk to inheriting the
    voice supplied in the next generate request.
    """
    db = get_project_db(slug)
    voice = req.voice or None
    db.update_chunk_voice(chunk_id, voice)
    return {"status": "updated", "chunk_id": chunk_id, "voice": voice}


@router.post("/{slug}/chunks/voice-bulk")
def bulk_update_chunk_voice(slug: str, req: BulkChunkVoiceUpdate):
    """Apply the same voice override to many chunks at once.

    Used by the Generate Audio UI when the user picks a voice for an
    entire scene or for the current selection.
    """
    db = get_project_db(slug)
    voice = req.voice or None
    n = db.bulk_update_chunk_voice(req.chunk_ids, voice)
    return {"status": "updated", "count": n, "voice": voice}


@router.post("/{slug}/title-chunks/refresh")
def refresh_title_chunks(slug: str, chapter_prefix: bool = True):
    """Rebuild only the per-chapter title chunks from current narration_title.

    Cheaper than re-running the full chunker — preserves existing content
    chunks (and their generations / QA / flags) and just deletes/recreates
    rows where ``is_title_chunk = 1``. Global indexes are kept stable.
    """
    db = get_project_db(slug)
    # Drop existing title chunks (and their downstream generations / QA
    # rows so we don't keep stale audio for a chapter the user just opted
    # out of narrating).
    title_rows = db.fetchall("SELECT id FROM chunks WHERE is_title_chunk=1")
    for row in title_rows:
        cid = row["id"]
        db.execute("DELETE FROM qa_results WHERE chunk_id=?", (cid,))
        db.execute("DELETE FROM generations WHERE chunk_id=?", (cid,))
        db.execute("DELETE FROM user_flags WHERE chunk_id=?", (cid,))
    db.execute("DELETE FROM chunks WHERE is_title_chunk=1")
    db.commit()

    # Recreate based on each chapter's narration_title (or auto default).
    chapters = db.get_chapters()
    new_titles = []
    for ch in chapters:
        title_text = chunk.resolve_narration_title(ch, chapter_prefix)
        if not title_text:
            continue
        # Place title chunk just before its chapter's first content chunk
        first_content = db.fetchone(
            "SELECT global_index FROM chunks WHERE chapter_id=? AND is_title_chunk=0 ORDER BY global_index LIMIT 1",
            (ch["id"],),
        )
        gi = first_content["global_index"] if first_content else 0
        new_titles.append({
            "id": f"ch{ch['idx']:03d}_title",
            "chapter_id": ch["id"],
            "local_index": -1,
            "global_index": gi,
            "original_text": title_text,
            "cleaned_text": title_text,
            "word_count": len(title_text.split()),
            "scene_break_after": 0,
            "chapter_break_after": 0,
            "is_title_chunk": 1,
        })

    if new_titles:
        db.insert_chunks(new_titles)
    return {"created": len(new_titles)}


# ─── Step 6-7: Word scanning ──────────────────────────────────────────

@router.get("/{slug}/words")
def get_nonstandard_words(slug: str):
    """Get all non-standard words (or scan if not done)."""
    db = get_project_db(slug)
    entries = db.get_pron_entries()
    if not entries:
        entries = analyze.find_nonstandard_words(db)
        db.set_meta("current_step", "words_scanned")
    return entries


@router.post("/{slug}/words/rescan")
def rescan_words(slug: str):
    """Clear existing word entries and rescan. Use after fixing detection."""
    db = get_project_db(slug)
    # Remove all pending/skipped entries (keep approved ones the user already set)
    db.execute(
        "DELETE FROM pron_attempts WHERE pron_entry_id IN "
        "(SELECT id FROM pron_entries WHERE status IN ('pending', 'skipped'))"
    )
    db.execute(
        "DELETE FROM pron_entries WHERE status IN ('pending', 'skipped')"
    )
    db.commit()
    entries = analyze.find_nonstandard_words(db)
    db.set_meta("current_step", "words_scanned")
    return entries


# ─── Step 8: Standard word overrides ──────────────────────────────────

@router.delete("/{slug}/pron/{entry_id}")
def remove_pron_entry(slug: str, entry_id: int):
    """Remove a word from the non-standard list (mark as standard)."""
    db = get_project_db(slug)
    db.execute("DELETE FROM pron_attempts WHERE pron_entry_id=?", (entry_id,))
    db.execute("DELETE FROM pron_entries WHERE id=?", (entry_id,))
    db.commit()
    return {"status": "removed"}


@router.post("/{slug}/pron/standard-override")
def add_standard_override(slug: str, word: str, phonetic: str):
    """Add a standard English word that should be pronounced differently."""
    db = get_project_db(slug)
    # Find context from the book
    context = analyze.get_context_for_word(db, word)
    chunk_id = ""
    chunks = db.get_chunks()
    for ch in chunks:
        if word.lower() in ch["original_text"].lower():
            chunk_id = ch["id"]
            break

    db.insert_pron_entry(
        word=word,
        frequency=0,  # manual override
        example_chunk_id=chunk_id,
        example_context=context,
        type_tag="standard-override",
    )
    return {"status": "added", "word": word}


# ─── Step 9: Pronunciation testing ────────────────────────────────────

@router.get("/{slug}/pron")
def get_pron_entries(slug: str, status: str = None):
    """Get pronunciation entries, optionally filtered by status."""
    db = get_project_db(slug)
    return db.get_pron_entries(status=status)


@router.get("/{slug}/pron/export")
def export_pron(slug: str):
    """Export pronunciation dictionary as a downloadable JSON file."""
    import json as _json
    from fastapi.responses import Response
    db = get_project_db(slug)
    entries = db.get_pron_entries()
    exportable = [
        {
            "word": e["word"],
            "phonetic": e["phonetic"],
            "type_tag": e["type_tag"] or "",
            "status": e["status"],
            "notes": e["notes"] or "",
            "is_global": e["is_global"] if e["is_global"] is not None else 1,
        }
        for e in entries
        if e["status"] in ("approved", "skipped")
    ]
    content = _json.dumps({"version": 1, "entries": exportable}, indent=2, ensure_ascii=False)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="pron_{slug}.json"'},
    )


@router.post("/{slug}/pron/import")
def import_pron(slug: str, payload: dict):
    """Import pronunciation entries from a JSON dictionary.

    For each entry: updates the existing word's phonetic/status if found,
    otherwise inserts it as a new entry with no book context.
    """
    db = get_project_db(slug)
    entries = payload.get("entries", [])
    imported = 0
    updated = 0
    skipped = 0
    for e in entries:
        word = (e.get("word") or "").strip()
        if not word:
            continue
        phonetic = e.get("phonetic") or None
        status = e.get("status") or "approved"
        type_tag = e.get("type_tag") or ""
        notes = e.get("notes") or ""
        is_global = int(e.get("is_global", 1))

        existing = db.fetchone("SELECT id, status FROM pron_entries WHERE word=?", (word,))
        if existing:
            if existing["status"] == "approved":
                skipped += 1
                continue
            db.update_pron_entry(
                existing["id"],
                phonetic=phonetic,
                status=status,
                notes=notes,
                is_global=is_global,
            )
            updated += 1
        else:
            db.insert_pron_entry(
                word=word,
                frequency=0,
                example_chunk_id="",
                example_context="",
                type_tag=type_tag,
            )
            new_id = db.fetchone("SELECT id FROM pron_entries WHERE word=?", (word,))["id"]
            db.update_pron_entry(
                new_id,
                phonetic=phonetic,
                status=status,
                notes=notes,
                is_global=is_global,
            )
            imported += 1

    return {"imported": imported, "updated": updated, "skipped": skipped}


@router.post("/{slug}/pron/{entry_id}/update")
def update_pron_entry(slug: str, entry_id: int, req: PronUpdate):
    """Update a pronunciation entry."""
    db = get_project_db(slug)
    updates = {}
    if req.phonetic is not None:
        updates["phonetic"] = req.phonetic
    if req.status is not None:
        updates["status"] = req.status
    if updates:
        db.update_pron_entry(entry_id, **updates)
    return {"status": "updated"}


@router.get("/{slug}/pron/{entry_id}/suggestions")
def get_suggestions(slug: str, entry_id: int):
    """Get phonetic suggestions for a pronunciation entry."""
    db = get_project_db(slug)
    entry = db.fetchone("SELECT * FROM pron_entries WHERE id=?", (entry_id,))
    if not entry:
        raise HTTPException(404, "Entry not found")

    from app.pipeline.pronunciation import get_phonetic_suggestions
    suggestions = get_phonetic_suggestions(entry["word"])
    return [s.model_dump() for s in suggestions]


@router.post("/{slug}/pron/{entry_id}/test")
def test_pronunciation(slug: str, entry_id: int, req: PronTestRequest):
    """Generate a test audio clip for a pronunciation attempt."""
    db = get_project_db(slug)
    settings = get_settings()
    project_dir = get_project_dir(slug)

    entry = db.fetchone("SELECT * FROM pron_entries WHERE id=?", (entry_id,))
    if not entry:
        raise HTTPException(404, "Entry not found")

    tts = get_tts(settings.engine.tts_engine)
    if not tts.is_loaded():
        try:
            tts.load_model(settings.engine.tts_config)
        except Exception as e:
            msg = str(e)
            # vLLM / Orpheus specific guidance
            if 'max seq len' in msg or 'KV cache' in msg or 'max_model_len' in msg:
                raise HTTPException(503,
                    f"TTS engine initialization failed: {msg}. "
                    f"Try lowering 'Max model context length' in Settings → TTS Engine (max_model_len), or increase GPU memory allocation.")
            if 'CUDA out of memory' in msg or 'out of memory' in msg:
                raise HTTPException(503,
                    f"CUDA out of memory while loading TTS engine: {msg}. "
                    f"Consider using a smaller model, increasing GPU RAM allowance (Settings → Resource → max_vram_gb), or set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.")
            raise HTTPException(503,
                f"Could not load TTS engine: {msg}. "
                f"Check Settings → TTS Engine → Model directory and max_model_len.")

    # Get context text
    if req.context_override:
        context = req.context_override
    elif entry["example_context"]:
        context = entry["example_context"]
    else:
        context = analyze.get_context_for_word(db, entry["word"])

    # Replace the word with the phonetic version in context
    import re
    test_text = re.sub(
        rf"\b{re.escape(entry['word'])}\b",
        req.phonetic,
        context,
        flags=re.IGNORECASE,
    )

    # Get next attempt number
    existing = db.fetchall(
        "SELECT * FROM pron_attempts WHERE pron_entry_id=?", (entry_id,)
    )
    attempt_num = len(existing) + 1

    voice = settings.engine.tts_config.get("default_voice", "tara")
    final_audio_path = project_dir / "test_clips" / f"pron_{entry_id}_v{attempt_num:02d}.wav"

    # Prefer using the pronunciation buffer if it already has a prefilled clip.
    buffer = get_or_create_buffer(slug, tts, project_dir / "test_clips", voice, settings.resource.pron_test_buffer_size)

    # If prefilled, promote to final attempt filename. Otherwise request urgent generation
    result_path = None
    try:
        buf_res = buffer.get_result(entry_id)
        if buf_res:
            # If this is a prefill file, check its metadata to ensure it was
            # generated with the same phonetic the user requested. If it
            # matches, promote it to the final attempt filename; otherwise
            # queue an urgent generation for the requested phonetic.
            try:
                from pathlib import Path
                pre = Path(buf_res)
                if str(buf_res).endswith('_prefill.wav'):
                    meta_path = pre.with_suffix('.json')
                    meta = None
                    if meta_path.exists():
                        try:
                            import json
                            with open(meta_path, 'r', encoding='utf-8') as mf:
                                meta = json.load(mf)
                        except Exception:
                            meta = None

                    # If meta indicates same phonetic, promote; otherwise request new generation
                    if meta and str(meta.get('phonetic_used','')).strip().lower() == str(req.phonetic).strip().lower():
                        import os
                        try:
                            os.replace(str(pre), str(final_audio_path))
                            # move metadata file too if present
                            try:
                                if meta_path.exists():
                                    new_meta = final_audio_path.with_suffix('.json')
                                    os.replace(str(meta_path), str(new_meta))
                            except Exception:
                                pass
                            result_path = str(final_audio_path)
                        except Exception:
                            result_path = str(buf_res)
                    else:
                        # Ask buffer to urgently generate the requested phonetic
                        buffer.request(entry_id, entry['word'], req.phonetic, test_text, attempt_num)
                        buf_res = None
                        result_path = None
                else:
                    # already written to a numbered file (unlikely), use it
                    result_path = str(buf_res)
            except Exception:
                result_path = str(buf_res)
        else:
            # Queue an urgent generation via the buffer and wait briefly for it to complete
            buffer.request(entry_id, entry['word'], req.phonetic, test_text, attempt_num)
            import time, os
            waited = 0.0
            timeout = 8.0
            interval = 0.25
            while waited < timeout:
                buf_res = buffer.get_result(entry_id)
                if buf_res:
                    # Promote if prefill (but only if metadata matches), otherwise use buffer result
                    try:
                        from pathlib import Path
                        br = Path(buf_res)
                        if str(buf_res).endswith('_prefill.wav'):
                            meta_path = br.with_suffix('.json')
                            meta = None
                            if meta_path.exists():
                                try:
                                    import json
                                    with open(meta_path, 'r', encoding='utf-8') as mf:
                                        meta = json.load(mf)
                                except Exception:
                                    meta = None
                            if meta and str(meta.get('phonetic_used','')).strip().lower() == str(req.phonetic).strip().lower():
                                import os
                                try:
                                    os.replace(str(br), str(final_audio_path))
                                    try:
                                        new_meta = final_audio_path.with_suffix('.json')
                                        if meta_path.exists():
                                            os.replace(str(meta_path), str(new_meta))
                                    except Exception:
                                        pass
                                    result_path = str(final_audio_path)
                                except Exception:
                                    result_path = str(buf_res)
                                break
                            else:
                                # Not matching; continue waiting for buffer.request to produce result
                                buf_res = None
                                result_path = None
                        else:
                            result_path = str(buf_res)
                            break
                    except Exception:
                        result_path = str(buf_res)
                        break
                    break
                time.sleep(interval)
                waited += interval

    except Exception:
        result_path = None

    # If buffer did not produce a result within timeout, fall back to synchronous generation
    from app.pipeline.pronunciation import generate_pron_test
    if not result_path:
        try:
            result_path = generate_pron_test(tts, test_text, voice, final_audio_path)
        except Exception as e:
            emsg = str(e)
            if 'CUDA out of memory' in emsg or 'out of memory' in emsg:
                raise HTTPException(500, f"TTS generation failed (CUDA OOM): {emsg}. Try reducing model size or adjust Resource → max_vram_gb in Settings.")
            raise HTTPException(500, f"TTS generation failed: {emsg}")

    attempt_id = db.insert_pron_attempt(
        entry_id, attempt_num, req.phonetic, result_path
    )

    return {
        "attempt_id": attempt_id,
        "attempt_number": attempt_num,
        "audio_url": f"/api/audio/{slug}/test_clips/pron_{entry_id}_v{attempt_num:02d}.wav",
        "phonetic": req.phonetic,
        "test_text": test_text,
    }


@router.delete("/{slug}/pron/{entry_id}/attempt/{attempt_id}")
def delete_pron_attempt(slug: str, entry_id: int, attempt_id: int):
    """Delete a generated pronunciation attempt and its file."""
    db = get_project_db(slug)
    row = db.fetchone("SELECT * FROM pron_attempts WHERE id=?", (attempt_id,))
    if not row:
        raise HTTPException(404, "Attempt not found")
    audio_path = row.get("audio_path")
    if audio_path:
        try:
            import os
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
    db.execute("DELETE FROM pron_attempts WHERE id=?", (attempt_id,))
    db.commit()
    return {"status": "deleted"}


@router.post("/{slug}/pron/{entry_id}/choose/{attempt_id}")
def choose_attempt(slug: str, entry_id: int, attempt_id: int):
    """Choose a pronunciation attempt as the final one."""
    db = get_project_db(slug)
    db.choose_pron_attempt(attempt_id)

    # Get the chosen phonetic and update the entry
    attempt = db.fetchone("SELECT * FROM pron_attempts WHERE id=?", (attempt_id,))
    if attempt:
        db.update_pron_entry(entry_id,
                             phonetic=attempt["phonetic_used"],
                             status="approved")

    return {"status": "chosen"}


@router.post("/{slug}/pron/skip-all")
def skip_remaining(slug: str):
    """Skip all remaining pending pronunciation entries."""
    db = get_project_db(slug)
    db.execute(
        "UPDATE pron_entries SET status='skipped' WHERE status='pending'"
    )
    db.commit()
    db.set_meta("current_step", "pron_tested")
    return {"status": "all_skipped"}


@router.post("/{slug}/pron/{entry_id}/test-original")
def test_original_pronunciation(slug: str, entry_id: int):
    """Play the word as-is (no phonetic substitution) so user can compare."""
    db = get_project_db(slug)
    settings = get_settings()
    project_dir = get_project_dir(slug)

    entry = db.fetchone("SELECT * FROM pron_entries WHERE id=?", (entry_id,))
    if not entry:
        raise HTTPException(404, "Entry not found")

    tts = get_tts(settings.engine.tts_engine)
    if not tts.is_loaded():
        try:
            tts.load_model(settings.engine.tts_config)
        except Exception as e:
            raise HTTPException(503,
                f"Could not load TTS engine: {e}. "
                f"Check Settings → TTS Engine → Model directory.")

    context = entry["example_context"] or f"The word {entry['word']} appears here."
    voice = settings.engine.tts_config.get("default_voice", "tara")
    audio_path = project_dir / "test_clips" / f"pron_{entry_id}_original.wav"

    from app.pipeline.pronunciation import generate_pron_test
    try:
        result_path = generate_pron_test(tts, context, voice, audio_path)
    except Exception as e:
        raise HTTPException(500, f"TTS generation failed: {e}")

    return {
        "audio_url": f"/api/audio/{slug}/test_clips/pron_{entry_id}_original.wav",
        "test_text": context,
    }


@router.get("/{slug}/pron/{entry_id}/attempts")
def get_pron_attempts(slug: str, entry_id: int):
    """Get all test attempts for a pronunciation entry."""
    db = get_project_db(slug)
    attempts = db.fetchall(
        "SELECT * FROM pron_attempts WHERE pron_entry_id=? ORDER BY attempt_number",
        (entry_id,),
    )
    for a in attempts:
        a["audio_url"] = f"/api/audio/{slug}/test_clips/pron_{entry_id}_v{a['attempt_number']:02d}.wav"
    # If no DB attempt exists, also expose any prefill file on disk so UI
    # can show instant variations generated by the background buffer.
    project_dir = get_project_dir(slug)
    prefill_path = project_dir / "test_clips" / f"pron_{entry_id}_prefill.wav"
    if prefill_path.exists():
        # Only add a prefill entry if it's not already represented in DB
        exists = any(str(prefill_path) == a.get("audio_path") for a in attempts)
        if not exists:
            ent = db.fetchone("SELECT phonetic, word FROM pron_entries WHERE id=?", (entry_id,)) or {}
            phonetic_used = ent.get("phonetic") or ent.get("word") or ''
            attempts.append({
                "id": "prefill",
                "pron_entry_id": entry_id,
                "attempt_number": 0,
                "phonetic_used": phonetic_used,
                "audio_path": str(prefill_path),
                "audio_url": f"/api/audio/{slug}/test_clips/pron_{entry_id}_prefill.wav",
                "chosen": 0,
                "created_at": None,
            })
    return attempts


@router.post("/{slug}/pron/prefill")
def prefill_pron_buffer(slug: str, current_entry_id: int = 0,
                        buffer_size: int = 5):
    """Pre-generate pronunciation test clips for upcoming entries."""
    db = get_project_db(slug)
    settings = get_settings()
    project_dir = get_project_dir(slug)

    tts = get_tts(settings.engine.tts_engine)
    if not tts.is_loaded():
        try:
            tts.load_model(settings.engine.tts_config)
        except Exception as e:
            msg = str(e)
            if 'max seq len' in msg or 'KV cache' in msg or 'max_model_len' in msg:
                raise HTTPException(503, f"TTS engine initialization failed: {msg}. Check Settings → TTS Engine and resource settings.")
            if 'CUDA out of memory' in msg or 'out of memory' in msg:
                raise HTTPException(503, f"CUDA out of memory while loading TTS engine: {msg}. Check Settings → Resource → max_vram_gb.")
            raise HTTPException(503, f"Could not load TTS engine: {msg}")

    voice = settings.engine.tts_config.get("default_voice", "tara")

    from app.pipeline.pron_buffer import get_or_create_buffer
    buffer = get_or_create_buffer(
        slug, tts, project_dir / "test_clips", voice, buffer_size
    )

    # Get pending entries and start prefill from the current entry (inclusive)
    entries = db.get_pron_entries(status="pending")
    start_from = 0
    if current_entry_id:
        for i, e in enumerate(entries):
            if e["id"] == current_entry_id:
                start_from = i
                break
    upcoming = entries[start_from:start_from + buffer_size]

    buffer.prefill(upcoming)
    return {"queued": len(upcoming)}


# ─── Step 10: Emotion tagging ─────────────────────────────────────────

@router.get("/{slug}/tagging/available-tags")
def get_available_tags(slug: str):
    """Get tags supported by the current TTS engine."""
    settings = get_settings()
    tts = get_tts(settings.engine.tts_engine)
    tags = tts.get_supported_tags()
    return [t.model_dump() for t in tags]


@router.post("/{slug}/tagging/test")
def test_tagging(slug: str, req: TaggingRequest):
    """Test emotion tagging on a few chunks."""
    db = get_project_db(slug)
    settings = get_settings()
    llm = get_llm(settings.engine.llm_engine)

    if not llm.is_available():
        raise HTTPException(400, "No LLM configured")

    results = tagging.tag_test_chunks(
        db, llm, req.system_prompt,
        chunk_ids=req.test_chunk_ids or None,
    )
    return results


@router.post("/{slug}/tagging/apply")
def apply_tagging(slug: str, req: TaggingApplyRequest):
    """Tag all chunks (runs in background)."""
    db = get_project_db(slug)
    settings = get_settings()
    llm = get_llm(settings.engine.llm_engine)

    if not llm.is_available():
        raise HTTPException(400, "No LLM configured")

    system_prompt = req.system_prompt
    if not system_prompt:
        config = db.get_tagging_config()
        if config:
            system_prompt = config["system_prompt"]
        else:
            tts = get_tts(settings.engine.tts_engine)
            system_prompt = tagging.build_tagging_prompt(tts)

    task_id = f"{slug}_tagging"
    _task_progress[task_id] = {"current": 0, "total": 0, "status": "running"}

    def run_tagging():
        def on_progress(current, total):
            _task_progress[task_id] = {
                "current": current, "total": total, "status": "running"
            }
        tagging.tag_all_chunks(db, llm, system_prompt, progress_callback=on_progress)
        _task_progress[task_id]["status"] = "done"

    if not _run_in_dedicated_thread(task_id, run_tagging):
        raise HTTPException(409, "Tagging task already running")
    return {"task_id": task_id, "status": "started"}


@router.get("/{slug}/tagging/default-prompt")
def get_default_tagging_prompt(slug: str):
    """Get the default tagging prompt based on current TTS engine."""
    db = get_project_db(slug)
    settings = get_settings()

    # Return saved prompt if it exists
    config = db.get_tagging_config()
    if config:
        return {"prompt": config["system_prompt"], "saved": True}

    tts = get_tts(settings.engine.tts_engine)
    prompt = tagging.build_tagging_prompt(tts)
    return {"prompt": prompt, "saved": False}


@router.post("/{slug}/tagging/chat")
def tagging_chat(slug: str, req: TaggingChatRequest):
    """Chat with the LLM to adjust the tagging prompt.

    The user can say things like "use fewer tags" or "don't tag narration"
    and the LLM will produce an updated system prompt.
    """
    db = get_project_db(slug)
    settings = get_settings()
    llm = get_llm(settings.engine.llm_engine)

    if not llm.is_available():
        raise HTTPException(400, "No LLM configured")

    system_prompt = req.system_prompt
    if not system_prompt:
        config = db.get_tagging_config()
        system_prompt = config["system_prompt"] if config else ""

    meta_prompt = f"""You are helping a user refine a system prompt for audiobook emotion tagging.

The current system prompt is:
---
{system_prompt}
---

The user wants to adjust it. Based on their message below, produce an UPDATED version
of the system prompt that incorporates their feedback. Return ONLY the updated prompt,
nothing else — no explanations, no markdown fences.

User's request: {req.message}"""

    updated = llm.complete(
        system="You are a prompt engineering assistant.",
        prompt=meta_prompt,
        max_tokens=2000,
    )

    # Save draft (not approved yet)
    db.save_tagging_config(updated.strip(), approved=False)

    return {"updated_prompt": updated.strip()}


@router.post("/{slug}/tagging/save-prompt")
def save_tagging_prompt(slug: str, req: TaggingSaveRequest):
    """Save and approve a tagging prompt."""
    db = get_project_db(slug)
    db.save_tagging_config(req.system_prompt, approved=True)
    return {"status": "saved"}


# ─── Step 11-12: Audio generation ─────────────────────────────────────

@router.post("/{slug}/generate")
def generate_audio(slug: str, req: GenerateRequest):
    """Generate audio for chunks (runs in background)."""
    db = get_project_db(slug)
    settings = get_settings()
    project_dir = get_project_dir(slug)

    tts = get_tts(settings.engine.tts_engine)
    if not tts.is_loaded():
        try:
            stt = get_stt(settings.engine.stt_engine)
            if stt.is_loaded():
                stt.unload_model()
            tts.load_model(settings.engine.tts_config)
        except Exception as e:
            raise HTTPException(503,
                f"Could not load TTS engine: {e}. "
                f"Check Settings → TTS Engine → Model directory.")

    voice = req.voice or settings.engine.tts_config.get("default_voice", "tara")
    audio_dir = project_dir / "audio"

    # Apply pronunciation to all chunks first
    from app.pipeline.pronunciation import apply_all_pronunciation
    apply_all_pronunciation(db)

    # Resolve which chunk IDs to generate based on explicit list or mode
    if req.chunk_ids:
        resolved_chunk_ids = req.chunk_ids
    elif req.mode == "all":
        resolved_chunk_ids = [c["id"] for c in db.get_chunks()]
    elif req.mode == "failed":
        resolved_chunk_ids = []
        for ch in db.get_chunks():
            gen = db.get_latest_generation(ch["id"])
            if gen and gen["status"] == "error":
                resolved_chunk_ids.append(ch["id"])
                continue
            qa_r = db.fetchone(
                "SELECT status FROM qa_results WHERE chunk_id=? ORDER BY id DESC LIMIT 1",
                (ch["id"],),
            )
            if qa_r and qa_r["status"] == "fail":
                resolved_chunk_ids.append(ch["id"])
                continue
            flags = db.fetchall(
                "SELECT id FROM user_flags WHERE chunk_id=? AND resolved=0",
                (ch["id"],),
            )
            if flags:
                resolved_chunk_ids.append(ch["id"])
    else:
        # "pending": generate_batch will select chunks without a successful generation
        resolved_chunk_ids = None

    task_id = f"{slug}_generate"
    # Persist chosen voice to project metadata so UI remembers it per-project
    try:
        db.set_meta("voice", voice)
    except Exception:
        pass

    _task_progress[task_id] = {"current": 0, "total": 0, "status": "running"}
    _task_controls[task_id] = {"stop": False, "pause": False}

    def make_stop_check():
        def stop_check():
            ctrl = _task_controls.get(task_id, {})
            # If paused, block here until resumed or stopped
            while ctrl.get("pause"):
                import time
                time.sleep(0.5)
                ctrl = _task_controls.get(task_id, {})
            return bool(ctrl.get("stop", False))
        return stop_check

    def run_gen():
        import time as _t
        ok_count = 0
        err_count = 0
        started_at = _t.time()

        def on_progress(current, total, result):
            nonlocal ok_count, err_count
            if result.get("status") == "starting":
                pass
            elif result.get("status") == "ok":
                ok_count += 1
            else:
                err_count += 1
            _task_progress[task_id] = {
                "current": current, "total": total,
                "status": "running", "ok": ok_count, "errors": err_count,
                "last_result": result,
                "started_at": started_at,
            }

        summary = generate.generate_batch(
            tts, db, voice, audio_dir,
            chunk_ids=resolved_chunk_ids,
            max_retries=req.max_retries,
            progress_callback=on_progress,
            stop_check=make_stop_check(),
        )
        _task_progress[task_id] = {
            "status": "done",
            "started_at": started_at,
            **{k: v for k, v in summary.items() if k != "results"},
        }

    if not _run_in_dedicated_thread(task_id, run_gen):
        raise HTTPException(409, "Generation task already running")
    return {"task_id": task_id, "status": "started"}


@router.post("/{slug}/generate_test")
def generate_test(slug: str, req: GenQATestRequest):
    """Run a short generate -> QA cycle on a sample of chunks.

    This runs up to `max_cycles` cycles: generate for the selected chunks,
    run QA, then regenerate failures and repeat. Progress is exposed via
    the task polling endpoint (`/api/pipeline/{slug}/task/generate_test`).
    """
    db = get_project_db(slug)
    settings = get_settings()
    project_dir = get_project_dir(slug)

    tts = get_tts(settings.engine.tts_engine)
    # Load TTS first (we generate audio first). Don't load STT here to avoid
    # holding both large models in GPU memory simultaneously.
    if not tts.is_loaded():
        try:
            tts.load_model(settings.engine.tts_config)
        except Exception as e:
            raise HTTPException(503, f"Could not load TTS engine: {e}. Check Settings → TTS Engine → Model directory.")

    stt = get_stt(settings.engine.stt_engine)
    # Do NOT auto-load STT here; it will be loaded/unloaded per QA phase to
    # avoid running out of GPU memory when both engines are resident.

    voice = req.voice or settings.engine.tts_config.get("default_voice", "tara")
    # persist chosen voice per-project
    try:
        db.set_meta("voice", voice)
    except Exception:
        pass

    # Determine chunk sample
    if req.chunk_ids:
        chunk_ids = [cid for cid in req.chunk_ids if db.get_chunk(cid)]
    else:
        all_chunks = db.get_chunks()
        total = len(all_chunks)
        if total == 0:
            raise HTTPException(400, "No chunks available to test")
        sample_n = min(req.sample_size, total)
        # deterministic-ish selection: random sample for now
        chunk_ids = [c["id"] for c in random.sample(all_chunks, sample_n)]

    task_id = f"{slug}_generate_test"
    _task_progress[task_id] = {"current": 0, "total": 0, "status": "running"}
    _task_controls[task_id] = {"stop": False, "pause": False}

    def make_stop_check():
        def stop_check():
            ctrl = _task_controls.get(task_id, {})
            while ctrl.get("pause"):
                import time
                time.sleep(0.5)
                ctrl = _task_controls.get(task_id, {})
            return bool(ctrl.get("stop", False))
        return stop_check

    from app.pipeline.pronunciation import build_replacement_map

    def run_test():
        remaining = set(chunk_ids)
        cycles = []

        def ensure_tts_loaded():
            # Ensure TTS is loaded; unload STT first if necessary to free GPU memory
            try:
                if stt.is_loaded():
                    try:
                        stt.unload_model()
                    except Exception:
                        pass
                if not tts.is_loaded():
                    tts.load_model(settings.engine.tts_config)
                return True
            except Exception as e:
                _task_progress[task_id] = {"status": "error", "message": f"Could not load TTS engine: {e}. Try reducing model size or freeing GPU memory (Settings → TTS Engine)."}
                return False


        def ensure_stt_loaded():
            # Ensure STT is loaded; unload TTS first if necessary to free GPU memory
            try:
                if tts.is_loaded():
                    try:
                        tts.unload_model()
                    except Exception:
                        pass
                if not stt.is_loaded():
                    stt.load_model(settings.engine.stt_config)
                return True
            except Exception as e:
                # Surface CUDA OOM guidance
                msg = str(e)
                guidance = ""
                if 'out of memory' in msg or 'CUDA out of memory' in msg:
                    guidance = " Try reducing the STT model size in Settings → STT Engine, run STT on CPU, or set environment variable PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation."
                _task_progress[task_id] = {"status": "error", "message": f"Could not load STT engine: {e}.{guidance}"}
                return False

        for cycle in range(1, max(1, req.max_cycles) + 1):
            if make_stop_check()():
                break

            # Generation phase for remaining chunks
            def on_gen_progress(current, total, result):
                _task_progress[task_id] = {
                    "phase": "generate",
                    "cycle": cycle,
                    "current": current, "total": total,
                    "status": "running",
                    "last_result": result,
                }

            if remaining:
                # Ensure TTS is loaded before attempting generation
                if not ensure_tts_loaded():
                    # ensure_tts_loaded writes an error message into _task_progress
                    break
                # For the first cycle, do a single-generation pass (no internal retries).
                # Subsequent cycles attempt regeneration with the configured max_retries.
                gen_max_retries = 1 if cycle == 1 else req.max_retries
                generate.generate_batch(
                    tts, db, voice, project_dir / "audio",
                    chunk_ids=list(remaining),
                    max_retries=gen_max_retries,
                    progress_callback=on_gen_progress,
                    stop_check=make_stop_check(),
                )

            # QA phase
            pron_map = build_replacement_map(db)

            passed = 0
            failed = 0
            failed_ids = []

            def on_qa_progress(current, total, result):
                nonlocal passed, failed
                if result.get("status") == "pass":
                    passed += 1
                else:
                    failed += 1
                _task_progress[task_id] = {
                    "phase": "qa",
                    "cycle": cycle,
                    "current": current, "total": total,
                    "status": "running",
                    "passed": passed, "failed": failed,
                    "last_result": result,
                }

            # Ensure STT is loaded before running QA
            if not ensure_stt_loaded():
                # ensure_stt_loaded writes an error message into _task_progress
                break

            qa_summary = qa.qa_batch(
                stt, db,
                chunk_ids=list(remaining),
                threshold=0.85,
                pron_map=pron_map,
                progress_callback=on_qa_progress,
                stop_check=make_stop_check(),
            )

            failed_ids = [r["chunk_id"] for r in qa_summary.get("results", []) if r.get("status") != "pass"]
            cycles.append({"cycle": cycle, "total": qa_summary.get("total", 0), "passed": qa_summary.get("passed", 0), "failed": qa_summary.get("failed", 0), "failed_ids": failed_ids})

            remaining = set(failed_ids)
            if not remaining:
                break

        # Build per-chunk summary
        results = []
        for cid in chunk_ids:
            gen = db.get_latest_generation(cid)
            qa_rows = db.fetchall("SELECT * FROM qa_results WHERE chunk_id=? ORDER BY created_at DESC", (cid,))
            last_qa = qa_rows[0] if qa_rows else None
            results.append({"chunk_id": cid, "generation": gen, "qa": last_qa})

        _task_progress[task_id] = {"status": "done", "cycles": cycles, "results": results}

    if not _run_in_dedicated_thread(task_id, run_test):
        raise HTTPException(409, "Generate-test task already running")
    return {"task_id": task_id, "status": "started"}


@router.post("/{slug}/qa")
def run_qa(slug: str, thresholds: QAThresholds = QAThresholds()):
    """Run STT QA on generated audio (runs in background)."""
    db = get_project_db(slug)
    settings = get_settings()

    stt = get_stt(settings.engine.stt_engine)
    if not stt.is_loaded():
        try:
            tts = get_tts(settings.engine.tts_engine)
            if tts.is_loaded():
                tts.unload_model()
            stt.load_model(settings.engine.stt_config)
        except Exception as e:
            raise HTTPException(503,
                f"Could not load STT engine: {e}. "
                f"Check Settings → STT Engine.")

    threshold = thresholds.min_similarity

    from app.pipeline.pronunciation import build_replacement_map
    pron_map = build_replacement_map(db)

    # When mode="new" (default), skip chunks that already have a passing/override QA
    # result for their current generation.  mode="all" re-checks everything.
    qa_chunk_ids: list[str] | None = None
    if thresholds.mode == "new":
        qa_chunk_ids = []
        for ch in db.get_chunks():
            gen = db.get_latest_generation(ch["id"])
            if not gen or gen["status"] != "ok":
                continue
            qa_r = db.fetchone(
                "SELECT status FROM qa_results WHERE chunk_id=? ORDER BY id DESC LIMIT 1",
                (ch["id"],),
            )
            if qa_r and qa_r["status"] in ("pass", "override"):
                continue
            qa_chunk_ids.append(ch["id"])

    task_id = f"{slug}_qa"
    _task_progress[task_id] = {"current": 0, "total": 0, "status": "running"}

    def run_qa_batch():
        pass_count = 0
        fail_count = 0

        def on_progress(current, total, result):
            nonlocal pass_count, fail_count
            if result.get("status") == "pass":
                pass_count += 1
            else:
                fail_count += 1
            _task_progress[task_id] = {
                "current": current, "total": total,
                "status": "running", "passed": pass_count, "failed": fail_count,
                "last_result": {
                    "chunk_id": result["chunk_id"],
                    "score": result["adjusted_score"],
                    "qa_status": result["status"],
                },
            }

        summary = qa.qa_batch(
            stt, db,
            chunk_ids=qa_chunk_ids,
            threshold=threshold,
            pron_map=pron_map,
            progress_callback=on_progress,
        )
        _task_progress[task_id] = {
            "status": "done", **{k: v for k, v in summary.items() if k != "results"}
        }

    if not _run_in_dedicated_thread(task_id, run_qa_batch):
        raise HTTPException(409, "QA task already running")
    return {"task_id": task_id, "status": "started"}


# ─── Task progress ────────────────────────────────────────────────────

@router.get("/{slug}/task/{task_id}")
def get_task_progress(slug: str, task_id: str):
    """Poll progress of a background task."""
    full_id = f"{slug}_{task_id}" if not task_id.startswith(slug) else task_id
    progress = _task_progress.get(full_id)
    if not progress:
        raise HTTPException(404, "Task not found")
    return progress


@router.post("/{slug}/task/{task_name}/pause")
def pause_task(slug: str, task_name: str):
    full = f"{slug}_{task_name}" if not task_name.startswith(slug) else task_name
    ctrl = _task_controls.setdefault(full, {"stop": False, "pause": False})
    ctrl["pause"] = True
    return {"status": "paused", "task": full}


@router.post("/{slug}/task/{task_name}/resume")
def resume_task(slug: str, task_name: str):
    full = f"{slug}_{task_name}" if not task_name.startswith(slug) else task_name
    ctrl = _task_controls.setdefault(full, {"stop": False, "pause": False})
    ctrl["pause"] = False
    return {"status": "resumed", "task": full}


@router.post("/{slug}/task/{task_name}/stop")
def stop_task(slug: str, task_name: str):
    full = f"{slug}_{task_name}" if not task_name.startswith(slug) else task_name
    ctrl = _task_controls.setdefault(full, {"stop": False, "pause": False})
    ctrl["stop"] = True
    # Also clear pause so it can exit quickly
    ctrl["pause"] = False
    return {"status": "stopped", "task": full}


@router.get("/{slug}/task/{task_name}/controls")
def get_task_controls(slug: str, task_name: str):
    full = f"{slug}_{task_name}" if not task_name.startswith(slug) else task_name
    return _task_controls.get(full, {"stop": False, "pause": False})


# ─── Step 15: Export ──────────────────────────────────────────────────

@router.post("/{slug}/export")
def export_audio(slug: str, req: ExportRequest):
    """Merge and export audio.

    The response includes ``total_seams``, ``flagged_seams``, and a
    ``flagged`` list with per-seam warnings (long silences, abrupt cuts,
    loudness jumps, sample-discontinuity pops). The UI surfaces these so
    the user can inspect or regenerate suspect chunks before publishing.
    """
    db = get_project_db(slug)
    settings = get_settings()
    project_dir = get_project_dir(slug)
    export_dir = project_dir / "export"

    if req.scope == "chapter":
        return merge.export_by_chapter(db, export_dir, settings.audio)
    else:
        return merge.export_full_book(db, export_dir, settings.audio)
