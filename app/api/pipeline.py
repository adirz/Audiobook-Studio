"""API routes for pipeline step execution and status."""

import json
import random
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.api.projects import get_project_db, get_project_dir
from app.handlers.registry import get_tts, get_stt, get_llm, get_extractor
from app.config import get_settings
from app.models import (
    RangeSelection, SymbolDecision, GenerateRequest,
    QAThresholds, ExportRequest, TaggingRequest,
    TaggingChatRequest, TaggingApplyRequest, TaggingSaveRequest,
    PronUpdate, PronTestRequest, GenQATestRequest,
)
from app.pipeline import analyze, clean, chunk, tagging, generate, qa, merge
from app.pipeline.pron_buffer import get_or_create_buffer

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# In-memory progress tracking for long-running tasks
_task_progress: dict[str, dict] = {}
_task_controls: dict[str, dict] = {}


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
def chunk_text(slug: str, max_words: int = 150, min_words: int = 30):
    """Chunk all chapters."""
    db = get_project_db(slug)
    count = chunk.chunk_all_chapters(db, max_words, min_words)
    return {"chunks_created": count}


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
def apply_tagging(slug: str, bg: BackgroundTasks,
                  req: TaggingApplyRequest):
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

    bg.add_task(run_tagging)
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
def generate_audio(slug: str, bg: BackgroundTasks, req: GenerateRequest):
    """Generate audio for chunks (runs in background)."""
    db = get_project_db(slug)
    settings = get_settings()
    project_dir = get_project_dir(slug)

    tts = get_tts(settings.engine.tts_engine)
    if not tts.is_loaded():
        try:
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
        ok_count = 0
        err_count = 0

        def on_progress(current, total, result):
            nonlocal ok_count, err_count
            if result.get("status") == "ok":
                ok_count += 1
            else:
                err_count += 1
            _task_progress[task_id] = {
                "current": current, "total": total,
                "status": "running", "ok": ok_count, "errors": err_count,
                "last_result": result,
            }

        summary = generate.generate_batch(
            tts, db, voice, audio_dir,
            chunk_ids=req.chunk_ids or None,
            max_retries=req.max_retries,
            progress_callback=on_progress,
            stop_check=make_stop_check(),
        )
        _task_progress[task_id] = {
            "status": "done", **{k: v for k, v in summary.items() if k != "results"}
        }

    bg.add_task(run_gen)
    return {"task_id": task_id, "status": "started"}


@router.post("/{slug}/generate_test")
def generate_test(slug: str, bg: BackgroundTasks, req: GenQATestRequest):
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
                    generate.generate_batch(
                        tts, db, voice, project_dir / "audio",
                        chunk_ids=list(remaining),
                        max_retries=req.max_retries,
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

    bg.add_task(run_test)
    return {"task_id": task_id, "status": "started"}


@router.post("/{slug}/qa")
def run_qa(slug: str, bg: BackgroundTasks,
           thresholds: QAThresholds = QAThresholds()):
    """Run STT QA on generated audio (runs in background)."""
    db = get_project_db(slug)
    settings = get_settings()

    stt = get_stt(settings.engine.stt_engine)
    if not stt.is_loaded():
        try:
            stt.load_model(settings.engine.stt_config)
        except Exception as e:
            raise HTTPException(503,
                f"Could not load STT engine: {e}. "
                f"Check Settings → STT Engine.")

    threshold = thresholds.min_similarity

    from app.pipeline.pronunciation import build_replacement_map
    pron_map = build_replacement_map(db)

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
            threshold=threshold,
            pron_map=pron_map,
            progress_callback=on_progress,
        )
        _task_progress[task_id] = {
            "status": "done", **{k: v for k, v in summary.items() if k != "results"}
        }

    bg.add_task(run_qa_batch)
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
    """Merge and export audio."""
    db = get_project_db(slug)
    settings = get_settings()
    project_dir = get_project_dir(slug)
    export_dir = project_dir / "export"

    if req.scope == "chapter":
        files = merge.export_by_chapter(db, export_dir, settings.audio)
        return {"files": files}
    else:
        path = merge.export_full_book(db, export_dir, settings.audio)
        return {"file": path}
