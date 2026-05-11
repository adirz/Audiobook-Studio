"""API routes for audio serving, playback, and review (Steps 13-14)."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api.projects import get_project_db, get_project_dir
from app.models import ChunkFlag, LocationOverride, ManualQAUpdate

router = APIRouter(prefix="/api/audio", tags=["audio"])


@router.get("/{slug}/test_clips/{filename}")
def serve_test_clip(slug: str, filename: str):
    """Serve a pronunciation test audio clip."""
    project_dir = get_project_dir(slug)
    path = project_dir / "test_clips" / filename
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(str(path), media_type="audio/wav")


@router.head("/{slug}/test_clips/{filename}")
def head_test_clip(slug: str, filename: str):
    """Respond to HEAD requests for pronunciation test clips."""
    project_dir = get_project_dir(slug)
    path = project_dir / "test_clips" / filename
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    # FileResponse will include headers (Content-Length, Content-Type)
    return FileResponse(str(path), media_type="audio/wav")


@router.get("/{slug}/chunk/{chunk_id}")
def serve_chunk_audio(slug: str, chunk_id: str):
    """Serve the latest successful audio for a chunk."""
    db = get_project_db(slug)
    gen = db.get_latest_generation(chunk_id)
    if not gen or gen["status"] != "ok" or not gen.get("wav_path"):
        raise HTTPException(404, "No audio available for this chunk")

    path = Path(gen["wav_path"])
    if not path.exists():
        raise HTTPException(404, "Audio file not found on disk")

    return FileResponse(str(path), media_type="audio/wav")


@router.head("/{slug}/chunk/{chunk_id}")
def head_chunk_audio(slug: str, chunk_id: str):
    """Respond to HEAD requests for chunk audio."""
    db = get_project_db(slug)
    gen = db.get_latest_generation(chunk_id)
    if not gen or gen["status"] != "ok" or not gen.get("wav_path"):
        raise HTTPException(404, "No audio available for this chunk")

    path = Path(gen["wav_path"])
    if not path.exists():
        raise HTTPException(404, "Audio file not found on disk")

    return FileResponse(str(path), media_type="audio/wav")


@router.get("/{slug}/chunk/{chunk_id}/qa")
def get_chunk_qa(slug: str, chunk_id: str):
    """Get QA results for a chunk."""
    db = get_project_db(slug)
    results = db.fetchall(
        """SELECT qr.*, g.attempt as gen_attempt
           FROM qa_results qr
           JOIN generations g ON qr.generation_id = g.id
           WHERE qr.chunk_id=?
           ORDER BY qr.id DESC""",
        (chunk_id,),
    )
    import json
    for r in results:
        if r.get("word_diff_json"):
            r["word_diff"] = json.loads(r["word_diff_json"])
    return results


@router.get("/{slug}/export/{filename}")
def serve_export(slug: str, filename: str):
    """Serve an exported audio file."""
    project_dir = get_project_dir(slug)
    path = project_dir / "export" / filename
    if not path.exists():
        raise HTTPException(404, "Export file not found")
    return FileResponse(str(path), media_type="audio/wav")


@router.head("/{slug}/export/{filename}")
def head_export(slug: str, filename: str):
    """Respond to HEAD requests for exported files."""
    project_dir = get_project_dir(slug)
    path = project_dir / "export" / filename
    if not path.exists():
        raise HTTPException(404, "Export file not found")
    return FileResponse(str(path), media_type="audio/wav")


# ─── Review / flagging (Steps 13-14) ──────────────────────────────────

# Common CTEs for "latest generation per chunk" and "latest QA per
# generation". Used by both /review and /review/chapters to replace the
# old per-chunk SELECT loops (which were ~3 round trips per chunk =
# 15k+ queries for a 5,000-chunk book) with a single joined query.
_LATEST_CTES = """
    WITH latest_gen AS (
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY chunk_id ORDER BY attempt DESC, id DESC
            ) AS _rn FROM generations
        ) WHERE _rn = 1
    ),
    latest_qa AS (
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY generation_id ORDER BY id DESC
            ) AS _rn FROM qa_results
        ) WHERE _rn = 1
    )
"""


@router.get("/{slug}/review")
def get_review_data(slug: str, chapter_id: int = None):
    """Get all chunks with their generation/QA status for review."""
    db = get_project_db(slug)

    where = "WHERE c.chapter_id = ?" if chapter_id is not None else ""
    params: tuple = (chapter_id,) if chapter_id is not None else ()
    sql = _LATEST_CTES + f"""
        SELECT c.*,
            g.id            AS gen_id,
            g.attempt       AS gen_attempt,
            g.status        AS gen_status,
            g.wav_path      AS gen_wav_path,
            g.duration_sec  AS gen_duration_sec,
            g.gen_time_sec  AS gen_gen_time_sec,
            g.error_msg     AS gen_error_msg,
            g.params_json   AS gen_params_json,
            g.created_at    AS gen_created_at,
            qa.id              AS qa_id,
            qa.similarity_score AS qa_score,
            qa.status          AS qa_status
        FROM chunks c
        LEFT JOIN latest_gen g  ON g.chunk_id = c.id
        LEFT JOIN latest_qa qa  ON qa.generation_id = g.id
        {where}
        ORDER BY c.global_index, c.local_index
    """
    rows = db.fetchall(sql, params)

    # Compute a scene index per chunk on the fly. The scene resets at
    # chapter boundaries and increments at every chunk whose
    # ``scene_break_after`` is set. Scene 1 covers everything before the
    # first scene break inside a chapter — i.e. the "default" / "other"
    # bucket the UI surfaces. We also propagate the *originating symbol*
    # of the most recent scene break so each chunk knows which kind of
    # scene break it sits behind (used by the Generate-Audio filter to
    # split "after ♦" from "after * * *", etc.).
    scene_by_chunk: dict[str, int] = {}
    sb_before_by_chunk: dict[str, str | None] = {}
    current_chapter = None
    current_scene = 1
    last_sb_symbol: str | None = None
    for ch in rows:
        if ch.get("chapter_id") != current_chapter:
            current_chapter = ch.get("chapter_id")
            current_scene = 1
            last_sb_symbol = None
        scene_by_chunk[ch["id"]] = current_scene
        sb_before_by_chunk[ch["id"]] = last_sb_symbol
        if ch.get("scene_break_after"):
            current_scene += 1
            last_sb_symbol = ch.get("scene_break_symbol")

    # One batched fetch for unresolved flags across every chunk in scope —
    # SQLite caps parameter lists at 999 by default so chunk into 500s.
    chunk_ids = [r["id"] for r in rows]
    flags_by_chunk: dict[str, list[dict]] = {}
    for i in range(0, len(chunk_ids), 500):
        batch = chunk_ids[i:i + 500]
        placeholders = ",".join("?" * len(batch))
        for f in db.fetchall(
            f"SELECT * FROM user_flags WHERE resolved=0 AND chunk_id IN ({placeholders})",
            tuple(batch),
        ):
            flags_by_chunk.setdefault(f["chunk_id"], []).append(f)

    enriched = []
    chunk_keys = {
        "id", "chapter_id", "local_index", "global_index",
        "original_text", "cleaned_text", "tagged_text", "pron_text",
        "scene_break_after", "chapter_break_after", "word_count",
        "is_title_chunk", "voice", "scene_break_symbol", "pron_text_locked",
    }
    for r in rows:
        gen = None
        if r.get("gen_id") is not None:
            gen = {
                "id": r["gen_id"],
                "chunk_id": r["id"],
                "attempt": r.get("gen_attempt"),
                "status": r.get("gen_status"),
                "wav_path": r.get("gen_wav_path"),
                "duration_sec": r.get("gen_duration_sec"),
                "gen_time_sec": r.get("gen_gen_time_sec"),
                "error_msg": r.get("gen_error_msg"),
                "params_json": r.get("gen_params_json"),
                "created_at": r.get("gen_created_at"),
            }
        qa = None
        if r.get("qa_id") is not None:
            qa = {"score": r.get("qa_score"), "status": r.get("qa_status")}

        chunk_only = {k: r.get(k) for k in chunk_keys if k in r}
        enriched.append({
            **chunk_only,
            "generation": gen,
            "qa": qa,
            "flags": flags_by_chunk.get(r["id"], []),
            "has_audio": bool(gen and gen["status"] == "ok"),
            "audio_url": f"/api/audio/{slug}/chunk/{r['id']}" if gen and gen["status"] == "ok" else None,
            "scene_index": scene_by_chunk.get(r["id"], 1),
            "scene_break_symbol_before": sb_before_by_chunk.get(r["id"]),
        })

    return enriched


@router.get("/{slug}/review/chapters")
def get_chapter_list(slug: str):
    """Get chapter list with summary stats for navigation."""
    db = get_project_db(slug)
    sql = _LATEST_CTES + """
        SELECT
            ch.id   AS id,
            ch.idx  AS idx,
            ch.title AS title,
            COALESCE(COUNT(c.id), 0)                                                AS total_chunks,
            COALESCE(SUM(CASE WHEN g.status = 'ok' THEN 1 ELSE 0 END), 0)           AS generated,
            COALESCE(SUM(CASE WHEN qa.status = 'pass' THEN 1 ELSE 0 END), 0)        AS qa_passed,
            COALESCE(SUM(CASE WHEN flag_counts.cnt > 0 THEN 1 ELSE 0 END), 0)       AS flagged
        FROM chapters ch
        LEFT JOIN chunks c     ON c.chapter_id = ch.id
        LEFT JOIN latest_gen g ON g.chunk_id   = c.id
        LEFT JOIN latest_qa qa ON qa.generation_id = g.id
        LEFT JOIN (
            SELECT chunk_id, COUNT(*) AS cnt
            FROM user_flags WHERE resolved = 0
            GROUP BY chunk_id
        ) flag_counts ON flag_counts.chunk_id = c.id
        GROUP BY ch.id, ch.idx, ch.title
        ORDER BY ch.idx
    """
    return db.fetchall(sql)


@router.post("/{slug}/review/qa-status")
def set_manual_qa_status(slug: str, req: ManualQAUpdate):
    """Manually mark a chunk as pass / fail, or clear the manual mark.

    Manual marks are stored as a fresh ``qa_results`` row with a NULL
    similarity score; clearing only drops that override row, leaving any
    previous automatic QA intact.
    """
    db = get_project_db(slug)
    if req.status == "clear":
        cleared = db.clear_manual_qa(req.chunk_id)
        return {"status": "cleared" if cleared else "noop", "chunk_id": req.chunk_id}
    if req.status not in ("pass", "fail"):
        raise HTTPException(400, "status must be one of pass / fail / clear")
    db.set_manual_qa_status(req.chunk_id, req.status)
    return {"status": req.status, "chunk_id": req.chunk_id}


@router.post("/{slug}/review/flag")
def flag_chunk(slug: str, flag: ChunkFlag):
    """Flag a chunk with an issue during review."""
    db = get_project_db(slug)
    db.insert_flag(
        chunk_id=flag.chunk_id,
        flag_type=flag.flag_type.value,
        word_range=flag.word_range,
        notes=flag.notes,
    )
    return {"status": "flagged"}


@router.post("/{slug}/review/flag/{flag_id}/resolve")
def resolve_flag(slug: str, flag_id: int):
    """Mark a flag as resolved."""
    db = get_project_db(slug)
    db.resolve_flag(flag_id)
    return {"status": "resolved"}


@router.get("/{slug}/review/flags")
def get_all_flags(slug: str, resolved: bool = None):
    """Get all flags, optionally filtered."""
    db = get_project_db(slug)
    return db.get_flags(resolved=resolved)


# ─── Chunk text editing ───────────────────────────────────────────────

@router.patch("/{slug}/chunk/{chunk_id}")
def edit_chunk_tts_text(slug: str, chunk_id: str, payload: dict):
    """Override the text sent to TTS for this chunk.

    Saves the user's edit as pron_text (highest priority in prepare_chunk_text).
    original_text is never modified — it stays as the manuscript source.
    Pass text=null/empty to clear the override and revert to automatic text.
    """
    db = get_project_db(slug)
    chunk = db.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(404, "Chunk not found")
    text = (payload.get("text") or "").strip() or None
    locked = 1 if text else 0
    db.execute("UPDATE chunks SET pron_text=?, pron_text_locked=? WHERE id=?", (text, locked, chunk_id))
    db.commit()
    return {"status": "updated", "chunk_id": chunk_id, "cleared": text is None}


# ─── Location overrides (from review, step 13) ────────────────────────

@router.post("/{slug}/review/location-override")
def add_location_override(slug: str, override: LocationOverride):
    """Add a per-location pronunciation override from review."""
    db = get_project_db(slug)
    db.insert_location_override(
        word=override.word,
        phonetic=override.phonetic,
        chunk_id=override.chunk_id,
        word_offset=override.word_offset,
        notes=override.notes,
    )
    return {"status": "added"}


@router.get("/{slug}/review/location-overrides")
def get_location_overrides(slug: str, chunk_id: str = None):
    """Get location-specific pronunciation overrides."""
    db = get_project_db(slug)
    return db.get_location_overrides(chunk_id)
