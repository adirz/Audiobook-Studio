"""API routes for audio serving, playback, and review (Steps 13-14)."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api.projects import get_project_db, get_project_dir
from app.models import ChunkFlag, LocationOverride

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

@router.get("/{slug}/review")
def get_review_data(slug: str, chapter_id: int = None):
    """Get all chunks with their generation/QA status for review."""
    db = get_project_db(slug)

    if chapter_id is not None:
        chunks = db.get_chunks(chapter_id=chapter_id)
    else:
        chunks = db.get_chunks()

    enriched = []
    for ch in chunks:
        gen = db.get_latest_generation(ch["id"])
        flags = db.fetchall(
            "SELECT * FROM user_flags WHERE chunk_id=? AND resolved=0",
            (ch["id"],),
        )
        qa_result = db.fetchone(
            """SELECT * FROM qa_results
               WHERE chunk_id=? ORDER BY id DESC LIMIT 1""",
            (ch["id"],),
        )

        enriched.append({
            **ch,
            "generation": gen,
            "qa": {
                "score": qa_result["similarity_score"] if qa_result else None,
                "status": qa_result["status"] if qa_result else None,
            } if qa_result else None,
            "flags": flags,
            "has_audio": bool(gen and gen["status"] == "ok"),
            "audio_url": f"/api/audio/{slug}/chunk/{ch['id']}" if gen and gen["status"] == "ok" else None,
        })

    return enriched


@router.get("/{slug}/review/chapters")
def get_chapter_list(slug: str):
    """Get chapter list with summary stats for navigation."""
    db = get_project_db(slug)
    chapters = db.get_chapters()

    result = []
    for ch in chapters:
        chunks = db.get_chunks(chapter_id=ch["id"])
        total = len(chunks)
        generated = 0
        qa_passed = 0
        flagged = 0

        for chunk in chunks:
            gen = db.get_latest_generation(chunk["id"])
            if gen and gen["status"] == "ok":
                generated += 1
            qa_r = db.fetchone(
                "SELECT status FROM qa_results WHERE chunk_id=? ORDER BY id DESC LIMIT 1",
                (chunk["id"],),
            )
            if qa_r and qa_r["status"] == "pass":
                qa_passed += 1
            flags = db.fetchall(
                "SELECT id FROM user_flags WHERE chunk_id=? AND resolved=0",
                (chunk["id"],),
            )
            if flags:
                flagged += 1

        result.append({
            "id": ch["id"],
            "idx": ch["idx"],
            "title": ch["title"],
            "total_chunks": total,
            "generated": generated,
            "qa_passed": qa_passed,
            "flagged": flagged,
        })

    return result


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
