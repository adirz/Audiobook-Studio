"""Audio generation with retry logic.

Step 11-12: Generate WAV files for each chunk, with configurable retries
and parameter variation on failure.
"""

import wave
import time
import json
from pathlib import Path

from app.database import ProjectDB
from app.handlers.tts_base import TTSHandler
from app.pipeline.pronunciation import apply_pronunciation, build_replacement_map


def prepare_chunk_text(db: ProjectDB, chunk: dict) -> str:
    """Get the final text to send to TTS for a chunk.

    Priority: pron_text > tagged_text > original_text.
    Also applies pronunciation if pron_text isn't pre-computed.
    """
    if chunk.get("pron_text"):
        text = chunk["pron_text"]
    elif chunk.get("tagged_text"):
        pron_map = build_replacement_map(db)
        overrides = db.get_location_overrides(chunk["id"])
        text = apply_pronunciation(chunk["tagged_text"], pron_map, overrides)
    else:
        pron_map = build_replacement_map(db)
        overrides = db.get_location_overrides(chunk["id"])
        text = apply_pronunciation(chunk["original_text"], pron_map, overrides)

    # Collapse newlines for TTS (most engines handle this poorly)
    text = text.replace("\n", "  ")

    return text


def generate_chunk(tts: TTSHandler, db: ProjectDB,
                   chunk: dict, voice: str,
                   audio_dir: Path,
                   params: dict | None = None,
                   attempt: int | None = None) -> dict:
    """Generate audio for a single chunk.

    Returns a result dict with status, duration, paths, etc.
    """
    if attempt is None:
        # Find next attempt number
        existing = db.get_generations(chunk_id=chunk["id"])
        attempt = len(existing) + 1

    # Register the generation attempt
    gen_id = db.insert_generation(
        chunk_id=chunk["id"],
        attempt=attempt,
        params=params or {},
    )

    db.update_generation(gen_id, status="generating")

    text = prepare_chunk_text(db, chunk)
    wav_path = audio_dir / f"{chunk['id']}_v{attempt:02d}.wav"

    start = time.time()
    try:
        pcm_data = tts.generate(text, voice, params)
        elapsed = time.time() - start

        # Write WAV
        sample_rate = tts.get_sample_rate()
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)

        duration = len(pcm_data) / (2 * sample_rate)  # 16-bit = 2 bytes/sample

        db.update_generation(gen_id,
                             status="ok",
                             wav_path=str(wav_path),
                             duration_sec=round(duration, 2),
                             gen_time_sec=round(elapsed, 2))

        return {
            "gen_id": gen_id,
            "chunk_id": chunk["id"],
            "wav_path": str(wav_path),
            "duration_sec": round(duration, 2),
            "gen_time_sec": round(elapsed, 2),
            "status": "ok",
            "attempt": attempt,
        }

    except Exception as e:
        elapsed = time.time() - start
        db.update_generation(gen_id,
                             status="error",
                             error_msg=str(e),
                             gen_time_sec=round(elapsed, 2))
        return {
            "gen_id": gen_id,
            "chunk_id": chunk["id"],
            "status": "error",
            "error": str(e),
            "attempt": attempt,
        }


def generate_with_retry(tts: TTSHandler, db: ProjectDB,
                        chunk: dict, voice: str,
                        audio_dir: Path,
                        max_retries: int = 3,
                        base_params: dict | None = None) -> dict:
    """Generate audio for a chunk with automatic retries on failure.

    Each retry varies parameters slightly:
    - Attempt 2: bump temperature +0.1
    - Attempt 3: bump repetition_penalty +0.1, lower max_tokens
    """
    params = dict(base_params or {})

    for attempt in range(1, max_retries + 1):
        if attempt == 2:
            params["temperature"] = params.get("temperature", 0.7) + 0.1
        elif attempt >= 3:
            params["repetition_penalty"] = params.get("repetition_penalty", 1.1) + 0.1
            params["max_tokens"] = max(2000, params.get("max_tokens", 8000) - 2000)

        result = generate_chunk(tts, db, chunk, voice, audio_dir,
                                params=params, attempt=attempt)

        if result["status"] == "ok":
            return result

    # All retries exhausted
    return result  # return last failed result


def generate_batch(tts: TTSHandler, db: ProjectDB,
                   voice: str, audio_dir: Path,
                   chunk_ids: list[str] | None = None,
                   max_retries: int = 3,
                   params: dict | None = None,
                   progress_callback=None,
                   stop_check=None) -> dict:
    """Generate audio for multiple chunks (or all pending).

    Args:
        chunk_ids: Specific chunks to generate. None = all without successful generation.
        progress_callback: Called with (current, total, result_dict).
        stop_check: Callable returning True if we should stop (graceful interrupt).

    Returns summary dict.
    """
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    if chunk_ids:
        chunks = [db.get_chunk(cid) for cid in chunk_ids]
        chunks = [c for c in chunks if c]
    else:
        # Find chunks without a successful generation
        all_chunks = db.get_chunks()
        chunks = []
        for ch in all_chunks:
            gen = db.get_latest_generation(ch["id"])
            if not gen or gen["status"] != "ok":
                chunks.append(ch)

    total = len(chunks)
    ok_count = 0
    error_count = 0
    results = []

    for i, chunk in enumerate(chunks):
        if stop_check and stop_check():
            break

        result = generate_with_retry(
            tts, db, chunk, voice, audio_dir,
            max_retries=max_retries,
            base_params=params,
        )
        results.append(result)

        if result["status"] == "ok":
            ok_count += 1
        else:
            error_count += 1

        if progress_callback:
            progress_callback(i + 1, total, result)

    return {
        "total": total,
        "completed": ok_count + error_count,
        "ok": ok_count,
        "errors": error_count,
        "results": results,
    }
