"""Quality assurance via speech-to-text comparison.

Step 11 (test) and 12 (full): Transcribe generated audio with Whisper,
compare against original text, produce word-level diffs.
"""

import re
import json
from difflib import SequenceMatcher

from app.database import ProjectDB
from app.handlers.stt_base import STTHandler


def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = re.sub(r"<[^>]+>", "", text)          # remove TTS tags
    text = text.replace("*", "")                  # remove italic markers
    # Unify apostrophe and quote forms so curly (’) and straight (')
    # don't end up on opposite sides of the diff. Without this,
    # "gods'" (curly) → "gods" while the STT-side "gods'" (straight)
    # stays "gods'", producing a phantom replace.
    text = (text
            .replace("‘", "'").replace("’", "'")
            .replace("“", "").replace("”", ""))
    text = re.sub(r"[^\w\s']", "", text.lower())  # keep only word chars and apostrophes
    return re.sub(r"\s+", " ", text).strip()


def _word_similarity(a: str, b: str) -> float:
    """Character-level similarity ratio in [0, 1]."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# Per-word similarity floor at which we still consider a paired word a
# "mispronunciation" rather than two unrelated words. Picked empirically:
# "Ithas" ↔ "Ethos" sits around 0.6, "lake" ↔ "kingdom" sits around 0.18.
_WORD_PAIR_SIM_THRESHOLD = 0.4


def word_level_diff(original: str, transcribed: str) -> list[dict]:
    """Produce word-level alignment between original and transcribed text.

    Returns a list of diff operations:
    - {"type": "equal", "original": "word", "transcribed": "word"}
    - {"type": "replace", "original": "Tanji", "transcribed": "tangy"}
    - {"type": "insert", "transcribed": "extra_word"}
    - {"type": "delete", "original": "missing_word"}

    A ``replace`` opcode from ``SequenceMatcher`` covers any aligned
    mismatch — including wholesale rewrites where the audio veered off
    script entirely. We only emit per-word ``replace`` entries when the
    two sides have the same word count AND every paired word looks like a
    plausible mishearing (character-level similarity ≥ threshold).
    Otherwise we split the block into deletes + inserts so the UI shows
    "removed words" / "added words" instead of fake one-to-one
    mispronunciations.
    """
    orig_words = normalize(original).split()
    trans_words = normalize(transcribed).split()

    matcher = SequenceMatcher(None, orig_words, trans_words)
    diff = []

    def emit_split(o_slice, t_slice, anchor):
        """Emit deletes for o_slice followed by inserts for t_slice."""
        for i, w in enumerate(o_slice):
            diff.append({
                "type": "delete",
                "original": w,
                "transcribed": "",
                "orig_index": anchor + i,
            })
        # Inserts anchor at the end of the deleted block so they render
        # adjacent to (after) the deleted region in the original-as-canvas
        # view.
        ins_anchor = anchor + max(0, len(o_slice) - 1)
        for w in t_slice:
            diff.append({
                "type": "insert",
                "original": "",
                "transcribed": w,
                "orig_index": ins_anchor,
            })

    for op, o_start, o_end, t_start, t_end in matcher.get_opcodes():
        if op == "equal":
            for i in range(o_end - o_start):
                diff.append({
                    "type": "equal",
                    "original": orig_words[o_start + i],
                    "transcribed": trans_words[t_start + i],
                    "orig_index": o_start + i,
                })
        elif op == "replace":
            o_slice = orig_words[o_start:o_end]
            t_slice = trans_words[t_start:t_end]
            same_length = len(o_slice) == len(t_slice) and o_slice
            paired_replaces = same_length and all(
                _word_similarity(o_slice[i], t_slice[i]) >= _WORD_PAIR_SIM_THRESHOLD
                for i in range(len(o_slice))
            )
            if paired_replaces:
                for i in range(len(o_slice)):
                    diff.append({
                        "type": "replace",
                        "original": o_slice[i],
                        "transcribed": t_slice[i],
                        "orig_index": o_start + i,
                    })
            else:
                emit_split(o_slice, t_slice, o_start)
        elif op == "insert":
            for i in range(t_end - t_start):
                diff.append({
                    "type": "insert",
                    "original": "",
                    "transcribed": trans_words[t_start + i],
                    "orig_index": o_start,
                })
        elif op == "delete":
            for i in range(o_end - o_start):
                diff.append({
                    "type": "delete",
                    "original": orig_words[o_start + i],
                    "transcribed": "",
                    "orig_index": o_start + i,
                })

    return diff


def compute_similarity(diff: list[dict]) -> float:
    """Compute similarity ratio from word-level diff."""
    if not diff:
        return 1.0
    equal = sum(1 for d in diff if d["type"] == "equal")
    total = len(diff)
    return equal / total if total > 0 else 1.0


def is_pron_substitution(original_word: str, transcribed_word: str,
                         pron_map: dict[str, str]) -> bool:
    """Check if a mismatch is expected due to pronunciation substitution.

    e.g. Original "Tanji" → TTS heard "Tahn-jee" → Whisper transcribes "tangy"
    These are expected differences and shouldn't count as failures.

    Possessive forms count too: a dictionary entry for "Harmonia" should
    cover "Harmonia's" — the chunker substitutes the base name (the
    apostrophe is a word boundary in the substitution regex), so the
    audio renders the phonetic version even for possessives.
    """
    lower = original_word.lower()
    # Strip trailing possessive 's / s' so possessives match the base entry.
    base = re.sub(r"'s?$", "", lower)
    for orig in pron_map:
        ol = orig.lower()
        if ol == lower or ol == base:
            return True
    return False


def qa_single_chunk(stt: STTHandler, db: ProjectDB,
                    chunk: dict, generation: dict,
                    pron_map: dict[str, str] | None = None,
                    threshold: float = 0.85) -> dict:
    """Run QA on a single chunk's generation.

    Returns result dict with score, diff, and status.
    """
    if not generation.get("wav_path"):
        return {"status": "error", "error": "No audio file"}

    # Transcribe
    transcribed = stt.transcribe(generation["wav_path"])

    # Build diff
    diff = word_level_diff(chunk["original_text"], transcribed)
    score = compute_similarity(diff)

    # Mark pronunciation-related mismatches if pron_map provided
    if pron_map:
        for entry in diff:
            if entry["type"] == "replace" and entry["original"]:
                entry["pron_expected"] = is_pron_substitution(
                    entry["original"], entry["transcribed"], pron_map
                )

    # Calculate adjusted score (ignoring expected pron differences)
    if pron_map:
        adjusted_diff = [
            d for d in diff
            if not (d["type"] == "replace" and d.get("pron_expected"))
        ]
        adjusted_score = compute_similarity(adjusted_diff) if adjusted_diff else 1.0
    else:
        adjusted_score = score

    status = "pass" if adjusted_score >= threshold else "fail"

    # Save to DB
    qa_id = db.insert_qa_result(
        chunk_id=chunk["id"],
        gen_id=generation["id"],
        transcribed=transcribed,
        score=adjusted_score,
        word_diff=diff,
        status=status,
    )

    return {
        "qa_id": qa_id,
        "chunk_id": chunk["id"],
        "raw_score": round(score, 3),
        "adjusted_score": round(adjusted_score, 3),
        "status": status,
        "transcribed": transcribed,
        "diff": diff,
    }


def qa_batch(stt: STTHandler, db: ProjectDB,
             chunk_ids: list[str] | None = None,
             threshold: float = 0.85,
             pron_map: dict[str, str] | None = None,
             progress_callback=None,
             stop_check=None) -> dict:
    """Run QA on multiple chunks.

    Args:
        chunk_ids: Specific chunks. None = all with successful generation.
        progress_callback: Called with (current, total, result).
        stop_check: Callable returning True to stop.
    """
    if chunk_ids:
        chunks = [db.get_chunk(cid) for cid in chunk_ids if db.get_chunk(cid)]
    else:
        all_chunks = db.get_chunks()
        chunks = []
        for ch in all_chunks:
            gen = db.get_latest_generation(ch["id"])
            if gen and gen["status"] == "ok":
                chunks.append(ch)

    total = len(chunks)
    pass_count = 0
    fail_count = 0
    results = []

    for i, chunk in enumerate(chunks):
        if stop_check and stop_check():
            break

        gen = db.get_latest_generation(chunk["id"])
        if not gen or gen["status"] != "ok":
            continue

        result = qa_single_chunk(stt, db, chunk, gen, pron_map, threshold)
        results.append(result)

        if result["status"] == "pass":
            pass_count += 1
        else:
            fail_count += 1

        if progress_callback:
            progress_callback(i + 1, total, result)

    return {
        "total": total,
        "completed": pass_count + fail_count,
        "passed": pass_count,
        "failed": fail_count,
        "results": results,
    }
