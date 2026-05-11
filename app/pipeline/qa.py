"""Quality assurance via speech-to-text comparison.

Step 11 (test) and 12 (full): Transcribe generated audio with Whisper,
compare against original text, produce word-level diffs.
"""

import re
import json
from difflib import SequenceMatcher

from app.database import ProjectDB
from app.handlers.stt_base import STTHandler


# Common English contractions → expanded form.
# Applied to BOTH original and transcribed so that "I'm" ↔ "I am",
# "we'll" ↔ "we will", etc. compare as equal.  Mapping keys are
# already lowercased; values are the long form (also lowercase).
_CONTRACTIONS: dict[str, str] = {
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "here's": "here is",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "that'll": "that will",
    "won't": "will not",
    "can't": "cannot",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "let's": "let us",
    "o'clock": "oclock",
}

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
         "sixty", "seventy", "eighty", "ninety"]


def _int_to_words(n: int) -> str:
    """Convert non-negative integer 0–9 999 to English words (no 'and')."""
    if n < 0 or n > 9999:
        return str(n)
    if n == 0:
        return "zero"
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (" " + _ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + (" " + _int_to_words(r) if r else "")
    th, r = divmod(n, 1000)
    return _ONES[th] + " thousand" + (" " + _int_to_words(r) if r else "")


def _expand_numbers(text: str) -> str:
    """Replace digit sequences with their word equivalents.

    Whisper often writes spoken numbers as digits ("150") while the
    manuscript has the word form ("hundred and fifty"). Converting digits
    to words on both sides gives the diff a fair chance to align them.
    The word "and" inside number phrases (e.g. "hundred and fifty") is
    harmless — it appears on both sides and cancels out.
    """
    def _replace(m: re.Match) -> str:
        try:
            return _int_to_words(int(m.group()))
        except (ValueError, OverflowError):
            return m.group()
    return re.sub(r"\b\d+\b", _replace, text)


def _expand_contractions(text: str) -> str:
    """Expand common English contractions word-by-word (text already lowercased)."""
    words = text.split()
    out = []
    for w in words:
        expanded = _CONTRACTIONS.get(w)
        out.extend(expanded.split() if expanded else [w])
    return " ".join(out)


def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = re.sub(r"<[^>]+>", "", text)          # remove TTS tags
    text = text.replace("*", "")                  # remove italic markers
    # Unify all apostrophe/quote variants to plain ASCII so curly and straight
    # forms never end up on opposite sides of the diff.
    text = (text
            .replace("’", "'").replace("‘", "'")  # curly single quotes
            .replace("“", "").replace("”", "")               # curly double quotes
            .replace("„", "").replace("‚", "")               # low double/single quotes
            .replace("′", "'").replace("″", ""))       # prime / double-prime
    text = text.replace("-", " ")                  # hyphens as word boundaries (well-off → well off)
    text = re.sub(r"[^\w\s']", "", text.lower())  # keep only word chars and apostrophes

    # Expand contractions BEFORE possessive stripping so "i'm" → "i am"
    # (not "im") and "we'll" → "we will" (not "well").
    text = _expand_contractions(text)
    # Convert digit sequences to word form so "150" == "hundred fifty".
    text = _expand_numbers(text)

    # Strip word-boundary apostrophes (used as single-quote marks) and
    # normalize possessive endings so "Davis's" == "Davis'" == "Davis".
    # Leading apostrophe: 'hello' (single-quoted) -> hello
    # Trailing 's / ':   possessive forms -> base word
    # Internal apostrophes in contractions (don't, it's, o'clock) are kept.
    cleaned = []
    for w in text.split():
        w = re.sub(r"^'+", "", w)   # strip leading apostrophes
        w = re.sub(r"'s?$", "", w)  # strip trailing possessive ('s or bare ')
        if w:
            cleaned.append(w)
    text = " ".join(cleaned)

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

# Similarity floor for collapsing compound-word splits (e.g. welloff ≈ wellof).
# Higher than _WORD_PAIR_SIM_THRESHOLD because we are comparing full
# concatenated strings (longer = more forgiving on absolute edits).
_COMPOUND_SIM_THRESHOLD = 0.85


def _collapse_compound_splits(diff: list[dict]) -> list[dict]:
    """Re-mark delete/insert runs as equal when they are compound-word variants.

    Handles cases where a hyphenated or closed compound word is transcribed
    with or without a space (or hyphen), e.g.:
      Pyrostick  ↔ Pyro stick   (one word vs two)
      well-off   ↔ well off     (normalised to two words vs two words already equal)
      welloff    ↔ well off     (one word vs two, exact concat match)
      welloff    ↔ wellof       (near-match — dropped double consonant)

    A contiguous run of delete/insert entries is collapsed to equal when the
    concatenation of all original words equals (or is very similar to) the
    concatenation of all transcribed words.
    """
    result = []
    i = 0
    while i < len(diff):
        entry = diff[i]
        if entry["type"] in ("equal", "replace"):
            result.append(entry)
            i += 1
            continue

        # Collect a contiguous block of delete/insert entries.
        run_start = i
        while i < len(diff) and diff[i]["type"] in ("delete", "insert"):
            i += 1
        run = diff[run_start:i]

        orig_concat = "".join(e.get("original", "") for e in run)
        trans_concat = "".join(e.get("transcribed", "") for e in run)

        if orig_concat and trans_concat and (
            orig_concat == trans_concat
            or _word_similarity(orig_concat, trans_concat) >= _COMPOUND_SIM_THRESHOLD
        ):
            # Compound equivalents — emit one equal entry per original word.
            orig_entries = [e for e in run if e.get("original")]
            if not orig_entries:
                orig_entries = [run[0]]
            for oe in orig_entries:
                result.append({
                    "type": "equal",
                    "original": oe["original"],
                    "transcribed": oe["original"],
                    "orig_index": oe["orig_index"],
                })
        else:
            result.extend(run)

    return result


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

    return _collapse_compound_splits(diff)


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
                    threshold: float = 0.95) -> dict:
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
             threshold: float = 0.95,
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
