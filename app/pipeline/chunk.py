"""Split cleaned chapters into TTS-friendly chunks.

Step 5b: Chunks respect sentence, paragraph, dialogue, and scene boundaries.
Chunks never cross chapter or scene boundaries.
"""

import re
from app.database import ProjectDB

DEFAULT_MAX_WORDS = 150
DEFAULT_MIN_WORDS = 30


# Strip a leading numeric prefix like "19 ", "19. ", "19: ", "19 - " from
# titles. Many manuscripts have chapter heading text such as
# "19 Order of war" — when narrated as a title chunk those numbers are
# noise (and the optional "Chapter N:" prefix already supplies the count).
_NUM_PREFIX_RE = re.compile(r"^\s*\d+\s*[\.\:\)\-\–—]?\s+")


def strip_leading_numbering(title: str) -> str:
    """Remove a leading "19 " / "19. " / "19: " style prefix from a title."""
    if not title:
        return title
    return _NUM_PREFIX_RE.sub("", title).strip()


def auto_narration_title(chapter_title: str, chapter_idx: int,
                         chapter_prefix: bool) -> str:
    """Compute the default narrated title for a chapter.

    - Strips a leading numeric prefix from the original title.
    - Optionally prepends ``"Chapter N: "`` (1-based on chapter_idx).
    """
    clean = strip_leading_numbering(chapter_title or "")
    if chapter_prefix:
        return f"Chapter {chapter_idx + 1}: {clean}" if clean else f"Chapter {chapter_idx + 1}"
    return clean


def split_into_chunks(text: str, max_words: int = DEFAULT_MAX_WORDS,
                      min_words: int = DEFAULT_MIN_WORDS) -> list[dict]:
    """Chunk text using full paragraphs as primary units. No overlap.

    Rules:
    - Each chunk is composed of as many consecutive full paragraphs as fit
      under ``max_words``. Adjacent chunks share no text.
    - Paragraphs longer than ``max_words`` are pre-split into sentence
      groups (each fitting under ``max_words``) so the greedy packing
      step never sees an over-budget unit.
    - A scene-break marker always closes the current chunk; the chunk
      immediately before the break has ``scene_break_after`` set.

    Returns a list of ``{"text": str, "scene_break_after": bool}``.
    """
    def split_sentences(para: str) -> list[str]:
        # Split on sentence-ending punctuation followed by whitespace and a capital/quote
        return re.split(r"(?<=[.!?…])\s+(?=[A-Z\"'\u201C\u2018\u2019])", para)

    def wc(s: str) -> int:
        return len(re.findall(r"\w+", s))

    def split_paragraph_to_units(para: str, maxw: int) -> list[str]:
        """Split a paragraph into one-or-more text units (each <= maxw words).

        If the paragraph is short, returns [para]. Otherwise groups sentences
        into successive units each not exceeding `maxw` words.
        """
        sents = split_sentences(para)
        if wc(para) <= maxw:
            return [para]

        groups: list[str] = []
        buf: list[str] = []
        buf_words = 0
        for sent in sents:
            sw = wc(sent)
            if buf and buf_words + sw > maxw:
                groups.append(" ".join(buf).strip())
                buf = []
                buf_words = 0
            buf.append(sent)
            buf_words += sw
        if buf:
            groups.append(" ".join(buf).strip())
        return groups

    # Build paragraph-like "units" for greedy packing; handle scene breaks
    raw_paras = [p.strip() for p in text.split("\n")]
    units: list[dict] = []
    for para in raw_paras:
        if not para:
            continue
        if para == "[SCENE_BREAK]":
            units.append({"is_scene_break": True})
            continue

        # split very long paragraphs into sentence groups that each fit
        parts = split_paragraph_to_units(para, max_words)
        for part in parts:
            units.append({
                "is_scene_break": False,
                "text": part,
                "word_count": wc(part),
            })

    chunks: list[dict] = []
    i = 0
    while i < len(units):
        u = units[i]
        if u.get("is_scene_break"):
            if chunks:
                chunks[-1]["scene_break_after"] = True
            i += 1
            continue

        # Greedy: pack as many consecutive non-break units as fit under max_words
        total = 0
        best_k = 0
        j = i
        while j < len(units) and not units[j].get("is_scene_break"):
            total += units[j]["word_count"]
            if total <= max_words:
                best_k = j - i + 1
                j += 1
                continue
            break

        # Safety net: a single unit that still exceeds max_words shouldn't
        # happen (split_paragraph_to_units pre-splits) but emit it whole
        # rather than loop forever.
        if best_k == 0:
            best_k = 1

        parts = [units[idx]["text"] for idx in range(i, i + best_k)]
        chunk_text = "\n".join(parts).strip()
        chunks.append({"text": chunk_text, "scene_break_after": False})
        i += best_k

    return chunks


def _split_long_paragraph(para: str, max_words: int,
                          chunks: list[dict]):
    """Split a paragraph longer than max_words by sentences."""
    # Split on sentence-ending punctuation followed by space + quote or capital
    sentences = re.split(r'(?<=[.!?…])\s+(?=[A-Z"\'\u201C\u2018\u2019])', para)

    buf = []
    buf_words = 0

    for sent in sentences:
        sw = len(sent.split())
        if buf_words + sw > max_words and buf:
            chunks.append({
                "text": " ".join(buf),
                "scene_break_after": False,
            })
            buf = []
            buf_words = 0
        buf.append(sent)
        buf_words += sw

    if buf:
        chunks.append({
            "text": " ".join(buf),
            "scene_break_after": False,
        })


def _is_open_dialogue(current_text: str, next_para: str) -> bool:
    """Check if splitting here would break mid-dialogue.

    Heuristic: if an opening quote has no closing quote yet, we're inside
    a dialogue turn and shouldn't split.
    """
    # For curly quotes, we can tell open vs close
    open_curly = current_text.count('\u201c')
    close_curly = current_text.count('\u201d')
    if open_curly > close_curly:
        return True  # unmatched curly opening quote

    # For straight quotes, we can only tell if the total is odd (mid-dialogue)
    straight_count = current_text.count('"')
    if straight_count % 2 == 1:
        return True

    return False


def resolve_narration_title(chapter: dict, chapter_prefix: bool) -> str:
    """Return the title-chunk text for a chapter, or "" to skip narration.

    Resolution order:
    - ``narration_title`` is NULL → auto default (chapter title with the
      leading numbering stripped, optionally with a "Chapter N: " prefix).
    - ``narration_title`` is empty string → user opted out, no title chunk.
    - Otherwise → use the explicit user value verbatim.
    """
    nt = chapter.get("narration_title")
    if nt is None:
        return auto_narration_title(chapter.get("title", ""), chapter["idx"], chapter_prefix)
    return nt


def chunk_all_chapters(db: ProjectDB,
                       max_words: int = DEFAULT_MAX_WORDS,
                       min_words: int = DEFAULT_MIN_WORDS,
                       narrate_titles: bool = False,
                       chapter_prefix: bool = True):
    """Chunk all cleaned chapters and store results in DB.

    Args:
        narrate_titles: When True, insert a title chunk at the start of each
            chapter that doesn't have an explicit empty narration_title.
        chapter_prefix: Used as the auto-default policy for chapters where
            ``narration_title`` is NULL. Chapters with an explicit override
            ignore this flag and use their own text.
    """
    chapters = db.get_chapters()
    global_idx = 0

    all_chunks = []
    for ch in chapters:
        text = ch["cleaned_text"] or ch["raw_text"]
        content_chunks = split_into_chunks(text, max_words, min_words)

        # Per-chapter title chunk: honor explicit narration_title override,
        # fall back to the auto default when narrate_titles is on.
        title_text = ""
        nt = ch.get("narration_title")
        if nt is None and narrate_titles:
            title_text = auto_narration_title(ch.get("title", ""), ch["idx"], chapter_prefix)
        elif nt:
            title_text = nt
        # nt == "" → user opted out; leave title_text empty

        if title_text:
            all_chunks.append({
                "id": f"ch{ch['idx']:03d}_title",
                "chapter_id": ch["id"],
                "local_index": -1,
                "global_index": global_idx,
                "original_text": title_text,
                "cleaned_text": title_text,
                "word_count": len(title_text.split()),
                "scene_break_after": 0,
                "chapter_break_after": 0,
                "is_title_chunk": 1,
            })
            global_idx += 1

        for i, chunk in enumerate(content_chunks):
            chunk_id = f"ch{ch['idx']:03d}_chunk{i:04d}"
            all_chunks.append({
                "id": chunk_id,
                "chapter_id": ch["id"],
                "local_index": i,
                "global_index": global_idx,
                "original_text": chunk["text"],
                "cleaned_text": chunk["text"],
                "word_count": len(chunk["text"].split()),
                "scene_break_after": 1 if chunk["scene_break_after"] else 0,
                "chapter_break_after": 1 if i == len(content_chunks) - 1 else 0,
                "is_title_chunk": 0,
            })
            global_idx += 1

    db.insert_chunks(all_chunks)
    db.set_meta("current_step", "chunked")
    return len(all_chunks)
