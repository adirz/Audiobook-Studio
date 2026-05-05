"""Split cleaned chapters into TTS-friendly chunks.

Step 5b: Chunks respect sentence, paragraph, dialogue, and scene boundaries.
Chunks never cross chapter or scene boundaries.
"""

import re
from app.database import ProjectDB

DEFAULT_MAX_WORDS = 150
DEFAULT_MIN_WORDS = 30


def split_into_chunks(text: str, max_words: int = DEFAULT_MAX_WORDS,
                      min_words: int = DEFAULT_MIN_WORDS) -> list[dict]:
    """Chunk text using full paragraphs as primary units with a 1-sentence overlap.

    Rules implemented:
    - Each chunk is composed of k full paragraphs plus one full sentence (from the
      following paragraph) when that fits under `max_words`.
    - k is chosen as the largest integer such that the chunk (including an optional
      carry-in sentence from the previous chunk) does not exceed `max_words`.
    - If adding the extra sentence would exceed `max_words` even for k=1, the
      chunk will be the single full paragraph and the *next* chunk will start
      with the last sentence of this paragraph (this creates the required overlap).
    - Very long paragraphs (longer than `max_words`) are split into sentence
      groups that each fit within `max_words`.

    Returns list of {"text": str, "scene_break_after": bool}.
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
            sents = split_sentences(part)
            units.append({
                "is_scene_break": False,
                "text": part,
                "sentences": sents,
                "word_count": wc(part),
                "skip_first_sentence": False,
            })

    chunks: list[dict] = []
    i = 0
    carry_sentence = None  # sentence to prepend to the next chunk (for overlap)

    while i < len(units):
        u = units[i]
        if u.get("is_scene_break"):
            # mark previous chunk as scene break end
            if chunks:
                chunks[-1]["scene_break_after"] = True
            i += 1
            continue

        carry_in = carry_sentence
        carry_sentence = None

        # Find largest k (number of full units/paragraphs) that fits when also
        # including the first sentence of the following unit (for overlap)
        total = 0
        best_k = 0
        j = i
        while j < len(units) and not units[j].get("is_scene_break"):
            total += units[j]["word_count"]
            # words in first sentence of next unit
            next_first_words = 0
            if j + 1 < len(units) and not units[j + 1].get("is_scene_break"):
                nxt0 = units[j + 1]["sentences"][0] if units[j + 1]["sentences"] else ""
                next_first_words = wc(nxt0)
            carry_words = wc(carry_in) if carry_in else 0
            if carry_words + total + next_first_words <= max_words:
                best_k = j - i + 1
                j += 1
                continue
            break

        if best_k > 0:
            end_idx = i + best_k - 1
            # prepare chunk parts
            parts: list[str] = []
            if carry_in:
                parts.append(carry_in.strip())

            for idx in range(i, i + best_k):
                unit = units[idx]
                if unit.get("skip_first_sentence") and unit.get("sentences"):
                    # omit the first sentence because it is carried in
                    rem = unit["sentences"][1:]
                    if rem:
                        parts.append(" ".join(rem).strip())
                else:
                    parts.append(unit["text"])

            # determine extra (first) sentence from the following unit to append
            extra_sentence = None
            if i + best_k < len(units) and not units[i + best_k].get("is_scene_break"):
                extra_sentence = units[i + best_k]["sentences"][0] if units[i + best_k]["sentences"] else None
                # ensure next unit won't repeat this sentence
                units[i + best_k]["skip_first_sentence"] = True
                carry_sentence = extra_sentence

            if extra_sentence:
                parts.append(extra_sentence.strip())

            chunk_text = "\n".join(p for p in parts if p).strip()
            chunks.append({"text": chunk_text, "scene_break_after": False})
            i = i + best_k
            continue

        # best_k == 0: cannot include any k with the extra-sentence; fallback to
        # single-unit chunk (include carry_in if present) and make the next chunk
        # start with the last sentence of this unit (for overlap)
        unit = units[i]
        parts = []
        if carry_in:
            parts.append(carry_in.strip())
        parts.append(unit["text"])
        # prepare carry for next chunk as the last sentence of this unit
        last_sent = unit["sentences"][-1] if unit.get("sentences") else None
        carry_sentence = last_sent
        chunk_text = "\n".join(p for p in parts if p).strip()
        chunks.append({"text": chunk_text, "scene_break_after": False})
        i += 1

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


def chunk_all_chapters(db: ProjectDB,
                       max_words: int = DEFAULT_MAX_WORDS,
                       min_words: int = DEFAULT_MIN_WORDS):
    """Chunk all cleaned chapters and store results in DB."""
    chapters = db.get_chapters()
    global_idx = 0

    all_chunks = []
    for ch in chapters:
        text = ch["cleaned_text"] or ch["raw_text"]
        chunks = split_into_chunks(text, max_words, min_words)

        for i, chunk in enumerate(chunks):
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
                "chapter_break_after": 1 if i == len(chunks) - 1 else 0,
            })
            global_idx += 1

    db.insert_chunks(all_chunks)
    db.set_meta("current_step", "chunked")
    return len(all_chunks)
