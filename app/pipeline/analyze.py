"""Analyze extracted text: find non-standard symbols and non-standard words.

Step 4: Symbol detection (scene break candidates)
Step 6: Non-standard word scanning (pronunciation dictionary candidates)
"""

import re
import unicodedata
from collections import Counter
from app.database import ProjectDB


# Characters that are "standard" in English prose
STANDARD_CHARS = set(
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789'
    ' \t\n'
    '.,;:!?\'"-—–…()'
    '""''*'  # curly quotes and italic markers
)


def find_nonstandard_symbols(db: ProjectDB) -> list[dict]:
    """Scan all chapter text for non-standard symbols.

    Returns list of {"symbol": str, "context_preview": str} and
    inserts them into the symbols table.
    """
    chapters = db.get_chapters()
    found = {}  # symbol -> first context

    for ch in chapters:
        text = ch["raw_text"]
        for i, char in enumerate(text):
            if char not in STANDARD_CHARS and char not in found:
                # Grab surrounding context
                start = max(0, i - 30)
                end = min(len(text), i + 30)
                context = text[start:end].replace("\n", " ")
                uname = unicodedata.name(char, "UNKNOWN")
                found[char] = {
                    "symbol": char,
                    "context_preview": f"[{uname}]  …{context}…",
                }

    # Also look for multi-character patterns that might be scene breaks
    # e.g. "***", "---", "♦ ♦ ♦", "* * *", "⁂"
    scene_break_patterns = [
        (r"^\*\s*\*\s*\*\s*$", "* * *"),
        (r"^-{3,}\s*$", "---"),
        (r"^={3,}\s*$", "==="),
        (r"^#{3,}\s*$", "###"),
        (r"^~{3,}\s*$", "~~~"),
        (r"^⁂\s*$", "⁂"),
    ]
    for ch in chapters:
        for line in ch["raw_text"].split("\n"):
            stripped = line.strip()
            for pattern, label in scene_break_patterns:
                if re.match(pattern, stripped) and label not in found:
                    found[label] = {
                        "symbol": label,
                        "context_preview": f"[Scene break pattern: {stripped}]",
                    }

    symbols = list(found.values())
    db.insert_symbols(symbols)
    return symbols


def find_nonstandard_words(db: ProjectDB) -> list[dict]:
    """Scan all chunks for words not in a standard English dictionary.

    Uses lemmatization to handle irregular forms (held→hold, children→child),
    handles contractions and possessives, and groups variants under their
    base form (Tanji + Tanji's = one entry).

    Returns list of word info dicts and inserts them as pron_entries.
    """
    # ─── Build the English word set ───────────────────────────
    english_words = set()
    try:
        import nltk
        # Ensure common corpora are present (quiet download if needed)
        for corpus_name in ("words", "wordnet"):
            try:
                nltk.data.find(f"corpora/{corpus_name}")
            except LookupError:
                nltk.download(corpus_name, quiet=True)

        # tagger name can vary; try a couple of common names
        for tagger in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
            try:
                nltk.data.find(f"taggers/{tagger}")
                break
            except LookupError:
                try:
                    nltk.download(tagger, quiet=True)
                    break
                except Exception:
                    pass

        english_words = set(w.lower() for w in nltk.corpus.words.words())

        # Also include WordNet lemmas which cover many inflections
        try:
            from nltk.corpus import wordnet as wn
            english_words.update(x.lower() for x in wn.all_lemma_names())
        except Exception:
            pass

        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
    except Exception:
        # NLTK not available or corpora couldn't be fetched — fall back
        _lemmatizer = None

    # Common contractions, possessives, and informal words
    english_words.update({
        "ok", "okay", "yeah", "nah", "gonna", "wanna", "gotta",
        "shouldn't", "couldn't", "wouldn't", "don't", "didn't",
        "isn't", "aren't", "wasn't", "weren't", "won't", "can't",
        "i'm", "i'll", "i'd", "i've", "he's", "she's", "it's",
        "we're", "they're", "you're", "we've", "they've", "you've",
        "that's", "there's", "here's", "what's", "who's",
        "let's", "who'd", "how's", "where's", "someone's",
        "everyone's", "nobody's", "everything's", "we'll", "you'll",
        "he'd", "she'd", "they'd", "they'll", "it'll",
        "hasn't", "hadn't", "haven't", "doesn't", "mustn't",
        "mightn't", "needn't", "shan't", "oughtn't",
        "who've", "would've", "could've", "should've",
        "might've", "must've", "where'd", "what'd",
    })

    def normalize_apostrophes(text: str) -> str:
        """Replace all curly/typographic quotes with straight ones."""
        return text.replace("\u2019", "'").replace("\u2018", "'").replace("\u2032", "'")

    def strip_possessive(word: str) -> str:
        """Remove trailing 's or ' from possessives (straight or curly)."""
        if word.endswith("'s") or word.endswith("\u2019s"):
            return word[:-2]
        if (word.endswith("'") or word.endswith("\u2019")) and len(word) > 1:
            return word[:-1]
        return word

    def is_english(word: str) -> bool:
        """Check if a word is standard English, including inflections."""
        low = normalize_apostrophes(word.lower())

        # Treat hyphenated/compound words as English if all parts are English
        if re.search(r"[-–—]", low):
            parts = re.split(r"[-–—]", low)
            if len(parts) > 1 and all(part and is_english(part) for part in parts):
                return True

        # Direct lookup
        if low in english_words:
            return True

        # Strip possessive and check base
        base = strip_possessive(low)
        if base in english_words:
            return True

        # Contraction: split on apostrophe, check if parts are known
        if "'" in low:
            parts = low.split("'")
            # "hadn't" → ["hadn", "t"] — check if "had" is a word
            # "we'll" → ["we", "ll"]
            contraction_suffixes = {"t", "s", "d", "ll", "ve", "re", "m", "nt"}
            if len(parts) == 2 and parts[1] in contraction_suffixes:
                # The base might need an extra letter: hadn → had, doesn → does
                stem = parts[0]
                if stem in english_words:
                    return True
                if stem + "d" in english_words:  # "hadn" → "hadn" + check, but "had" → yes
                    return True
                if stem.endswith("n") and stem[:-1] in english_words:  # hadn → had
                    return True
                if stem.endswith("s") and stem[:-1] in english_words:  # does → doe? no...
                    return True
                # Try lemmatizing the stem
                if _lemmatizer:
                    for pos in ('v', 'n', 'a', 'r'):
                        lemma = _lemmatizer.lemmatize(stem, pos)
                        if lemma in english_words:
                            return True

        # Lemmatize with all POS tags (handles held→hold, children→child, etc.)
        if _lemmatizer:
            for pos in ('v', 'n', 'a', 'r'):
                lemma = _lemmatizer.lemmatize(base, pos)
                if lemma in english_words:
                    return True

        # Common suffix stripping for words lemmatizer misses
        SUFFIXES = ["s", "es", "ed", "ing", "er", "est", "ly", "ness",
                     "ment", "tion", "sion", "ful", "less", "able",
                     "ive", "al", "ize", "ise", "ify"]
        for suffix in SUFFIXES:
            if base.endswith(suffix) and len(base) > len(suffix) + 1:
                stem = base[:-len(suffix)]
                if stem in english_words or stem + "e" in english_words:
                    return True
                # Doubled consonant: "nodded" → stripped to "nodd" → "nod"
                if len(stem) >= 2 and stem[-1] == stem[-2] and stem[:-1] in english_words:
                    return True

        # -ied → -y: replied → reply
        if base.endswith("ied") and len(base) > 4:
            if base[:-3] + "y" in english_words:
                return True
        # -ies → -y: stories → story
        if base.endswith("ies") and len(base) > 4:
            if base[:-3] + "y" in english_words:
                return True

        return False

    # ─── Scan chunks ──────────────────────────────────────────
    chunks = db.get_chunks()
    word_data = {}  # base_form -> {word, count, first_chunk_id, context}

    # Pattern: matches words including those with apostrophes/hyphens
    word_pattern = re.compile(
        r"[A-Za-zÀ-ÿ](?:[A-Za-zÀ-ÿ'''\u2018\u2019\u2032-]*[A-Za-zÀ-ÿ])?"
    )

    for chunk in chunks:
        text = chunk["original_text"]
        clean_text = re.sub(r"<[^>]+>", "", text)
        clean_text = clean_text.replace("*", "")  # remove italic markers

        for match in word_pattern.finditer(clean_text):
            word = match.group()

            # Skip very short
            if len(word) <= 1:
                continue

            # Normalize and check
            normalized = normalize_apostrophes(word.lower())

            if is_english(normalized):
                continue

            # Skip orphaned contraction fragments
            if normalized in ("s", "t", "d", "ll", "ve", "re", "m",
                              "didn", "couldn", "wouldn", "shouldn",
                              "isn", "aren", "wasn", "weren", "won", "don",
                              "hasn", "hadn", "haven", "doesn", "mustn"):
                continue

            # Group key: strip possessive and lowercase for dedup
            # So "Tanji" and "Tanji's" map to the same entry
            base_key = strip_possessive(normalize_apostrophes(word)).lower()

            if base_key not in word_data:
                # Use the sentence-aware context extractor so stored examples
                # match what the pronunciation test will use. Prefer the
                # context coming from the chunk we're currently scanning.
                ctx = get_context_for_word(db, strip_possessive(word), prefer_chunk_id=chunk["id"])

                word_data[base_key] = {
                    "word": strip_possessive(word),  # store the base form
                    "count": 0,
                    "first_chunk_id": chunk["id"],
                    "context": ctx,
                }
            word_data[base_key]["count"] += 1

    # ─── Insert into DB ───────────────────────────────────────
    entries = []
    for key, data in sorted(word_data.items(), key=lambda x: -x[1]["count"]):
        entry_id = db.insert_pron_entry(
            word=data["word"],
            frequency=data["count"],
            example_chunk_id=data["first_chunk_id"],
            example_context=data["context"],
        )
        entries.append({
            "id": entry_id,
            "word": data["word"],
            "frequency": data["count"],
            "context": data["context"],
        })

    return entries


def get_context_for_word(db: ProjectDB, word: str,
                         context_chars: int = 120,
                         prefer_chunk_id: str | None = None) -> str:
    """Find a word in the chunks and return its surrounding context.

    Used for pronunciation testing — plays the word in its natural
    book context rather than a synthetic sentence.
    Preserves original punctuation and formatting.
    """
    MIN_WORDS_AROUND = 5

    # Regex to match sentences ending with punctuation, keeping trailing
    # closing quotes/parens. Uses DOTALL so sentences can span newlines.
    sentence_re = re.compile(r'.+?(?:[.!?…][\)"\'\u201D\u2019]*)+(?=\s+|$)', re.S)

    chunks = db.get_chunks()

    # Group chunks into contiguous segments that do NOT cross scene or chapter breaks.
    segments: list[str] = []
    seg_chunk_map: dict[str, int] = {}
    cur_parts: list[str] = []
    cur_chunk_ids: list[str] = []
    for ch in chunks:
        t = re.sub(r"<[^>]+>", "", ch["original_text"]).replace("*", " ").strip()
        cur_parts.append(t)
        cur_chunk_ids.append(ch["id"])
        # break segment at scene or chapter boundaries
        if ch.get("scene_break_after") or ch.get("chapter_break_after"):
            seg_idx = len(segments)
            seg_text = " ".join(p for p in cur_parts if p)
            segments.append(seg_text)
            for cid in cur_chunk_ids:
                seg_chunk_map[cid] = seg_idx
            cur_parts = []
            cur_chunk_ids = []
    if cur_parts:
        seg_idx = len(segments)
        seg_text = " ".join(p for p in cur_parts if p)
        segments.append(seg_text)
        for cid in cur_chunk_ids:
            seg_chunk_map[cid] = seg_idx

    if not segments:
        return f"The word {word} appears in this text."

    def wcount(s: str) -> int:
        return len(re.findall(r"\w+", s))

    word_re = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)

    # Collect candidates across segments
    candidates = []
    for seg_idx, seg_text in enumerate(segments):
        if not seg_text:
            continue
        sentences = list(sentence_re.finditer(seg_text))
        if not sentences:
            # No sentence boundaries; try bounded matches
            for m in word_re.finditer(seg_text):
                candidates.append({"seg": seg_idx, "before": 0, "after": 0, "start": 0, "end": 0, "match_span": m.span()})
            continue

        for match in word_re.finditer(seg_text):
            pos = match.start()
            # find the sentence index containing this match
            idx = None
            for i, s in enumerate(sentences):
                if s.start() <= pos < s.end():
                    idx = i
                    break
            if idx is None:
                continue

            sent_text = sentences[idx].group(0)
            within_offset = pos - sentences[idx].start()
            before_in_sent = wcount(sent_text[:within_offset])
            after_in_sent = wcount(sent_text[within_offset + len(match.group(0)):])

            before_count = before_in_sent
            after_count = after_in_sent
            selected_start = idx
            selected_end = idx

            # Expand backward if needed
            i = idx - 1
            while before_count < MIN_WORDS_AROUND and i >= 0:
                before_count += wcount(sentences[i].group(0))
                selected_start = i
                i -= 1

            # Expand forward if needed
            j = idx + 1
            while after_count < MIN_WORDS_AROUND and j < len(sentences):
                after_count += wcount(sentences[j].group(0))
                selected_end = j
                j += 1

            candidates.append({
                "seg": seg_idx,
                "before": before_count,
                "after": after_count,
                "start": selected_start,
                "end": selected_end,
            })

    # Determine preferred segment (if caller provided a chunk id)
    preferred_seg = None
    if prefer_chunk_id and prefer_chunk_id in seg_chunk_map:
        preferred_seg = seg_chunk_map[prefer_chunk_id]

    def pick_best(cands):
        return max(cands, key=lambda c: (min(c["before"], c["after"]), c["before"] + c["after"]))

    # 1) Prefer a candidate in the preferred segment that satisfies both sides
    if preferred_seg is not None:
        pref_cands = [c for c in candidates if c["seg"] == preferred_seg]
        good = [c for c in pref_cands if c["before"] >= MIN_WORDS_AROUND and c["after"] >= MIN_WORDS_AROUND]
        if good:
            best = pick_best(good)
            seg_sentences = list(sentence_re.finditer(segments[best["seg"]]))
            parts = [seg_sentences[k].group(0).strip() for k in range(best["start"], best["end"] + 1)]
            # dedupe adjacent identical sentences
            deduped = []
            prev_norm = None
            for p in parts:
                norm = " ".join(p.split())
                if norm != prev_norm:
                    deduped.append(p)
                    prev_norm = norm
            return " ".join(deduped).strip()

    # 2) Any segment candidate that satisfies both sides
    good_all = [c for c in candidates if c["before"] >= MIN_WORDS_AROUND and c["after"] >= MIN_WORDS_AROUND]
    if good_all:
        best = pick_best(good_all)
        seg_sentences = list(sentence_re.finditer(segments[best["seg"]]))
        parts = [seg_sentences[k].group(0).strip() for k in range(best["start"], best["end"] + 1)]
        deduped = []
        prev_norm = None
        for p in parts:
            norm = " ".join(p.split())
            if norm != prev_norm:
                deduped.append(p)
                prev_norm = norm
        return " ".join(deduped).strip()

    # 3) Otherwise prefer best candidate in preferred segment
    if preferred_seg is not None:
        pref_cands = [c for c in candidates if c["seg"] == preferred_seg]
        if pref_cands:
            best = pick_best(pref_cands)
            seg_sentences = list(sentence_re.finditer(segments[best["seg"]]))
            parts = [seg_sentences[k].group(0).strip() for k in range(best["start"], best["end"] + 1)]
            deduped = []
            prev_norm = None
            for p in parts:
                norm = " ".join(p.split())
                if norm != prev_norm:
                    deduped.append(p)
                    prev_norm = norm
            return " ".join(deduped).strip()

    # 4) Otherwise pick the best candidate anywhere
    if candidates:
        best = pick_best(candidates)
        seg_sentences = list(sentence_re.finditer(segments[best["seg"]]))
        parts = [seg_sentences[k].group(0).strip() for k in range(best["start"], best["end"] + 1)]
        deduped = []
        prev_norm = None
        for p in parts:
            norm = " ".join(p.split())
            if norm != prev_norm:
                deduped.append(p)
                prev_norm = norm
        return " ".join(deduped).strip()

    # 5) Fallback: bounded snippet around first occurrence in preferred segment, else global
    if preferred_seg is not None and segments[preferred_seg]:
        seg_text = segments[preferred_seg]
        m = word_re.search(seg_text)
        if m:
            s = max(0, m.start() - context_chars)
            e = min(len(seg_text), m.end() + context_chars)
            return seg_text[s:e].strip()

    # global fallback
    joined = " ".join(segments)
    m = word_re.search(joined)
    if m:
        s = max(0, m.start() - context_chars)
        e = min(len(joined), m.end() + context_chars)
        return joined[s:e].strip()

    return f"The word {word} appears in this text."
