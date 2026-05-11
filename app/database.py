"""SQLite database layer — one database per project."""

import sqlite3
import json
import threading
from functools import wraps
from pathlib import Path
from contextlib import contextmanager
from typing import Any

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS project_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idx INTEGER NOT NULL,
    title TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    cleaned_text TEXT,
    status TEXT DEFAULT 'extracted'  -- extracted, cleaned, chunked
);

CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    is_scene_break INTEGER,  -- NULL = undecided, 1 = yes, 0 = no
    context_preview TEXT,
    user_decided INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,         -- e.g. ch001_chunk0003
    chapter_id INTEGER NOT NULL,
    local_index INTEGER NOT NULL,
    global_index INTEGER NOT NULL,
    original_text TEXT NOT NULL,
    cleaned_text TEXT,
    tagged_text TEXT,            -- NULL until emotion tagging
    pron_text TEXT,              -- text with pronunciation subs applied
    scene_break_after INTEGER DEFAULT 0,
    chapter_break_after INTEGER DEFAULT 0,
    word_count INTEGER NOT NULL,
    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
);

CREATE TABLE IF NOT EXISTS pron_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    phonetic TEXT,              -- NULL = sounds fine as-is
    type_tag TEXT,              -- character, place, object, custom, standard-override
    status TEXT DEFAULT 'pending',  -- pending, testing, approved, skipped
    frequency INTEGER DEFAULT 1,
    example_chunk_id TEXT,
    example_context TEXT,       -- surrounding text for test playback
    notes TEXT,
    is_global INTEGER DEFAULT 1, -- 1 = replace everywhere, 0 = location-specific
    FOREIGN KEY (example_chunk_id) REFERENCES chunks(id)
);

CREATE TABLE IF NOT EXISTS pron_location_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    phonetic TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    word_offset INTEGER,        -- word position in chunk for precision
    notes TEXT,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);

CREATE TABLE IF NOT EXISTS pron_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pron_entry_id INTEGER NOT NULL,
    attempt_number INTEGER NOT NULL,
    phonetic_used TEXT NOT NULL,
    audio_path TEXT,
    chosen INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pron_entry_id) REFERENCES pron_entries(id)
);

CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    wav_path TEXT,
    duration_sec REAL,
    gen_time_sec REAL,
    params_json TEXT,
    status TEXT DEFAULT 'pending',  -- pending, generating, ok, error, flagged
    error_msg TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);

CREATE TABLE IF NOT EXISTS qa_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL,
    generation_id INTEGER NOT NULL,
    transcribed_text TEXT,
    similarity_score REAL,
    word_diff_json TEXT,        -- word-level alignment data
    status TEXT DEFAULT 'pending',  -- pending, pass, fail, override
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id),
    FOREIGN KEY (generation_id) REFERENCES generations(id)
);

CREATE TABLE IF NOT EXISTS user_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL,
    flag_type TEXT NOT NULL,    -- garbled, missing, repetition, added_words, pronunciation, other
    word_range TEXT,            -- "12-15" = words 12 through 15
    notes TEXT,
    resolved INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);

CREATE TABLE IF NOT EXISTS tagging_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_prompt TEXT NOT NULL,
    user_approved INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(chapter_id);
CREATE INDEX IF NOT EXISTS idx_chunks_global ON chunks(global_index);
CREATE INDEX IF NOT EXISTS idx_pron_word ON pron_entries(word);
CREATE INDEX IF NOT EXISTS idx_generations_chunk ON generations(chunk_id);
CREATE INDEX IF NOT EXISTS idx_qa_chunk ON qa_results(chunk_id);
CREATE INDEX IF NOT EXISTS idx_flags_chunk ON user_flags(chunk_id);
"""


def _locked(method):
    """Hold the per-DB RLock for the duration of the call.

    sqlite3.Connection is not thread-safe even in WAL mode with
    check_same_thread=False — concurrent execute/commit calls from
    different threads (HTTP handlers vs background workers) corrupt
    cursor state and can interleave a logical multi-statement operation.
    Re-entrant so cross-method calls within the same thread don't
    deadlock.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class ProjectDB:
    """Wrapper around a per-project SQLite database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    @_locked
    def _init_schema(self):
        self.conn.executescript(SCHEMA_SQL)
        # Migrate: add is_title_chunk column to existing databases
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(chunks)").fetchall()]
        if "is_title_chunk" not in cols:
            self.conn.execute(
                "ALTER TABLE chunks ADD COLUMN is_title_chunk INTEGER DEFAULT 0"
            )
        # Migrate: add narration_title to chapters. NULL = use auto default
        # (derived from the chapter title), empty string = no narration.
        ch_cols = [r[1] for r in self.conn.execute("PRAGMA table_info(chapters)").fetchall()]
        if "narration_title" not in ch_cols:
            self.conn.execute(
                "ALTER TABLE chapters ADD COLUMN narration_title TEXT"
            )
        # Migrate: add per-chunk voice override. NULL = use the voice
        # supplied at generate-time (the request's default).
        if "voice" not in cols:
            self.conn.execute(
                "ALTER TABLE chunks ADD COLUMN voice TEXT"
            )
        # Migrate: record which scene-break symbol terminated a chunk
        # (NULL when no break, or for legacy data cleaned before the
        # symbol was tracked).
        if "scene_break_symbol" not in cols:
            self.conn.execute(
                "ALTER TABLE chunks ADD COLUMN scene_break_symbol TEXT"
            )
        # Migrate: flag that the user manually edited pron_text from the
        # review screen. When set, apply_all_pronunciation skips this chunk
        # so the edit is not overwritten on the next generate run.
        if "pron_text_locked" not in cols:
            self.conn.execute(
                "ALTER TABLE chunks ADD COLUMN pron_text_locked INTEGER DEFAULT 0"
            )
        # Set schema version if new
        existing = self.get_meta("schema_version")
        if existing is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
            self.set_meta("current_step", "new")
        self.conn.commit()

    @_locked
    def close(self):
        self.conn.close()

    # -- Meta helpers --

    @_locked
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM project_meta WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    @_locked
    def set_meta(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO project_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    # -- Generic helpers --

    @_locked
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    @_locked
    def executemany(self, sql: str, params_list: list) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params_list)

    @_locked
    def commit(self):
        self.conn.commit()

    @_locked
    def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    @_locked
    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # -- Chapter operations --

    @_locked
    def insert_chapters(self, chapters: list[dict]):
        self.executemany(
            "INSERT INTO chapters (idx, title, raw_text) VALUES (?, ?, ?)",
            [(ch["idx"], ch["title"], ch["raw_text"]) for ch in chapters],
        )
        self.commit()

    @_locked
    def get_chapters(self) -> list[dict]:
        return self.fetchall("SELECT * FROM chapters ORDER BY idx")

    @_locked
    def update_chapter_cleaned(self, chapter_id: int, cleaned_text: str):
        self.execute(
            "UPDATE chapters SET cleaned_text=?, status='cleaned' WHERE id=?",
            (cleaned_text, chapter_id),
        )
        self.commit()

    @_locked
    def update_chapter_narration_title(self, chapter_id: int, narration_title):
        """Set the narration title used for the chapter's title audio chunk.

        ``None`` means "use the auto default" (derived from the title).
        Empty string means "do not narrate this chapter's title" — no
        title chunk will be created.
        """
        self.execute(
            "UPDATE chapters SET narration_title=? WHERE id=?",
            (narration_title, chapter_id),
        )
        self.commit()

    # -- Symbol operations --

    @_locked
    def insert_symbols(self, symbols: list[dict]):
        for sym in symbols:
            self.execute(
                "INSERT OR IGNORE INTO symbols (symbol, context_preview) VALUES (?, ?)",
                (sym["symbol"], sym.get("context_preview", "")),
            )
        self.commit()

    @_locked
    def get_symbols(self, undecided_only: bool = False) -> list[dict]:
        if undecided_only:
            return self.fetchall(
                "SELECT * FROM symbols WHERE user_decided=0"
            )
        return self.fetchall("SELECT * FROM symbols")

    @_locked
    def update_symbol(self, symbol_id: int, is_scene_break: bool):
        self.execute(
            "UPDATE symbols SET is_scene_break=?, user_decided=1 WHERE id=?",
            (1 if is_scene_break else 0, symbol_id),
        )
        self.commit()

    # -- Chunk operations --

    @_locked
    def insert_chunks(self, chunks: list[dict]):
        # Perform per-row insert with fallback to update when a chunk ID
        # already exists. This makes chunking idempotent / re-runnable
        # without raising UNIQUE constraint errors when chunk IDs collide.
        for c in chunks:
            is_title = int(bool(c.get("is_title_chunk", 0)))
            sb_sym = c.get("scene_break_symbol")
            vals = (
                c["id"], c["chapter_id"], c["local_index"], c["global_index"],
                c["original_text"], c.get("cleaned_text"),
                c["word_count"], c.get("scene_break_after", 0),
                c.get("chapter_break_after", 0), is_title, sb_sym,
            )
            try:
                self.execute(
                    """INSERT INTO chunks
                       (id, chapter_id, local_index, global_index,
                        original_text, cleaned_text, word_count,
                        scene_break_after, chapter_break_after, is_title_chunk,
                        scene_break_symbol)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    vals,
                )
            except sqlite3.IntegrityError:
                # Chunk already exists — update its fields instead of failing.
                self.execute(
                    """UPDATE chunks SET
                       chapter_id=?, local_index=?, global_index=?,
                       original_text=?, cleaned_text=?, word_count=?,
                       scene_break_after=?, chapter_break_after=?, is_title_chunk=?,
                       scene_break_symbol=?
                       WHERE id=?""",
                    (
                        c["chapter_id"], c["local_index"], c["global_index"],
                        c["original_text"], c.get("cleaned_text"),
                        c["word_count"], c.get("scene_break_after", 0),
                        c.get("chapter_break_after", 0), is_title, sb_sym, c["id"],
                    ),
                )
        self.commit()

    @_locked
    def get_chunks(self, chapter_id: int | None = None) -> list[dict]:
        # Tiebreaker on local_index: title chunks (local_index=-1) inserted
        # by /title-chunks/refresh share the global_index of the first
        # content chunk in their chapter, so the secondary sort puts the
        # title before the body where it belongs.
        if chapter_id is not None:
            return self.fetchall(
                "SELECT * FROM chunks WHERE chapter_id=? ORDER BY global_index, local_index",
                (chapter_id,),
            )
        return self.fetchall("SELECT * FROM chunks ORDER BY global_index, local_index")

    @_locked
    def get_chunk(self, chunk_id: str) -> dict | None:
        return self.fetchone("SELECT * FROM chunks WHERE id=?", (chunk_id,))

    @_locked
    def update_chunk_tagged(self, chunk_id: str, tagged_text: str):
        self.execute(
            "UPDATE chunks SET tagged_text=? WHERE id=?",
            (tagged_text, chunk_id),
        )
        self.commit()

    @_locked
    def update_chunk_pron(self, chunk_id: str, pron_text: str):
        self.execute(
            "UPDATE chunks SET pron_text=? WHERE id=?",
            (pron_text, chunk_id),
        )
        self.commit()

    @_locked
    def update_chunk_voice(self, chunk_id: str, voice):
        """Set per-chunk voice override. Pass None to clear the override."""
        self.execute(
            "UPDATE chunks SET voice=? WHERE id=?",
            (voice, chunk_id),
        )
        self.commit()

    @_locked
    def bulk_update_chunk_voice(self, chunk_ids: list[str], voice):
        """Set the same voice override on many chunks at once."""
        if not chunk_ids:
            return 0
        self.executemany(
            "UPDATE chunks SET voice=? WHERE id=?",
            [(voice, cid) for cid in chunk_ids],
        )
        self.commit()
        return len(chunk_ids)

    # -- Pronunciation operations --

    @_locked
    def insert_pron_entry(self, word: str, frequency: int, example_chunk_id: str,
                          example_context: str, type_tag: str = "") -> int:
        cur = self.execute(
            """INSERT INTO pron_entries
               (word, frequency, example_chunk_id, example_context, type_tag)
               VALUES (?, ?, ?, ?, ?)""",
            (word, frequency, example_chunk_id, example_context, type_tag),
        )
        self.commit()
        return cur.lastrowid

    @_locked
    def get_pron_entries(self, status: str | None = None) -> list[dict]:
        if status:
            return self.fetchall(
                "SELECT * FROM pron_entries WHERE status=? ORDER BY frequency DESC",
                (status,),
            )
        return self.fetchall(
            "SELECT * FROM pron_entries ORDER BY frequency DESC"
        )

    @_locked
    def update_pron_entry(self, entry_id: int, **kwargs):
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [entry_id]
        self.execute(f"UPDATE pron_entries SET {sets} WHERE id=?", tuple(vals))
        self.commit()

    @_locked
    def insert_pron_attempt(self, entry_id: int, attempt_num: int,
                            phonetic: str, audio_path: str) -> int:
        cur = self.execute(
            """INSERT INTO pron_attempts
               (pron_entry_id, attempt_number, phonetic_used, audio_path)
               VALUES (?, ?, ?, ?)""",
            (entry_id, attempt_num, phonetic, audio_path),
        )
        self.commit()
        return cur.lastrowid

    @_locked
    def choose_pron_attempt(self, attempt_id: int):
        """Mark one attempt as chosen, unmark others for same entry."""
        row = self.fetchone(
            "SELECT pron_entry_id FROM pron_attempts WHERE id=?", (attempt_id,)
        )
        if row:
            self.execute(
                "UPDATE pron_attempts SET chosen=0 WHERE pron_entry_id=?",
                (row["pron_entry_id"],),
            )
            self.execute(
                "UPDATE pron_attempts SET chosen=1 WHERE id=?", (attempt_id,),
            )
            self.commit()

    # -- Location overrides --

    @_locked
    def insert_location_override(self, word: str, phonetic: str,
                                 chunk_id: str, word_offset: int = None,
                                 notes: str = ""):
        self.execute(
            """INSERT INTO pron_location_overrides
               (word, phonetic, chunk_id, word_offset, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (word, phonetic, chunk_id, word_offset, notes),
        )
        self.commit()

    @_locked
    def get_location_overrides(self, chunk_id: str = None) -> list[dict]:
        if chunk_id:
            return self.fetchall(
                "SELECT * FROM pron_location_overrides WHERE chunk_id=?",
                (chunk_id,),
            )
        return self.fetchall("SELECT * FROM pron_location_overrides")

    # -- Generation operations --

    @_locked
    def insert_generation(self, chunk_id: str, attempt: int,
                          params: dict) -> int:
        cur = self.execute(
            """INSERT INTO generations (chunk_id, attempt, params_json, status)
               VALUES (?, ?, ?, 'pending')""",
            (chunk_id, attempt, json.dumps(params)),
        )
        self.commit()
        return cur.lastrowid

    @_locked
    def update_generation(self, gen_id: int, **kwargs):
        if "params" in kwargs:
            kwargs["params_json"] = json.dumps(kwargs.pop("params"))
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [gen_id]
        self.execute(f"UPDATE generations SET {sets} WHERE id=?", tuple(vals))
        self.commit()

    @_locked
    def get_latest_generation(self, chunk_id: str) -> dict | None:
        # Tiebreaker on id DESC: if two rows share the same attempt number
        # (can happen after an interrupted run that's been retried), the
        # most recently inserted row always wins.
        return self.fetchone(
            """SELECT * FROM generations
               WHERE chunk_id=? ORDER BY attempt DESC, id DESC LIMIT 1""",
            (chunk_id,),
        )

    @_locked
    def reset_stale_generations(self) -> int:
        """Mark any 'generating' or 'pending' rows as 'error'.

        Call this only when no generation task is running for this project
        (e.g. at server startup). Such rows are leftovers from an
        interrupted run and would otherwise hide the chunk from selection
        / re-generation flows.

        Returns the number of rows updated.
        """
        cur = self.execute(
            """UPDATE generations
               SET status='error', error_msg='interrupted (server restart)'
               WHERE status IN ('generating', 'pending')"""
        )
        self.commit()
        return cur.rowcount

    @_locked
    def get_generations(self, chunk_id: str = None,
                        status: str = None) -> list[dict]:
        sql = "SELECT * FROM generations WHERE 1=1"
        params = []
        if chunk_id:
            sql += " AND chunk_id=?"
            params.append(chunk_id)
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY chunk_id, attempt"
        return self.fetchall(sql, tuple(params))

    # -- QA operations --

    @_locked
    def insert_qa_result(self, chunk_id: str, gen_id: int,
                         transcribed: str, score: float,
                         word_diff: list, status: str = None) -> int:
        if status is None:
            status = "pass" if score >= 0.85 else "fail"
        cur = self.execute(
            """INSERT INTO qa_results
               (chunk_id, generation_id, transcribed_text,
                similarity_score, word_diff_json, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (chunk_id, gen_id, transcribed, score,
             json.dumps(word_diff), status),
        )
        self.commit()
        return cur.lastrowid

    # Manual QA overrides: a chunk's QA status (pass/fail) can be set or
    # cleared by the user from the review UI. We represent these as a new
    # qa_results row whose ``similarity_score`` is NULL — that NULL is
    # the marker the system uses to tell "manually marked" rows apart
    # from automatic Whisper-driven ones (and is what ``clear_manual_qa``
    # looks for when reverting).
    @_locked
    def set_manual_qa_status(self, chunk_id: str, status: str) -> int:
        gen = self.get_latest_generation(chunk_id)
        gen_id = gen["id"] if gen else 0
        cur = self.execute(
            """INSERT INTO qa_results
               (chunk_id, generation_id, transcribed_text,
                similarity_score, word_diff_json, status)
               VALUES (?, ?, ?, NULL, NULL, ?)""",
            (chunk_id, gen_id, "(manual override)", status),
        )
        self.commit()
        return cur.lastrowid

    @_locked
    def clear_manual_qa(self, chunk_id: str) -> bool:
        """Drop the most recent manual override row, if any.

        Returns True when a row was deleted. Manual overrides are
        identified by ``similarity_score IS NULL``.
        """
        row = self.fetchone(
            """SELECT id, similarity_score FROM qa_results
               WHERE chunk_id=? ORDER BY id DESC LIMIT 1""",
            (chunk_id,),
        )
        if not row or row.get("similarity_score") is not None:
            return False
        self.execute("DELETE FROM qa_results WHERE id=?", (row["id"],))
        self.commit()
        return True

    # -- User flags --

    @_locked
    def insert_flag(self, chunk_id: str, flag_type: str,
                    word_range: str = None, notes: str = ""):
        self.execute(
            """INSERT INTO user_flags (chunk_id, flag_type, word_range, notes)
               VALUES (?, ?, ?, ?)""",
            (chunk_id, flag_type, word_range, notes),
        )
        self.commit()

    @_locked
    def get_flags(self, resolved: bool = None) -> list[dict]:
        if resolved is not None:
            return self.fetchall(
                "SELECT * FROM user_flags WHERE resolved=?",
                (1 if resolved else 0,),
            )
        return self.fetchall("SELECT * FROM user_flags")

    @_locked
    def resolve_flag(self, flag_id: int):
        self.execute(
            "UPDATE user_flags SET resolved=1 WHERE id=?", (flag_id,)
        )
        self.commit()

    # -- Tagging config --

    @_locked
    def save_tagging_config(self, prompt: str, approved: bool = False) -> int:
        cur = self.execute(
            "INSERT INTO tagging_config (system_prompt, user_approved) VALUES (?, ?)",
            (prompt, 1 if approved else 0),
        )
        self.commit()
        return cur.lastrowid

    @_locked
    def get_tagging_config(self) -> dict | None:
        return self.fetchone(
            "SELECT * FROM tagging_config ORDER BY id DESC LIMIT 1"
        )
