# Audiobook Studio — Project Context

Use this document as the seed for new Claude conversations about this project. It replaces needing to scroll through prior conversation history.

---

## What this project is

A **local-first desktop web app** that turns a `.docx` manuscript into an audiobook using local TTS (currently Orpheus via vLLM). The user wanted it to have a **comfortable GUI** and to handle the full workflow end-to-end: installation, text extraction, cleanup, pronunciation dictionary, optional emotion tagging, batch audio generation with retries, STT-based QA, manual review, and export.

The existing starting point was a set of standalone Python scripts that operated on a single hardcoded book (`Dealing in Shadows.docx`). The user wanted to transform them into a proper application with:

- Multi-project support (can work on several books, save and return to any step)
- Pluggable engine handlers (so other TTS/STT/LLM engines can be added later)
- Real settings UI (hardware limits, model paths, auth tokens, etc.)
- A pronunciation workflow where words are tested in their actual book context, with phonetic suggestions

---

## Decisions made during design

- **Stack**: FastAPI backend + vanilla HTML/CSS/JS frontend (no npm, no build step). The user dislikes JS toolchains breaking — this avoids all of that while still giving the reactive UI that audio playback + highlighted text needs.
- **State model**: One SQLite database per project (in `projects/<slug>/project.db`). Every step writes to the DB, resuming = reading current state. No scattered JSON pipeline files.
- **Engine abstraction**: All TTS/STT/LLM/extractor engines go through abstract handler interfaces. The app queries handlers for capabilities (voices, tags, IPA support, etc.) and adapts the UI. Only Orpheus/Whisper/docx are implemented but adding another is a subclass of the base handler.
- **Pronunciation context**: Tests use real text from the book (not invented sentences). The word is found in its actual passage, ±120 chars of surrounding context, with the word highlighted.
- **Chunk boundaries**: Chunks never cross chapters or scenes. Dialogue-aware splitting.
- **QA scoring**: Word-level diff with `SequenceMatcher`. Pronunciation substitutions are marked `pron_expected` and don't count against the score.
- **LLM tagging (optional)**: Chat interface where user refines the prompt with natural language, test on a few chunks, apply to all.
- **Multi-voice**: Not implemented. Single narrator per project for now. Design note: would need per-character voice assignment, probably detectable via quoted dialogue attribution.
- **Resource management**: Not implemented. Models stay loaded once loaded. Most steps are set-it-and-forget-it, so serial use is fine. An async buffer exists for pronunciation test clips (pre-generates N ahead while user is listening).

---

## Workflow (11 steps)

1. **Upload** — drag-drop .docx
2. **Select Range** — click heading for start, click again for end
3. **Extract** — parse into chapters, preserve italics as `*text*`
4. **Review Symbols** — user marks which non-standard chars are scene breaks
5. **Clean & Chunk** — normalize dashes/ellipses, split into ~150-word chunks respecting sentence/dialogue/scene/chapter boundaries
6. **Scan Words** — find non-standard words using lemmatization (nltk WordNetLemmatizer tries all POS), contraction decomposition, possessive stripping, variant grouping
7. **Pronunciation** — for each word, hear it in-context, adjust phonetic spelling with clickable suggestion chips, re-test. Async buffer pre-generates upcoming test clips.
8. **Standard-word overrides** (part of step 6 UI) — force standard words to be pronounced differently
9. **Emotion Tags** (optional, needs LLM) — chat with LLM to refine prompt, test on samples, apply to all chunks
10. **Generate Audio** — batch TTS with retry-on-failure (varies temperature/repetition_penalty per retry)
11. **QA Check** — Whisper transcribes, word-level diff produced, pronunciation-aware scoring
12. **Review & Fix** — continuous player across chunks with chapter nav, filter buttons, flag modal, location pronunciation override, per-chunk regenerate
13. **Export** — merge with LUFS normalization, crossfading, silence gaps; full book or per-chapter as WAV/MP3/M4B

---

## File structure

```
audiobook-studio/
├── run.sh                       ← creates .venv, installs deps, runs server
├── requirements.txt
├── README.md
├── settings.json                ← global settings (created on first run)
├── projects/
│   └── <slug>/
│       ├── project.db           ← SQLite per project, all state
│       ├── source/              ← uploaded manuscript
│       ├── audio/               ← generated chunk WAVs
│       ├── test_clips/          ← pronunciation test WAVs
│       └── export/              ← final merged audio
└── app/
    ├── main.py                  ← FastAPI entry
    ├── config.py                ← AppSettings dataclass, JSON persistence
    ├── database.py              ← SQLite schema (10 tables) + ProjectDB helper
    ├── models.py                ← Pydantic request/response models
    ├── handlers/
    │   ├── tts_base.py          ← abstract TTS interface
    │   ├── tts_orpheus.py       ← Orpheus implementation
    │   ├── stt_base.py          ← abstract STT
    │   ├── stt_whisper.py       ← Whisper implementation
    │   ├── llm_base.py          ← abstract LLM + NoLLM placeholder
    │   ├── extractor.py         ← docx extractor with heading scan
    │   └── registry.py          ← engine singleton manager
    ├── pipeline/
    │   ├── analyze.py           ← symbol detection + non-standard word scanning
    │   ├── clean.py             ← text normalization, scene break markers
    │   ├── chunk.py             ← chapter-respecting chunker
    │   ├── pronunciation.py     ← phonetic rules, application, test generation
    │   ├── pron_buffer.py       ← async queue for pre-generating test clips
    │   ├── tagging.py           ← LLM emotion tagging
    │   ├── generate.py          ← batch generation with retry
    │   ├── qa.py                ← STT comparison, word-level diff
    │   └── merge.py             ← LUFS normalize, crossfade, export
    ├── api/
    │   ├── projects.py          ← project CRUD + upload
    │   ├── pipeline.py          ← all step endpoints + background tasks
    │   ├── audio.py             ← audio serving, review, flagging
    │   └── settings.py          ← engine config + capabilities
    └── static/
        ├── index.html           ← project picker
        ├── workspace.html       ← step-driven workspace (1 big file)
        ├── css/main.css         ← warm bookish dark theme
        └── js/
            ├── api.js           ← API client, toast, audio controller, diff render
            └── player.js        ← continuous playback controller
```

---

## Database schema (all in `projects/<slug>/project.db`)

| Table | Purpose |
|-------|---------|
| `project_meta` | key-value: project name, current_step, source file, etc. |
| `chapters` | id, idx, title, raw_text, cleaned_text, status |
| `symbols` | detected non-standard chars + scene-break decisions |
| `chunks` | id (like `ch001_chunk0003`), text, tagged_text, pron_text, break flags |
| `pron_entries` | word, phonetic (null = sounds fine), status, frequency, example context |
| `pron_location_overrides` | per-chunk pronunciation overrides |
| `pron_attempts` | each test clip attempt with audio path |
| `generations` | each TTS attempt with params, wav path, status |
| `qa_results` | transcribed text, score, word_diff_json, pass/fail |
| `user_flags` | review-step flags (garbled, missing, etc.) |
| `tagging_config` | saved LLM prompt for emotion tagging |

---

## Key engineering details

- **Keyboard shortcuts (review step)**: Space play/pause, ←/→ chunk, [/] scene
- **Audio playback**: Each chunk is a separate WAV served by the backend; the continuous player preloads next chunk and inserts silence gaps based on `scene_break_after`/`chapter_break_after` flags
- **Pronunciation buffer**: Priority queue in a background thread. User clicks = URGENT, pre-fill = NORMAL. Results cached by entry_id.
- **Word detection**: Uses nltk `WordNetLemmatizer` trying all POS tags (v/n/a/r) to catch irregular forms. Plus contraction decomposition, possessive stripping, variant deduplication.
- **Scene break regex**: Matches lines with repeated symbols (`♦ ♦ ♦`, `* * *`, `---`, `⁂`). User confirms which symbols mean scene break.
- **Apostrophe handling**: Word regex includes U+2019 (right single quote, used by Word for contractions) so "didn't" stays as one word.
- **Orpheus path resolution**: Absolute-path resolution with `os.path.expanduser` + `os.path.abspath` to avoid vLLM treating `./orpheus-model` as a HuggingFace repo ID.

---

## What's tested vs. untested

**Manually tested by user (bugs reported and fixed):**
- Project creation, docx upload
- Heading scan + range selection
- Symbol detection + scene break marking (bug: state wasn't persisting — fixed)
- Clean + chunk
- Non-standard word scanning (bug: many false positives, contractions split — fixed)

**Not tested yet** (Orpheus model path issue blocked testing past word scanning):
- Pronunciation test audio playback (needs working TTS)
- Emotion tagging (needs LLM configured)
- Batch audio generation
- Whisper QA
- Continuous player in review
- Export merge

**Likely subtle bugs** (based on code review, not live testing):
- M4B chapter markers are TODO (export still produces a valid M4B, just without navigable chapter marks)
- Multi-voice support not implemented
- Pronunciation buffer may have thread-safety issues around `_results` dict (no lock)
- Generate retry parameter variation is simple (temp +0.1, rep_penalty +0.1) — may not help enough for genuinely problematic chunks

---

## Important caveats about Orpheus

1. **Model path must be absolute**. vLLM interprets relative paths like `./orpheus-model` as HuggingFace repo IDs and fails with "Repo id must use alphanumeric chars..."
2. The user has a **patched local copy** of `orpheus_tts_pypi` with `max_model_len=8196` added to `_setup_engine`. They set `orpheus_pypi_path` in settings to point to it.
3. Default config expanded from `./orpheus-model` → `~/orpheus-model` and the load-time auto-migration catches the old stale value in existing `settings.json`.

---

## Known places for continued work

1. **End-to-end test with real Orpheus**: user was stuck at the model path issue. After fixing, they need to retest from pronunciation onwards.
2. **Tagging LLM connector**: only `NoLLMHandler` exists. An actual LLM connector (Anthropic, OpenAI, or local) needs to be added to `app/handlers/`. The abstract `LLMHandler` interface is already defined.
3. **Multi-voice narration**: design work needed. Probably: per-character voice assignment, dialogue-attribution auto-detection, voice field on chunks or sentences.
4. **M4B chapter markers**: add ffmpeg metadata file generation in `merge.py::_convert_to_m4b`.
5. **Resource manager**: currently models stay loaded. A manager that unloads the TTS before loading STT (when VRAM-constrained) would help low-VRAM users.
6. **Continuous playback — word-level highlighting during playback**: the player auto-scrolls between chunks but doesn't highlight individual words. Would need per-word timing from TTS or forced alignment.

---

## If starting a new conversation

Paste this summary plus the current archive. Key things for Claude to know:

- The code is in `/home/claude/audiobook-studio/` when unpacked
- All pipeline endpoints use Pydantic body models (PronUpdate, TaggingChatRequest, etc.)
- Workspace UI is a single HTML file with inline `<script>` — large, but intentional to avoid JS build tooling
- The user wants **warm bookish UI**, not generic SaaS aesthetic. CSS uses Crimson Pro + DM Sans with an amber accent on deep warm dark.
- The user prefers Python over JS for any new work. JS should be minimal, no frameworks, no npm.
- Don't refactor across files unless necessary. The user has been reviewing changes carefully.

---

## Quick resume checklist

```bash
tar xzf audiobook-studio.tar.gz
cd audiobook-studio
chmod +x run.sh
./run.sh
# → open http://localhost:8899
# → Settings → set Model directory to absolute path
# → (optional) set Orpheus pypi path
```

If something broke across runs: delete `settings.json` and restart. If a project is broken: delete it from the home page.
