# Audiobook Studio

Turn a manuscript into an audiobook using local TTS. Pipeline: extract → clean → chunk → pronunciation testing → (optional) emotion tagging → generation → QA → review → export.

## Quick start

```bash
tar xzf audiobook-studio.tar.gz
cd audiobook-studio
chmod +x run.sh
./run.sh
```

Opens at `http://localhost:8899`. The script creates a `.venv/` on first run and installs everything into it — your system Python is untouched.

## First-run setup

1. Download the Orpheus model somewhere on your disk. Keep it wherever it currently lives.
2. Open the app, click ⚙ Settings on the home page.
3. Set **TTS Engine → Model directory** to the **absolute path** to that folder (e.g. `/home/you/models/orpheus-3b`). Don't use relative paths — vLLM interprets them as HuggingFace repo IDs.
4. If you have a patched local copy of `orpheus_tts_pypi`, set **Orpheus pypi path** to its absolute path too. Otherwise leave default.
5. Save. The setting is global — all projects share it.

## Workflow

Each project is a folder in `projects/` containing one SQLite database (`project.db`) plus `source/`, `audio/`, `test_clips/`, `export/` subdirectories. You can create new projects, open existing ones, or delete projects from the home page at any time.

The sidebar lists every pipeline step. You can jump around freely; each step reads its state from the DB. Nothing is lost between sessions.

### Key steps

- **Select Range** — click the heading where narration should start, click again for the end.
- **Review Symbols** — tick which non-standard symbols are scene breaks. Your choices persist if you leave and come back.
- **Scan Words** — finds words not in the English dictionary. Each word gets a ✕ to remove it (mark as standard). **↻ Rescan** re-runs detection while keeping your approved entries.
- **Pronunciation** — for each word, hear it in its real book context, adjust the phonetic spelling, re-test until right. Suggestion chips auto-apply common substitutions. **View / Edit Full Dictionary** shows all entries with click-to-edit phonetic values.
- **Emotion Tags** — (optional, needs LLM) Chat with the LLM to tune the tagging prompt. Test on a few chunks, approve, apply to all.
- **Generate Audio** — batch generation with auto-retry (each retry varies params).
- **QA Check** — Whisper transcribes each chunk, compares against original, produces word-level diffs. Pronunciation substitutions are marked as expected and don't count against the score.
- **Review & Fix** — continuous player with auto-advance across chunks, respecting silence gaps for scene and chapter breaks. Chapter nav sidebar, filter buttons (all/flagged/QA fail/no audio), per-chunk flag and override modals. Keyboard: Space play/pause, ←/→ chunk, [/] scene.
- **Export** — merge with LUFS normalization, crossfading, silence gaps. Full book or per-chapter.

## Resetting things

- **Single word**: click ✕ in the word table or pronunciation dictionary view.
- **All pending pronunciation**: click **↻ Rescan** in the Scan Words step.
- **Full project**: delete the project from the home page (wipes the folder).
- **Global settings**: delete `settings.json` at the app root and restart.
