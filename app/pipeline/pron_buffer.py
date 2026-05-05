"""Pronunciation test clip pre-generation buffer.

Runs in a background thread, pre-generates N pronunciation test clips
ahead of the user's current position so playback is instant.
"""

import asyncio
import threading
import queue
import json
import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from app.handlers.tts_base import TTSHandler
from app.pipeline.pronunciation import generate_pron_test


class Priority(Enum):
    URGENT = 0   # User-requested regeneration
    NORMAL = 1   # Pre-generation buffer
    LOW = 2      # Background fill


@dataclass
class PronJob:
    entry_id: int
    word: str
    phonetic: str
    context: str
    voice: str
    output_path: Path
    priority: Priority = Priority.NORMAL
    callback: object = None   # callable(result_path) or None


class PronBuffer:
    """Async buffer that pre-generates pronunciation test clips.

    Usage:
        buffer = PronBuffer(tts_handler, project_dir, voice)
        buffer.start()

        # Queue a user-triggered test (urgent)
        buffer.request(entry_id, word, phonetic, context, callback=on_ready)

        # Queue background pre-generation
        buffer.prefill(entries_ahead)

        buffer.stop()
    """

    def __init__(self, tts: TTSHandler, clips_dir: Path, voice: str,
                 buffer_size: int = 5):
        self.tts = tts
        self.clips_dir = clips_dir
        self.voice = voice
        self.buffer_size = buffer_size

        self._queue = queue.PriorityQueue()
        self._thread = None
        self._stop_event = threading.Event()
        self._results: dict[int, str] = {}  # entry_id -> audio_path
        self._generating: set[int] = set()
        self._queued: set[int] = set()
        self._pending_callbacks: dict[int, list] = {}
        self._lock = threading.Lock()
        self._counter = 0  # tie-breaker for priority queue

    def start(self):
        """Start the background generation thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def request(self, entry_id: int, word: str, phonetic: str,
                context: str, attempt_num: int = 1,
                callback=None):
        """Queue an urgent generation (user clicked Play)."""
        output_path = self.clips_dir / f"pron_{entry_id}_v{attempt_num:02d}.wav"

        with self._lock:
            # If already generated, return immediately
            if entry_id in self._results:
                if callback:
                    try:
                        callback(self._results[entry_id])
                    except Exception:
                        pass
                return self._results[entry_id]

            # If already queued or generating, attach callback and return
            if entry_id in self._queued or entry_id in self._generating:
                if callback:
                    self._pending_callbacks.setdefault(entry_id, []).append(callback)
                return None

            # Otherwise queue an urgent job and mark queued
            job = PronJob(
                entry_id=entry_id,
                word=word,
                phonetic=phonetic,
                context=context,
                voice=self.voice,
                output_path=output_path,
                priority=Priority.URGENT,
                callback=callback,
            )
            self._counter += 1
            self._queued.add(entry_id)
            self._queue.put((job.priority.value, self._counter, job))
            return None  # will be ready async

    def prefill(self, entries: list[dict], attempt_num: int = 1):
        """Pre-generate clips for upcoming entries."""
        for entry in entries[:self.buffer_size]:
            eid = entry.get('id', 0)
            with self._lock:
                if eid in self._results or eid in self._generating or eid in self._queued:
                    continue

            word = entry.get('word', '')
            phonetic = entry.get('phonetic') or word
            context = entry.get('example_context', f'The word {word} appears here.')

            import re
            test_context = re.sub(
                rf"\b{re.escape(word)}\b",
                phonetic,
                context,
                flags=re.IGNORECASE,
            )

            # Prefill files use a special suffix so they can be promoted to
            # a numbered attempt when the user actually chooses to test.
            output_path = self.clips_dir / f"pron_{eid}_prefill.wav"

            job = PronJob(
                entry_id=eid,
                word=word,
                phonetic=phonetic,
                context=test_context,
                voice=self.voice,
                output_path=output_path,
                priority=Priority.NORMAL,
            )
            with self._lock:
                self._queued.add(eid)
                self._counter += 1
                self._queue.put((job.priority.value, self._counter, job))

    def get_result(self, entry_id: int) -> str | None:
        """Check if a clip has been generated for this entry."""
        # Check in-memory results first
        with self._lock:
            res = self._results.get(entry_id)
        if res:
            return res
        # Also check prefill file on disk
        pref = self.clips_dir / f"pron_{entry_id}_prefill.wav"
        if pref.exists():
            return str(pref)
        return None

    def is_ready(self, entry_id: int) -> bool:
        return entry_id in self._results

    def clear(self):
        """Clear the queue and results."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._results.clear()
        self._generating.clear()

    def _worker(self):
        """Background thread that processes the generation queue."""
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        while not self._stop_event.is_set():
            try:
                priority, counter, job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._stop_event.is_set():
                break
            with self._lock:
                # mark as no longer queued and now generating
                self._queued.discard(job.entry_id)
                self._generating.add(job.entry_id)

            try:
                result_path = generate_pron_test(
                    self.tts,
                    job.context,
                    job.voice,
                    job.output_path,
                )

                with self._lock:
                    self._results[job.entry_id] = result_path

                # If this was a prefill file, write a small metadata JSON so
                # callers can determine what phonetic was used when it was
                # generated. This allows the API to decide whether promoting
                # a prefill to a numbered attempt is appropriate.
                try:
                    outp = Path(result_path)
                    if outp.name.endswith('_prefill.wav'):
                        meta = {
                            'phonetic_used': job.phonetic,
                            'context': job.context,
                        }
                        meta_path = outp.with_suffix('.json')
                        # Write atomically
                        tmp = meta_path.with_suffix('.json.tmp')
                        with open(tmp, 'w', encoding='utf-8') as mf:
                            json.dump(meta, mf)
                        os.replace(str(tmp), str(meta_path))
                except Exception:
                    pass

                # Call any pending callbacks registered while queued
                cbs = []
                with self._lock:
                    cbs = self._pending_callbacks.pop(job.entry_id, [])

                for cb in cbs:
                    try:
                        cb(result_path)
                    except Exception:
                        pass

                if job.callback:
                    try:
                        job.callback(result_path)
                    except Exception:
                        pass

            except Exception as e:
                print(f"PronBuffer: failed to generate for entry {job.entry_id}: {e}")

            finally:
                with self._lock:
                    self._generating.discard(job.entry_id)


# Singleton buffer per project
_buffers: dict[str, PronBuffer] = {}


def get_or_create_buffer(slug: str, tts: TTSHandler,
                         clips_dir: Path, voice: str,
                         buffer_size: int = 5) -> PronBuffer:
    """Get or create a pronunciation buffer for a project."""
    if slug not in _buffers:
        _buffers[slug] = PronBuffer(tts, clips_dir, voice, buffer_size)
        _buffers[slug].start()
    return _buffers[slug]


def stop_buffer(slug: str):
    if slug in _buffers:
        _buffers[slug].stop()
        del _buffers[slug]
