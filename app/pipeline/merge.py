"""Merge and export audio chunks into final audiobook files.

Step 15: Concatenate chunks with crossfading, silence padding, and
LUFS normalization. Export as chapters, scenes, or full book.
"""

import wave
import struct
import subprocess
import json
from pathlib import Path
from io import BytesIO

from app.database import ProjectDB
from app.config import AudioSettings


def read_wav_pcm(wav_path: str) -> tuple[bytes, int]:
    """Read a WAV file and return (pcm_bytes, sample_rate)."""
    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1, "Expected mono audio"
        assert wf.getsampwidth() == 2, "Expected 16-bit audio"
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sample_rate


def generate_silence(duration_ms: int, sample_rate: int) -> bytes:
    """Generate silence as PCM bytes."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples  # 16-bit silence


def crossfade_pcm(pcm_a: bytes, pcm_b: bytes, crossfade_ms: int,
                  sample_rate: int) -> bytes:
    """Crossfade between two PCM buffers.

    The last crossfade_ms of pcm_a fades out while the first
    crossfade_ms of pcm_b fades in, overlapping.
    """
    fade_samples = int(sample_rate * crossfade_ms / 1000)

    if fade_samples == 0 or len(pcm_a) < fade_samples * 2 or len(pcm_b) < fade_samples * 2:
        return pcm_a + pcm_b

    # Convert to sample arrays
    fmt = f"<{len(pcm_a)//2}h"
    samples_a = list(struct.unpack(fmt, pcm_a))

    fmt = f"<{len(pcm_b)//2}h"
    samples_b = list(struct.unpack(fmt, pcm_b))

    # The overlap region
    a_tail = samples_a[-fade_samples:]
    b_head = samples_b[:fade_samples]

    crossfaded = []
    for i in range(fade_samples):
        t = i / fade_samples  # 0 → 1
        mixed = int(a_tail[i] * (1 - t) + b_head[i] * t)
        mixed = max(-32768, min(32767, mixed))
        crossfaded.append(mixed)

    # Reconstruct
    result_samples = samples_a[:-fade_samples] + crossfaded + samples_b[fade_samples:]
    return struct.pack(f"<{len(result_samples)}h", *result_samples)


def normalize_lufs(pcm: bytes, sample_rate: int,
                   target_lufs: float = -16.0) -> bytes:
    """Normalize audio to target LUFS using pyloudnorm."""
    try:
        import numpy as np
        import pyloudnorm as pyln

        # Convert PCM to float array
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0

        meter = pyln.Meter(sample_rate)
        current_lufs = meter.integrated_loudness(samples)

        if current_lufs == float("-inf"):
            return pcm  # silence, nothing to normalize

        normalized = pyln.normalize.loudness(samples, current_lufs, target_lufs)

        # Clip and convert back to int16
        normalized = np.clip(normalized, -1.0, 1.0)
        return (normalized * 32767).astype(np.int16).tobytes()

    except ImportError:
        # pyloudnorm not available — skip normalization
        return pcm


def merge_chunks(db: ProjectDB, chunk_ids: list[str],
                 output_path: Path,
                 audio_settings: AudioSettings) -> str:
    """Merge a sequence of chunks into a single WAV file.

    Applies normalization, crossfading, and appropriate silence
    between chunks/scenes/chapters.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = audio_settings.sample_rate

    merged_pcm = b""

    for i, chunk_id in enumerate(chunk_ids):
        chunk = db.get_chunk(chunk_id)
        gen = db.get_latest_generation(chunk_id)

        if not gen or gen["status"] != "ok" or not gen.get("wav_path"):
            continue

        pcm, sr = read_wav_pcm(gen["wav_path"])
        assert sr == sample_rate, f"Sample rate mismatch: {sr} != {sample_rate}"

        # Normalize this chunk
        pcm = normalize_lufs(pcm, sample_rate, audio_settings.target_lufs)

        if not merged_pcm:
            merged_pcm = pcm
        else:
            # Determine silence duration based on break type
            if chunk and chunk.get("chapter_break_after"):
                silence = generate_silence(
                    audio_settings.chapter_break_silence_ms, sample_rate
                )
                merged_pcm += silence + pcm
            elif chunk and chunk.get("scene_break_after"):
                silence = generate_silence(
                    audio_settings.scene_break_silence_ms, sample_rate
                )
                merged_pcm += silence + pcm
            else:
                # Normal chunk boundary — crossfade
                if audio_settings.crossfade_ms > 0:
                    # Add small silence then crossfade
                    silence = generate_silence(
                        audio_settings.chunk_silence_ms, sample_rate
                    )
                    merged_pcm = crossfade_pcm(
                        merged_pcm + silence, pcm,
                        audio_settings.crossfade_ms, sample_rate
                    )
                else:
                    silence = generate_silence(
                        audio_settings.chunk_silence_ms, sample_rate
                    )
                    merged_pcm += silence + pcm

    # Write final WAV
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(merged_pcm)

    return str(output_path)


def export_by_chapter(db: ProjectDB, export_dir: Path,
                      audio_settings: AudioSettings) -> list[str]:
    """Export one file per chapter."""
    export_dir.mkdir(parents=True, exist_ok=True)
    chapters = db.get_chapters()
    exports = []

    for ch in chapters:
        chunks = db.get_chunks(chapter_id=ch["id"])
        if not chunks:
            continue

        chunk_ids = [c["id"] for c in chunks]
        filename = f"ch{ch['idx']:03d}_{_slugify(ch['title'])}.wav"
        output_path = export_dir / filename

        merge_chunks(db, chunk_ids, output_path, audio_settings)
        exports.append(str(output_path))

    return exports


def export_full_book(db: ProjectDB, export_dir: Path,
                     audio_settings: AudioSettings) -> str:
    """Export the entire book as a single file."""
    export_dir.mkdir(parents=True, exist_ok=True)
    chunks = db.get_chunks()
    chunk_ids = [c["id"] for c in chunks]

    book_name = db.get_meta("project_name") or "audiobook"
    output_path = export_dir / f"{_slugify(book_name)}.wav"

    merge_chunks(db, chunk_ids, output_path, audio_settings)

    # Convert to other formats if requested
    if audio_settings.export_format == "mp3":
        mp3_path = output_path.with_suffix(".mp3")
        _convert_with_ffmpeg(str(output_path), str(mp3_path))
        return str(mp3_path)
    elif audio_settings.export_format == "m4b":
        m4b_path = output_path.with_suffix(".m4b")
        _convert_to_m4b(db, str(output_path), str(m4b_path))
        return str(m4b_path)

    return str(output_path)


def _convert_with_ffmpeg(input_path: str, output_path: str):
    """Convert audio using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-q:a", "2", output_path],
        check=True, capture_output=True,
    )


def _convert_to_m4b(db: ProjectDB, wav_path: str, m4b_path: str):
    """Convert to M4B with chapter markers."""
    # First convert to m4a
    m4a_path = m4b_path.replace(".m4b", ".m4a")
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-c:a", "aac", "-b:a", "128k", m4a_path],
        check=True, capture_output=True,
    )
    # Rename to m4b (m4b is just m4a with chapter markers)
    import shutil
    shutil.move(m4a_path, m4b_path)
    # TODO: Add proper chapter markers via ffmpeg metadata


def _slugify(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]
