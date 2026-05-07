"""Merge and export audio chunks into final audiobook files.

Step 15: Concatenate chunks with crossfading, silence padding, and
LUFS normalization. Export as chapters, scenes, or full book.

Chunks have no shared text (the chunker emits clean cuts at paragraph
boundaries), so merging is straightforward concatenation. The exporter
also runs a per-seam anomaly scan (see ``analyze_seam``) and returns the
findings so the UI can flag suspicious boundaries — long silences,
truncated words, sample-discontinuity pops, and unexpected loudness
jumps within a single voice.
"""

import re
import wave
import struct
import subprocess
from pathlib import Path

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


# ─── Seam anomaly analysis ────────────────────────────────────────────

# Silence floor used when measuring trailing/leading quiet regions
# (-46 dBFS — generous enough to ignore room tone but tight enough to
# detect genuine speech).
_SILENCE_THRESHOLD = 0.005

# Window length used to compare loudness on either side of a seam.
_SEAM_WINDOW_SEC = 0.2


def analyze_seam(prev_pcm: bytes, curr_pcm: bytes,
                 sample_rate: int,
                 same_voice: bool = True,
                 prev_chunk_id: str = "",
                 curr_chunk_id: str = "") -> dict:
    """Inspect the boundary between two PCM buffers for likely defects.

    Returns a dict with raw measurements and a ``warnings`` list of
    human-readable strings. Each warning is short and actionable so the
    UI can show a punch-list. Volume-jump checks are skipped when the
    two chunks use different voice overrides (the jump is intentional).
    """
    try:
        import numpy as np
    except ImportError:
        return {"warnings": [], "skipped": "numpy not available"}

    if not prev_pcm or not curr_pcm:
        return {"warnings": []}

    a = np.frombuffer(prev_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    b = np.frombuffer(curr_pcm, dtype=np.int16).astype(np.float32) / 32768.0

    # Trailing silence in prev chunk: distance from last non-silent sample
    # to end of buffer. Long values mean TTS over-padded the chunk.
    nonsilent_a = np.where(np.abs(a) >= _SILENCE_THRESHOLD)[0]
    trailing_silence = (len(a) - (nonsilent_a[-1] + 1)) / sample_rate if len(nonsilent_a) else len(a) / sample_rate

    # Leading silence in curr chunk: distance from start to first
    # non-silent sample.
    nonsilent_b = np.where(np.abs(b) >= _SILENCE_THRESHOLD)[0]
    leading_silence = (nonsilent_b[0] / sample_rate) if len(nonsilent_b) else len(b) / sample_rate

    # Loudness on both sides of the seam (RMS of the last/first 200 ms).
    win = int(_SEAM_WINDOW_SEC * sample_rate)
    prev_tail = a[-win:] if len(a) >= win else a
    curr_head = b[:win] if len(b) >= win else b
    rms_prev = float(np.sqrt(np.mean(prev_tail ** 2))) if len(prev_tail) else 0.0
    rms_curr = float(np.sqrt(np.mean(curr_head ** 2))) if len(curr_head) else 0.0

    # Sample-value discontinuity right at the join. A large jump tends to
    # produce an audible click/pop.
    last_a = float(a[-1]) if len(a) else 0.0
    first_b = float(b[0]) if len(b) else 0.0
    discontinuity = abs(last_a - first_b)

    warnings: list[str] = []
    if trailing_silence > 1.5:
        warnings.append(
            f"Long trailing silence ({trailing_silence:.2f}s) in {prev_chunk_id} — TTS over-padded"
        )
    if leading_silence > 1.5:
        warnings.append(
            f"Long leading silence ({leading_silence:.2f}s) in {curr_chunk_id} — TTS over-padded"
        )
    # Truncation detection: chunk ends with energy still high and almost
    # no silence buffer.
    if trailing_silence < 0.05 and rms_prev > 0.08:
        warnings.append(
            f"Possible truncation at end of {prev_chunk_id} — no trailing silence and signal still hot"
        )
    if leading_silence < 0.05 and rms_curr > 0.08:
        warnings.append(
            f"Abrupt start in {curr_chunk_id} — verify the first word isn't clipped"
        )
    # Volume mismatch: only meaningful when both sides have signal AND
    # they share the same voice (different voices legitimately differ).
    if same_voice and rms_prev > 0.01 and rms_curr > 0.01:
        ratio = max(rms_prev, rms_curr) / min(rms_prev, rms_curr)
        if ratio > 2.0:
            db = 20.0 * float(np.log10(ratio))
            warnings.append(
                f"Loudness jump {db:.1f}dB at {prev_chunk_id} → {curr_chunk_id}"
            )
    # Sample-jump pop heuristic.
    if discontinuity > 0.25:
        warnings.append(
            f"Sample discontinuity {discontinuity:.2f} at {prev_chunk_id} → {curr_chunk_id} (potential pop)"
        )

    return {
        "warnings": warnings,
        "trailing_silence_sec": round(trailing_silence, 3),
        "leading_silence_sec": round(leading_silence, 3),
        "rms_prev_tail": round(rms_prev, 4),
        "rms_curr_head": round(rms_curr, 4),
        "discontinuity": round(discontinuity, 4),
    }


def merge_chunks(db: ProjectDB, chunk_ids: list[str],
                 output_path: Path,
                 audio_settings: AudioSettings) -> tuple[str, list[dict]]:
    """Merge a sequence of chunks into a single WAV file.

    Returns ``(output_path, seam_reports)``. Each seam report includes
    the two chunk IDs around it, the boundary type (``"chunk"`` /
    ``"scene"`` / ``"chapter"``), the analysis output from
    ``analyze_seam`` for raw chunk-vs-chunk boundaries, and any human
    warnings.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = audio_settings.sample_rate

    merged_pcm = b""
    prev_chunk = None
    prev_chunk_pcm = b""
    seam_reports: list[dict] = []

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
            # Decide silence/transition strategy by break type, and
            # capture an analysis of the boundary while we still have
            # the un-faded prev/curr buffers in hand.
            if prev_chunk and prev_chunk.get("chapter_break_after"):
                boundary = "chapter"
            elif prev_chunk and prev_chunk.get("scene_break_after"):
                boundary = "scene"
            else:
                boundary = "chunk"

            same_voice = (
                prev_chunk is not None and chunk is not None
                and (prev_chunk.get("voice") or "") == (chunk.get("voice") or "")
            )
            analysis = analyze_seam(
                prev_chunk_pcm, pcm, sample_rate,
                same_voice=same_voice,
                prev_chunk_id=prev_chunk["id"] if prev_chunk else "",
                curr_chunk_id=chunk_id,
            )
            seam_reports.append({
                "boundary": boundary,
                "prev_chunk_id": prev_chunk["id"] if prev_chunk else "",
                "curr_chunk_id": chunk_id,
                "same_voice": same_voice,
                **analysis,
            })

            if boundary == "chapter":
                silence = generate_silence(audio_settings.chapter_break_silence_ms, sample_rate)
                merged_pcm += silence + pcm
            elif boundary == "scene":
                silence = generate_silence(audio_settings.scene_break_silence_ms, sample_rate)
                merged_pcm += silence + pcm
            else:
                # Plain chunk-to-chunk join: small silence + optional
                # crossfade. With clean (no-overlap) chunks this is just
                # a transition smoother, not a dedup mechanism.
                if audio_settings.crossfade_ms > 0:
                    silence = generate_silence(audio_settings.chunk_silence_ms, sample_rate)
                    merged_pcm = crossfade_pcm(
                        merged_pcm + silence, pcm,
                        audio_settings.crossfade_ms, sample_rate,
                    )
                else:
                    silence = generate_silence(audio_settings.chunk_silence_ms, sample_rate)
                    merged_pcm += silence + pcm

        prev_chunk = chunk
        prev_chunk_pcm = pcm

    # Write final WAV
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(merged_pcm)

    return str(output_path), seam_reports


def _summarize_seams(seam_reports: list[dict]) -> dict:
    """Collapse a flat list of seam reports into UI-friendly counts."""
    flagged = [s for s in seam_reports if s.get("warnings")]
    return {
        "total_seams": len(seam_reports),
        "flagged_seams": len(flagged),
        "flagged": flagged,
    }


def export_by_chapter(db: ProjectDB, export_dir: Path,
                      audio_settings: AudioSettings) -> dict:
    """Export one file per chapter. Returns paths plus seam summary."""
    export_dir.mkdir(parents=True, exist_ok=True)
    chapters = db.get_chapters()
    files: list[str] = []
    all_seams: list[dict] = []

    for ch in chapters:
        chunks = db.get_chunks(chapter_id=ch["id"])
        if not chunks:
            continue

        chunk_ids = [c["id"] for c in chunks]
        filename = f"ch{ch['idx']:03d}_{_slugify(ch['title'])}.wav"
        output_path = export_dir / filename

        path, seam_reports = merge_chunks(db, chunk_ids, output_path, audio_settings)
        files.append(path)
        all_seams.extend(seam_reports)

    return {"files": files, **_summarize_seams(all_seams)}


def export_full_book(db: ProjectDB, export_dir: Path,
                     audio_settings: AudioSettings) -> dict:
    """Export the entire book as a single file. Returns path + seam summary."""
    export_dir.mkdir(parents=True, exist_ok=True)
    chunks = db.get_chunks()
    chunk_ids = [c["id"] for c in chunks]

    book_name = db.get_meta("project_name") or "audiobook"
    output_path = export_dir / f"{_slugify(book_name)}.wav"

    path, seam_reports = merge_chunks(db, chunk_ids, output_path, audio_settings)

    # Convert to other formats if requested
    if audio_settings.export_format == "mp3":
        mp3_path = output_path.with_suffix(".mp3")
        _convert_with_ffmpeg(str(output_path), str(mp3_path))
        path = str(mp3_path)
    elif audio_settings.export_format == "m4b":
        m4b_path = output_path.with_suffix(".m4b")
        _convert_to_m4b(db, str(output_path), str(m4b_path))
        path = str(m4b_path)

    return {"file": path, **_summarize_seams(seam_reports)}


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
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]
