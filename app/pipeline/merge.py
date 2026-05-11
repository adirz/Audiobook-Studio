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


def _apply_fades(pcm: bytes, sample_rate: int,
                 fade_in_ms: int, fade_out_ms: int) -> bytes:
    """Apply equal-power (cos/sin) fade-in and/or fade-out to a chunk.

    Replaces the old "crossfade through silence" hack: the previous code
    appended silence to the running buffer and then crossfaded the next
    chunk's head against that silence, which is just a fade-in. Doing the
    fade per chunk lets us stream the output to disk and produces clean,
    click-free joins at boundaries.
    """
    if fade_in_ms <= 0 and fade_out_ms <= 0:
        return pcm
    try:
        import numpy as np
    except ImportError:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n = len(samples)
    if n == 0:
        return pcm

    if fade_in_ms > 0:
        k = min(int(sample_rate * fade_in_ms / 1000), n)
        if k > 0:
            ramp = np.sin(np.linspace(0.0, np.pi / 2.0, k, dtype=np.float32))
            samples[:k] *= ramp
    if fade_out_ms > 0:
        k = min(int(sample_rate * fade_out_ms / 1000), n)
        if k > 0:
            ramp = np.cos(np.linspace(0.0, np.pi / 2.0, k, dtype=np.float32))
            samples[-k:] *= ramp

    np.clip(samples, -32768.0, 32767.0, out=samples)
    return samples.astype(np.int16).tobytes()


def _measure_chunk_lufs(pcm: bytes, sample_rate: int) -> float | None:
    """Integrated LUFS for one chunk, or None when measurement is unavailable."""
    try:
        import numpy as np
        import pyloudnorm as pyln
    except ImportError:
        return None
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
    if samples.size == 0:
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            lufs = pyln.Meter(sample_rate).integrated_loudness(samples)
    except Exception:
        return None
    if lufs == float("-inf") or lufs != lufs:  # also catches NaN
        return None
    return float(lufs)


def _book_gain_db(measurements: list[tuple[float | None, float]],
                  target_lufs: float) -> float:
    """Energy-weighted mean LUFS across chunks → global gain in dB.

    Per-chunk normalization (the previous behavior) flattens dynamic
    range — quiet passages get amplified to match loud ones, and any
    breath / room tone in soft chunks comes up with the signal. Computing
    one gain from a duration-weighted energy mean preserves the relative
    loudness between chunks while still hitting the target on average.
    """
    import math
    total_energy = 0.0
    total_duration = 0.0
    for lufs, duration in measurements:
        if lufs is None or duration <= 0:
            continue
        total_energy += (10.0 ** (lufs / 10.0)) * duration
        total_duration += duration
    if total_duration <= 0 or total_energy <= 0:
        return 0.0
    book_lufs = 10.0 * math.log10(total_energy / total_duration)
    return target_lufs - book_lufs


def _apply_gain_db(pcm: bytes, gain_db: float) -> bytes:
    """Apply a constant gain (in dB) to int16 PCM, clipping at int16 range."""
    if abs(gain_db) < 0.01:
        return pcm
    try:
        import numpy as np
    except ImportError:
        return pcm
    scale = 10.0 ** (gain_db / 20.0)
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * scale
    np.clip(samples, -32768.0, 32767.0, out=samples)
    return samples.astype(np.int16).tobytes()


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
                 audio_settings: AudioSettings,
                 progress_callback=None) -> tuple[str, list[dict]]:
    """Merge a sequence of chunks into a single WAV file.

    Returns ``(output_path, seam_reports)``. Each seam report includes
    the two chunk IDs around it, the boundary type (``"chunk"`` /
    ``"scene"`` / ``"chapter"``), the analysis output from
    ``analyze_seam`` for raw chunk-vs-chunk boundaries, and any human
    warnings.

    ``progress_callback(current, total)`` is called after each chunk is
    processed (including skipped ones) so the caller can track progress.

    Two passes:
      1. Read every chunk, measure integrated LUFS + duration, then
         compute one global gain. This avoids the per-chunk normalization
         that flattened dynamic range.
      2. Read every chunk again, apply the gain + small fades, and stream
         frames straight into the output WAV. Memory stays O(one chunk)
         instead of growing with the whole book.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = audio_settings.sample_rate

    # Resolve which chunks have usable audio. Store everything we'll need
    # again in pass 2 so we don't have to re-query the DB per chunk.
    valid: list[tuple[str, dict, str]] = []  # (chunk_id, chunk_row, wav_path)
    for chunk_id in chunk_ids:
        chunk = db.get_chunk(chunk_id)
        gen = db.get_latest_generation(chunk_id)
        if not chunk or not gen or gen.get("status") != "ok" or not gen.get("wav_path"):
            continue
        valid.append((chunk_id, chunk, gen["wav_path"]))

    total = len(chunk_ids)

    # Pass 1 — measure each chunk's integrated loudness for the global gain.
    measurements: list[tuple[float | None, float]] = []
    for _, _, wav_path in valid:
        pcm, sr = read_wav_pcm(wav_path)
        if sr != sample_rate:
            raise ValueError(f"Sample rate mismatch in {wav_path}: {sr} != {sample_rate}")
        duration = len(pcm) / (2 * sample_rate)
        measurements.append((_measure_chunk_lufs(pcm, sample_rate), duration))

    gain_db = _book_gain_db(measurements, audio_settings.target_lufs)

    # Pass 2 — stream-write the output.
    seam_reports: list[dict] = []
    prev_chunk: dict | None = None
    prev_chunk_id = ""
    prev_tail = b""  # last ~200 ms of previous chunk for seam analysis
    seam_window_bytes = int(_SEAM_WINDOW_SEC * sample_rate * 2)
    cf_ms = audio_settings.crossfade_ms
    valid_set = {cid for cid, _, _ in valid}

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        idx = 0  # tracks position within ``chunk_ids`` for progress
        for chunk_id, chunk, wav_path in valid:
            # Advance the progress index past any skipped chunk_ids that
            # came before this valid one.
            while idx < len(chunk_ids) and chunk_ids[idx] not in valid_set:
                idx += 1
                if progress_callback:
                    try:
                        progress_callback(idx, total)
                    except Exception:
                        pass
            idx += 1

            pcm, _ = read_wav_pcm(wav_path)
            pcm = _apply_gain_db(pcm, gain_db)
            # Per-chunk fades in/out so every join is click-free without
            # accumulating the whole book in RAM to do a true crossfade.
            pcm = _apply_fades(pcm, sample_rate, cf_ms, cf_ms)

            if prev_chunk is not None:
                if prev_chunk.get("chapter_break_after"):
                    boundary = "chapter"
                    silence_ms = audio_settings.chapter_break_silence_ms
                elif prev_chunk.get("scene_break_after"):
                    boundary = "scene"
                    silence_ms = audio_settings.scene_break_silence_ms
                else:
                    boundary = "chunk"
                    silence_ms = audio_settings.chunk_silence_ms

                same_voice = (prev_chunk.get("voice") or "") == (chunk.get("voice") or "")
                head = pcm[:seam_window_bytes] if seam_window_bytes else pcm
                analysis = analyze_seam(
                    prev_tail, head, sample_rate,
                    same_voice=same_voice,
                    prev_chunk_id=prev_chunk_id,
                    curr_chunk_id=chunk_id,
                )
                seam_reports.append({
                    "boundary": boundary,
                    "prev_chunk_id": prev_chunk_id,
                    "curr_chunk_id": chunk_id,
                    "same_voice": same_voice,
                    **analysis,
                })

                if silence_ms > 0:
                    wf.writeframes(generate_silence(silence_ms, sample_rate))

            wf.writeframes(pcm)

            prev_tail = pcm[-seam_window_bytes:] if seam_window_bytes and len(pcm) >= seam_window_bytes else pcm
            prev_chunk = chunk
            prev_chunk_id = chunk_id

            if progress_callback:
                try:
                    progress_callback(idx, total)
                except Exception:
                    pass

        # Flush progress for any trailing skipped chunk_ids.
        while idx < total:
            idx += 1
            if progress_callback:
                try:
                    progress_callback(idx, total)
                except Exception:
                    pass

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
                      audio_settings: AudioSettings,
                      progress_callback=None) -> dict:
    """Export one file per chapter. Returns paths plus seam summary."""
    export_dir.mkdir(parents=True, exist_ok=True)
    chapters = db.get_chapters()

    # Pre-compute per-chapter chunk ID lists so we know the grand total.
    chapter_data = []
    grand_total = 0
    for ch in chapters:
        chunks = db.get_chunks(chapter_id=ch["id"])
        if not chunks:
            continue
        cids = [c["id"] for c in chunks]
        chapter_data.append((ch, cids))
        grand_total += len(cids)

    files: list[str] = []
    all_seams: list[dict] = []
    offset = 0

    for ch, cids in chapter_data:
        ch_offset = offset

        def _make_cb(o):
            def _cb(current, _total):
                if progress_callback:
                    progress_callback(o + current, grand_total)
            return _cb

        filename = f"ch{ch['idx']:03d}_{_slugify(ch['title'])}.wav"
        output_path = export_dir / filename
        path, seam_reports = merge_chunks(
            db, cids, output_path, audio_settings, _make_cb(ch_offset)
        )
        files.append(path)
        all_seams.extend(seam_reports)
        offset += len(cids)

    return {"files": files, **_summarize_seams(all_seams)}


def export_full_book(db: ProjectDB, export_dir: Path,
                     audio_settings: AudioSettings,
                     progress_callback=None) -> dict:
    """Export the entire book as a single file. Returns path + seam summary."""
    export_dir.mkdir(parents=True, exist_ok=True)
    chunks = db.get_chunks()
    chunk_ids = [c["id"] for c in chunks]

    book_name = db.get_meta("project_name") or "audiobook"
    output_path = export_dir / f"{_slugify(book_name)}.wav"

    path, seam_reports = merge_chunks(
        db, chunk_ids, output_path, audio_settings, progress_callback
    )

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
