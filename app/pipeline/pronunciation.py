"""Pronunciation handling: build dictionary, suggest phonetics, apply substitutions.

Step 7-9: The core pronunciation workflow.
"""

import re
import wave
import io
from pathlib import Path
from app.database import ProjectDB
from app.handlers.tts_base import TTSHandler
from app.models import PhoneticSuggestion


# ─── Phonetic suggestion rules ─────────────────────────────────────────

PHONETIC_RULES = [
    # Vowel patterns
    (r"ae", [("ay", "'ae' as in 'day'"), ("ee", "'ae' as in 'Caesar'"), ("eh", "'ae' as short 'a'")]),
    (r"ei", [("ay", "'ei' as in 'vein'"), ("ee", "'ei' as in 'receive'"), ("eye", "'ei' as in 'height'")]),
    (r"ou", [("oo", "'ou' as in 'you'"), ("ow", "'ou' as in 'out'"), ("uh", "'ou' as in 'tough'")]),
    (r"eu", [("yoo", "'eu' as in 'feud'"), ("oy", "'eu' as in German")]),
    (r"au", [("aw", "'au' as in 'caught'"), ("ow", "'au' as in 'haus'")]),
    (r"oo", [("oo", "'oo' as in 'food'"), ("uh", "'oo' as in 'blood'")]),
    (r"ea", [("ee", "'ea' as in 'read/reed'"), ("eh", "'ea' as in 'read/red'")]),

    # Consonant patterns
    (r"c(?=[ei])", [("s", "'c' before e/i as 's'"), ("k", "'c' before e/i as 'k'")]),
    (r"ch", [("ch", "'ch' as in 'church'"), ("k", "'ch' as in 'chorus'"), ("sh", "'ch' as in 'machine'")]),
    (r"gh", [("", "'gh' silent as in 'night'"), ("g", "'gh' as hard 'g'"), ("f", "'gh' as in 'enough'")]),
    (r"ph", [("f", "'ph' as 'f'")]),
    (r"th", [("th", "'th' as in 'think'"), ("t", "'th' as hard 't'"), ("dh", "'th' as in 'the'")]),
    (r"kh", [("k", "'kh' as hard 'k'"), ("kh", "'kh' as guttural (use k'h)")]),
    (r"bh", [("v", "'bh' as 'v' (Gaelic)"), ("b", "'bh' as plain 'b'")]),
    (r"dh", [("th", "'dh' as 'th' (Gaelic)"), ("d", "'dh' as plain 'd'")]),
    (r"mh", [("v", "'mh' as 'v' (Gaelic)"), ("m", "'mh' as plain 'm'")]),

    # Final patterns
    (r"e$", [("", "final 'e' silent"), ("eh", "final 'e' pronounced"), ("ay", "final 'e' as 'ay'")]),
    (r"i$", [("ee", "final 'i' as 'ee'"), ("eye", "final 'i' as 'eye'"), ("ih", "final 'i' short")]),
    (r"a$", [("ah", "final 'a' as 'ah'"), ("uh", "final 'a' as schwa")]),

    # Stress markers (always offer)
    (r"[aeiou]", [("'", "add stress mark before vowel")]),
]


def get_phonetic_suggestions(word: str) -> list[PhoneticSuggestion]:
    """Generate phonetic modification suggestions for a word.

    Returns a list of suggested changes the user can apply,
    shown as clickable chips in the UI.
    """
    suggestions = []
    lower = word.lower()

    for pattern, replacements in PHONETIC_RULES:
        for match in re.finditer(pattern, lower):
            segment = match.group()
            for replacement, rule_desc in replacements:
                if replacement != segment:  # don't suggest no-change
                    suggestions.append(PhoneticSuggestion(
                        original_segment=segment,
                        suggested_replacement=replacement,
                        rule=rule_desc,
                    ))

    # Deduplicate
    seen = set()
    unique = []
    for s in suggestions:
        key = (s.original_segment, s.suggested_replacement)
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def apply_suggestion(word: str, original_segment: str,
                     replacement: str) -> str:
    """Apply a single phonetic suggestion to a word.

    Replaces the first occurrence of original_segment.
    """
    return word.replace(original_segment, replacement, 1)


# ─── Pronunciation application ─────────────────────────────────────────

def build_replacement_map(db: ProjectDB) -> dict[str, str]:
    """Build the final word→phonetic map from approved pron entries.

    Sorted longest-first to avoid partial matches (e.g. "Caelibre Guard"
    before "Caeli").
    """
    entries = db.get_pron_entries()
    replacements = {}

    for entry in entries:
        if entry["phonetic"] and entry["status"] == "approved":
            replacements[entry["word"]] = entry["phonetic"]

    # Sort longest-first
    return dict(sorted(replacements.items(), key=lambda x: -len(x[0])))


def apply_pronunciation(text: str, pron_map: dict[str, str],
                        location_overrides: list[dict] | None = None) -> str:
    """Apply pronunciation substitutions to text.

    Args:
        text: Original text.
        pron_map: Global word→phonetic replacements.
        location_overrides: Per-location overrides (from user flags in stage 13).
    """
    result = text

    # Apply location-specific overrides first (higher priority)
    if location_overrides:
        for override in location_overrides:
            word = override["word"]
            phonetic = override["phonetic"]
            # Replace only at the specific position if word_offset is set
            result = re.sub(
                rf"\b{re.escape(word)}\b",
                phonetic,
                result,
                count=1,  # only first occurrence in this chunk
                flags=re.IGNORECASE,
            )

    # Apply global replacements (longest-first)
    for original, phonetic in pron_map.items():
        result = re.sub(
            rf"\b{re.escape(original)}\b",
            phonetic,
            result,
            flags=re.IGNORECASE,
        )

    return result


def apply_all_pronunciation(db: ProjectDB):
    """Apply pronunciation substitutions to all chunks, storing results.

    Skips chunks where pron_text_locked=1 (user manually edited TTS text).
    """
    pron_map = build_replacement_map(db)
    chunks = db.get_chunks()

    for chunk in chunks:
        if chunk.get("pron_text_locked"):
            continue
        overrides = db.get_location_overrides(chunk["id"])
        source = chunk.get("tagged_text") or chunk["original_text"]
        pron_text = apply_pronunciation(source, pron_map, overrides)
        db.update_chunk_pron(chunk["id"], pron_text)


# ─── Test audio generation ──────────────────────────────────────────────

def generate_pron_test(tts: TTSHandler, text: str, voice: str,
                       output_path: Path, params: dict | None = None) -> str:
    """Generate a short test audio clip for pronunciation checking.

    Args:
        tts: Active TTS handler.
        text: Context text with the word embedded.
        voice: Voice to use.
        output_path: Where to save the WAV file.
        params: Optional TTS params override.

    Returns:
        Path to the generated WAV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcm_data = tts.generate(text, voice, params)
    sample_rate = tts.get_sample_rate()

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    return str(output_path)
