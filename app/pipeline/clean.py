"""Clean extracted chapter text.

Step 5a: Normalize dashes, ellipses, whitespace.
Convert user-confirmed scene break symbols into [SCENE_BREAK] markers.
"""

import re
from app.database import ProjectDB


def clean_chapter_text(text: str, scene_break_symbols: set[str]) -> str:
    """Clean a single chapter's text.

    Args:
        text: Raw chapter text.
        scene_break_symbols: Set of symbols/patterns the user confirmed as scene breaks.
    """
    # Remove standalone page numbers
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # Normalize dashes to em-dashes (TTS reads pauses on these)
    text = text.replace(" - ", " — ")
    text = text.replace("--", "—")

    # Normalize ellipses
    text = text.replace("...", "…")

    # Convert confirmed scene break symbols to markers. The originating
    # symbol is embedded so downstream stages (chunking, the Generate
    # Audio filter) can distinguish "after a ♦ break" from "after a
    # * * * break" when a manuscript uses more than one kind. Keep
    # ``[SCENE_BREAK]`` (no symbol) recognized for backward compatibility.
    for sym in scene_break_symbols:
        marker = f"[SCENE_BREAK:{sym}]"
        if len(sym) == 1:
            # Single character repeated as scene break: ♦ ♦ ♦, etc.
            escaped = re.escape(sym)
            text = re.sub(
                rf"^{escaped}(\s*{escaped})*\s*$",
                marker,
                text,
                flags=re.MULTILINE,
            )
        else:
            # Multi-char pattern like "* * *", "---", "⁂"
            escaped = re.escape(sym)
            text = re.sub(
                rf"^{escaped}\s*$",
                marker,
                text,
                flags=re.MULTILINE,
            )

    # Clean up multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_all_chapters(db: ProjectDB):
    """Clean all extracted chapters using user-confirmed scene break symbols."""
    # Get confirmed scene break symbols
    symbols = db.get_symbols()
    scene_breaks = {s["symbol"] for s in symbols if s["is_scene_break"]}

    chapters = db.get_chapters()
    for ch in chapters:
        cleaned = clean_chapter_text(ch["raw_text"], scene_breaks)
        db.update_chapter_cleaned(ch["id"], cleaned)

    db.set_meta("current_step", "cleaned")
