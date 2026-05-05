"""LLM-based emotion tagging for TTS chunks.

Step 10: If an LLM is configured and the TTS supports tags, the user can
have chunks tagged with expression markers (<laugh>, <sigh>, etc.).
The process: build a prompt → test on a few chunks → user approves → tag all.
"""

from app.database import ProjectDB
from app.handlers.llm_base import LLMHandler
from app.handlers.tts_base import TTSHandler


DEFAULT_SYSTEM_PROMPT = """You are an audiobook director. You receive a chunk of fiction text
and return it with emotion tags inserted where they would sound natural.

Available tags: {available_tags}

Rules:
- Be SPARING. Most sentences need NO tags. Only add where emotion is clearly implied.
- Never add tags inside a word.
- Preserve ALL original text exactly — only insert tags.
- Don't add tags to narration unless the narrator is clearly emotional.
- Dialogue is the main place for tags — match the character's described emotion.
- Return ONLY the tagged text, nothing else. No explanations.

Examples:
Input:  "I can't believe he's gone," she whispered, wiping her eyes.
Output: "I can't believe he's gone," <sniffle> she whispered, wiping her eyes.

Input:  He took a deep breath and stared at the horizon.
Output: He took a deep breath <sigh> and stared at the horizon.

Input:  The soldiers marched in formation through the gate.
Output: The soldiers marched in formation through the gate.
(no tags needed — neutral narration)
"""


def build_tagging_prompt(tts: TTSHandler) -> str:
    """Build the system prompt using the active TTS engine's supported tags."""
    tags = tts.get_supported_tags()
    if not tags:
        return ""

    tag_list = ", ".join(t.tag for t in tags)
    return DEFAULT_SYSTEM_PROMPT.format(available_tags=tag_list)


def tag_single_chunk(llm: LLMHandler, system_prompt: str,
                     chunk_text: str) -> str:
    """Send one chunk to the LLM for tagging."""
    response = llm.complete(
        system=system_prompt,
        prompt=chunk_text,
        max_tokens=len(chunk_text.split()) * 3,
    )
    return response.strip()


def tag_test_chunks(db: ProjectDB, llm: LLMHandler,
                    system_prompt: str,
                    chunk_ids: list[str] | None = None,
                    count: int = 5) -> list[dict]:
    """Tag a few test chunks for user review before committing to full tagging.

    Returns list of {"chunk_id", "original", "tagged"}.
    """
    if chunk_ids:
        chunks = [db.get_chunk(cid) for cid in chunk_ids if db.get_chunk(cid)]
    else:
        # Pick evenly spaced chunks for variety
        all_chunks = db.get_chunks()
        if not all_chunks:
            return []
        step = max(1, len(all_chunks) // count)
        chunks = all_chunks[::step][:count]

    results = []
    for chunk in chunks:
        tagged = tag_single_chunk(llm, system_prompt, chunk["original_text"])
        results.append({
            "chunk_id": chunk["id"],
            "original": chunk["original_text"],
            "tagged": tagged,
        })

    return results


def tag_all_chunks(db: ProjectDB, llm: LLMHandler,
                   system_prompt: str,
                   progress_callback=None) -> int:
    """Tag all chunks with emotion markers.

    Args:
        progress_callback: Called with (current, total) after each chunk.

    Returns number of chunks tagged.
    """
    chunks = db.get_chunks()
    tagged_count = 0

    # Save the approved prompt
    db.save_tagging_config(system_prompt, approved=True)

    for i, chunk in enumerate(chunks):
        # Skip already-tagged chunks
        if chunk.get("tagged_text"):
            tagged_count += 1
            if progress_callback:
                progress_callback(i + 1, len(chunks))
            continue

        try:
            tagged = tag_single_chunk(llm, system_prompt, chunk["original_text"])
            db.update_chunk_tagged(chunk["id"], tagged)
            tagged_count += 1
        except Exception as e:
            # Log error but continue
            print(f"Tagging failed for {chunk['id']}: {e}")

        if progress_callback:
            progress_callback(i + 1, len(chunks))

    db.set_meta("current_step", "tagged")
    return tagged_count
