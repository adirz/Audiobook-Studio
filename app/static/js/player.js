/* ═══════════════════════════════════════════════════════════════
   Continuous Playback Controller
   Plays audio across chunks seamlessly, highlights current text,
   allows jump to chapter/scene/chunk, prev/next, flag in-place.
   ═══════════════════════════════════════════════════════════════ */

class ContinuousPlayer {
    constructor({ slug, onChunkChange, onStateChange, getVisibleIds }) {
        this.slug = slug;
        this.onChunkChange = onChunkChange || (() => {});
        this.onStateChange = onStateChange || (() => {});
        // Optional: a callback that returns a Set of chunk IDs currently
        // visible in the UI. When provided, navigation (next/prev/scene
        // jumps and auto-advance after a chunk ends) skips chunks not in
        // that set so the audio "follows" what the user can see.
        this.getVisibleIds = getVisibleIds || (() => null);

        this.chunks = [];          // full ordered list of review chunks
        this.chapters = [];        // chapter index for navigation
        this.currentIdx = 0;       // index into this.chunks
        this.state = 'stopped';    // stopped | playing | paused | loading

        this.audio = new Audio();
        this.audio.addEventListener('ended', () => this._onTrackEnded());
        this.audio.addEventListener('timeupdate', () => this._onTimeUpdate());
        this.audio.addEventListener('error', (e) => this._onError(e));
        this.audio.addEventListener('canplay', () => {
            if (this.state === 'loading') {
                this.state = 'playing';
                const p = this.audio.play();
                if (p !== undefined && typeof p.then === 'function') {
                    p.catch(e => console.debug('ContinuousPlayer play interrupted', e));
                }
                this.onStateChange(this.state);
            }
        });

        // Preload next chunk's audio
        this._preloadAudio = new Audio();
    }

    async init() {
        this.chunks = await API.get(`/api/audio/${this.slug}/review`);
        try {
            this.chapters = await API.get(`/api/audio/${this.slug}/review/chapters`);
        } catch (_) {
            this.chapters = [];
        }
        return this;
    }

    // ─── Playback controls ───────────────────────

    play() {
        if (this.chunks.length === 0) return;
        const chunk = this.chunks[this.currentIdx];
        if (!chunk || !chunk.has_audio) {
            this._skipToNextPlayable();
            return;
        }

        // Ensure any currently playing audio is stopped immediately
        try {
            this.audio.pause();
            this.audio.currentTime = 0;
        } catch (e) { /* ignore */ }

        this.state = 'loading';
        this.onStateChange(this.state);

        // Assign new source and load. Play will start once `canplay` fires
        // (we rely on the existing `canplay` handler to transition to playing).
        try {
            this.audio.src = chunk.audio_url;
            this.audio.load();
        } catch (e) {
            console.error('Failed to load audio src', e);
        }

        this.onChunkChange(this.currentIdx, chunk);

        // Preload next
        this._preloadNext();
    }

    pause() {
        this.audio.pause();
        this.state = 'paused';
        this.onStateChange(this.state);
    }

    resume() {
        const p = this.audio.play();
        if (p !== undefined && typeof p.then === 'function') {
            p.then(() => { this.state = 'playing'; this.onStateChange(this.state); })
             .catch(e => { console.debug('ContinuousPlayer resume interrupted', e); this.state = 'paused'; this.onStateChange(this.state); });
        } else {
            this.state = 'playing';
            this.onStateChange(this.state);
        }
    }

    togglePlay() {
        if (this.state === 'playing') this.pause();
        else if (this.state === 'paused') this.resume();
        else this.play();
    }

    stop() {
        try { this.audio.pause(); } catch(_){}
        try { this.audio.currentTime = 0; } catch(_){}
        try { this.audio.src = ''; } catch(_){}
        this.state = 'stopped';
        this.onStateChange(this.state);
    }

    // ─── Navigation ──────────────────────────────

    _visibleSet() {
        try {
            const v = this.getVisibleIds && this.getVisibleIds();
            return v instanceof Set ? v : null;
        } catch (_) { return null; }
    }
    _isVisible(idx) {
        const v = this._visibleSet();
        if (!v) return true;
        return v.has(this.chunks[idx]?.id);
    }
    _findNextVisible(fromIdx) {
        for (let i = fromIdx; i < this.chunks.length; i++) {
            if (this._isVisible(i)) return i;
        }
        return -1;
    }
    _findPrevVisible(fromIdx) {
        for (let i = fromIdx; i >= 0; i--) {
            if (this._isVisible(i)) return i;
        }
        return -1;
    }
    _commitMove(idx) {
        this.currentIdx = idx;
        if (this.state === 'playing' || this.state === 'loading') this.play();
        else this.onChunkChange(this.currentIdx, this.chunks[this.currentIdx]);
    }

    next() {
        const idx = this._findNextVisible(this.currentIdx + 1);
        if (idx >= 0) this._commitMove(idx);
    }

    prev() {
        // If we're more than 3 seconds in, restart current chunk
        if (this.audio.currentTime > 3) {
            this.audio.currentTime = 0;
            return;
        }
        const idx = this._findPrevVisible(this.currentIdx - 1);
        if (idx >= 0) this._commitMove(idx);
    }

    jumpToChunk(chunkId) {
        // Direct user choice — don't constrain by visibility.
        const idx = this.chunks.findIndex(c => c.id === chunkId);
        if (idx >= 0) this._commitMove(idx);
    }

    jumpToChapter(chapterIdx) {
        const pfx = `ch${String(chapterIdx).padStart(3, '0')}`;
        const matches = (c) => c.chapter_id === chapterIdx || c.id?.startsWith(pfx);
        // Prefer the first chapter chunk that's currently visible.
        const v = this._visibleSet();
        let idx = -1;
        for (let i = 0; i < this.chunks.length; i++) {
            if (matches(this.chunks[i]) && (!v || v.has(this.chunks[i].id))) { idx = i; break; }
        }
        if (idx < 0) idx = this.chunks.findIndex(matches);
        if (idx >= 0) this._commitMove(idx);
    }

    jumpToNextScene() {
        // Walk forward through visible chunks looking for a scene/chapter
        // boundary; jump to the first visible chunk after it.
        for (let i = this.currentIdx; i < this.chunks.length; i++) {
            if (!this._isVisible(i)) continue;
            const c = this.chunks[i];
            if (c.scene_break_after || c.chapter_break_after) {
                const nxt = this._findNextVisible(i + 1);
                if (nxt >= 0) this._commitMove(nxt);
                return;
            }
        }
    }

    jumpToPrevScene() {
        // Find the previous visible scene/chapter boundary, then jump
        // to the first visible chunk after that boundary.
        for (let i = this.currentIdx - 2; i >= 0; i--) {
            if (!this._isVisible(i)) continue;
            const c = this.chunks[i];
            if (c.scene_break_after || c.chapter_break_after || i === 0) {
                const target = (i === 0) ? this._findNextVisible(0) : this._findNextVisible(i + 1);
                if (target >= 0) this._commitMove(target);
                return;
            }
        }
    }

    seek(fraction) {
        if (this.audio.duration) {
            this.audio.currentTime = fraction * this.audio.duration;
        }
    }

    // ─── Getters ─────────────────────────────────

    get currentChunk() {
        return this.chunks[this.currentIdx] || null;
    }

    get progress() {
        if (!this.audio.duration) return 0;
        return this.audio.currentTime / this.audio.duration;
    }

    get currentTime() {
        return this.audio.currentTime || 0;
    }

    get duration() {
        return this.audio.duration || 0;
    }

    get globalProgress() {
        if (this.chunks.length === 0) return 0;
        return (this.currentIdx + this.progress) / this.chunks.length;
    }

    get playableChunks() {
        return this.chunks.filter(c => c.has_audio);
    }

    // ─── Filter views ────────────────────────────

    getChunksByFilter(filter, opts = {}) {
        const hasScore = c => c.qa && c.qa.score !== null && c.qa.score !== undefined;
        switch (filter) {
            case 'all':       return this.chunks;
            case 'flagged':   return this.chunks.filter(c => c.flags && c.flags.length > 0);
            case 'qa_fail':   return this.chunks.filter(c => c.qa?.status === 'fail');
            case 'qa_pass':   return this.chunks.filter(c => c.qa?.status === 'pass' || c.qa?.status === 'override');
            case 'unchecked': return this.chunks.filter(c => c.has_audio && (!c.qa || !c.qa.status || c.qa.status === 'pending'));
            case 'no_audio':  return this.chunks.filter(c => !c.has_audio);
            case 'sim_range': {
                // Two-bound inclusive range. Defaults capture everything
                // when the user hasn't picked numbers yet.
                const lo = (opts.min ?? 0);
                const hi = (opts.max ?? 1);
                return this.chunks.filter(c => hasScore(c) && c.qa.score >= lo && c.qa.score <= hi);
            }
            default:          return this.chunks;
        }
    }

    getChunksForChapter(chapterIdx) {
        return this.chunks.filter(c =>
            c.id?.startsWith(`ch${String(chapterIdx).padStart(3, '0')}`)
        );
    }

    // ─── Internal ────────────────────────────────

    _onTrackEnded() {
        // Insert silence gap based on break type
        const chunk = this.chunks[this.currentIdx];
        let delayMs = 300; // default inter-chunk silence

        if (chunk?.scene_break_after) delayMs = 1500;
        if (chunk?.chapter_break_after) delayMs = 3000;

        // Auto-advance walks the *visible* list so playback stays inside
        // whatever filter the user has set in the review pane.
        const nxt = this._findNextVisible(this.currentIdx + 1);
        if (nxt >= 0) {
            setTimeout(() => {
                this.currentIdx = nxt;
                this._skipToNextPlayable();
            }, delayMs);
        } else {
            this.state = 'stopped';
            this.onStateChange(this.state);
        }
    }

    _skipToNextPlayable() {
        // Find the next chunk that is both visible (in current filter)
        // and has audio. Stop if none.
        while (this.currentIdx < this.chunks.length) {
            const c = this.chunks[this.currentIdx];
            if (c?.has_audio && this._isVisible(this.currentIdx)) {
                this.play();
                return;
            }
            const nxt = this._findNextVisible(this.currentIdx + 1);
            if (nxt < 0) break;
            this.currentIdx = nxt;
        }
        this.state = 'stopped';
        this.onStateChange(this.state);
    }

    _onTimeUpdate() {
        this.onStateChange(this.state);
    }

    _onError(e) {
        console.error('Audio error:', e);
        // Skip to next on error
        if (this.state === 'playing' || this.state === 'loading') {
            this.next();
        }
    }

    _preloadNext() {
        const nextIdx = this.currentIdx + 1;
        if (nextIdx < this.chunks.length && this.chunks[nextIdx]?.has_audio) {
            try{
                this._preloadAudio.pause();
            }catch(_){ }
            this._preloadAudio.src = this.chunks[nextIdx].audio_url;
            try{ this._preloadAudio.load(); }catch(_){ }
        }
    }

    destroy() {
        this.stop();
        try { this._preloadAudio.pause(); } catch(_){}
        try { this._preloadAudio.src = ''; } catch(_){}
    }
}
