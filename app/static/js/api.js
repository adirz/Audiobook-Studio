/* ═══════════════════════════════════════════════════════════════
   Audiobook Studio — API client & utilities
   ═══════════════════════════════════════════════════════════════ */

const API = {
    async request(method, path, body = null) {
        const opts = {
            method,
            headers: { 'Content-Type': 'application/json' },
        };
        if (body) opts.body = JSON.stringify(body);
        const res = await fetch(path, opts);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || JSON.stringify(err));
        }
        return res.json();
    },

    get(path)        { return this.request('GET', path); },
    post(path, body) { return this.request('POST', path, body); },
    del(path)        { return this.request('DELETE', path); },

    async upload(path, file) {
        const form = new FormData();
        form.append('file', file);
        const res = await fetch(path, { method: 'POST', body: form });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Upload failed');
        }
        return res.json();
    },

    // Poll a background task until done
    async pollTask(slug, taskId, onProgress, intervalMs = 1500) {
        const fullId = taskId.startsWith(slug) ? taskId : `${slug}_${taskId}`;
        while (true) {
            const data = await this.get(`/api/pipeline/${slug}/task/${fullId}`);
            if (onProgress) onProgress(data);
            if (data.status === 'done' || data.status === 'error') return data;
            await new Promise(r => setTimeout(r, intervalMs));
        }
    },
};


/* ─── Toast notifications ──────────────────────────────────── */

function getToastContainer() {
    let c = document.getElementById('toast-container');
    if (!c) {
        c = document.createElement('div');
        c.id = 'toast-container';
        c.className = 'toast-container';
        document.body.appendChild(c);
    }
    return c;
}

function showToast(message, type = 'info', durationMs = 4000) {
    const container = getToastContainer();
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = message;
    container.appendChild(el);
    // Keep error toasts visible longer and make them clickable to open
    // a simple in-app error log. Non-error toasts use the provided duration.
    if (type === 'error') {
        window.__errorLog = window.__errorLog || [];
        window.__errorLog.push({ message, ts: new Date().toISOString() });
        el.style.cursor = 'pointer';
        el.title = 'Click to view error log';
        el.addEventListener('click', () => openErrorModal());
        const ms = durationMs || 15000;
        setTimeout(() => { el.remove(); }, ms);
    } else {
        setTimeout(() => { el.remove(); }, durationMs);
    }
}

function openErrorModal() {
    const logs = window.__errorLog || [];
    const modalId = 'error-log-modal';
    let modal = document.getElementById(modalId);
    if (!modal) {
        modal = document.createElement('div');
        modal.id = modalId;
        modal.className = 'modal-overlay';
        modal.innerHTML = `<div class="modal"><h2>Recent errors</h2><div id="error-log-list" style="max-height:60vh;overflow:auto;margin-bottom:12px;"></div><div class="modal-actions"><button class="btn btn-secondary" onclick="closeErrorModal()">Close</button><button class="btn btn-primary" onclick="clearErrorLog()">Clear</button></div></div>`;
        document.body.appendChild(modal);
    }
    const list = modal.querySelector('#error-log-list');
    list.innerHTML = '';
    for (let i = logs.length - 1; i >= 0; i--) {
        const it = logs[i];
        const row = document.createElement('div');
        row.style.padding = '6px 0';
        row.style.borderBottom = '1px solid var(--bg-elevated)';
        row.textContent = `${it.ts} — ${it.message}`;
        list.appendChild(row);
    }
    modal.classList.add('visible');
}

function closeErrorModal() {
    const m = document.getElementById('error-log-modal');
    if (m) m.classList.remove('visible');
}

function clearErrorLog() {
    window.__errorLog = [];
    const list = document.querySelector('#error-log-list');
    if (list) list.innerHTML = '';
    showToast('Cleared error log', 'info', 2000);
}


/* ─── Audio player helper ──────────────────────────────────── */

class AudioPlayerController {
    constructor(containerEl) {
        this.container = containerEl;
        this.audio = new Audio();
        this.playBtn = containerEl.querySelector('.play-btn');
        this.progress = containerEl.querySelector('.progress');
        this.waveform = containerEl.querySelector('.waveform');
        this.timeEl = containerEl.querySelector('.time');

        this.playBtn.addEventListener('click', () => this.togglePlay());
        this.waveform.addEventListener('click', (e) => this.seek(e));
        this.audio.addEventListener('timeupdate', () => this.updateProgress());
        this.audio.addEventListener('ended', () => this.onEnded());
        this.audio.addEventListener('loadedmetadata', () => this.onLoaded());
    }

    load(url) {
        this.audio.src = url;
        this.audio.load();
        this.playBtn.textContent = '▶';
    }

    togglePlay() {
        if (this.audio.paused) {
            const p = this.audio.play();
            if (p !== undefined && typeof p.then === 'function') {
                p.then(() => { this.playBtn.textContent = '⏸'; })
                 .catch(e => { console.debug('Audio play interrupted', e); this.playBtn.textContent = '▶'; });
            } else {
                this.playBtn.textContent = '⏸';
            }
        } else {
            try { this.audio.pause(); } catch(_) {}
            this.playBtn.textContent = '▶';
        }
    }

    seek(e) {
        if (!this.audio.duration) return;
        const rect = this.waveform.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        this.audio.currentTime = pct * this.audio.duration;
    }

    updateProgress() {
        if (!this.audio.duration) return;
        const pct = (this.audio.currentTime / this.audio.duration) * 100;
        this.progress.style.width = pct + '%';
        this.timeEl.textContent = this.formatTime(this.audio.currentTime);
    }

    onEnded() {
        this.playBtn.textContent = '▶';
        this.progress.style.width = '0%';
    }

    onLoaded() {
        this.timeEl.textContent = this.formatTime(this.audio.duration);
    }

    formatTime(sec) {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    destroy() {
        this.audio.pause();
        this.audio.src = '';
    }
}


/* ─── Render helpers ───────────────────────────────────────── */

function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === 'class' || k === 'className') e.className = v;
        else if (k.startsWith('on')) e.addEventListener(k.slice(2).toLowerCase(), v);
        else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
        else e.setAttribute(k, v);
    }
    for (const child of children) {
        if (typeof child === 'string') e.appendChild(document.createTextNode(child));
        else if (child) e.appendChild(child);
    }
    return e;
}

function renderAudioPlayer() {
    return el('div', { class: 'audio-player' },
        el('button', { class: 'play-btn' }, '▶'),
        el('div', { class: 'waveform' },
            el('div', { class: 'progress' })
        ),
        el('span', { class: 'time' }, '0:00')
    );
}

function renderWordDiff(diffData) {
    const container = el('div', { class: 'word-diff reading-text' });
    for (const d of diffData) {
        let cls = 'word word-' + d.type;
        let text = d.type === 'delete' ? d.original : (d.transcribed || d.original);
        let title = '';
        if (d.type === 'replace') {
            title = `Expected: "${d.original}" → Heard: "${d.transcribed}"`;
        }
        if (d.pron_expected) {
            cls += ' pron-expected';
            title += ' (pronunciation substitution — expected)';
        }
        container.appendChild(el('span', { class: cls, title }, text + ' '));
    }
    return container;
}

function renderProgressBar(current, total) {
    const pct = total > 0 ? (current / total * 100) : 0;
    return el('div', { class: 'progress-bar' },
        el('div', { class: 'fill', style: { width: pct + '%' } })
    );
}

// Debounce utility
function debounce(fn, ms) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
    };
}
