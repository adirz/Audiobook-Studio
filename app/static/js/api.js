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

    get(path)          { return this.request('GET', path); },
    post(path, body)   { return this.request('POST', path, body); },
    patch(path, body)  { return this.request('PATCH', path, body); },
    del(path)          { return this.request('DELETE', path); },

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


/* ─── Background task panel (follows user across all steps) ─────────── */
function ensureTaskPanel() {
    let p = document.getElementById('bg-task-panel');
    if (!p) {
        p = document.createElement('div');
        p.id = 'bg-task-panel';
        Object.assign(p.style, {
            position: 'fixed', right: '16px', bottom: '16px', width: '280px',
            maxHeight: '70vh', overflowY: 'auto', zIndex: 9999,
            display: 'flex', flexDirection: 'column', gap: '10px', pointerEvents: 'auto'
        });
        document.body.appendChild(p);
    }
    return p;
}

function pauseTask(taskName){
    API.post(`/api/pipeline/${SLUG}/task/${taskName}/pause`).then(()=>showToast('Paused','info')).catch(e=>showToast(e.message,'error'));
}
function resumeTask(taskName){
    API.post(`/api/pipeline/${SLUG}/task/${taskName}/resume`).then(()=>showToast('Resumed','success')).catch(e=>showToast(e.message,'error'));
}
function stopTask(taskName){
    if(!confirm('Stop this background task?')) return;
    API.post(`/api/pipeline/${SLUG}/task/${taskName}/stop`).then(()=>showToast('Stop requested','info')).catch(e=>showToast(e.message,'error'));
}

async function monitorTaskPanel(taskId, displayName){
    const panel = ensureTaskPanel();
    const fullId = taskId.startsWith(SLUG) ? taskId : `${SLUG}_${taskId}`;
    const taskName = fullId.replace(`${SLUG}_`, '');
    const elId = `task-${fullId}`;
    let card = document.getElementById(elId);
    if (!card) {
        card = document.createElement('div');
        card.id = elId;
        Object.assign(card.style, {
            padding: '12px 14px',
            background: '#2e2b27',
            border: '1px solid #d4a05640',
            borderTop: '3px solid #d4a056',
            borderRadius: '8px',
            color: '#e8e0d4',
            fontSize: '0.88rem',
            boxShadow: '0 4px 20px rgba(0,0,0,0.55)',
        });
        panel.appendChild(card);
    }

    const fmtDuration = (sec) => {
        if (!isFinite(sec) || sec < 0) return '—';
        sec = Math.round(sec);
        const h = Math.floor(sec / 3600);
        const m = Math.floor((sec % 3600) / 60);
        const s = sec % 60;
        if (h > 0) return `${h}h ${String(m).padStart(2,'0')}m`;
        if (m > 0) return `${m}m ${String(s).padStart(2,'0')}s`;
        return `${s}s`;
    };

    const updateCard = (d) => {
        const isDone = d.status === 'done' || d.status === 'error';
        const pct = (d.total && d.current !== undefined) ? Math.round(d.current / d.total * 100) : null;

        // Compute ETA from started_at (if backend provided it) and progress.
        let etaStr = null, elapsedStr = null;
        if (d.started_at && d.current && d.total && !isDone) {
            const elapsed = (Date.now() / 1000) - d.started_at;
            elapsedStr = fmtDuration(elapsed);
            if (d.current > 0) {
                const perItem = elapsed / d.current;
                const remaining = (d.total - d.current) * perItem;
                etaStr = fmtDuration(remaining);
            }
        }

        let html = `<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">` +
            `<span style="font-weight:600;color:#e8e0d4;font-size:0.9rem">${displayName || taskName}</span>` +
            (isDone
                ? `<span style="font-size:0.75rem;color:${d.status==='error'?'#c55a5a':'#6dba6d'}">${d.status==='error'?'error':'done'}</span>`
                : `<span style="font-size:0.75rem;color:#a89f90">${pct !== null ? pct+'%' : 'starting…'}</span>`) +
        `</div>`;

        if (pct !== null && !isDone) {
            html += `<div style="height:4px;background:#1a1714;border-radius:2px;margin-bottom:8px;overflow:hidden">` +
                `<div style="height:100%;width:${pct}%;background:#d4a056;border-radius:2px;transition:width 0.3s"></div></div>`;
        }

        if (d.phase) html += `<div style="font-size:0.8rem;color:#7a7268;margin-bottom:4px">Phase: ${d.phase}${d.cycle ? ' · cycle '+d.cycle : ''}</div>`;
        if (d.current !== undefined && d.total) html += `<div style="font-size:0.82rem;color:#a89f90">${d.current} / ${d.total}</div>`;
        if (etaStr) html += `<div style="font-size:0.78rem;color:#7a7268;margin-top:2px">elapsed ${elapsedStr} · ETA ${etaStr}</div>`;

        if (d.passed !== undefined || d.failed !== undefined) {
            html += `<div style="margin-top:6px;display:flex;gap:8px">` +
                `<span class="badge badge-success">${d.passed||0} pass</span>` +
                `<span class="badge badge-error">${d.failed||0} fail</span></div>`;
        }
        if (d.ok !== undefined || d.errors !== undefined) {
            html += `<div style="margin-top:6px;display:flex;gap:8px">` +
                `<span class="badge badge-success">${d.ok||0} ok</span>` +
                `<span class="badge badge-error">${d.errors||0} err</span></div>`;
        }
        if (d.last_result && !isDone) {
            html += `<div style="font-size:0.78rem;margin-top:6px;color:#504a42;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${d.last_result.chunk_id||''}</div>`;
        }

        if (!isDone) {
            html += `<div style="margin-top:10px;display:flex;gap:5px;justify-content:flex-end">` +
                `<button class="btn btn-ghost btn-small" onclick="pauseTask('${taskName}')">Pause</button>` +
                `<button class="btn btn-ghost btn-small" onclick="resumeTask('${taskName}')">Resume</button>` +
                `<button class="btn btn-ghost btn-small" onclick="stopTask('${taskName}')">Stop</button>` +
            `</div>`;
        }

        card.innerHTML = html;
        if (isDone) card.style.borderTop = `3px solid ${d.status==='error'?'#c55a5a':'#6dba6d'}`;
    };

    // Independent poll — continues even if the user navigates to a different step
    API.pollTask(SLUG, fullId, (d) => {
        updateCard(d);

        // Dispatch global event so other parts of the UI can react
        try {
            window.dispatchEvent(new CustomEvent('bg-task-update', { detail: { taskId: fullId, data: d } }));
        } catch (e) { /* ignore */ }

        // Emit chunk-level events for the review live-update system
        try {
            const lr = d.last_result;
            if (lr && lr.chunk_id) {
                if (lr.status === 'starting') {
                    window.dispatchEvent(new CustomEvent('chunk-start', { detail: { taskId: fullId, chunk_id: lr.chunk_id } }));
                } else if (lr.status === 'ok') {
                    window.dispatchEvent(new CustomEvent('chunk-done', { detail: { taskId: fullId, chunk_id: lr.chunk_id } }));
                } else if (lr.status === 'error') {
                    window.dispatchEvent(new CustomEvent('chunk-error', { detail: { taskId: fullId, chunk_id: lr.chunk_id, error: lr.error } }));
                }
            }
        } catch (e) { /* ignore */ }

        if (d.status === 'done' || d.status === 'error') {
            setTimeout(() => { try{ card.remove(); } catch(_){} }, 20000);
        }
    }).catch(e => {
        card.innerHTML = `<div style="color:#7a7268;font-size:0.82rem">Monitor error: ${e.message}</div>`;
        setTimeout(() => { try{ card.remove(); } catch(_){} }, 8000);
    });
}
