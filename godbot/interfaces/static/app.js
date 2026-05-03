// godbot/interfaces/static/app.js
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let state = { session: null, model: '?', rag: 'none', currentAssistantBuf: null, pendingTools: {} };

const TOOL_GROUPS = {
  read_file: 'fs', list_dir: 'fs', glob: 'fs', grep: 'fs',
  write_file: 'fs', edit_file: 'fs', read_blob: 'fs',
  run_powershell: 'shell', run_bash: 'shell',
  web_fetch: 'web', web_search: 'web',
  run_python: 'python',
  save_note: 'memory', recall_notes: 'memory',
  todo_set: 'task', todo_check: 'task',
  search_knowledge: 'rag',
};

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

function bind(name, value) {
  $$(`[data-bind="${name}"]`).forEach(el => el.textContent = value);
}

function appendUser(text) {
  const div = document.createElement('div');
  div.className = 'msg-user';
  div.textContent = text;
  $('#conversation').appendChild(div);
  scrollBottom();
}

function ensureAssistantBubble() {
  if (!state.currentAssistantBuf) {
    const div = document.createElement('div');
    div.className = 'msg-assistant';
    $('#conversation').appendChild(div);
    state.currentAssistantBuf = div;
  }
  return state.currentAssistantBuf;
}

function appendToken(t) {
  ensureAssistantBubble().textContent += t;
  scrollBottom();
}

function appendToolCall(ev) {
  state.currentAssistantBuf = null;
  const card = document.createElement('div');
  card.className = 'tool-card collapsed';
  card.dataset.group = TOOL_GROUPS[ev.name] || 'misc';
  const args = Object.entries(ev.args).map(([k,v]) => `${k}=${JSON.stringify(v)}`).join(', ');
  card.innerHTML = `<div class="head">▸ <strong>${ev.name}</strong>(${args}) <span class="status">…</span></div><div class="body"></div>`;
  card.querySelector('.head').onclick = () => card.classList.toggle('collapsed');
  $('#conversation').appendChild(card);
  state.pendingTools[ev.id] = card;
  scrollBottom();
}

function appendToolResult(ev) {
  const card = state.pendingTools[ev.id];
  if (!card) return;
  card.querySelector('.status').textContent = `[${ev.duration_ms}ms]`;
  const body = card.querySelector('.body');
  body.innerHTML = `<pre>${escape(ev.preview)}</pre>` + (ev.blob ? `<small>full result in blob ${ev.blob}</small>` : '');
  delete state.pendingTools[ev.id];
}

function appendGate(ev) {
  const card = document.createElement('div');
  card.className = 'gate-card';
  const args = JSON.stringify(ev.args, null, 2);
  card.innerHTML = `<div><strong>Approve ${ev.name}?</strong></div><pre>${escape(args)}</pre>
    <button data-d="allow">Approve</button>
    <button data-d="always">Always allow</button>
    <button data-d="deny">Deny</button>`;
  for (const b of card.querySelectorAll('button')) {
    b.onclick = async () => {
      const decision = b.dataset.d;
      await api(`/api/gate/${ev.id}`, { method: 'POST', body: JSON.stringify({ session_id: state.session, decision }) });
      card.remove();
    };
  }
  $('#conversation').appendChild(card);
  scrollBottom();
}

function appendError(ev) {
  const div = document.createElement('div');
  div.className = 'error-banner';
  div.textContent = `error: ${ev.message}`;
  $('#conversation').appendChild(div);
}

function escape(s) { return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function scrollBottom() { const c = $('#conversation'); c.scrollTop = c.scrollHeight; }

function openStream() {
  if (!state.session) return;
  const es = new EventSource(`/api/chat/stream?session_id=${state.session}`);
  es.addEventListener('token', (e) => appendToken(JSON.parse(e.data).text));
  es.addEventListener('tool_call', (e) => appendToolCall(JSON.parse(e.data)));
  es.addEventListener('tool_result', (e) => appendToolResult(JSON.parse(e.data)));
  es.addEventListener('gate', (e) => appendGate(JSON.parse(e.data)));
  es.addEventListener('error', (e) => appendError(JSON.parse(e.data)));
  es.addEventListener('done', () => { state.currentAssistantBuf = null; });
  es.addEventListener('ping', () => {});
  state.es = es;
}

async function send() {
  const text = $('#input').value.trim();
  if (!text || !state.session) return;
  $('#input').value = '';
  appendUser(text);
  await api('/api/chat', { method: 'POST', body: JSON.stringify({ session_id: state.session, message: text }) });
}

async function newSession() {
  const r = await api('/api/sessions/new', { method: 'POST', body: JSON.stringify({}) });
  state.session = r.session_id;
  bind('session', state.session);
  $('#conversation').innerHTML = '';
  if (state.es) state.es.close();
  openStream();
  await refreshSessions();
}

async function refreshSessions() {
  const list = await api('/api/sessions');
  $('#session-list').innerHTML = '';
  for (const s of list.reverse()) {
    const li = document.createElement('li');
    li.textContent = s.id;
    li.onclick = () => loadSession(s.id);
    $('#session-list').appendChild(li);
  }
}

async function loadSession(sid) {
  const s = await api(`/api/sessions/${sid}`);
  state.session = sid;
  bind('session', sid);
  $('#conversation').innerHTML = '';
  for (const m of s.messages) {
    if (m.role === 'user') appendUser(m.content);
    else if (m.role === 'assistant' && m.content) {
      ensureAssistantBubble().textContent = m.content;
      state.currentAssistantBuf = null;
    }
  }
  if (state.es) state.es.close();
  openStream();
}

async function refreshTools() {
  const tools = await api('/api/tools');
  $('#tool-list').innerHTML = '';
  for (const t of tools) {
    const li = document.createElement('li');
    li.innerHTML = `<label><input type="checkbox" checked data-name="${t.name}"> ${t.name}${t.dangerous ? ' ⚠' : ''}</label>`;
    li.querySelector('input').onchange = (e) => api('/api/tools/toggle', {
      method: 'POST',
      body: JSON.stringify({ session_id: state.session, name: t.name, enabled: e.target.checked }),
    });
    $('#tool-list').appendChild(li);
  }
}

async function init() {
  await refreshTools();
  await newSession();
  $('#new-session').onclick = newSession;
  $('#send').onclick = send;
  $('#stop').onclick = () => api('/api/stop', { method: 'POST', body: JSON.stringify({ session_id: state.session }) });
  $('#input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
}

init();
