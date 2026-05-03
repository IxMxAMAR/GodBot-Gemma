// @ts-nocheck — runs in webview (browser-ish), not Node
const vscode = acquireVsCodeApi();
const $ = (s) => document.querySelector(s);

const TOOL_GROUPS = {
  read_file: 'fs', write_file: 'fs', edit_file: 'fs', glob: 'fs', grep: 'fs', list_dir: 'fs', read_blob: 'fs',
  run_powershell: 'shell', run_bash: 'shell',
  web_fetch: 'web', web_search: 'web',
  run_python: 'python',
  save_note: 'memory', recall_notes: 'memory',
  todo_set: 'task', todo_check: 'task',
  search_knowledge: 'rag',
};

let currentAssistant = null;
let pendingTools = {};

function escape(s) { return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]); }
function scrollBottom() { const c = $('#conversation'); c.scrollTop = c.scrollHeight; }

function appendUser(text) {
  const div = document.createElement('div');
  div.className = 'msg-user';
  div.textContent = text;
  $('#conversation').appendChild(div);
  scrollBottom();
  currentAssistant = null;
}

let _tokenQueue = [];
let _rafScheduled = false;

function _flushTokens() {
  _rafScheduled = false;
  if (_tokenQueue.length === 0) return;
  if (!currentAssistant) {
    currentAssistant = document.createElement('div');
    currentAssistant.className = 'msg-assistant';
    $('#conversation').appendChild(currentAssistant);
  }
  currentAssistant.textContent += _tokenQueue.join('');
  _tokenQueue = [];
  scrollBottom();
}

function appendToken(text) {
  _tokenQueue.push(text);
  if (!_rafScheduled) {
    _rafScheduled = true;
    requestAnimationFrame(_flushTokens);
  }
}

function finalizeAssistant() {
  // Flush any pending tokens immediately so the JSON parse sees the full text.
  if (_tokenQueue.length > 0) _flushTokens();
  if (!currentAssistant) return;
  try {
    const parsed = JSON.parse(currentAssistant.textContent);
    if (parsed && typeof parsed === 'object' && typeof parsed.final_answer === 'string') {
      currentAssistant.textContent = parsed.final_answer;
    }
  } catch { /* leave as raw */ }
  currentAssistant = null;
}

function appendToolCall(ev) {
  currentAssistant = null;
  const card = document.createElement('div');
  card.className = 'tool-card';
  card.dataset.group = TOOL_GROUPS[ev.name] || 'default';
  const args = Object.entries(ev.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ');
  card.innerHTML = `<div>▸ <strong>${escape(ev.name)}</strong>(${escape(args)}) <span class="status">…</span></div><div class="body"></div>`;
  $('#conversation').appendChild(card);
  pendingTools[ev.id] = card;
  scrollBottom();
}

function appendToolResult(ev) {
  const card = pendingTools[ev.id];
  if (!card) return;
  card.querySelector('.status').textContent = `[${ev.duration_ms}ms]`;
  const body = card.querySelector('.body');
  body.textContent = ev.preview || '';
  if (ev.blob) body.textContent += `\n(blob: ${ev.blob})`;
  delete pendingTools[ev.id];
}

function appendGate(ev) {
  const card = document.createElement('div');
  card.className = 'gate-card';
  card.innerHTML = `<div><strong>⚠ Approve ${escape(ev.name)}?</strong></div><pre>${escape(JSON.stringify(ev.args, null, 2))}</pre>
    <button data-d="allow">Allow</button>
    <button data-d="always">Always</button>
    <button data-d="deny">Deny</button>`;
  for (const b of card.querySelectorAll('button')) {
    b.onclick = () => {
      vscode.postMessage({ type: 'gate_decision', callId: ev.id, decision: b.dataset.d });
      card.remove();
    };
  }
  $('#conversation').appendChild(card);
  scrollBottom();
}

function appendError(msg) {
  const div = document.createElement('div');
  div.className = 'error-banner';
  div.textContent = `error: ${msg}`;
  $('#conversation').appendChild(div);
  scrollBottom();
}

function dispatchEvent(ev) {
  switch (ev.type) {
    case 'token': appendToken(ev.text); break;
    case 'tool_call': appendToolCall(ev); break;
    case 'tool_result': appendToolResult(ev); break;
    case 'gate': appendGate(ev); break;
    case 'agent_error': appendError(ev.message); break;
    case 'done': finalizeAssistant(); break;
  }
}

window.addEventListener('message', (e) => {
  const m = e.data;
  switch (m.type) {
    case 'user_message': appendUser(m.text); break;
    case 'event': dispatchEvent(m.payload); break;
    case 'session': $('#session-id').textContent = m.sid; break;
    case 'tools': renderTools(m.tools); break;
    case 'cleared': $('#conversation').innerHTML = ''; pendingTools = {}; currentAssistant = null; break;
    case 'error': appendError(m.message); break;
  }
});

function renderTools(tools) {
  const list = $('#tool-list');
  list.innerHTML = '';
  for (const t of tools) {
    const li = document.createElement('li');
    const dangerous = t.dangerous ? ' ⚠' : '';
    li.innerHTML = `<label><input type="checkbox" checked data-name="${escape(t.name)}"> ${escape(t.name)}${dangerous}</label>`;
    li.querySelector('input').onchange = (e) => {
      vscode.postMessage({ type: 'toggle_tool', name: t.name, enabled: e.target.checked });
    };
    list.appendChild(li);
  }
}

function send() {
  const text = $('#input').value.trim();
  if (!text) return;
  $('#input').value = '';
  vscode.postMessage({ type: 'send', text });
}

$('#send').onclick = send;
$('#new').onclick = () => vscode.postMessage({ type: 'new_session' });
$('#stop').onclick = () => vscode.postMessage({ type: 'stop' });
$('#input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
