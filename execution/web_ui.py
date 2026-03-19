"""
Personal Assistant Web UI
=========================
FastAPI + WebSocket server with a real-time chat interface for the agent.

Usage:
    .venv/bin/python execution/web_ui.py
    # Then open http://localhost:7860
"""

import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

load_dotenv()

# Shared LLM factory (single source of truth — see execution/llm_utils.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_utils import get_llm  # noqa: E402


# ── Global app state (shared across routes) ───────────────────────────────────

class AppState:
    """Tracks the active agent task and overlay subscribers."""
    def __init__(self):
        self.agent_task: asyncio.Task | None = None
        self.running: bool = False
        self.status_subs: list[WebSocket] = []

    async def broadcast(self, running: bool):
        """Push running state to all overlay subscribers."""
        self.running = running
        dead = []
        for sub in self.status_subs:
            try:
                await sub.send_json({"running": running})
            except Exception:
                dead.append(sub)
        for sub in dead:
            self.status_subs.remove(sub)

state = AppState()


# ── Session ──────────────────────────────────────────────────────────────────

class AgentSession:
    """Holds state for one active WebSocket + agent session."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.permission_queue: asyncio.Queue = asyncio.Queue()
        self.running = False

    async def send(self, msg: dict):
        try:
            await self.ws.send_json(msg)
        except Exception:
            pass

    async def log(self, text: str, level: str = "agent"):
        ts = datetime.now().strftime("%H:%M:%S")
        await self.send({"type": "log", "text": text, "level": level, "ts": ts})

    async def request_permission(self, question: str) -> str:
        await self.send({"type": "permission_request", "question": question})
        return await self.permission_queue.get()


# ── Logging bridge ────────────────────────────────────────────────────────────

class WSLogHandler(logging.Handler):
    """Forwards browser_use log records to the WebSocket."""

    def __init__(self, session: AgentSession):
        super().__init__()
        self.session = session

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        if not msg.strip():
            return
        level = "warn" if record.levelno >= logging.WARNING else "agent"
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.session.log(msg, level))
        except Exception:
            pass


# ── Agent runner ──────────────────────────────────────────────────────────────

# Responses that mean "no" — kept in sync with personal_assistant.py
DENY_WORDS = {"no", "n", "deny", "cancel", "stop"}


async def run_agent(task: str, session: AgentSession):
    from browser_use import Agent, BrowserSession, Controller
    from browser_use.agent.views import ActionResult

    controller = Controller()

    @controller.action(
        "Ask the human user for permission before performing any sensitive action "
        "(purchases, payments, form submissions, deletions, sending messages, sign-ups). "
        "Also use this to request missing info like login credentials or delivery addresses."
    )
    async def ask_human(question: str) -> ActionResult:
        response = await session.request_permission(question)
        if response.strip().lower() in DENY_WORDS:
            return ActionResult(
                extracted_content="User denied this action. Do NOT proceed.",
                error="Permission denied by user.",
            )
        return ActionResult(extracted_content=f"User approved and said: {response}")

    browser_session = BrowserSession(headless=False, keep_alive=False)
    llm = get_llm()

    agent = Agent(
        task=task,
        llm=llm,
        controller=controller,
        browser_session=browser_session,
        max_actions_per_step=5,
        max_failures=3,
        extend_system_message="""
You are a helpful personal assistant that controls the browser.

PERMISSION RULES — follow strictly:
- Before any purchase, payment, or order → call ask_human FIRST
- Before submitting forms with personal or financial info → call ask_human FIRST
- Before deleting any data → call ask_human FIRST
- Before sending emails or messages → call ask_human FIRST
- Before signing up anywhere → call ask_human FIRST
- If you need credentials, card details, or personal info → call ask_human to request them
- If unsure whether an action is sensitive → call ask_human to be safe

Narrate clearly what you are doing. If you hit a CAPTCHA, call ask_human.
""",
    )

    history = await agent.run(max_steps=50)
    return history


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Personal Assistant Agent</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0f1117;--surface:#1a1d27;--surface2:#222535;--border:#2d3148;
    --accent:#6366f1;--accent-h:#818cf8;--green:#22c55e;--red:#ef4444;
    --yellow:#f59e0b;--text:#e2e8f0;--muted:#64748b;--dim:#94a3b8;
    --r:12px;--rs:8px;
  }
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}

  /* Header */
  header{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0}
  header h1{font-size:17px;font-weight:600;letter-spacing:-.3px}
  header h1 span{color:var(--accent)}
  .status{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--dim)}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--muted);transition:background .3s}
  .dot.on{background:var(--green);box-shadow:0 0 6px var(--green)}
  .dot.busy{background:var(--yellow);box-shadow:0 0 6px var(--yellow);animation:pulse 1s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

  /* Layout */
  main{display:grid;grid-template-columns:1fr 400px;flex:1;overflow:hidden}

  /* Chat */
  .chat{display:flex;flex-direction:column;border-right:1px solid var(--border);overflow:hidden}
  .panel-hdr{padding:12px 20px;border-bottom:1px solid var(--border);font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;flex-shrink:0}
  .msgs{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
  .msg{display:flex;flex-direction:column;gap:5px;max-width:85%}
  .msg.user{align-self:flex-end;align-items:flex-end}
  .msg.agent{align-self:flex-start;align-items:flex-start}
  .msg-lbl{font-size:11px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px}
  .bubble{padding:11px 15px;border-radius:var(--r);font-size:14px;line-height:1.6;word-break:break-word}
  .msg.user .bubble{background:var(--accent);color:#fff;border-bottom-right-radius:4px}
  .msg.agent .bubble{background:var(--surface2);border:1px solid var(--border);border-bottom-left-radius:4px}
  .msg.err .bubble{background:#2d1515;border-color:var(--red);color:#fca5a5}

  /* Input */
  .inp-area{padding:14px 18px;border-top:1px solid var(--border);display:flex;gap:10px;align-items:flex-end;flex-shrink:0}
  .inp-area textarea{flex:1;background:var(--surface2);border:1px solid var(--border);border-radius:var(--rs);color:var(--text);font-size:14px;padding:11px 13px;resize:none;outline:none;font-family:inherit;line-height:1.5;min-height:46px;max-height:120px;transition:border-color .2s}
  .inp-area textarea:focus{border-color:var(--accent)}
  .inp-area textarea::placeholder{color:var(--muted)}
  .inp-area textarea:disabled{opacity:.5;cursor:not-allowed}
  .send{background:var(--accent);color:#fff;border:none;border-radius:var(--rs);padding:0 20px;font-size:14px;font-weight:600;cursor:pointer;height:46px;transition:background .2s,transform .1s;white-space:nowrap}
  .send:hover{background:var(--accent-h)}
  .send:active{transform:scale(.97)}
  .send:disabled{background:var(--border);color:var(--muted);cursor:not-allowed}

  /* Log panel */
  .log-panel{display:flex;flex-direction:column;overflow:hidden;background:var(--surface)}
  .log-hdr{padding:12px 18px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-shrink:0}
  .log-hdr span{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}
  .clr{background:none;border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer;transition:all .2s}
  .clr:hover{border-color:var(--dim);color:var(--text)}
  .log-out{flex:1;overflow-y:auto;padding:10px 14px;font-family:'SF Mono','Consolas',monospace;font-size:12px;display:flex;flex-direction:column;gap:3px}
  .le{display:flex;gap:8px;padding:4px 7px;border-radius:6px;line-height:1.5}
  .le:hover{background:var(--surface2)}
  .ts{color:var(--muted);white-space:nowrap;flex-shrink:0;font-size:11px;padding-top:1px}
  .lt{color:var(--dim);word-break:break-word;flex:1}
  .le.info .lt{color:#60a5fa}
  .le.warn .lt{color:#fbbf24}
  .le.error .lt{color:#f87171}
  .le.success .lt{color:var(--green)}
  .le.agent .lt{color:#c4b5fd}

  /* Permission modal */
  .overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);backdrop-filter:blur(4px);z-index:100;align-items:center;justify-content:center}
  .overlay.on{display:flex}
  .modal{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:28px;max-width:480px;width:90%;box-shadow:0 20px 60px rgba(0,0,0,.5);animation:up .2s ease}
  @keyframes up{from{transform:translateY(16px);opacity:0}to{transform:translateY(0);opacity:1}}
  .modal-icon{font-size:28px;margin-bottom:10px}
  .modal h3{font-size:17px;font-weight:600;margin-bottom:8px;color:var(--yellow)}
  .modal-q{font-size:14px;color:var(--dim);line-height:1.6;margin-bottom:18px;padding:11px;background:var(--surface2);border-radius:var(--rs);border-left:3px solid var(--yellow)}
  .modal textarea{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:var(--rs);color:var(--text);font-size:14px;padding:10px 12px;font-family:inherit;resize:none;outline:none;margin-bottom:8px;height:68px}
  .modal textarea:focus{border-color:var(--accent)}
  .modal-hint{font-size:12px;color:var(--muted);margin-bottom:16px}
  .modal-btns{display:flex;gap:10px;justify-content:flex-end}
  .deny{background:transparent;border:1px solid var(--red);color:var(--red);border-radius:var(--rs);padding:10px 20px;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
  .deny:hover{background:#2d1515}
  .approve{background:var(--green);border:none;color:#fff;border-radius:var(--rs);padding:10px 20px;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
  .approve:hover{background:#16a34a}

  /* Welcome */
  .welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:10px;color:var(--muted);text-align:center;padding:40px}
  .wi{font-size:44px;margin-bottom:6px}
  .welcome h2{font-size:19px;color:var(--dim);font-weight:600}
  .welcome p{font-size:14px;line-height:1.7;max-width:340px}
  .chips{display:flex;flex-direction:column;gap:8px;margin-top:10px;width:100%;max-width:380px}
  .chip{background:var(--surface2);border:1px solid var(--border);border-radius:20px;padding:8px 16px;font-size:13px;color:var(--dim);cursor:pointer;transition:all .2s;text-align:left}
  .chip:hover{border-color:var(--accent);color:var(--accent)}

  ::-webkit-scrollbar{width:5px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
  ::-webkit-scrollbar-thumb:hover{background:var(--muted)}

  /* ── Floating Agent Badge ── */
  #agent-badge{
    display:none;position:fixed;bottom:30px;right:30px;z-index:9999;
    background:var(--surface);border:1.5px solid var(--accent);
    border-radius:50px;padding:10px 14px 10px 12px;
    box-shadow:0 6px 28px rgba(99,102,241,.35);
    align-items:center;gap:10px;
    cursor:grab;user-select:none;
    animation:badge-in .25s cubic-bezier(.34,1.56,.64,1);
    transition:box-shadow .2s;
  }
  #agent-badge.active{display:flex}
  #agent-badge.dragging{cursor:grabbing;box-shadow:0 10px 36px rgba(99,102,241,.55)}
  #agent-badge.cancelling{border-color:var(--red);box-shadow:0 6px 28px rgba(239,68,68,.3)}
  @keyframes badge-in{from{transform:scale(.75) translateY(14px);opacity:0}to{transform:scale(1) translateY(0);opacity:1}}

  .badge-icon{font-size:22px;animation:wiggle 2.4s ease-in-out infinite;line-height:1}
  @keyframes wiggle{0%,100%{transform:rotate(-8deg) scale(1)}50%{transform:rotate(8deg) scale(1.08)}}
  #agent-badge.cancelling .badge-icon{animation:spin .6s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  .badge-pulse{width:8px;height:8px;border-radius:50%;background:var(--yellow);
    box-shadow:0 0 6px var(--yellow);animation:pulse 1s infinite;flex-shrink:0}
  #agent-badge.cancelling .badge-pulse{background:var(--red);box-shadow:0 0 6px var(--red)}

  .badge-label{font-size:13px;font-weight:600;color:var(--text);white-space:nowrap}

  .badge-cancel{
    width:22px;height:22px;border-radius:50%;background:var(--red);
    border:none;color:#fff;font-size:12px;font-weight:700;
    cursor:pointer;display:flex;align-items:center;justify-content:center;
    flex-shrink:0;transition:background .15s,transform .1s;line-height:1;
    pointer-events:all;
  }
  .badge-cancel:hover{background:#dc2626;transform:scale(1.15)}
  .badge-cancel:active{transform:scale(.92)}
  .badge-cancel:disabled{background:var(--muted);cursor:not-allowed;transform:none}
</style>
</head>
<body>

<header>
  <h1>Personal <span>Assistant</span> Agent</h1>
  <div class="status">
    <div class="dot" id="dot"></div>
    <span id="status-txt">Connecting...</span>
  </div>
</header>

<main>
  <!-- Chat Panel -->
  <div class="chat">
    <div class="panel-hdr">Conversation</div>
    <div class="msgs" id="msgs">
      <div class="welcome" id="welcome">
        <div class="wi">🤖</div>
        <h2>Ready to help</h2>
        <p>Type any task in plain English. The agent opens a real browser and gets it done — and asks before any sensitive action.</p>
        <div class="chips">
          <div class="chip" onclick="useEx(this)">Search for a Logitech MX Master mouse on Amazon</div>
          <div class="chip" onclick="useEx(this)">Find top Python tutorials on YouTube</div>
          <div class="chip" onclick="useEx(this)">Check flights from NYC to LA for next Friday on Google Flights</div>
        </div>
      </div>
    </div>
    <div class="inp-area">
      <textarea id="inp" placeholder="Type your command... (Enter to send, Shift+Enter for newline)" rows="1"></textarea>
      <button class="send" id="send-btn" onclick="send()">Send</button>
    </div>
  </div>

  <!-- Log Panel -->
  <div class="log-panel">
    <div class="log-hdr">
      <span>Agent Activity</span>
      <button class="clr" onclick="clearLog()">Clear</button>
    </div>
    <div class="log-out" id="log"></div>
  </div>
</main>

<!-- Floating Agent Badge -->
<div id="agent-badge">
  <div class="badge-icon">🤖</div>
  <div class="badge-pulse"></div>
  <span class="badge-label" id="badge-lbl">Agent Working...</span>
  <button class="badge-cancel" id="badge-cancel-btn" onclick="cancelTask()" title="Cancel task">✕</button>
</div>

<!-- Permission Modal -->
<div class="overlay" id="overlay">
  <div class="modal">
    <div class="modal-icon">⚠️</div>
    <h3>Permission Required</h3>
    <div class="modal-q" id="modal-q"></div>
    <textarea id="modal-inp" placeholder="Type yes / no, or provide the requested info..."></textarea>
    <p class="modal-hint">Press Enter or click Approve to confirm. Click Deny to cancel this action.</p>
    <div class="modal-btns">
      <button class="deny" onclick="respond('no')">Deny</button>
      <button class="approve" onclick="respond('yes')">Approve</button>
    </div>
  </div>
</div>

<script>
  const dot       = document.getElementById('dot');
  const stxt      = document.getElementById('status-txt');
  const btn       = document.getElementById('send-btn');
  const inp       = document.getElementById('inp');
  const msgs      = document.getElementById('msgs');
  const log       = document.getElementById('log');
  const overlay   = document.getElementById('overlay');
  const modalQ    = document.getElementById('modal-q');
  const modalI    = document.getElementById('modal-inp');
  const badge     = document.getElementById('agent-badge');
  const badgeLbl  = document.getElementById('badge-lbl');
  const badgeCanBtn = document.getElementById('badge-cancel-btn');
  let running     = false;
  let ws          = null;
  // Track whether badge was manually dragged (so we don't reset its position)
  let badgeDragged = false;

  // ── Floating badge drag ───────────────────────────────────────────────────
  let dragging=false, dragOffX=0, dragOffY=0;

  function badgePointerDown(e) {
    if (e.target === badgeCanBtn || badgeCanBtn.contains(e.target)) return;
    dragging = true;
    const r = badge.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    dragOffX = cx - r.left;
    dragOffY = cy - r.top;
    badge.classList.add('dragging');
    // Switch from right/bottom anchoring to left/top absolute positioning
    badge.style.right  = 'auto';
    badge.style.bottom = 'auto';
    badge.style.left   = r.left + 'px';
    badge.style.top    = r.top  + 'px';
    badgeDragged = true;
    e.preventDefault && e.preventDefault();
  }

  function badgePointerMove(e) {
    if (!dragging) return;
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    // Clamp inside viewport
    const bw = badge.offsetWidth, bh = badge.offsetHeight;
    const x  = Math.max(0, Math.min(window.innerWidth  - bw, cx - dragOffX));
    const y  = Math.max(0, Math.min(window.innerHeight - bh, cy - dragOffY));
    badge.style.left = x + 'px';
    badge.style.top  = y + 'px';
  }

  function badgePointerUp() {
    if (!dragging) return;
    dragging = false;
    badge.classList.remove('dragging');
  }

  badge.addEventListener('mousedown',   badgePointerDown);
  badge.addEventListener('touchstart',  badgePointerDown, {passive:false});
  document.addEventListener('mousemove',  badgePointerMove);
  document.addEventListener('touchmove',  badgePointerMove, {passive:true});
  document.addEventListener('mouseup',    badgePointerUp);
  document.addEventListener('touchend',   badgePointerUp);

  // Clamp badge inside viewport on window resize
  window.addEventListener('resize', () => {
    if (!badge.classList.contains('active')) return;
    if (badge.style.left) {
      const bw = badge.offsetWidth, bh = badge.offsetHeight;
      const x = Math.max(0, Math.min(window.innerWidth  - bw, parseInt(badge.style.left)));
      const y = Math.max(0, Math.min(window.innerHeight - bh, parseInt(badge.style.top)));
      badge.style.left = x + 'px';
      badge.style.top  = y + 'px';
    }
  });

  function cancelTask() {
    if (!running) return;
    badgeLbl.textContent = 'Cancelling...';
    badge.classList.add('cancelling');
    badgeCanBtn.disabled = true;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({type:'cancel'}));
    addLog('Cancelling task...', 'warn');
  }

  function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);

    ws.onopen = () => {
      dot.className = 'dot on';
      stxt.textContent = 'Connected · Idle';
      addLog('Connected to agent server.', 'success');
    };

    ws.onclose = () => {
      dot.className = 'dot';
      stxt.textContent = 'Reconnecting...';
      setRunning(false);
      // Auto-reconnect after 2 seconds
      setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      addLog('WebSocket error — retrying...', 'error');
    };

    ws.onmessage = ({data}) => {
      let m;
      try { m = JSON.parse(data); } catch(e) { return; }
      const t = m.type || '';
      if (t === 'log')                    addLog(m.text||'', m.level||'agent', m.ts);
      else if (t === 'permission_request') showModal(m.question||'');
      else if (t === 'done')  { setRunning(false); addMsg('agent', m.result||'Task completed.'); addLog('Task completed.','success'); }
      else if (t === 'error') { setRunning(false); addMsg('agent', m.text||'Unknown error.', true); addLog('Error: '+(m.text||''),'error'); }
    };
  }

  connect(); // initial connection

  function send() {
    const text = inp.value.trim();
    if (!text || running) return;
    if (!ws || ws.readyState !== 1) {
      addLog('Not connected — please wait for reconnection.', 'warn');
      return;
    }
    document.getElementById('welcome')?.remove();
    addMsg('user', text);
    addLog('Task: ' + text, 'info');
    ws.send(JSON.stringify({type:'command', text}));
    inp.value = ''; inp.style.height = 'auto';
    setRunning(true);
  }

  function showModal(q) {
    modalQ.textContent = q; modalI.value = '';
    overlay.classList.add('on'); modalI.focus();
    addLog('Permission requested: ' + q, 'warn');
  }

  function respond(def) {
    const text = modalI.value.trim() || def;
    overlay.classList.remove('on');
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({type:'permission_response', text}));
    addLog('Response: "' + text + '"', 'info');
  }

  function addMsg(role, text, err=false) {
    const d = document.createElement('div');
    d.className = 'msg ' + role + (err?' err':'');
    d.innerHTML = '<div class="msg-lbl">'+(role==='user'?'You':'Agent')+'</div><div class="bubble">'+esc(text)+'</div>';
    msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight;
  }

  function addLog(text, level='agent', ts=null) {
    const t = ts || new Date().toLocaleTimeString('en-US',{hour12:false});
    const d = document.createElement('div');
    d.className = 'le ' + level;
    d.innerHTML = '<span class="ts">'+t+'</span><span class="lt">'+esc(text)+'</span>';
    log.appendChild(d); log.scrollTop = log.scrollHeight;
  }

  function setRunning(v) {
    running = v; btn.disabled = v; inp.disabled = v;
    dot.className = v ? 'dot busy' : 'dot on';
    stxt.textContent = v ? 'Agent Running...' : 'Connected · Idle';
    if (v) {
      // Reset badge state and anchor it bottom-right (unless user already dragged it)
      badge.classList.remove('cancelling');
      badge.classList.add('active');
      badgeLbl.textContent = 'Agent Working...';
      badgeCanBtn.disabled = false;
      if (!badgeDragged) {
        badge.style.left = ''; badge.style.top = '';
        badge.style.right = '30px'; badge.style.bottom = '30px';
      }
    } else {
      badge.classList.remove('active', 'cancelling');
      // Reset drag state so next task starts at default position
      badgeDragged = false;
      badge.style.left = ''; badge.style.top = '';
      badge.style.right = '30px'; badge.style.bottom = '30px';
    }
  }

  function clearLog() { log.innerHTML = ''; }
  function useEx(el)  { inp.value = el.textContent; inp.focus(); }
  function esc(s)     {
    return String(s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .split('\\n').join('<br>');
  }

  inp.addEventListener('input', () => { inp.style.height='auto'; inp.style.height=Math.min(inp.scrollHeight,120)+'px'; });
  inp.addEventListener('keydown', e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();} });
  modalI.addEventListener('keydown', e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();respond('yes');} });
</script>
</body>
</html>"""


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()


@app.get("/")
async def index():
    return HTMLResponse(HTML)


@app.get("/status")
async def status_route():
    """Polled by the native overlay to know if the agent is running."""
    return {"running": state.running}


@app.post("/cancel")
async def cancel_route():
    """HTTP cancel endpoint — used by the native overlay button."""
    if state.agent_task and not state.agent_task.done():
        state.agent_task.cancel()
    state.running = False
    return {"ok": True}


@app.websocket("/status-ws")
async def status_ws_route(ws: WebSocket):
    """
    Lightweight WebSocket for the native overlay.
    Pushes {running: bool} whenever agent state changes.
    """
    await ws.accept()
    state.status_subs.append(ws)
    try:
        # Send current state immediately on connect
        await ws.send_json({"running": state.running})
        # Keep alive with periodic pings; disconnect raises an exception
        while True:
            await asyncio.sleep(20)
            await ws.send_json({"ping": True})
    except Exception:
        pass
    finally:
        if ws in state.status_subs:
            state.status_subs.remove(ws)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = AgentSession(ws)
    agent_task: asyncio.Task | None = None

    # Bridge browser_use logs → WebSocket
    handler = WSLogHandler(session)
    handler.setFormatter(logging.Formatter("%(message)s"))
    bu_logger = logging.getLogger("browser_use")
    bu_logger.addHandler(handler)

    async def run_and_notify(task: str):
        """Runs the agent and sends done/error back — as a background task."""
        await state.broadcast(True)   # ← show native overlay
        try:
            history = await run_agent(task, session)
            result = history.final_result() or "Task completed."
            await ws.send_json({"type": "done", "result": result})
        except asyncio.CancelledError:
            await ws.send_json({"type": "error", "text": "Task cancelled by user."})
        except Exception as e:
            await ws.send_json({"type": "error", "text": str(e)})
        finally:
            session.running = False
            state.agent_task = None        # ← clear reference after task ends
            await state.broadcast(False)  # ← hide native overlay

    try:
        while True:
            data = await ws.receive_json()

            # Guard against missing/malformed messages
            msg_type = data.get("type") if isinstance(data, dict) else None
            if not msg_type:
                continue

            if msg_type == "command":
                if session.running:
                    await session.log("Agent is already running — please wait.", "warn")
                    continue
                task_text = data.get("text", "").strip()
                if not task_text:
                    await session.log("Empty command — please type a task.", "warn")
                    continue
                session.running = True
                # Run agent as a background task so this loop keeps receiving
                # permission_response messages while the agent is working.
                agent_task = asyncio.create_task(run_and_notify(task_text))
                state.agent_task = agent_task  # expose for /cancel endpoint

            elif msg_type == "permission_response":
                await session.permission_queue.put(data.get("text", "no"))

            elif msg_type == "cancel":
                if state.agent_task and not state.agent_task.done():
                    state.agent_task.cancel()
                    await session.log("Task cancelled by user.", "warn")

    except WebSocketDisconnect:
        if agent_task and not agent_task.done():
            agent_task.cancel()
    finally:
        bu_logger.removeHandler(handler)


# ── Start ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Personal Assistant Web UI")
    print("=" * 50)
    print("  Open: http://localhost:7860")
    print("  Press Ctrl+C to stop")
    print("=" * 50 + "\n")

    # Validate LLM config before starting — fail fast with a clear message
    try:
        get_llm()
        print("  ✓ LLM config OK")
    except ValueError as e:
        print(f"\n  [Config Error] {e}")
        print("  Please update your .env file and try again.\n")
        sys.exit(1)

    # Kill any stale overlay process, then launch a fresh one.
    subprocess.run(["pkill", "-f", "agent_overlay.py"], capture_output=True)
    overlay_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_overlay.py")
    subprocess.Popen(
        [sys.executable, overlay_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("  ✓ Native overlay launched\n")

    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
