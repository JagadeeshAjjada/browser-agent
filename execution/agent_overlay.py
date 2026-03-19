#!/usr/bin/env python3
"""
Agent Overlay  —  always-on-top floating pill window
=====================================================
Runs with the built-in macOS system Python (/usr/bin/python3).
Polls http://localhost:7860/status every 500 ms.
Shows a draggable pill above ALL apps when the agent is running.
Click  ✕  to cancel.  Drag to move anywhere on screen.

Auto-launched by web_ui.py.  Can also run standalone:
    /usr/bin/python3 execution/agent_overlay.py
"""

import json
import threading
import tkinter as tk
import urllib.request

# ── Config ────────────────────────────────────────────────────────────────────

STATUS_URL = "http://localhost:7860/status"
CANCEL_URL = "http://localhost:7860/cancel"
POLL_MS    = 500          # how often to check /status  (ms)
TICK_MS    = 450          # animation tick interval      (ms)

BG      = "#1a1d27"
BORDER  = "#6366f1"
RED     = "#ef4444"
YELLOW  = "#f59e0b"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"

# ── Overlay ───────────────────────────────────────────────────────────────────

class AgentOverlay:
    def __init__(self):
        self._visible    = False
        self._pulse_on   = True
        self._dot_frame  = 0
        self._cancelling = False
        self._drag_ox    = 0
        self._drag_oy    = 0

        self._build_window()
        self._build_ui()
        self._bind_drag()

        # Start polling and animation loops
        self.root.after(POLL_MS, self._poll)
        self.root.after(TICK_MS, self._tick)

    # ── Window ────────────────────────────────────────────────────────────────

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("Agent Overlay")
        self.root.configure(bg=BORDER)          # outer 1-px border colour
        self.root.overrideredirect(True)         # no title bar
        self.root.attributes("-topmost", True)   # float above every app
        self.root.attributes("-alpha", 0.97)
        self.root.resizable(False, False)
        self.root.withdraw()                     # hidden until agent starts

        # macOS: promote to floating panel level so it sits above Chrome etc.
        try:
            self.root.tk.call(
                "::tk::unsupported::MacWindowStyle",
                "style", self.root._w,
                "floating", "",
            )
        except Exception:
            pass

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Inner frame — the 1-px gap around it becomes the accent border
        inner = tk.Frame(self.root, bg=BG, padx=13, pady=10)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        self._inner = inner

        # 🤖  Robot icon
        self._icon_lbl = tk.Label(
            inner, text="🤖", bg=BG, font=("", 20), cursor="fleur",
        )
        self._icon_lbl.pack(side="left", padx=(0, 8))

        # Pulse dot — small canvas circle we can recolour
        self._dot_cv = tk.Canvas(
            inner, width=10, height=10, bg=BG,
            highlightthickness=0, cursor="fleur",
        )
        self._dot_cv.pack(side="left", padx=(0, 9), pady=2)
        self._dot = self._dot_cv.create_oval(1, 1, 9, 9, fill=YELLOW, outline="")

        # Status label
        self._label_var = tk.StringVar(value="Agent Working...")
        self._label = tk.Label(
            inner, textvariable=self._label_var,
            bg=BG, fg=TEXT,
            font=("Helvetica Neue", 12, "bold"),
            cursor="fleur",
        )
        self._label.pack(side="left", padx=(0, 14))

        # ✕  Cancel button
        self._cancel_btn = tk.Button(
            inner,
            text=" ✕ ",
            bg=RED, fg="white",
            font=("Helvetica Neue", 10, "bold"),
            bd=0, relief="flat", cursor="hand2",
            activebackground="#dc2626", activeforeground="white",
            command=self._on_cancel,
        )
        self._cancel_btn.pack(side="right")

    # ── Drag ──────────────────────────────────────────────────────────────────

    def _bind_drag(self):
        for w in [self._inner, self._icon_lbl, self._label, self._dot_cv]:
            w.bind("<Button-1>",  self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)

    def _drag_start(self, e):
        self._drag_ox = e.x_root - self.root.winfo_x()
        self._drag_oy = e.y_root - self.root.winfo_y()

    def _drag_move(self, e):
        x = e.x_root - self._drag_ox
        y = e.y_root - self._drag_oy
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w  = self.root.winfo_width()
        h  = self.root.winfo_height()
        x  = max(0, min(sw - w, x))
        y  = max(0, min(sh - h, y))
        self.root.geometry(f"+{x}+{y}")

    # ── Show / hide ───────────────────────────────────────────────────────────

    def _show(self):
        if self._visible:
            return
        self._visible    = True
        self._cancelling = False

        # Reset UI
        self._label_var.set("Agent Working...")
        self._cancel_btn.config(state="normal", bg=RED)
        self._dot_cv.itemconfig(self._dot, fill=YELLOW)
        self.root.configure(bg=BORDER)

        # Position: bottom-right of screen
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        rw = self.root.winfo_reqwidth()
        rh = self.root.winfo_reqheight()
        self.root.geometry(f"+{sw - rw - 40}+{sh - rh - 90}")

        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)   # re-assert after show

    def _hide(self):
        if not self._visible:
            return
        self._visible = False
        self.root.withdraw()

    # ── Cancel ────────────────────────────────────────────────────────────────

    def _on_cancel(self):
        if self._cancelling:
            return
        self._cancelling = True
        self._label_var.set("Cancelling...")
        self._cancel_btn.config(state="disabled", bg=MUTED)
        self.root.configure(bg=RED)
        threading.Thread(target=self._http_cancel, daemon=True).start()

    def _http_cancel(self):
        try:
            req = urllib.request.Request(
                CANCEL_URL, data=b"", method="POST",
            )
            urllib.request.urlopen(req, timeout=4)
        except Exception as e:
            print(f"[overlay] cancel error: {e}")

    # ── Poll /status ──────────────────────────────────────────────────────────

    def _poll(self):
        threading.Thread(target=self._fetch_status, daemon=True).start()
        self.root.after(POLL_MS, self._poll)

    def _fetch_status(self):
        try:
            with urllib.request.urlopen(STATUS_URL, timeout=3) as r:
                data = json.loads(r.read())
            running = bool(data.get("running", False))
            # Schedule UI update on main thread
            self.root.after(0, self._on_status, running)
        except Exception:
            pass  # server not up yet — retry next poll

    def _on_status(self, running: bool):
        if running and not self._visible:
            self._show()
        elif not running and self._visible:
            self._hide()

    # ── Animation ─────────────────────────────────────────────────────────────

    def _tick(self):
        if self._visible and not self._cancelling:
            # Blink dot
            self._pulse_on = not self._pulse_on
            self._dot_cv.itemconfig(
                self._dot,
                fill=YELLOW if self._pulse_on else "#4a3a10",
            )
            # Animate "..." suffix
            self._dot_frame = (self._dot_frame + 1) % 4
            self._label_var.set("Agent Working" + "." * self._dot_frame)
        self.root.after(TICK_MS, self._tick)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    AgentOverlay().run()
