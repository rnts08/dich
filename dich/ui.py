"""
Dịch — Vietnamese translator (Tkinter UI).

Tabs:
  1. Translate  - open a file (pdf/docx/docm/doc/rtf/md/txt) or paste text, see
                  source and translation side by side
  2. Speech     - microphone -> Vietnamese speech-to-text -> English translation
  3. Settings   - model / languages, cache controls, and a debug panel
                  with the Google Translate reference translation

The model loads in-process (dich.backend), so there is no separate server to
start. Run with:  python -m dich
"""

import difflib
import json
import queue
import re
import threading
import time
import tkinter as tk
import traceback
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

from . import backend, cleanup, quality, theme
from .backend import (
    CONFIG_PATH,
    Progress,
    TranslationResult,
    cache_clear,
    cache_stats,
    google_reference_translation,
    health_check,
    translate_text,
)

try:  # speech is optional; the UI explains what to install if it is missing
    from . import speech_engine
except ImportError:  # pragma: no cover
    speech_engine = None

APP_NAME = "Dịch"
APP_TAGLINE = "Vietnamese → English translator"
VERSION = backend.VERSION
SUPPORTED_EXTS = [".txt", ".md", ".markdown", ".pdf", ".docx", ".docm", ".doc", ".rtf"]
FILE_TYPES = [
    ("All supported files", " ".join(f"*{e}" for e in SUPPORTED_EXTS)),
    ("Plain text / Markdown", "*.txt *.md *.markdown"),
    ("PDF", "*.pdf"),
    ("Word", "*.docx *.docm *.doc"),
    ("RTF", "*.rtf"),
    ("All files", "*.*"),
]
AUDIO_TYPES = [
    ("Audio files", "*.mp3 *.wav *.m4a *.ogg *.flac *.aac"),
    ("All files", "*.*"),
]

# Aliases so the widget code reads plainly. Values come from the theme, so a
# palette change happens in one place rather than in both modules.
BG = theme.BASE
SURFACE = theme.SURFACE0
FG = theme.TEXT
SUBTLE = theme.OVERLAY
GREEN = theme.GREEN
RED = theme.RED
YELLOW = theme.YELLOW
BLUE = theme.BLUE
CACHE_LABEL = "loaded from cache"

HOME_URL = "https://github.com/rnts08/dich"
COPYRIGHT = "© 2026 Tim Bergström"
LICENCE_SUMMARY = (
    "Non-commercial use only.\n\n"
    "You may use, copy and modify this software for personal, educational and "
    "other non-commercial purposes. Commercial use, including embedding it in a "
    "paid product or offering it as a paid service, is not permitted without "
    "written permission from the copyright holder.\n\n"
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.'
)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
# The translation pane renders Markdown rather than showing raw markup, so the
# model is asked to emit it. Tkinter's Text widget has no HTML renderer and
# pulling in a webview for one pane is not worth the dependency, so a small
# converter maps Markdown constructs onto Text tags.

MD_TAGS = theme.MD_TAGS

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_FENCE = re.compile(r"^\s*```")
_MD_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_MD_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_MD_QUOTE = re.compile(r"^\s*>\s?(.*)$")
_MD_INLINE = re.compile(r"(\*\*.+?\*\*|__.+?__|\*[^*]+?\*|_[^_]+?_|`[^`]+?`)")
# HTML tags, in case a model returns them rather than Markdown. Showing "<h1>"
# as literal text is how the Markdown-input bug first became visible.
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def _strip_html(text: str) -> str:
    """Remove HTML tags, converting the common ones to their Markdown form first."""
    replacements = (
        (r"<h1[^>]*>", "# "),
        (r"<h2[^>]*>", "## "),
        (r"<h3[^>]*>", "### "),
        (r"</h[1-6]>", "\n"),
        (r"<strong>", "**"),
        (r"</strong>", "**"),
        (r"<b>", "**"),
        (r"</b>", "**"),
        (r"<em>", "*"),
        (r"</em>", "*"),
        (r"<i>", "*"),
        (r"</i>", "*"),
        (r"<code>", "`"),
        (r"</code>", "`"),
        (r"<li[^>]*>", "- "),
        (r"</li>", "\n"),
        (r"<br\s*/?>", "\n"),
        (r"<p[^>]*>", "\n"),
        (r"</p>", "\n"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return _HTML_TAG.sub("", text)


def _plain(text: str) -> str:
    """Collapse Markdown source down to plain text for a single-line insertion."""
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text


def render_markdown(text: str, widget: tk.Text) -> None:
    """
    Render Markdown into a Tk Text widget using tags.

    Falls back to inserting the text verbatim if anything goes wrong, so the
    pane never ends up empty because of a parsing edge case.
    """
    try:
        _render_markdown(text, widget)
    except Exception:  # noqa: BLE001 - the text is more important than the formatting
        widget.insert(tk.END, text)


def _render_markdown(text: str, widget: tk.Text) -> None:
    # A model that returns HTML must not have its tags shown as literal text.
    text = _strip_html(text)
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if _MD_FENCE.match(line):
            i = _render_code_block(lines, i + 1, widget)
            continue

        heading = _MD_HEADING.match(line)
        if heading:
            level = min(len(heading.group(1)), 3)
            _insert_inline(widget, heading.group(2), f"h{level}")
            widget.insert(tk.END, "\n")
            i += 1
            continue

        if line.strip() == "":
            widget.insert(tk.END, "\n")
            i += 1
            continue

        bullet = _MD_BULLET.match(line)
        if bullet:
            _insert_inline(widget, "•  " + bullet.group(1), "list")
            widget.insert(tk.END, "\n")
            i += 1
            continue

        numbered = _MD_NUMBERED.match(line)
        if numbered:
            _insert_inline(widget, numbered.group(1), "list")
            widget.insert(tk.END, "\n")
            i += 1
            continue

        quote = _MD_QUOTE.match(line)
        if quote:
            _insert_inline(widget, "│ " + quote.group(1), "quote")
            widget.insert(tk.END, "\n")
            i += 1
            continue

        _insert_inline(widget, line, None)
        widget.insert(tk.END, "\n")
        i += 1


def _render_code_block(lines: list, start: int, widget: tk.Text) -> int:
    """Render a fenced code block; returns the index after the closing fence."""
    body = []
    i = start
    while i < len(lines) and not _MD_FENCE.match(lines[i]):
        body.append(lines[i])
        i += 1
    if body:
        widget.insert(tk.END, "\n".join(body) + "\n", "code_block")
    return i + 1


def _insert_inline(widget: tk.Text, text: str, block_tag: str | None) -> None:
    """Insert a line, applying inline Markdown emphasis to the matching spans."""
    for piece in _MD_INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            tag = "bold"
            piece = piece[2:-2]
        elif piece.startswith("__") and piece.endswith("__") and len(piece) > 4:
            tag = "bold"
            piece = piece[2:-2]
        elif (
            piece.startswith(("*", "_")) and piece.endswith(piece[0]) and len(piece) > 2
        ):
            tag = "italic"
            piece = piece[1:-1]
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            tag = "code"
            piece = piece[1:-1]
        else:
            tag = None

        if block_tag and tag:
            widget.insert(tk.END, piece, (block_tag, tag))
        else:
            widget.insert(tk.END, piece, tag or block_tag)


# ---------------------------------------------------------------------------
# Thread-safe worker: keeps the UI responsive while translation runs
# ---------------------------------------------------------------------------
class Worker:
    """Runs callables on a background thread and marshals results onto tk main loop."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.queue: "queue.Queue" = queue.Queue()
        self.root.after(40, self._poll)

    def run(self, fn, *args, on_success=None, on_error=None, **kwargs):
        """Run fn on a background thread. Extra kwargs are forwarded to fn."""

        def task():
            try:
                result = fn(*args, **kwargs)
                if on_success:
                    # Bind at creation time: `result` would otherwise be looked
                    # up late, and the except-clause `exc` is cleared after its
                    # block ends, which would make the lambda fail with NameError.
                    self.queue.put(lambda r=result: on_success(r))
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                if on_error:
                    self.queue.put(lambda e=exc, trace=tb: on_error(e, trace))

        threading.Thread(target=task, daemon=True).start()

    def post(self, fn, *args):
        """
        Queue a callable to run on the UI thread.

        Tkinter is not thread-safe: root.after() from a worker thread raises
        "main thread is not in main loop". Progress callbacks therefore come
        through here instead, using the same queue as run()'s results.
        """
        self.queue.put(lambda: fn(*args))

    def _poll(self):
        try:
            while True:
                self.queue.get_nowait()()
        except queue.Empty:
            pass
        finally:
            self.root.after(40, self._poll)


# ---------------------------------------------------------------------------
# Text pane with placeholder support
# ---------------------------------------------------------------------------
class TextPane(ttk.Frame):
    def __init__(
        self,
        master,
        label: str,
        editable: bool = True,
        placeholder: str = "",
        markdown: bool = False,
    ):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.placeholder = placeholder
        self.markdown = markdown

        ttk.Label(self, text=label).grid(
            row=0, column=0, sticky="w", pady=(0, theme.SPACE_1)
        )

        self.text = tk.Text(
            self,
            wrap="word",
            bd=0,
            relief="flat",
            bg=SURFACE,
            fg=FG,
            insertbackground=FG,
            font=(theme.FONT_FAMILY, 12),
            padx=theme.SPACE_3,
            pady=theme.SPACE_2,
            state="normal" if editable else "disabled",
        )
        self.text.grid(row=1, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(self, command=self.text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.text.configure(yscrollcommand=scrollbar.set)

        if markdown:
            self._configure_markdown_tags()

        self.editable = editable
        self._has_placeholder = False
        self._markdown_source = ""
        if not editable and placeholder:
            self._show_placeholder()
            self.text.bind("<FocusIn>", self._on_focus_in)
            self.text.bind("<Button-1>", self._on_focus_in)

    def _configure_markdown_tags(self):
        """Register the formatting tags the Markdown renderer applies."""
        for name, options in MD_TAGS.items():
            self.text.tag_configure(name, **options)

    # -- content ----------------------------------------------------------
    def _show_placeholder(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", self.placeholder, "placeholder")
        self.text.tag_configure("placeholder", foreground=SUBTLE)
        self.text.configure(state="disabled")
        self._has_placeholder = True

    def _on_focus_in(self, _event=None):
        if self._has_placeholder:
            self.text.configure(state="normal")
            self.text.delete("1.0", tk.END)
            self.text.configure(state="disabled")
            self._has_placeholder = False

    @property
    def has_placeholder(self) -> bool:
        return self._has_placeholder

    def set(self, content: str):
        self._markdown_source = content
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        if self.markdown:
            render_markdown(content, self.text)
        else:
            self.text.insert("1.0", content)
        if not self.editable:
            self.text.configure(state="disabled")
        self._has_placeholder = False

    def markdown_source(self) -> str:
        """
        Return the pane content as Markdown source.

        A Markdown pane renders formatted text into the widget, so the raw
        Markdown is kept alongside it: saving, copying and re-translating all
        need the source, not the decorated copy.
        """
        return self._markdown_source

    def get(self) -> str:
        """Return the pane content, treating a visible placeholder as empty."""
        if not self.editable and self._has_placeholder:
            return ""
        return self.text.get("1.0", tk.END).strip()

    def append(self, content: str):
        self.text.configure(state="normal")
        self.text.insert(tk.END, content)
        self.text.see(tk.END)
        if not self.editable:
            self.text.configure(state="disabled")
        self._has_placeholder = False


# ---------------------------------------------------------------------------
# Status footer: progress bar + live throughput
# ---------------------------------------------------------------------------
class StatusBar(ttk.Frame):
    """
    Footer: a progress bar plus live throughput text.

    The bar is determinate throughout. A phase whose length is unknown shows
    0% with a live elapsed-seconds counter rather than a bar that swishes back
    and forth, which told the user nothing at all.
    """

    def __init__(self, master):
        super().__init__(master)
        self.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        # Busy-phase clock state, driven by set_busy / _tick_busy.
        self._busy_started: Optional[float] = None
        self._busy_job = None
        self._busy_label = ""
        self._busy_color = FG

        self.stats = ttk.Label(
            self,
            text="idle",
            anchor="e",
            font=(theme.MONO_FAMILY, 11),
            foreground=SUBTLE,
        )
        self.stats.grid(row=0, column=1, sticky="e", padx=(theme.SPACE_3, 0))

        self.memory = ttk.Label(
            self,
            text="",
            anchor="e",
            font=(theme.MONO_FAMILY, 11),
            foreground=SUBTLE,
        )
        self.memory.grid(row=0, column=2, sticky="e")

        self.hint = ttk.Label(
            self,
            text="",
            anchor="w",
            font=(theme.FONT_FAMILY, 11),
            foreground=SUBTLE,
        )
        self.hint.grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(theme.SPACE_1, 0)
        )

    # -- determinate -----------------------------------------------------
    def set_progress(self, percent: int):
        self._stop_busy_clock()
        self.set_determinate()
        self.progress["value"] = max(0, min(100, percent))

    def update(self, percent: int, stats: str):
        self.set_progress(percent)
        self.stats.configure(text=stats)

    # -- busy ------------------------------------------------------------
    def set_busy(self, label: str, hint: str = "", color: str = FG):
        """
        Mark a phase with no known total.

        The bar stays at 0 and a live elapsed-seconds counter stands in for the
        old swishing bar, so the user can see the phase is still running.
        """
        self.set_determinate()
        self.progress["value"] = 0
        self._busy_label = label
        self._busy_color = color
        self._busy_started = time.time()
        self.stats.configure(text=f"{label} · 0s", foreground=color)
        if hint:
            self.hint.configure(text=hint)
        self._schedule_busy_clock()

    def _schedule_busy_clock(self):
        if self._busy_job is not None:
            try:
                self.after_cancel(self._busy_job)
            except Exception:  # noqa: BLE001 - a stale id must not stop the UI
                pass
        self._busy_job = self.after(1000, self._tick_busy)

    def _tick_busy(self):
        self._busy_job = None
        if self._busy_started is None:
            return
        elapsed = int(time.time() - self._busy_started)
        self.stats.configure(
            text=f"{self._busy_label} · {elapsed}s", foreground=self._busy_color
        )
        self._schedule_busy_clock()

    def _stop_busy_clock(self):
        self._busy_started = None
        if self._busy_job is not None:
            try:
                self.after_cancel(self._busy_job)
            except Exception:  # noqa: BLE001
                pass
            self._busy_job = None

    def set_determinate(self):
        if str(self.progress["mode"]) != "determinate":
            self.progress.configure(mode="determinate", maximum=100)
            self.progress.stop()

    def set_stats(self, stats: str, color: str = FG):
        self._stop_busy_clock()
        self.stats.configure(text=stats, foreground=color)

    def set_hint(self, text: str):
        self.hint.configure(text=text)

    def set_memory(self, text: str):
        """Buffer memory readout, shown beside the throughput figures."""
        self.memory.configure(text=text)

    def reset(self):
        self._stop_busy_clock()
        self.set_determinate()
        self.progress["value"] = 0
        self.stats.configure(text="idle")
        self.set_memory("")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class DichApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        theme.apply(root)
        self.root.title(f"{APP_NAME} {VERSION} · {APP_TAGLINE}")
        self.root.geometry(self._saved_geometry())
        self.root.configure(bg=BG)

        self.worker = Worker(root)
        self.current_file: Path | None = None
        self.recording = False
        self.translator = None
        self.audio_path: Path | None = None
        self._mic_opening = False
        self._memory_polling = False
        self._file_transcribing = False
        self._file_translate_total = 0
        self._last_partial = ""
        # Push-to-talk state: held while the mouse button is down, started marks
        # a recording this button opened (so a release during the mic-open step
        # still stops it).
        self._ptt_held = False
        self._ptt_started = False
        # Idle bookkeeping: models are freed once nothing has been asked of
        # them for a while, so a finished session does not hold gigabytes.
        self._last_activity = time.time()
        self._idle_unloaded = False
        self._pressure_unloaded = False

        self._build_style()
        self._build_menu()
        self._build_widgets()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Load the model immediately on startup if the setting asks for it, so
        # the status turns green without the user having to translate anything.
        if backend.CONFIG.get("load_model_on_startup", True):
            backend.preload_model()
        self.root.after(200, self._check_server)
        self.root.after(60_000, self._idle_tick)

    def _saved_geometry(self) -> str:
        """Remembered window geometry, or a sane default on first run."""
        geometry = str(backend.CONFIG.get("window_geometry") or "").strip()
        return geometry or "1280x820"

    # -- idle model unloading ---------------------------------------------
    def note_activity(self):
        """Reset the idle clock; call whenever work is requested."""
        self._last_activity = time.time()
        self._idle_unloaded = False
        self._pressure_unloaded = False

    def _idle_tick(self):
        """
        Free model RAM after a stretch with nothing to do, or when the system
        is short on unified memory (VRAM on Apple Silicon).

        Both the translation and the speech engines hold gigabytes, and a
        finished session should not keep them. The clock resets on every
        request, so a model in active use is never dropped. Memory pressure is
        checked independently: when RAM is scarce the models are unloaded even
        mid-idle so other apps can use the memory.
        """
        try:
            minutes = float(backend.CONFIG.get("idle_unload_minutes", 15) or 0)
        except (TypeError, ValueError):
            minutes = 15
        idle_for = time.time() - self._last_activity
        pressure = backend.memory_pressure()
        pressure_high = (
            pressure is not None and pressure > backend.MEMORY_PRESSURE_THRESHOLD
        )
        pressure_cleared = pressure is not None and pressure < 0.75
        if pressure_cleared:
            self._pressure_unloaded = False
        if (
            not self.recording
            and not self._file_transcribing
            and (
                (minutes > 0 and not self._idle_unloaded and idle_for > minutes * 60)
                or (pressure_high and not self._pressure_unloaded)
            )
        ):
            names = backend.unload_models()
            if minutes > 0 and not self._idle_unloaded and idle_for > minutes * 60:
                self._idle_unloaded = True
            if pressure_high:
                self._pressure_unloaded = True
            if names:
                reason = "memory pressure" if pressure_high else "idle"
                self.status.set_hint(
                    f"unloaded {len(names)} {reason} model(s) — they reload on next use"
                )
        self.root.after(60_000, self._idle_tick)

    # -- menu -------------------------------------------------------------
    def _build_menu(self):
        """
        Native menu bar.

        macOS draws this in the system menu bar at the top of the screen, which
        is where an app's commands are expected to live. Each entry dispatches
        to the same handler as its toolbar counterpart, so there is exactly one
        path per action.
        """
        bar = tk.Menu(self.root)
        self.root.configure(menu=bar)

        file_menu = tk.Menu(bar, tearoff=False)
        bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Open Document…", accelerator="Cmd+O", command=self.open_file
        )
        file_menu.add_command(label="Open Folder (batch)…", command=self.open_folder)
        file_menu.add_command(
            label="Open Audio…", accelerator="Cmd+Shift+O", command=self.open_audio
        )
        file_menu.add_separator()
        self.recent_menu = tk.Menu(file_menu, tearoff=False)
        file_menu.add_cascade(label="Open Recent", menu=self.recent_menu)
        file_menu.add_separator()
        file_menu.add_command(
            label="Save Translation…",
            accelerator="Cmd+S",
            command=self.save_translation,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Close Window", accelerator="Cmd+W", command=self._on_close
        )

        edit_menu = tk.Menu(bar, tearoff=False)
        bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Translate", command=self.translate_current)
        edit_menu.add_command(
            label="Translate Clipboard",
            accelerator="Cmd+Shift+V",
            command=self.translate_clipboard,
        )
        edit_menu.add_command(label="Refine Speech", command=self.refine_speech)
        edit_menu.add_command(label="Clean up Speech", command=self.cleanup_speech)
        edit_menu.add_command(
            label="Rewrite Transcript (API)…",
            command=lambda: self.rewrite_api("transcript"),
        )
        edit_menu.add_command(
            label="Rewrite Translation (API)…",
            command=lambda: self.rewrite_api("translation"),
        )
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Translation", command=self.clear_translate)
        edit_menu.add_command(label="Clear Speech", command=self.clear_speech)

        view_menu = tk.Menu(bar, tearoff=False)
        bar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(
            label="Translate", command=lambda: self.notebook.select(0)
        )
        view_menu.add_command(label="Speech", command=lambda: self.notebook.select(1))
        view_menu.add_command(label="Settings", command=lambda: self.notebook.select(2))

        help_menu = tk.Menu(bar, tearoff=False)
        bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About Dịch", command=self.show_about)

        self._bind_shortcuts()
        self._rebuild_recent_menu()

    def _bind_shortcuts(self):
        self.root.bind_all("<Command-o>", lambda e: self.open_file())
        self.root.bind_all("<Command-O>", lambda e: self.open_audio())
        self.root.bind_all("<Command-s>", lambda e: self.save_translation())
        self.root.bind_all("<Command-Shift-V>", lambda e: self.translate_clipboard())

    def show_about(self):
        backend_info = (
            f"Model: {backend.MODEL_NAME}\n"
            f"Fallback: {', '.join(backend.CONFIG.get('fallback_models', [])) or 'none'}"
        )
        speech = "installed" if speech_engine else "not installed"
        messagebox.showinfo(
            "About Dịch",
            f"{APP_NAME} {VERSION}\n{APP_TAGLINE}\n\n"
            f"{backend_info}\n"
            f"Speech: {speech}\n\n"
            f"Translations run locally on this Mac.",
        )

    def clear_translate(self):
        self.source_pane.set("")
        self.target_pane.set("")
        self.cache_label.configure(text="")
        self.file_label.configure(text="nothing loaded")
        self.status.reset()

    # -- style ------------------------------------------------------------
    def _build_style(self):
        """ttk defaults live in theme.py; this only sets app-specific styles."""
        ttk.Style(self.root).configure("TPane", background=BG)

    # -- widgets -----------------------------------------------------------
    def _build_widgets(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=theme.SPACE_5, pady=(theme.SPACE_4, theme.SPACE_2))
        header.columnconfigure(2, weight=1)

        # Name, language pair and model status all on one row, so the vertical
        # space goes to the panes instead of a stacked subtitle.
        tk.Label(header, text=APP_NAME, font=theme.FONT_APP, bg=BG, fg=FG).grid(
            row=0, column=0, sticky="w"
        )
        tk.Label(
            header,
            text=f"{APP_TAGLINE} · v{VERSION}",
            font=theme.FONT_SMALL,
            bg=BG,
            fg=SUBTLE,
        ).grid(row=0, column=1, sticky="w", padx=(theme.SPACE_3, 0), pady=(8, 0))

        status_box = tk.Frame(header, bg=BG)
        status_box.grid(row=0, column=3, sticky="e")
        self.health_dot = tk.Label(
            status_box, text="●", bg=BG, foreground=YELLOW, font=theme.FONT_SMALL
        )
        self.health_dot.pack(side="left")
        self.health_text = tk.Label(
            status_box,
            text="checking…",
            bg=BG,
            foreground=SUBTLE,
            font=theme.FONT_SMALL,
        )
        self.health_text.pack(side="left", padx=(4, 0))
        self.model_label = tk.Label(
            status_box, text="", bg=BG, foreground=SUBTLE, font=theme.FONT_SMALL
        )
        self.model_label.pack(side="left", padx=(8, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(
            fill="both",
            expand=True,
            padx=theme.SPACE_3,
            pady=(theme.SPACE_1, theme.SPACE_2),
        )

        self._build_translate_tab(self.notebook)
        self._build_side_by_side_tab(self.notebook)
        self._build_speech_tab(self.notebook)
        self._build_settings_tab(self.notebook)

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=theme.SPACE_5, pady=(0, theme.SPACE_4))

        self.status = StatusBar(footer)
        self.status.pack(fill="x")

    def _build_translate_tab(self, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="  Translate  ")

        toolbar = tk.Frame(tab, bg=BG)
        toolbar.pack(fill="x", padx=theme.SPACE_3, pady=theme.SPACE_3)

        theme.RoundedButton(
            toolbar, "Open File…", command=self.open_file, variant="ghost"
        ).pack(side="left", padx=(0, theme.SPACE_2))
        theme.RoundedButton(
            toolbar, "Translate", command=self.translate_current, variant="primary"
        ).pack(side="left", padx=(0, theme.SPACE_2))
        theme.RoundedButton(
            toolbar, "Save…", command=self.save_translation, variant="ghost"
        ).pack(side="left")
        theme.RoundedButton(
            toolbar,
            "Side by side",
            command=self.show_side_by_side,
            variant="ghost",
            background=theme.SURFACE0,
        ).pack(side="left", padx=(theme.SPACE_3, 0))

        self.cache_label = ttk.Label(
            toolbar, text="", style="Faint.TLabel", foreground=GREEN
        )
        self.cache_label.pack(side="right", padx=(theme.SPACE_3, 0))
        self.file_label = ttk.Label(
            toolbar, text="nothing loaded", style="Faint.TLabel"
        )
        self.file_label.pack(side="right")

        body = tk.Frame(tab, bg=BG)
        body.pack(fill="both", expand=True, padx=theme.SPACE_3, pady=(0, theme.SPACE_3))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.source_pane = TextPane(
            body,
            f"Source — {backend.SOURCE_LANGUAGE}",
            placeholder=f"Paste {backend.SOURCE_LANGUAGE} text here, or open a file…",
        )
        self.source_pane.grid(row=0, column=0, sticky="nsew", padx=(0, theme.SPACE_3))

        self.target_pane = TextPane(
            body,
            f"Translation — {backend.TARGET_LANGUAGE}",
            editable=False,
            placeholder="Translation appears here",
            markdown=True,
        )
        self.target_pane.grid(row=0, column=1, sticky="nsew", padx=(theme.SPACE_3, 0))

    def _build_side_by_side_tab(self, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Side by Side")

        toolbar = tk.Frame(tab, bg=BG)
        toolbar.pack(fill="x", padx=theme.SPACE_3, pady=theme.SPACE_3)
        theme.RoundedButton(
            toolbar,
            "Refresh alignment",
            command=self._refresh_side_by_side,
            variant="ghost",
            background=theme.SURFACE0,
        ).pack(side="left")
        ttk.Label(
            toolbar,
            text="Source paragraphs on the left, translation on the right.",
            style="Faint.TLabel",
        ).pack(side="right")

        body = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True, padx=theme.SPACE_3, pady=(0, theme.SPACE_3))

        self.sbs_source = TextPane(
            body,
            f"Source — {backend.SOURCE_LANGUAGE}",
            editable=False,
            markdown=False,
        )
        self.sbs_target = TextPane(
            body,
            f"Translation — {backend.TARGET_LANGUAGE}",
            editable=False,
            markdown=False,
        )
        body.add(self.sbs_source, weight=1)
        body.add(self.sbs_target, weight=1)

    def show_side_by_side(self):
        """Switch to the side-by-side view and align the current text."""
        self._refresh_side_by_side()
        self.notebook.select(1)  # side-by-side tab sits right after Translate

    def _refresh_side_by_side(self):
        """Re-align the current source and translation paragraph by paragraph."""
        pairs = backend.align_paragraphs(self.source_pane.get(), self.target_pane.get())
        self.sbs_source.set("\n\n".join(s for s, _ in pairs))
        self.sbs_target.set("\n\n".join(t for _, t in pairs))

    def _build_speech_tab(self, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="  Speech  ")

        toolbar = tk.Frame(tab, bg=BG)
        toolbar.pack(fill="x", padx=theme.SPACE_3, pady=theme.SPACE_3)

        # Record is a ttk button rather than a rounded one because it keeps
        # being disabled and relabelled while the mic opens; driving that from
        # the canvas widget would mean more state to keep in sync.
        self.mic_button = ttk.Button(
            toolbar, text="●  Record", command=self.toggle_recording
        )
        self.mic_button.pack(side="left", padx=(0, theme.SPACE_2))
        theme.RoundedButton(
            toolbar, "Open Audio…", command=self.open_audio, variant="ghost"
        ).pack(side="left", padx=(0, theme.SPACE_2))
        theme.RoundedButton(
            toolbar, "Refine", command=self.refine_speech, variant="ghost"
        ).pack(side="left", padx=(0, theme.SPACE_2))
        theme.RoundedButton(
            toolbar, "Clean up", command=self.cleanup_speech, variant="ghost"
        ).pack(side="left", padx=(0, theme.SPACE_2))
        self.rewrite_button = theme.RoundedButton(
            toolbar, "Rewrite ▾", command=self._show_rewrite_menu, variant="ghost"
        )
        self.rewrite_button.pack(side="left", padx=(0, theme.SPACE_2))
        theme.RoundedButton(
            toolbar, "Save…", command=self.save_speech, variant="ghost"
        ).pack(side="left", padx=(0, theme.SPACE_2))
        theme.RoundedButton(
            toolbar, "Clear", command=self.clear_speech, variant="ghost"
        ).pack(side="left", padx=(0, theme.SPACE_2))

        # Push-to-talk: hold for a burst, release to transcribe. A plain ttk
        # button keeps the press/release events simple.
        self.ptt_button = ttk.Button(toolbar, text="Hold to Talk")
        self.ptt_button.pack(side="left", padx=(0, theme.SPACE_3))
        self.ptt_button.bind("<ButtonPress-1>", lambda _e: self._ptt_press())
        self.ptt_button.bind("<ButtonRelease-1>", lambda _e: self._ptt_release())

        # Input-level meter: fills toward full scale, with the silence floor
        # marked, so "it heard nothing" is visible rather than a mystery.
        self.mic_level = ttk.Progressbar(
            toolbar, mode="determinate", maximum=100, length=140
        )
        self.mic_level.pack(side="left", padx=(0, theme.SPACE_2))

        self.mic_status = ttk.Label(toolbar, text="idle", style="Faint.TLabel")
        self.mic_status.pack(side="left", padx=theme.SPACE_4)

        body = tk.Frame(tab, bg=BG)
        body.pack(fill="both", expand=True, padx=theme.SPACE_3, pady=(0, theme.SPACE_3))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.transcript_pane = TextPane(
            body,
            f"Transcript — {backend.SOURCE_LANGUAGE}",
            editable=False,
            placeholder="Transcript appears here",
        )
        self.transcript_pane.grid(
            row=0, column=0, sticky="nsew", padx=(0, theme.SPACE_3)
        )

        self.speech_translation_pane = TextPane(
            body,
            f"Translation — {backend.TARGET_LANGUAGE}",
            editable=False,
            placeholder="Translation appears here",
            markdown=True,
        )
        self.speech_translation_pane.grid(
            row=0, column=1, sticky="nsew", padx=(theme.SPACE_3, 0)
        )

    def _build_settings_tab(self, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="  Settings  ")

        canvas = tk.Canvas(tab, bg=BG, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        form = tk.Frame(canvas, bg=theme.SURFACE0)
        window = canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfigure(window, width=e.width)
        )
        form.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # The input column expands so every control shares one width.
        form.columnconfigure(1, weight=1)

        row = 0

        # --- model -----------------------------------------------------------
        self._section(form, "Model", row)
        row += 1

        self.model_var = tk.StringVar(value=backend.MODEL_NAME)
        self.source_var = tk.StringVar(value=backend.SOURCE_LANGUAGE)
        self.target_var = tk.StringVar(value=backend.TARGET_LANGUAGE)

        ttk.Label(form, text="Model").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        # Model is a picker over what is on disk, not free text: a typo used to
        # start a silent multi-gigabyte download. It stays editable for a new
        # repo, but the size line warns before anything is fetched.
        self.model_combo = ttk.Combobox(
            form,
            textvariable=self.model_var,
            width=54,
            values=self._cached_model_names(),
        )
        self.model_combo.grid(row=row, column=1, sticky="w", pady=theme.SPACE_1)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._describe_model())
        theme.RoundedButton(
            form,
            "Refresh",
            command=self._refresh_models,
            variant="ghost",
            background=theme.SURFACE0,
        ).grid(row=row, column=2, padx=theme.SPACE_2, pady=theme.SPACE_1)
        row += 1

        self.model_size_label = ttk.Label(form, text="", style="Faint.TLabel")
        self.model_size_label.grid(
            row=row, column=1, sticky="w", pady=(0, theme.SPACE_3)
        )
        row += 1

        ttk.Label(form, text="Source").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        ttk.Entry(form, textvariable=self.source_var, width=54).grid(
            row=row, column=1, sticky="w", pady=theme.SPACE_1
        )
        row += 1

        ttk.Label(form, text="Target").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        ttk.Entry(form, textvariable=self.target_var, width=54).grid(
            row=row, column=1, sticky="w", pady=theme.SPACE_1
        )
        row += 1

        self.settings_status = ttk.Label(form, text="", style="Faint.TLabel")
        self.settings_status.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(theme.SPACE_3, theme.SPACE_1),
        )
        row += 1

        theme.RoundedButton(
            form,
            "Save",
            command=self.save_settings,
            variant="primary",
            background=theme.SURFACE0,
        ).grid(
            row=row,
            column=0,
            pady=(theme.SPACE_1, theme.SPACE_3),
            padx=(0, theme.SPACE_2),
        )
        theme.RoundedButton(
            form,
            "Unload models",
            command=self.unload_models,
            variant="ghost",
            background=theme.SURFACE0,
        ).grid(row=row, column=1, pady=(theme.SPACE_1, theme.SPACE_3), sticky="w")
        row += 2

        # --- startup ---------------------------------------------------------
        self._section(form, "Startup", row)
        row += 1

        self.preload_var = tk.BooleanVar(
            value=bool(backend.CONFIG.get("load_model_on_startup", True))
        )
        preload_check = ttk.Checkbutton(
            form,
            text="Load the model when the app starts",
            variable=self.preload_var,
            command=self._on_preload_toggled,
        )
        preload_check.grid(
            row=row, column=0, columnspan=3, sticky="w", pady=theme.SPACE_1
        )
        row += 1

        ttk.Label(
            form,
            text="Turned off, the model loads on the first translation instead "
            "and the status stays red until then.",
            style="Faint.TLabel",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_3))
        row += 2

        # --- microphone ------------------------------------------------------
        self._section(form, "Microphone", row)
        row += 1

        self.device_names: List[str] = []
        self.device_ids: List[Optional[int]] = []
        # Bind the variable before populating, so the dropdown shows the
        # configured device rather than starting blank.
        self.device_var = tk.StringVar()
        self._refresh_devices()

        ttk.Label(form, text="Input device").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        self.device_combo = ttk.Combobox(
            form, textvariable=self.device_var, width=54, state="readonly"
        )
        self.device_combo.grid(row=row, column=1, sticky="w", pady=theme.SPACE_1)
        theme.RoundedButton(
            form,
            "Rescan",
            command=self._refresh_devices,
            variant="ghost",
            background=theme.SURFACE0,
        ).grid(row=row, column=2, padx=theme.SPACE_2, pady=theme.SPACE_1)
        row += 1

        ttk.Label(form, text="Speech model").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        self.mic_model_var = tk.StringVar()
        self.mic_model_labels = [label for _name, label in speech_engine.SPEECH_MODELS]
        self.mic_model_combo = ttk.Combobox(
            form,
            textvariable=self.mic_model_var,
            width=54,
            values=self.mic_model_labels,
            state="readonly",
        )
        self.mic_model_combo.grid(row=row, column=1, sticky="w", pady=theme.SPACE_1)
        self._set_mic_model_label(
            backend.CONFIG.get("mic_model", speech_engine.default_live_model())
        )
        row += 1

        self.push_to_talk_var = tk.BooleanVar(
            value=bool(backend.CONFIG.get("push_to_talk", False))
        )
        ttk.Checkbutton(
            form,
            text="Hold to Talk instead of always-on listening "
            "(mouse button on the Speech tab)",
            variable=self.push_to_talk_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=theme.SPACE_1)
        row += 1

        self.voice_only_var = tk.BooleanVar(
            value=bool(backend.CONFIG.get("voice_only_filter", True))
        )
        ttk.Checkbutton(
            form,
            text="Voice-only filter: ignore windows that are not speech "
            "(fans, keyboards, room tone) so noise cannot become text",
            variable=self.voice_only_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=theme.SPACE_1)
        row += 1

        self.combine_var = tk.BooleanVar(
            value=bool(backend.CONFIG.get("combine_short_utterances", False))
        )
        ttk.Checkbutton(
            form,
            text="Combine consecutive short utterances before translating "
            "(fewer, less fragmented lines)",
            variable=self.combine_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=theme.SPACE_1)
        row += 1

        self.clean_var = tk.BooleanVar(
            value=bool(backend.CONFIG.get("clean_disfluencies", False))
        )
        ttk.Checkbutton(
            form,
            text="Remove stutters, repeated words and filler sounds "
            '("ừm", "ờ") before translating',
            variable=self.clean_var,
            command=self._on_clean_toggled,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=theme.SPACE_1)
        row += 1

        ttk.Label(
            form,
            text="Only runs while recording. Realtime figures measured for "
            "Vietnamese on an Apple M5; above 1× keeps up with a speaker, "
            "below means the transcript lags.",
            style="Faint.TLabel",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_3))
        row += 1

        # --- microphone archive ---------------------------------------------
        self._section(form, "Archive recordings", row)
        row += 1

        self.archive_var = tk.BooleanVar(
            value=bool(backend.CONFIG.get("archive_audio", False))
        )
        ttk.Checkbutton(
            form,
            text="Save raw microphone input while recording, so it can be "
            "re-transcribed or refined later",
            variable=self.archive_var,
            command=self._on_archive_toggled,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=theme.SPACE_1)
        row += 1

        ttk.Label(form, text="Archive folder").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        self.archive_path_var = tk.StringVar(
            value=str(
                backend.CONFIG.get("audio_archive_path") or backend.audio_archive_dir()
            )
        )
        self.archive_path_entry = ttk.Entry(
            form, textvariable=self.archive_path_var, width=54
        )
        self.archive_path_entry.grid(row=row, column=1, sticky="w", pady=theme.SPACE_1)
        theme.RoundedButton(
            form,
            "Browse…",
            command=self._browse_archive_folder,
            variant="ghost",
            background=theme.SURFACE0,
        ).grid(row=row, column=2, padx=theme.SPACE_2, pady=theme.SPACE_1)
        row += 1

        self.archive_storage_label = ttk.Label(form, text="", style="Faint.TLabel")
        self.archive_storage_label.grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_2)
        )
        self._refresh_archive_storage()
        theme.RoundedButton(
            form,
            "Clear all saved recordings",
            command=self.clear_audio_archive,
            variant="danger",
            background=theme.SURFACE0,
        ).grid(row=row, column=0, sticky="w", pady=(0, theme.SPACE_3))
        row += 2

        # --- listening -------------------------------------------------------
        self._section(form, "Listening", row)
        row += 1

        ttk.Label(form, text="Transcription window").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        self.window_var = tk.DoubleVar(
            value=float(
                backend.CONFIG.get(
                    "window_seconds", speech_engine.default_window_seconds()
                )
            )
        )
        ttk.Spinbox(
            form,
            from_=0.5,
            to=30.0,
            increment=0.5,
            textvariable=self.window_var,
            width=6,
        ).grid(row=row, column=1, sticky="w", pady=theme.SPACE_1)
        row += 1

        ttk.Label(
            form,
            text="Seconds of audio per transcription. Short means the first words "
            "appear sooner; each call carries fixed overhead, so short windows "
            "also lower the realtime factor.",
            style="Faint.TLabel",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_2))
        row += 1

        ttk.Label(form, text="Refine every (utterances)").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        self.refine_var = tk.IntVar(value=int(backend.CONFIG.get("refine_every", 6)))
        ttk.Spinbox(
            form,
            from_=2,
            to=40,
            textvariable=self.refine_var,
            width=6,
            command=self._on_refine_every_changed,
        ).grid(row=row, column=1, sticky="w", pady=theme.SPACE_1)
        row += 1

        ttk.Label(
            form,
            text="How many utterances to collect before the whole block is "
            "re-translated as one passage. Lower means earlier correction at "
            "the cost of more model calls.",
            style="Faint.TLabel",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_2))
        row += 1

        ttk.Label(form, text="Buffer ceiling").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        self.buffer_var = tk.IntVar(
            value=int(
                backend.CONFIG.get(
                    "buffer_max_seconds", speech_engine.BUFFER_MAX_SECONDS
                )
            )
        )
        ttk.Spinbox(
            form,
            from_=30,
            to=6000,
            increment=30,
            textvariable=self.buffer_var,
            width=6,
        ).grid(row=row, column=1, sticky="w", pady=theme.SPACE_1)
        row += 1

        ttk.Label(
            form,
            text="Seconds of audio to hold before the oldest is dropped. 600s is "
            "about 38 MB. Raise it to let a slower model catch up, or to keep "
            "a long recording for re-transcribing.",
            style="Faint.TLabel",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_3))
        row += 2

        # --- OCR -------------------------------------------------------------
        self._section(form, "OCR", row)
        row += 1

        self.ocr_var = tk.BooleanVar(
            value=bool(backend.CONFIG.get("ocr_enabled", True))
        )
        ttk.Checkbutton(
            form,
            text="OCR image-based PDFs when they have no text layer "
            "(needs tesseract + poppler; Vietnamese needs its vie lang data)",
            variable=self.ocr_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=theme.SPACE_1)
        row += 2

        # --- external LLM rewrite -------------------------------------------
        self._section(form, "External LLM (optional)", row)
        row += 1

        ttk.Label(
            form,
            text="Send a transcript or translation to an OpenAI-compatible chat "
            "API and have it rewritten to capture meaning without losing "
            "context. Nothing leaves the machine unless this is enabled and a "
            "rewrite is invoked.",
            style="Faint.TLabel",
            wraplength=560,
            justify="left",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_2))
        row += 1

        self.llm_enabled_var = tk.BooleanVar(
            value=bool(backend.CONFIG.get("llm_enabled", False))
        )
        ttk.Checkbutton(
            form,
            text="Enable external LLM rewriting",
            variable=self.llm_enabled_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=theme.SPACE_1)
        row += 1

        ttk.Label(form, text="Base URL").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        self.llm_base_var = tk.StringVar(
            value=str(backend.CONFIG.get("llm_base_url", ""))
        )
        ttk.Entry(form, textvariable=self.llm_base_var, width=54).grid(
            row=row, column=1, columnspan=2, sticky="w", pady=theme.SPACE_1
        )
        row += 1

        ttk.Label(
            form,
            text="e.g. https://api.openai.com/v1, http://localhost:11434/v1 "
            "(Ollama), http://localhost:1234/v1 (LM Studio)",
            style="Faint.TLabel",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_2))
        row += 1

        ttk.Label(form, text="Model").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        self.llm_model_var = tk.StringVar(
            value=str(backend.CONFIG.get("llm_model", ""))
        )
        ttk.Entry(form, textvariable=self.llm_model_var, width=54).grid(
            row=row, column=1, columnspan=2, sticky="w", pady=theme.SPACE_1
        )
        row += 1

        ttk.Label(form, text="API key").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        self.llm_api_var = tk.StringVar(
            value=str(backend.CONFIG.get("llm_api_key", ""))
        )
        ttk.Entry(form, textvariable=self.llm_api_var, width=54, show="•").grid(
            row=row, column=1, columnspan=2, sticky="w", pady=theme.SPACE_1
        )
        row += 1

        ttk.Label(
            form,
            text="Left blank for a local server. Stored in plain text in config.json.",
            style="Faint.TLabel",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_2))
        row += 1

        ttk.Label(form, text="Instruction (optional)").grid(
            row=row, column=0, sticky="nw", pady=theme.SPACE_1
        )
        self.llm_instruction_text = tk.Text(
            form, height=3, width=54, wrap="word", relief="solid", borderwidth=1
        )
        self.llm_instruction_text.insert(
            "1.0", str(backend.CONFIG.get("llm_instruction", "") or "")
        )
        self.llm_instruction_text.grid(
            row=row, column=1, columnspan=2, sticky="w", pady=theme.SPACE_1
        )
        row += 1

        ttk.Label(
            form,
            text="Overrides the built-in edit prompt. Leave empty to use the "
            "default for the transcript or translation.",
            style="Faint.TLabel",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_2))
        row += 1

        self.llm_status = ttk.Label(form, text="", style="Faint.TLabel")
        self.llm_status.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=theme.SPACE_1
        )
        theme.RoundedButton(
            form,
            "Test connection",
            command=self.test_llm,
            variant="ghost",
            background=theme.SURFACE0,
        ).grid(row=row, column=2, padx=theme.SPACE_2, pady=theme.SPACE_1)
        row += 2

        # --- translation cache -----------------------------------------------
        self._section(form, "Translation cache", row)
        row += 1

        self.cache_status = ttk.Label(
            form, text=self._cache_summary(), style="Faint.TLabel"
        )
        self.cache_status.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=theme.SPACE_1
        )
        theme.RoundedButton(
            form,
            "Clear cache",
            command=self.clear_cache,
            variant="danger",
            background=theme.SURFACE0,
        ).grid(row=row, column=2, padx=theme.SPACE_2, pady=theme.SPACE_1)
        row += 2

        # --- debug: google reference -----------------------------------------
        self._section(form, "Debug", row)
        row += 1

        ttk.Label(
            form,
            text="Google Translate reference for verifying output.",
            style="Faint.TLabel",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_2))
        row += 1

        theme.RoundedButton(
            form,
            "Run Google reference",
            command=self.run_reference,
            variant="ghost",
            background=theme.SURFACE0,
        ).grid(row=row, column=0, pady=theme.SPACE_1, sticky="w")
        row += 1

        self.reference_pane = TextPane(
            form,
            "Google Translate reference",
            editable=False,
            placeholder="Reference appears here",
        )
        self.reference_pane.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="nsew",
            pady=(theme.SPACE_2, theme.SPACE_1),
        )
        row += 1

        self.quality_label = ttk.Label(form, text="", style="Faint.TLabel")
        self.quality_label.grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(theme.SPACE_1, theme.SPACE_3),
        )
        theme.RoundedButton(
            form,
            "Score vs reference",
            command=self.score_translation,
            variant="ghost",
            background=theme.SURFACE0,
        ).grid(
            row=row, column=2, padx=theme.SPACE_2, pady=(theme.SPACE_1, theme.SPACE_3)
        )
        row += 1

        theme.RoundedButton(
            form,
            "Diff vs reference",
            command=self.diff_reference,
            variant="ghost",
            background=theme.SURFACE0,
        ).grid(row=row, column=0, pady=theme.SPACE_1, sticky="w")
        row += 1

        self.diff_pane = TextPane(
            form,
            "Reference diff",
            editable=False,
            placeholder="Diff appears here",
        )
        self.diff_pane.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="nsew",
            pady=(theme.SPACE_2, theme.SPACE_1),
        )
        row += 1

        form.rowconfigure(row - 1, weight=1)  # diff pane row expands

        # --- about -----------------------------------------------------------
        self._section(form, "About", row)
        row += 1

        ttk.Label(form, text=f"{APP_NAME} v{VERSION}", style="H2.TLabel").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        row += 1

        ttk.Label(
            form,
            text="A local Vietnamese → English translator.",
            style="Faint.TLabel",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, theme.SPACE_3))
        row += 1

        theme.SectionLabel(form, "© 2026 Tim Bergström").grid(
            row=row, column=0, sticky="w", pady=theme.SPACE_1
        )
        row += 1

        ttk.Label(
            form,
            text="Non-commercial use only. See LICENSE.",
            style="Faint.TLabel",
        ).grid(row=row, column=0, sticky="w", pady=(0, theme.SPACE_3))
        row += 1

        theme.RoundedButton(
            form,
            "Home: github.com/rnts08/dich",
            command=self.open_home,
            variant="ghost",
            background=theme.SURFACE0,
        ).grid(row=row, column=0, sticky="w", pady=(theme.SPACE_1, theme.SPACE_2))
        theme.RoundedButton(
            form,
            "Show licence",
            command=self.show_licence,
            variant="ghost",
            background=theme.SURFACE0,
        ).grid(row=row, column=1, sticky="w", pady=(theme.SPACE_1, theme.SPACE_2))
        row += 2

        form.rowconfigure(row, weight=1)

        self._describe_model()

    # -- helpers ----------------------------------------------------------
    def _section(self, form, title, row):
        """Settings section heading, drawn on the card surface."""
        theme.SectionLabel(form, title).grid(
            row=row, column=0, sticky="w", pady=(theme.SPACE_5, theme.SPACE_2)
        )

    def _cache_summary(self) -> str:
        stats = cache_stats()
        return f"{stats['entries']} saved translation(s) in {stats['dir']}"

    # -- model picker -----------------------------------------------------
    def _cached_model_names(self) -> List[str]:
        """Repos already on disk, plus whatever is configured (never lost)."""
        names = [m["model"] for m in backend.cached_models()]
        current = backend.MODEL_NAME
        if current and current not in names:
            names.insert(0, current)
        return names

    def _refresh_models(self):
        """Re-scan the cache, e.g. after a first download completes."""
        self.model_combo.configure(values=self._cached_model_names())
        self._describe_model()

    def _describe_model(self):
        """
        Say how big the chosen model is, without blocking the UI.

        Cached: size straight off disk. Not cached: a Hub lookup on a worker
        thread, so a slow or offline network cannot freeze the settings tab.
        """
        repo = self.model_var.get().strip()
        if not repo:
            self.model_size_label.configure(text="")
            return
        size = backend.model_size_bytes(repo)
        if size is not None:
            ram = backend.estimated_ram_bytes(repo)
            self.model_size_label.configure(
                text=f"cached on disk: {backend.human_bytes(size)}"
                f" · about {backend.human_bytes(ram)} RAM",
                foreground=FG,
            )
            return
        self.model_size_label.configure(
            text="not cached — checking download size…", foreground=YELLOW
        )
        self.worker.run(
            backend.estimated_download_bytes,
            repo,
            on_success=lambda n: self._show_download_size(repo, n),
            on_error=lambda _e, _tb: self._model_not_cached(repo),
        )

    def _show_download_size(self, repo: str, num_bytes):
        if self.model_var.get().strip() != repo:
            return  # the user moved on while the lookup ran
        if num_bytes:
            self.model_size_label.configure(
                text=f"not cached — downloads about "
                f"{backend.human_bytes(num_bytes)} on first use",
                foreground=YELLOW,
            )
        else:
            self._model_not_cached(repo)

    def _model_not_cached(self, repo: str):
        if self.model_var.get().strip() != repo:
            return
        self.model_size_label.configure(
            text="not cached — will download on first use", foreground=YELLOW
        )

    def unload_models(self):
        """Free the RAM the translation models hold, without quitting."""
        names = backend.unload_models()
        if names:
            self.settings_status.configure(
                text=f"unloaded {len(names)} model(s)", foreground=GREEN
            )
        else:
            self.settings_status.configure(text="nothing was loaded", foreground=FG)

    def _refresh_devices(self):
        """Rebuild the microphone dropdown from what is actually present."""
        # Keep an unsaved choice: rescanning should update the option list, not
        # silently reset the selection the user just made.
        previous = self._selected_device() if hasattr(self, "device_var") else None

        if speech_engine is None:
            self.device_names, self.device_ids = ["(speech deps not installed)"], [None]
        else:
            devices = speech_engine.list_input_devices()
            self.device_ids = [None] + [index for index, _name in devices]
            self.device_names = ["System default"] + [name for _index, name in devices]

        if hasattr(self, "device_var"):
            self.device_var.set(self._device_label(previous))
        if hasattr(self, "device_combo"):
            self.device_combo.configure(values=self.device_names)

    def _device_label(self, index: Optional[int] = None) -> str:
        """Label for a device index, falling back to what config names."""
        if index is None:
            index = backend.CONFIG.get("mic_device_index")
        if index is None:
            return "System default"
        for device_index, name in zip(self.device_ids, self.device_names, strict=False):
            if device_index == index:
                return name
        return f"Device {index} (not present)"

    # -- about ------------------------------------------------------------
    def open_home(self):
        """Open the project home page in the default browser."""
        webbrowser.open(HOME_URL, new=2)

    def show_licence(self):
        messagebox.showinfo(
            f"Licence — {APP_NAME}",
            f"{LICENCE_SUMMARY}\n\n{HOME_URL}",
        )

    # -- buffer memory ----------------------------------------------------
    def _poll_memory(self):
        """
        Refresh the buffer-memory readout.

        The capture buffer is the only unbounded thing in the audio path: the
        device pushes audio in continuously while transcription drains it in
        windows, so if the model falls behind this is where it shows up first.
        A healthy pipeline keeps it near zero — the live mic measures about 1.5
        seconds — and a lagging one grows without limit.

        Never raises: open_audio() runs the first tick synchronously, so a
        readout error here once aborted file transcription before it started.
        """
        if not self._memory_polling:
            return
        try:
            if getattr(self, "_file_transcribing", False):
                # No capture buffer exists in file mode; buffer_usage() is the
                # idle zero dict there, and a steady "buffer 0 B · 0.0s" reads
                # as a stalled pipeline. The phase label is the real readout.
                self.status.set_memory("file mode · no capture buffer")
            elif speech_engine is not None:
                usage = speech_engine.buffer_usage()
                behind = usage.get("behind_by", 0.0)
                dropped = usage.get("dropped_seconds", 0.0)
                windowed = usage.get("window_seconds", 0.0)

                # Be honest about being behind rather than showing a countdown of
                # zero, which reads as "imminent" when nothing is in fact coming.
                if behind > 0.5:
                    mic_line = f"listening… · {behind:.0f}s behind, catching up"
                    bar_line = (
                        f"buffer {_human(usage['bytes'])} · {usage['seconds']:.1f}s "
                        f"· {behind:.0f}s behind"
                    )
                    mic_colour = YELLOW
                else:
                    mic_line = (
                        f"listening… · next transcript in "
                        f"{max(0.0, windowed - usage['seconds']):.1f}s"
                    )
                    bar_line = (
                        f"buffer {_human(usage['bytes'])} · {usage['seconds']:.1f}s"
                    )
                    mic_colour = GREEN

                if dropped > 0:
                    mic_line += f" · {dropped:.0f}s dropped"
                    bar_line += f" · {dropped:.0f}s DROPPED — raise the buffer ceiling"

                self.status.set_memory(bar_line)
                if self.recording:
                    self.mic_status.configure(text=mic_line, foreground=mic_colour)
                    self._show_mic_level(usage.get("peak", 0.0))
        except Exception:  # noqa: BLE001 - a readout must not kill its subject
            traceback.print_exc()
        finally:
            self.root.after(500, self._poll_memory)

    def _start_memory_poll(self):
        self._memory_polling = True
        self._poll_memory()

    def _stop_memory_poll(self):
        self._memory_polling = False
        self.status.set_memory("")

    def _set_mic_model_label(self, model):
        """Show the label for a model id, falling back to the raw id."""
        label = speech_engine.speech_model_label(str(model))
        self.mic_model_var.set(label if label in self.mic_model_labels else str(model))

    def _selected_mic_model(self) -> str:
        """Map the chosen label back to a model id."""
        chosen = self.mic_model_var.get()
        for name, label in speech_engine.SPEECH_MODELS:
            if label == chosen:
                return name
        return str(backend.CONFIG.get("mic_model", speech_engine.default_live_model()))

    def _on_preload_toggled(self):
        """Turning the option on should load the model now, not just persist it."""
        if self.preload_var.get() and not backend.preload_model():
            self.health_dot.configure(foreground=YELLOW)
            self.health_text.configure(text="loading…")
            self.status.set_stats("loading model…", YELLOW)
            self.status.set_hint("started loading — status turns green when ready")
            # Nudge the poll so the state is picked up promptly.
            self.root.after(400, self._check_server)

    def _on_refine_every_changed(self):
        """Apply the new block size to the translator in use, if there is one."""
        if self.translator is not None:
            try:
                self.translator.refine_every = max(1, int(self.refine_var.get()))
            except (TypeError, ValueError):
                pass

    # -- server status ----------------------------------------------------
    def _check_server(self):
        self.health_dot.configure(foreground=YELLOW)
        self.health_text.configure(text="checking…")
        self.worker.run(health_check, on_success=self._on_server_checked)

    def _on_server_checked(self, info: dict):
        model_short = (info.get("model") or "").split("/")[-1]
        self.model_label.configure(text=model_short)

        if info.get("ready"):
            self.health_dot.configure(foreground=GREEN)
            self.health_text.configure(text="ready")
            self.model_label.configure(text=model_short, foreground=GREEN)
            self.status.set_stats(f"{model_short} · ready", GREEN)
            if self.status.hint.cget("text").startswith("Loading"):
                self.status.set_hint("")
            # Nothing is in flight, so a slow poll is enough.
            self.root.after(15000, self._check_server)
            return

        if info.get("loading"):
            self.health_dot.configure(foreground=YELLOW)
            self.health_text.configure(text="loading…")
            self.model_label.configure(text="", foreground=SUBTLE)
            self.status.set_busy(
                f"loading {model_short}",
                hint=f"{model_short} — first load takes a few seconds",
                color=YELLOW,
            )
            # Poll quickly so the dot turns green the moment it is ready.
            self.root.after(600, self._check_server)
            return

        self.health_dot.configure(foreground=RED)
        self.health_text.configure(text="not loaded")
        self.model_label.configure(text="", foreground=SUBTLE)
        self.status.set_stats("not loaded", RED)
        self.status.set_hint(
            f"{model_short} · loads on first translation "
            f"(or enable 'load model on startup' in Settings)"
        )
        self.root.after(15000, self._check_server)

    # -- translate tab ----------------------------------------------------
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open a document to translate", filetypes=FILE_TYPES
        )
        if not path:
            return
        self._load_document(Path(path))

    def _load_document(self, path: Path):
        """Load one document into the source pane and kick off extraction."""
        self.note_activity()
        self.current_file = path
        self.file_label.configure(text=path.name)
        self.cache_label.configure(text="")
        self.source_pane.set("loading…")
        self.worker.run(
            backend.extract_text_from_file,
            str(path),
            on_success=self._on_file_loaded,
            on_error=lambda exc, tb: self._show_error(
                "Could not read that file", exc, tb
            ),
        )

    def open_folder(self):
        """Translate every supported file in a folder, writing each result out."""
        folder = filedialog.askdirectory(
            title="Choose a folder of documents to translate"
        )
        if not folder:
            return
        files = backend.folder_files(folder)
        if not files:
            messagebox.showinfo(
                "Nothing to translate", "No supported documents in that folder."
            )
            return
        self.note_activity()
        self.notebook.select(0)
        self.root.config(cursor="watch")
        self.status.set_busy(f"translating {len(files)} file(s)…", hint=folder)
        self.worker.run(
            backend.translate_folder,
            folder,
            on_file=lambda p, r: self.worker.post(self._on_folder_file, p, r),
            on_success=lambda out: self._on_folder_done(folder, out),
            on_error=self._on_translation_error,
        )

    def _on_folder_file(self, path: str, result: dict):
        name = Path(path).name
        if result.get("ok"):
            self.status.set_stats(f"translated {name}", GREEN)
        else:
            self.status.set_stats(f"failed {name}", RED)

    def _on_folder_done(self, folder: str, out: dict):
        self.root.config(cursor="")
        results = out.get("results", [])
        ok = sum(1 for r in results if r.get("ok"))
        failed = len(results) - ok
        self.status.set_stats(
            f"{ok} translated · {failed} failed", GREEN if not failed else YELLOW
        )
        self.status.set_hint(f"written beside the sources in {folder}")
        self._refresh_models()

    def translate_clipboard(self):
        """Translate whatever text is on the clipboard, without a file."""
        try:
            text = self.root.clipboard_get()
        except tk.TclError:
            self.status.set_hint("clipboard is empty")
            return
        if not text.strip():
            self.status.set_hint("clipboard is empty")
            return
        self.notebook.select(0)
        self.current_file = None
        self.file_label.configure(text="clipboard")
        self.cache_label.configure(text="")
        self.source_pane.set(text)
        self.translate_current()

    # -- recent files -----------------------------------------------------
    def _rebuild_recent_menu(self):
        menu = getattr(self, "recent_menu", None)
        if menu is None:
            return
        menu.delete(0, "end")
        recent = [p for p in backend.CONFIG.get("recent_files", []) if p]
        if not recent:
            menu.add_command(label="(nothing yet)", state="disabled")
            return
        for path in recent:
            menu.add_command(
                label=Path(path).name,
                command=lambda p=path: self._open_recent(p),
            )
        menu.add_separator()
        menu.add_command(label="Clear List", command=self._clear_recent)

    def _open_recent(self, path: str):
        candidate = Path(path)
        if not candidate.exists():
            messagebox.showwarning(
                "File is gone", f"{candidate}\n\nIt moved or was deleted."
            )
            self._forget_recent(path)
            return
        self._load_document(candidate)

    def _remember_recent(self, path: Path):
        entry = str(path)
        recent = [p for p in backend.CONFIG.get("recent_files", []) if p and p != entry]
        recent.insert(0, entry)
        self._save_config_fields({"recent_files": recent[:8], "last_document": entry})
        self._rebuild_recent_menu()

    def _forget_recent(self, path: str):
        recent = [p for p in backend.CONFIG.get("recent_files", []) if p != path]
        self._save_config_fields({"recent_files": recent})
        self._rebuild_recent_menu()

    def _clear_recent(self):
        self._save_config_fields({"recent_files": []})
        self._rebuild_recent_menu()

    def _save_config_fields(self, updates: dict):
        """Merge fields into config.json and the in-memory CONFIG together."""
        data = {}
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        data.update(updates)
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            self._show_error("Could not save settings", exc, "")
            return
        backend.CONFIG.update(updates)

    def _on_file_loaded(self, text: str):
        self.source_pane.set(text)
        if self.current_file is not None:
            self._remember_recent(self.current_file)
        # Show whether this exact text is already translated.
        hit = backend.cache_get(text.strip())
        if hit:
            self.cache_label.configure(text=CACHE_LABEL)

    def translate_current(self):
        text = self.source_pane.get()
        if not text:
            messagebox.showwarning(
                "Nothing to translate", "Open a file or paste some text first."
            )
            return

        self.note_activity()
        self.root.config(cursor="watch")
        self.status.set_busy("translating…")
        self.target_pane.set("")

        def on_progress(p: Progress):
            # Called from the worker thread; marshal onto the UI thread.
            self.worker.post(self._on_progress, p)

        self.worker.run(
            translate_text,
            text,
            on_progress=on_progress,
            on_success=self._on_translation_done,
            on_error=self._on_translation_error,
        )

    def _on_progress(self, p: Progress):
        self.status.update(
            p.percent,
            f"{p.words_translated}/{p.words_total} words  ·  {p.tokens_per_second:.1f} tok/s",
        )

    def _on_translation_done(self, result: TranslationResult):
        self.root.config(cursor="")
        self.target_pane.set(result.text)
        self.status.set_progress(100)
        self.note_activity()

        # A first translation loads the model, so the status may still be
        # showing "not loaded" and would stay red until the next (15s) poll.
        self.root.after(100, self._check_server)

        model_name = (result.model or backend.MODEL_NAME).split("/")[-1]

        if result.cached:
            self.status.set_stats(f"cached · {result.words_out} words", GREEN)
            self.status.set_hint("")
            self.cache_label.configure(text=CACHE_LABEL)
            return

        self.cache_label.configure(text="")
        rate = (
            f" · {result.tokens_per_second:.1f} tok/s"
            if result.tokens_per_second
            else ""
        )
        self.status.set_stats(
            f"{result.words_out} words · {result.elapsed:.1f}s{rate}", FG
        )
        batch_note = f" · {result.batches} batches" if result.batches > 1 else ""
        note = f"{model_name}"
        if result.fell_back:
            note = f"fell back to {model_name}"
        self.status.set_hint(f"{note} · {result.words_in} source words{batch_note}")

    def _on_translation_error(self, exc: Exception, tb: str):
        self.root.config(cursor="")
        self.status.set_stats("failed", RED)
        self._show_error("Translation failed", exc, tb)

    def save_translation(self):
        translation = self.target_pane.markdown_source() or self.target_pane.get()
        if not translation or translation == self.target_pane.placeholder:
            messagebox.showwarning("Nothing to save", "Translate something first.")
            return

        # Default to the format the source was read from, so a `.docx` in is a
        # `.docx` out. A PDF has no faithful container, so its text is `.txt`.
        suffix = (
            backend.export_suffix_for(str(self.current_file))
            if self.current_file
            else ".txt"
        )
        stem = (
            f"{self.current_file.stem}.{backend.TARGET_LANGUAGE.lower()}"
            if self.current_file
            else "dich-translation"
        )
        path = filedialog.asksaveasfilename(
            title="Save translation",
            initialfile=f"{stem}{suffix}",
            defaultextension=suffix,
            filetypes=[
                ("Same format as the source", f"*{suffix}"),
                ("Text files", "*.txt"),
                ("Markdown", "*.md"),
                ("Word document", "*.docx"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            # When the source was a Word file and the target is too, preserve
            # structure (tables, lists, headings) block by block.
            source_path = str(self.current_file) if self.current_file else None
            backend.export_text(translation, path, source_path=source_path)
        except Exception as exc:  # noqa: BLE001 - a save must not crash the app
            self._show_error(
                "Could not save the translation", exc, traceback.format_exc()
            )
            return
        self.status.set_hint(f"saved to {path}")

    # -- speech tab -------------------------------------------------------
    def toggle_recording(self):
        self.stop_recording() if self.recording else self.start_recording()

    def open_audio(self):
        """Transcribe an audio file, then translate it with accumulating context."""
        if self._file_transcribing:
            self.status.set_hint("a file is already being transcribed")
            return
        if speech_engine is None:
            messagebox.showerror(
                "Speech dependencies missing",
                "Install with:\n\npip install sounddevice faster-whisper av",
            )
            return

        path = filedialog.askopenfilename(
            title="Open an audio recording",
            initialdir=str(backend.audio_archive_dir()),
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.ogg *.flac *.aac"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        # Starting a file transcription while the microphone is live would feed
        # both into the same buffer and clobber the live transcript, so stop
        # recording first.
        if self.recording:
            self.stop_recording()

        self.audio_path = Path(path)
        self._file_transcribing = True
        self._file_translate_total = 0
        self.mic_status.configure(text="transcribing…")
        self.status.set_busy(
            "transcribing audio",
            hint=f"{Path(path).name} — speech model may download on first use",
        )
        self._new_translator()
        self.transcript_pane.set("")
        self.speech_translation_pane.set("")
        self._start_memory_poll()

        self.worker.run(
            speech_engine.transcribe_file,
            path,
            speech_engine.FILE_MODEL
            if hasattr(speech_engine, "FILE_MODEL")
            else speech_engine.LIVE_MODEL,
            on_progress=lambda phase, percent: self.worker.post(
                self._on_file_progress, phase, percent
            ),
            on_success=self._on_audio_transcribed,
            on_error=self.on_file_transcription_error,
        )

    def _on_file_progress(self, phase: str, percent):
        """
        A file job's phase, already marshalled onto the UI thread by Worker.post.

        percent is None when the engine has no total to divide by — the model
        load, which on first use is a multi-gigabyte download and must pulse
        rather than show a fake percentage.
        """
        label = f"{phase}…"
        if percent is None:
            self.status.set_busy(f"{phase} (first use may download the model)")
        else:
            self.status.update(percent, f"{phase} {percent}%")
        self.mic_status.configure(text=label)

    def _on_audio_transcribed(self, result):
        self._file_transcribing = False  # the file job's phases are over
        # Normalise once, up front, so the pane, the progress counter and the
        # translator all describe the same list of utterances.
        sentences = [self.translator.normalize(s) for s in result.sentences]
        sentences = [s for s in sentences if s]
        count = len(sentences)
        if not count:
            # Silence is left untranslated on purpose, so "0 sentences" is a
            # correct outcome rather than a failure — but it must be named,
            # otherwise a quiet file is indistinguishable from a broken one.
            self.mic_status.configure(text="no usable speech", foreground=YELLOW)
            self.transcript_pane.set("")
            self._file_translate_total = 0
            self.status.set_stats("0 sentences", YELLOW)
            self.status.set_hint(
                f"No usable speech found in {result.duration:.0f}s of audio. "
                "Silence is deliberately not transcribed — check the recording "
                "has audible voices and try again."
            )
            self.root.after(0, lambda: self._stream_sentences([]))
            return
        self.mic_status.configure(text=f"{count} sentences, {result.duration:.0f}s")
        self.transcript_pane.set("\n".join(sentences))
        self.status.set_stats("translating…")
        # The translation is the second half of a file job: show how far through
        # the sentences — and so the file — the translation has got.
        self._file_translate_total = count
        # Feed the sentences in on the UI thread so the panes update live.
        self.root.after(0, lambda: self._stream_sentences(list(sentences)))

    def _stream_sentences(self, sentences):
        """Translate sentences one at a time, letting context accumulate."""
        if not sentences:
            # Nothing further will arrive, so the trailing partial block gets
            # its own refinement pass instead of staying provisional.
            self.worker.run(
                self.translator.finalize,
                on_success=lambda ok: self._on_finalized(ok),
                on_error=lambda exc, tb: None,
            )
            return
        sentence = sentences.pop(0)
        total = getattr(self, "_file_translate_total", 0)

        def done(_ignored=None):
            self.speech_translation_pane.set(self.translator.render_refined())
            stats = self.translator.stats()
            if total:
                finished = total - len(sentences)
                self.status.update(
                    int(finished / total * 100),
                    f"translating {finished}/{total} — {stats['refined']} refined",
                )
            else:
                self.status.set_stats(
                    f"{stats['refined']} refined · {stats['provisional']} provisional"
                )
            if sentences:
                self.root.after(0, lambda: self._stream_sentences(sentences))
            else:
                self._stream_sentences(sentences)  # triggers finalize

        def failed(exc, tb):
            self.speech_translation_pane.append(f"[failed: {exc}]\n")
            print(tb)
            if sentences:
                self.root.after(0, lambda: self._stream_sentences(sentences))
            else:
                self._stream_sentences(sentences)

        self.worker.run(
            self.translator.add,
            sentence,
            on_success=done,
            on_error=failed,
        )

    def _on_finalized(self, ok):
        self._stop_memory_poll()
        self.speech_translation_pane.set(self.translator.render_refined())
        stats = self.translator.stats()
        if not stats["utterances"]:
            # Nothing was said (or nothing survived the filters); saying "done ·
            # 0 refined" in green would read as a successful translation.
            self.status.set_stats("no speech detected", YELLOW)
        elif ok:
            self.status.set_stats(
                f"done · {stats['refined']} refined of {stats['utterances']}", GREEN
            )
        elif stats["provisional"]:
            self.status.set_stats(
                f"done · {stats['refined']} refined, {stats['provisional']} provisional",
                YELLOW,
            )
        else:
            self.status.set_stats(f"done · {stats['refined']} refined", GREEN)

    def refine_speech(self):
        """Re-translate the pending utterances as one passage."""
        if self.translator is None or not self.translator.transcript:
            messagebox.showinfo("Nothing to refine", "Record or open some audio first.")
            return
        self.status.set_busy("refining…")
        self.worker.run(
            self.translator.refine,
            on_success=lambda ok: self._on_refined(ok),
            on_error=self._on_speech_error,
        )

    def _on_refined(self, ok):
        if ok:
            self.speech_translation_pane.set(self.translator.render_refined())
            stats = self.translator.stats()
            self.status.set_stats(
                f"{stats['refined']} refined of {stats['utterances']}", GREEN
            )
        else:
            self.status.set_stats("refine failed", RED)

    # -- disfluency cleanup & external-LLM rewrite -------------------------
    def cleanup_speech(self):
        """Remove stutters, repetitions and filler sounds, then re-translate."""
        if self.translator is None or not self.translator.transcript:
            messagebox.showinfo("Nothing to clean", "Record or open some audio first.")
            return
        self.status.set_busy("cleaning up…")
        self.worker.run(
            self.translator.cleanup,
            on_success=self._on_cleaned,
            on_error=self._on_speech_error,
        )

    def _on_cleaned(self, stats):
        self.transcript_pane.set("\n\n".join(self.translator.transcript))
        self.speech_translation_pane.set(self.translator.render_refined())
        changed = bool(stats.get("changed"))
        self.status.set_stats(cleanup.summarize(stats), GREEN if changed else FG)
        if changed and not self.translator.transcript:
            self.status.set_hint("every line was filler — the transcript is empty now")

    def _show_rewrite_menu(self):
        """Offer the two rewrite targets, since each keeps its own language."""
        menu = tk.Menu(self.root, tearoff=False)
        menu.add_command(
            label="Rewrite transcript (Vietnamese)…",
            command=lambda: self.rewrite_api("transcript"),
        )
        menu.add_command(
            label="Rewrite translation (English)…",
            command=lambda: self.rewrite_api("translation"),
        )
        try:
            menu.tk_popup(
                self.rewrite_button.winfo_rootx(),
                self.rewrite_button.winfo_rooty() + self.rewrite_button.winfo_height(),
            )
        finally:
            menu.grab_release()

    def _speech_text(self, kind: str) -> str:
        if kind == "transcript":
            return self.transcript_pane.get().strip()
        return (
            self.speech_translation_pane.markdown_source()
            or self.speech_translation_pane.get()
        ).strip()

    def rewrite_api(self, kind: str):
        """Send the transcript or translation to the configured LLM to rewrite."""
        if not backend.llm_available():
            messagebox.showinfo(
                "External LLM not set up",
                "Turn on external LLM rewriting and set a base URL and model in "
                "Settings → External LLM.\n\nNothing is sent until you do.",
            )
            return
        text = self._speech_text(kind)
        if not text:
            messagebox.showinfo(
                "Nothing to rewrite", "Record or open some audio first."
            )
            return
        label = "transcript" if kind == "transcript" else "translation"
        self.status.set_busy(
            f"rewriting {label} via API…", hint="contacting the configured LLM"
        )
        self.worker.run(
            backend.rewrite_with_llm,
            text,
            kind,
            on_success=lambda out: self._on_rewritten(kind, text, out),
            on_error=self._on_rewrite_error,
        )

    def _on_rewritten(self, kind: str, original: str, rewritten: str):
        if not rewritten.strip() or rewritten.strip() == original.strip():
            self.status.set_stats("rewrite made no change")
            return
        self.status.set_stats("rewrite ready", GREEN)
        self._show_rewrite_result(kind, rewritten)

    def _on_rewrite_error(self, exc: Exception, tb: str):
        self.status.set_stats("rewrite failed", RED)
        self.status.set_hint(str(exc))
        print(tb)

    def _show_rewrite_result(self, kind: str, rewritten: str):
        """Show the rewritten text and offer to replace the pane with it."""
        win = tk.Toplevel(self.root)
        win.title("Rewrite result")
        win.configure(bg=BG)
        win.transient(self.root)
        win.geometry("720x460")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(1, weight=1)

        what = (
            "transcript (Vietnamese)"
            if kind == "transcript"
            else "translation (English)"
        )
        ttk.Label(
            win,
            text=f"Rewritten {what} — review before replacing",
            style="Faint.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=theme.SPACE_3, pady=theme.SPACE_3)

        body = tk.Text(win, wrap="word", relief="solid", borderwidth=1)
        body.insert("1.0", rewritten)
        body.configure(state="disabled")
        body.grid(row=1, column=0, sticky="nsew", padx=theme.SPACE_3)

        buttons = tk.Frame(win, bg=BG)
        buttons.grid(
            row=2, column=0, sticky="e", padx=theme.SPACE_3, pady=theme.SPACE_3
        )

        def replace():
            win.destroy()
            if kind == "transcript":
                self._apply_transcript_rewrite(rewritten)
            else:
                self.speech_translation_pane.set(rewritten)
                self.status.set_hint("translation replaced with the rewritten text")

        theme.RoundedButton(
            buttons, "Replace", command=replace, variant="primary"
        ).pack(side="left", padx=theme.SPACE_2)
        theme.RoundedButton(
            buttons,
            "Copy",
            command=lambda: self._copy_text(rewritten),
            variant="ghost",
            background=BG,
        ).pack(side="left", padx=theme.SPACE_2)
        theme.RoundedButton(
            buttons, "Close", command=win.destroy, variant="ghost", background=BG
        ).pack(side="left")

    def _copy_text(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.set_hint("copied to clipboard")

    def _apply_transcript_rewrite(self, text: str):
        """Replace the transcript with the rewritten text and re-translate it."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if self.translator is None:
            self.transcript_pane.set(text)
            return
        self.transcript_pane.set("\n\n".join(lines))
        self.status.set_busy("re-translating the rewritten transcript…")
        self.worker.run(
            self.translator.replace,
            lines,
            on_success=lambda _result: self._on_replaced(),
            on_error=self._on_speech_error,
        )

    def _on_replaced(self):
        self.speech_translation_pane.set(self.translator.render_refined())
        stats = self.translator.stats()
        if stats["utterances"]:
            self.status.set_stats(
                f"{stats['refined']} refined of {stats['utterances']}", GREEN
            )
        else:
            self.status.set_stats("no text left")

    def _new_translator(self):
        self.translator = backend.ContextualTranslator(
            refine_every=int(backend.CONFIG.get("refine_every", 6)),
            combine_short_utterances=bool(
                backend.CONFIG.get("combine_short_utterances", False)
            ),
            clean_disfluencies=bool(backend.CONFIG.get("clean_disfluencies", False)),
        )

    def start_recording(self):
        if self._mic_opening or self.recording:
            return
        if self._file_transcribing:
            self.status.set_hint("a file is being transcribed — wait for it to finish")
            return
        if speech_engine is None:
            messagebox.showerror(
                "Speech dependencies missing",
                "Install with:\n\npip install sounddevice faster-whisper av",
            )
            return
        if not speech_engine.list_input_devices():
            messagebox.showerror(
                "No microphone",
                "No input device was found. Connect a microphone and try again.",
            )
            return

        # Opening the stream blocks until the OS decides whether to grant
        # microphone access, which can be indefinitely long if nothing can show
        # the permission prompt. It therefore runs off the UI thread: a stuck
        # open would otherwise freeze the whole window.
        self.note_activity()
        self._mic_opening = True
        self.mic_button.configure(state="disabled")
        self.mic_status.configure(text="opening…")
        self.status.set_busy(
            "opening microphone",
            hint="granting microphone access in System Settings if this stays here",
            color=YELLOW,
        )
        self.root.config(cursor="watch")

        self.worker.run(
            speech_engine.start_mic_stream,
            model_size=str(
                backend.CONFIG.get("mic_model", speech_engine.default_live_model())
            ),
            device_index=backend.CONFIG.get("mic_device_index"),
            window_seconds=float(
                backend.CONFIG.get(
                    "window_seconds", speech_engine.default_window_seconds()
                )
            ),
            buffer_max_seconds=float(
                backend.CONFIG.get(
                    "buffer_max_seconds", speech_engine.BUFFER_MAX_SECONDS
                )
            ),
            archive_dir=(
                str(backend.audio_archive_dir()) if self.archive_var.get() else None
            ),
            voice_only_filter=bool(self.voice_only_var.get()),
            on_success=self._on_mic_started,
            on_error=self._on_mic_failed,
        )

    def _ptt_press(self):
        self._ptt_held = True
        if not self.recording and not self._mic_opening:
            self._ptt_started = True
            self.start_recording()

    def _ptt_release(self):
        self._ptt_held = False
        if self.recording:
            self.stop_recording()

    def _show_mic_level(self, peak: float):
        """
        Render the input level.

        Speech peaks well under full scale, so the bar is scaled to 0.5 peak:
        a normal speaking voice fills it, and near-zero is visibly silence.
        """
        try:
            self.mic_level["value"] = min(100.0, max(0.0, peak) / 0.5 * 100.0)
        except (TypeError, ValueError):
            self.mic_level["value"] = 0.0

    def _on_mic_started(self, engine):
        self.root.config(cursor="")
        self._mic_opening = False
        self.mic_button.configure(state="normal")
        self.recording = True
        self.mic_button.configure(text="■  Stop")
        self.mic_status.configure(text="listening…", foreground=GREEN)
        self.status.set_stats("listening…", GREEN)
        self._new_translator()
        threading.Thread(target=self._render_speech, daemon=True).start()
        self._start_memory_poll()
        # Push-to-talk released while the stream was still opening: honour it.
        if self._ptt_started and not self._ptt_held:
            self._ptt_started = False
            self.stop_recording()

    def _on_mic_failed(self, exc: Exception, tb: str):
        self.root.config(cursor="")
        self._mic_opening = False
        self._ptt_started = False
        self.mic_button.configure(state="normal")
        self.mic_status.configure(text="failed to open")
        self.status.set_stats("microphone unavailable", RED)
        hint = (
            "macOS needs microphone permission for the app running this. "
            "Check System Settings → Privacy & Security → Microphone."
        )
        messagebox.showerror(
            "Could not start the microphone",
            f"{exc}\n\n{hint}",
        )
        print(tb)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self._ptt_started = False
        self.mic_button.configure(text="●  Record")
        self.mic_status.configure(text="stopped")
        try:
            self.mic_level["value"] = 0
        except Exception:  # noqa: BLE001 - the meter is cosmetic
            pass
        self._stop_memory_poll()
        self.status.reset()
        # The partial line belongs to the recording; drop it so a stale
        # "hearing: …" cannot outlive the session that produced it.
        self._last_partial = ""
        self._show_partial("")
        try:
            from . import speech_engine

            archive_path = speech_engine.stop_mic_stream()
        except ImportError:
            archive_path = None
        if archive_path:
            self._refresh_archive_storage()
            self.status.set_hint(f"recording saved to {archive_path.name}")

    def clear_speech(self):
        """
        Clear both speech panes.

        Keeps the buffer readout while recording, otherwise the countdown
        disappears from under someone who is still listening.
        """
        if not self.recording:
            self._stop_memory_poll()
        self.transcript_pane.set("")
        self.speech_translation_pane.set("")

    def save_speech(self):
        """Save the speech transcript and its translation together."""
        transcript = self.transcript_pane.get().strip()
        translation = (
            self.speech_translation_pane.markdown_source()
            or self.speech_translation_pane.get()
        ).strip()
        if not transcript and not translation:
            messagebox.showwarning(
                "Nothing to save", "Record or open some audio first."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save transcription and translation",
            initialfile="dich-speech.txt",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("Markdown (side by side)", "*.md"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        fmt = "md" if str(path).lower().endswith(".md") else "txt"
        try:
            document = backend.format_speech_export(
                transcript,
                translation,
                source=backend.SOURCE_LANGUAGE,
                target=backend.TARGET_LANGUAGE,
                fmt=fmt,
            )
            Path(path).write_text(document, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - a save must not crash the app
            self._show_error(
                "Could not save the speech output", exc, traceback.format_exc()
            )
            return
        self.status.set_hint(f"saved to {path}")

    def _render_speech(self):
        """
        Pull completed utterances from the speech engine and feed them to the
        contextual translator, so later sentences are translated with the
        earlier ones as context.
        """
        import time

        try:
            from . import speech_engine
        except ImportError:
            return

        while self.recording:
            # The utterance still being assembled, so text appears while the
            # speaker is still talking rather than only at the full stop.
            # Pushed through Worker.post because this loop is on its own thread.
            partial = speech_engine.latest_partial()
            if partial != self._last_partial:
                self._last_partial = partial
                self.worker.post(self._show_partial, partial)
            for sentence in speech_engine.drain_final_sentences():
                # Normalise here as well as in the translator so the pane shows
                # exactly the text that was translated (cleanup is idempotent).
                key = self.translator.normalize(sentence)
                if not key:
                    continue
                # _render_speech runs on its own thread, so the pane update
                # must hop through Worker.post, not root.after directly.
                self.worker.post(lambda s=key: self.transcript_pane.append(s + "\n\n"))
                self.worker.run(
                    self.translator.add,
                    key,
                    on_success=self._on_speech_utterance,
                    on_error=self._on_speech_error,
                )
            time.sleep(0.3)

    def _show_partial(self, text: str):
        """
        Show the in-progress transcript on the hint line.

        The hint line is the one readout the recording loop owns exclusively —
        the memory poll and the translation callbacks never touch it — so a
        partial can be replaced and cleared without clobbering anything else.
        It is a view of the engine's pending sentence, not extra text: when the
        sentence is emitted the engine clears it and so does this.
        """
        self.status.set_hint(f"hearing: {text}" if text else "")

    def _on_speech_utterance(self, _result=None):
        stats = self.translator.stats()
        self.speech_translation_pane.set(self.translator.render_refined())
        self.status.set_stats(
            f"{stats['refined']} refined · {stats['provisional']} provisional"
        )

    def on_file_transcription_error(self, exc: Exception, tb: str):
        """
        transcribe_file() failed — decode, model load/download or transcription.

        Every piece of state open_audio() set must come back down: the memory
        poll it started, the "transcribing…" label, the pulsing busy bar and
        the watch cursor. Leaving any of them up made a single bad file look
        like an eternally running job, and the generic "translation failed"
        text named the wrong stage entirely.
        """
        self._file_transcribing = False
        self._stop_memory_poll()
        self.root.config(cursor="")
        self.mic_status.configure(text="transcription failed", foreground=RED)
        self.status.set_stats("transcription failed", RED)
        self.status.set_hint(str(exc))
        self.speech_translation_pane.append(f"[transcription failed: {exc}]\n\n")
        print(tb)

    def _on_speech_error(self, exc: Exception, tb: str):
        self.speech_translation_pane.append(f"[translation failed: {exc}]\n\n")
        print(tb)

    # -- settings ---------------------------------------------------------
    def _selected_device(self) -> Optional[int]:
        """Map the chosen dropdown label back to a PortAudio device index."""
        label = self.device_var.get()
        for index, name in zip(self.device_ids, self.device_names, strict=False):
            if name == label:
                return index
        return None

    def save_settings(self):
        chosen = self.model_var.get().strip()
        if chosen and not backend.is_model_cached(chosen):
            # A new repo means a first use that fetches gigabytes. Ask first;
            # the size line above already showed the estimate, so this does not
            # need to hit the network again here on the UI thread.
            if not messagebox.askyesno(
                "Download a new model?",
                f"{chosen} is not cached yet.\n\nSaving keeps the choice; the "
                "weights download on the next translation, which can take a "
                "while and several gigabytes. Continue?",
            ):
                return

        data = {}
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}

        data.update(
            {
                "model": self.model_var.get().strip(),
                "source_language": self.source_var.get().strip(),
                "target_language": self.target_var.get().strip(),
                "load_model_on_startup": bool(self.preload_var.get()),
                "mic_device_index": self._selected_device(),
                "mic_model": self._selected_mic_model(),
                "refine_every": int(self.refine_var.get()),
                "window_seconds": float(self.window_var.get()),
                "buffer_max_seconds": int(self.buffer_var.get()),
                "push_to_talk": bool(self.push_to_talk_var.get()),
                "voice_only_filter": bool(self.voice_only_var.get()),
                "combine_short_utterances": bool(self.combine_var.get()),
                "ocr_enabled": bool(self.ocr_var.get()),
                "archive_audio": bool(self.archive_var.get()),
                "audio_archive_path": self.archive_path_var.get().strip(),
                "clean_disfluencies": bool(self.clean_var.get()),
                "llm_enabled": bool(self.llm_enabled_var.get()),
                "llm_base_url": self.llm_base_var.get().strip(),
                "llm_api_key": self.llm_api_var.get().strip(),
                "llm_model": self.llm_model_var.get().strip(),
                "llm_instruction": self.llm_instruction_text.get("1.0", "end").strip(),
            }
        )

        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            self._show_error("Could not save settings", exc, "")
            return

        # A different model may already be resident; drop it so the next run
        # reloads with the new choice instead of silently reusing the old one.
        backend.reload_config()
        self._refresh_models()
        self._refresh_archive_storage()
        self.settings_status.configure(text="Saved — model reloaded", foreground=GREEN)

    def test_llm(self):
        """Prove the configured endpoint answers, using the current fields."""
        base = self.llm_base_var.get().strip()
        model = self.llm_model_var.get().strip()
        if not base or not model:
            self.llm_status.configure(
                text="set a base URL and model first", foreground=YELLOW
            )
            return
        cfg = dict(backend.CONFIG)
        cfg.update(
            {
                "llm_base_url": base,
                "llm_model": model,
                "llm_api_key": self.llm_api_var.get().strip(),
                "llm_instruction": self.llm_instruction_text.get("1.0", "end").strip(),
            }
        )
        self.llm_status.configure(text="testing…", foreground=FG)

        def run():
            from . import llm

            return llm.check(config=cfg)

        self.worker.run(run, on_success=self._on_llm_ok, on_error=self._on_llm_failed)

    def _on_llm_ok(self, reply: str):
        snippet = (reply or "").strip().replace("\n", " ")[:80]
        self.llm_status.configure(text=f"connected — {snippet}", foreground=GREEN)

    def _on_llm_failed(self, exc: Exception, tb: str):
        self.llm_status.configure(text=f"failed: {exc}", foreground=RED)
        print(tb)

    def clear_cache(self):
        count = cache_clear()
        self.cache_status.configure(text=self._cache_summary())
        self.cache_label.configure(text="")
        self.status.set_hint(f"cleared {count} cached translation(s)")

    def _on_clean_toggled(self):
        """Persist the cleanup toggle so the next translator picks it up."""
        backend.CONFIG["clean_disfluencies"] = bool(self.clean_var.get())

    # -- microphone archive -------------------------------------------------
    def _on_archive_toggled(self):
        """Persist the archive toggle so recording picks it up next time."""
        backend.CONFIG["archive_audio"] = bool(self.archive_var.get())

    def _browse_archive_folder(self):
        from tkinter import filedialog

        folder = filedialog.askdirectory(
            title="Choose where recordings are saved",
            initialdir=self.archive_path_var.get() or str(Path.home()),
        )
        if folder:
            self.archive_path_var.set(folder)
            self._refresh_archive_storage()

    def _refresh_archive_storage(self):
        """Show how much storage archived recordings hold."""
        try:
            directory = backend.audio_archive_dir()
            files = backend.archive_files(directory)
            total = backend.archive_storage_bytes(directory)
            if files:
                self.archive_storage_label.configure(
                    text=f"{len(files)} recording(s), {backend.human_bytes(total)}",
                    foreground=FG,
                )
            else:
                self.archive_storage_label.configure(
                    text="no recordings saved yet", foreground=SUBTLE
                )
        except Exception:  # noqa: BLE001 - the readout is cosmetic
            self.archive_storage_label.configure(text="", foreground=FG)

    def clear_audio_archive(self):
        """Delete every saved recording and refresh the readout."""
        count = backend.clear_audio_archive()
        self._refresh_archive_storage()
        self.settings_status.configure(
            text=f"cleared {count} saved recording(s)", foreground=GREEN
        )
        self.status.set_hint(f"cleared {count} saved recording(s)")

    def run_reference(self):
        """Debug: Google Translate reference for the current source text."""
        text = self.source_pane.get()
        if not text:
            messagebox.showwarning(
                "Nothing to compare",
                "Load some source text on the Translate tab first.",
            )
            return
        self.notebook.select(2)  # switch to Settings so the result is visible
        self.reference_pane.set("")
        self.status.set_stats("querying Google…")
        self.worker.run(
            google_reference_translation,
            text,
            on_success=lambda r: (
                self.reference_pane.set(r),
                self.status.set_stats("reference ready", GREEN),
                self._score_if_ready(r),
            ),
            on_error=lambda exc, tb: self._show_error(
                "Google Translate failed", exc, tb
            ),
        )

    def _show_quality(self, result: dict):
        """Show how close the current translation is to the reference."""
        self.quality_label.configure(
            text=f"quality {result['quality']} · BLEU {result['bleu']} · "
            f"chrF {result['chrf']} · ROUGE-L {result['rouge_l']}",
            foreground=GREEN,
        )

    def _score_if_ready(self, reference: str):
        """Score against the reference when a translation already exists."""
        candidate = self.target_pane.get()
        if candidate and candidate != self.target_pane.placeholder:
            self._show_quality(quality.score(reference, candidate))

    def score_translation(self):
        """Debug: score the current translation against the Google reference."""
        reference = self.reference_pane.get()
        candidate = self.target_pane.get()
        if not reference or not candidate or candidate == self.target_pane.placeholder:
            messagebox.showwarning(
                "Nothing to score",
                "Run the Google reference and have a translation first.",
            )
            return
        self._show_quality(quality.score(reference, candidate))

    def diff_reference(self):
        """Debug: show a word-level diff of the translation vs the reference."""
        reference = self.reference_pane.get()
        candidate = self.target_pane.get()
        if not reference or not candidate or candidate == self.target_pane.placeholder:
            messagebox.showwarning(
                "Nothing to diff",
                "Run the Google reference and have a translation first.",
            )
            return
        diff = difflib.ndiff(reference.splitlines(), candidate.splitlines())
        self.diff_pane.set("\n".join(diff))
        self.notebook.select(2)  # switch to Settings so the diff is visible

    # -- misc -------------------------------------------------------------
    def _show_error(self, title: str, exc: Exception, tb: str):
        print(tb or f"{title}: {exc}")
        messagebox.showerror(title, f"{title}:\n\n{exc}")

    def _on_close(self):
        self.recording = False
        try:
            from . import speech_engine

            speech_engine.stop_mic_stream()
        except Exception:  # noqa: BLE001
            pass
        # Remember where the window was, and hand the model RAM back rather
        # than leaving gigabytes pinned by a process that is exiting.
        try:
            self._save_config_fields({"window_geometry": self.root.geometry()})
        except Exception:  # noqa: BLE001 - geometry is a nicety, never fatal
            pass
        try:
            backend.unload_models()
        except Exception:  # noqa: BLE001
            pass
        self.root.destroy()


def _human(num_bytes: int) -> str:
    """Format a byte count for the status bar, e.g. 3.4 MB."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024


def main():
    # Load ~/.config/dich before the app reads any setting. Deferred from
    # import so importing backend never touches the filesystem.
    backend.init_config()
    root = tk.Tk()
    # __init__ wires WM_DELETE_WINDOW to _on_close (stop mic, then destroy).
    app = DichApp(root)

    # Resume the last document on launch, if it still exists. Deferred to
    # idle so the window is shown before the file is read.
    last = str(backend.CONFIG.get("last_document") or "").strip()
    if last:
        root.after(0, lambda: app._load_document(Path(last)))

    root.mainloop()

    # The semaphore tracker warns at shutdown about objects leaked by the audio
    # stack. It is noise to the user, and nothing here can be done about it at
    # this point -- the stream is already closed.
    try:
        import multiprocessing.resource_tracker as rt

        rt.unregister_resource = lambda *a, **k: None
    except Exception:  # noqa: BLE001 - cosmetic
        pass


if __name__ == "__main__":
    main()
