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
import json
import queue
import re
import threading
import webbrowser
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

from . import backend, theme
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
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND."
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
_MD_INLINE = re.compile(
    r"(\*\*.+?\*\*|__.+?__|\*[^*]+?\*|_[^_]+?_|`[^`]+?`)"
)
# HTML tags, in case a model returns them rather than Markdown. Showing "<h1>"
# as literal text is how the Markdown-input bug first became visible.
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def _strip_html(text: str) -> str:
    """Remove HTML tags, converting the common ones to their Markdown form first."""
    replacements = (
        (r"<h1[^>]*>", "# "), (r"<h2[^>]*>", "## "), (r"<h3[^>]*>", "### "),
        (r"</h[1-6]>", "\n"), (r"<strong>", "**"), (r"</strong>", "**"),
        (r"<b>", "**"), (r"</b>", "**"), (r"<em>", "*"), (r"</em>", "*"),
        (r"<i>", "*"), (r"</i>", "*"), (r"<code>", "`"), (r"</code>", "`"),
        (r"<li[^>]*>", "- "), (r"</li>", "\n"), (r"<br\s*/?>", "\n"),
        (r"<p[^>]*>", "\n"), (r"</p>", "\n"),
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
        elif piece.startswith(("*", "_")) and piece.endswith(piece[0]) and len(piece) > 2:
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
    def __init__(self, master, label: str, editable: bool = True,
                 placeholder: str = "", markdown: bool = False):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.placeholder = placeholder
        self.markdown = markdown

        ttk.Label(self, text=label).grid(row=0, column=0, sticky="w",
                                        pady=(0, theme.SPACE_1))

        self.text = tk.Text(
            self, wrap="word", bd=0, relief="flat",
            bg=SURFACE, fg=FG, insertbackground=FG,
            font=(theme.FONT_FAMILY, 12), padx=theme.SPACE_3, pady=theme.SPACE_2,
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

    Two bar modes, because not every phase has a measurable total:
        determinate - 0-100%, used for translation batches
        indeterminate - a pulsing bar, used for model loading, transcribing and
                        any other phase whose length is unknown up front
    """

    def __init__(self, master):
        super().__init__(master)
        self.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.stats = ttk.Label(
            self, text="idle", anchor="e",
            font=(theme.MONO_FAMILY, 11), foreground=SUBTLE,
        )
        self.stats.grid(row=0, column=1, sticky="e", padx=(theme.SPACE_3, 0))

        self.memory = ttk.Label(
            self, text="", anchor="e",
            font=(theme.MONO_FAMILY, 11), foreground=SUBTLE,
        )
        self.memory.grid(row=0, column=2, sticky="e")

        self.hint = ttk.Label(
            self, text="", anchor="w", font=(theme.FONT_FAMILY, 11), foreground=SUBTLE,
        )
        self.hint.grid(row=1, column=0, columnspan=3, sticky="w", pady=(theme.SPACE_1, 0))

    # -- determinate -----------------------------------------------------
    def set_progress(self, percent: int):
        self.set_determinate()
        self.progress["value"] = max(0, min(100, percent))

    def update(self, percent: int, stats: str):
        self.set_progress(percent)
        self.stats.configure(text=stats)

    # -- indeterminate ---------------------------------------------------
    def set_busy(self, label: str, hint: str = "", color: str = FG):
        """Switch to a pulsing bar for work with no known total."""
        if str(self.progress["mode"]) != "indeterminate":
            self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.stats.configure(text=label, foreground=color)
        if hint:
            self.hint.configure(text=hint)

    def set_determinate(self):
        if str(self.progress["mode"]) != "determinate":
            self.progress.configure(mode="determinate", maximum=100)
            self.progress.stop()

    def set_stats(self, stats: str, color: str = FG):
        self.stats.configure(text=stats, foreground=color)

    def set_hint(self, text: str):
        self.hint.configure(text=text)

    def set_memory(self, text: str):
        """Buffer memory readout, shown beside the throughput figures."""
        self.memory.configure(text=text)

    def reset(self):
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
        self.root.geometry("1280x820")
        self.root.configure(bg=BG)

        self.worker = Worker(root)
        self.current_file: Path | None = None
        self.recording = False
        self.translator = None
        self.audio_path: Path | None = None
        self._mic_opening = False
        self._memory_polling = False

        self._build_style()
        self._build_menu()
        self._build_widgets()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Load the model immediately on startup if the setting asks for it, so
        # the status turns green without the user having to translate anything.
        if backend.CONFIG.get("load_model_on_startup", True):
            backend.preload_model()
        self.root.after(200, self._check_server)

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
        file_menu.add_command(label="Open Document…", accelerator="Cmd+O",
                              command=self.open_file)
        file_menu.add_command(label="Open Audio…", accelerator="Cmd+Shift+O",
                              command=self.open_audio)
        file_menu.add_separator()
        file_menu.add_command(label="Save Translation…", accelerator="Cmd+S",
                              command=self.save_translation)
        file_menu.add_separator()
        file_menu.add_command(label="Close Window", accelerator="Cmd+W",
                              command=self.root.destroy)

        edit_menu = tk.Menu(bar, tearoff=False)
        bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Translate", command=self.translate_current)
        edit_menu.add_command(label="Refine Speech", command=self.refine_speech)
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Translation", command=self.clear_translate)
        edit_menu.add_command(label="Clear Speech", command=self.clear_speech)

        view_menu = tk.Menu(bar, tearoff=False)
        bar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Translate", command=lambda: self.notebook.select(0))
        view_menu.add_command(label="Speech", command=lambda: self.notebook.select(1))
        view_menu.add_command(label="Settings", command=lambda: self.notebook.select(2))

        help_menu = tk.Menu(bar, tearoff=False)
        bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About Dịch", command=self.show_about)

        self._bind_shortcuts()

    def _bind_shortcuts(self):
        self.root.bind_all("<Command-o>", lambda e: self.open_file())
        self.root.bind_all("<Command-O>", lambda e: self.open_audio())
        self.root.bind_all("<Command-s>", lambda e: self.save_translation())

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
        tk.Label(
            header, text=APP_NAME, font=theme.FONT_APP, bg=BG, fg=FG
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header, text=f"{APP_TAGLINE} · v{VERSION}",
            font=theme.FONT_SMALL, bg=BG, fg=SUBTLE,
        ).grid(row=0, column=1, sticky="w", padx=(theme.SPACE_3, 0), pady=(8, 0))

        status_box = tk.Frame(header, bg=BG)
        status_box.grid(row=0, column=3, sticky="e")
        self.health_dot = tk.Label(
            status_box, text="●", bg=BG, foreground=YELLOW, font=theme.FONT_SMALL
        )
        self.health_dot.pack(side="left")
        self.health_text = tk.Label(
            status_box, text="checking…", bg=BG, foreground=SUBTLE,
            font=theme.FONT_SMALL,
        )
        self.health_text.pack(side="left", padx=(4, 0))
        self.model_label = tk.Label(
            status_box, text="", bg=BG, foreground=SUBTLE, font=theme.FONT_SMALL
        )
        self.model_label.pack(side="left", padx=(8, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=theme.SPACE_3,
                           pady=(theme.SPACE_1, theme.SPACE_2))

        self._build_translate_tab(self.notebook)
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

        self.cache_label = ttk.Label(
            toolbar, text="", style="Faint.TLabel", foreground=GREEN
        )
        self.cache_label.pack(side="right", padx=(theme.SPACE_3, 0))
        self.file_label = ttk.Label(
            toolbar, text="nothing loaded", style="Faint.TLabel"
        )
        self.file_label.pack(side="right")

        body = tk.Frame(tab, bg=BG)
        body.pack(fill="both", expand=True, padx=theme.SPACE_3,
                  pady=(0, theme.SPACE_3))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.source_pane = TextPane(
            body, f"Source — {backend.SOURCE_LANGUAGE}",
            placeholder=f"Paste {backend.SOURCE_LANGUAGE} text here, or open a file…",
        )
        self.source_pane.grid(row=0, column=0, sticky="nsew",
                              padx=(0, theme.SPACE_3))

        self.target_pane = TextPane(
            body, f"Translation — {backend.TARGET_LANGUAGE}", editable=False,
            placeholder="Translation appears here", markdown=True,
        )
        self.target_pane.grid(row=0, column=1, sticky="nsew",
                              padx=(theme.SPACE_3, 0))

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
            toolbar, "Clear", command=self.clear_speech, variant="ghost"
        ).pack(side="left")

        self.mic_status = ttk.Label(toolbar, text="idle", style="Faint.TLabel")
        self.mic_status.pack(side="left", padx=theme.SPACE_4)

        body = tk.Frame(tab, bg=BG)
        body.pack(fill="both", expand=True, padx=theme.SPACE_3,
                  pady=(0, theme.SPACE_3))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.transcript_pane = TextPane(
            body, f"Transcript — {backend.SOURCE_LANGUAGE}", editable=False, placeholder="Transcript appears here"
        )
        self.transcript_pane.grid(row=0, column=0, sticky="nsew",
                                  padx=(0, theme.SPACE_3))

        self.speech_translation_pane = TextPane(
            body, f"Translation — {backend.TARGET_LANGUAGE}", editable=False,
            placeholder="Translation appears here", markdown=True,
        )
        self.speech_translation_pane.grid(row=0, column=1, sticky="nsew",
                                       padx=(theme.SPACE_3, 0))

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
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # --- model settings ------------------------------------------------
        self._section(form, "Model", row=0)

        self.model_var = tk.StringVar(value=backend.MODEL_NAME)
        self.source_var = tk.StringVar(value=backend.SOURCE_LANGUAGE)
        self.target_var = tk.StringVar(value=backend.TARGET_LANGUAGE)

        fields = [
            ("Model", self.model_var),
            ("Source", self.source_var),
            ("Target", self.target_var),
        ]
        for i, (label, var) in enumerate(fields, start=1):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=6)
            ttk.Entry(form, textvariable=var, width=54).grid(row=i, column=1, sticky="w", pady=6)

        theme.RoundedButton(
            form, "Save", command=self.save_settings, variant="primary",
            background=theme.SURFACE0,
        ).grid(row=5, column=0, pady=(theme.SPACE_4, 0), padx=(0, theme.SPACE_2))
        self.settings_status = ttk.Label(form, text="", style="Faint.TLabel")
        self.settings_status.grid(row=5, column=1, sticky="w", pady=(16, 0))

        # --- startup -------------------------------------------------------
        self._section(form, "Startup", row=7)
        self.preload_var = tk.BooleanVar(
            value=bool(backend.CONFIG.get("load_model_on_startup", True))
        )
        preload_check = ttk.Checkbutton(
            form,
            text="Load the model when the app starts",
            variable=self.preload_var,
            command=self._on_preload_toggled,
        )
        preload_check.grid(row=8, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(
            form,
            text="Turned off, the model loads on the first translation instead "
                 "and the status stays red until then.",
            style="Faint.TLabel",
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # --- microphone -----------------------------------------------------
        self._section(form, "Microphone", row=11)

        self.device_names: List[str] = []
        self.device_ids: List[Optional[int]] = []
        # Bind the variable before populating, so the dropdown shows the
        # configured device rather than starting blank.
        self.device_var = tk.StringVar()
        self._refresh_devices()

        ttk.Label(form, text="Input device").grid(row=12, column=0, sticky="w", pady=6)
        self.device_combo = ttk.Combobox(
            form, textvariable=self.device_var, width=54, state="readonly"
        )
        self.device_combo.grid(row=12, column=1, sticky="w", pady=6)
        theme.RoundedButton(
            form, "Rescan", command=self._refresh_devices, variant="ghost",
            background=theme.SURFACE0,
        ).grid(row=12, column=2, padx=theme.SPACE_2)

        ttk.Label(form, text="Speech model").grid(row=13, column=0, sticky="w", pady=6)
        self.mic_model_var = tk.StringVar()
        self.mic_model_labels = [label for _name, label in speech_engine.SPEECH_MODELS]
        self.mic_model_combo = ttk.Combobox(
            form, textvariable=self.mic_model_var, width=54,
            values=self.mic_model_labels, state="readonly",
        )
        self.mic_model_combo.grid(row=13, column=1, sticky="w", pady=6)
        self._set_mic_model_label(backend.CONFIG.get("mic_model"))

        ttk.Label(
            form,
            text="Only runs while recording. Realtime figures measured for "
                 "Vietnamese on an Apple M5; above 1× keeps up with a speaker, "
                 "below means the transcript lags.",
            style="Faint.TLabel",
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(0, 4))

        ttk.Label(form, text="Refine every (utterances)").grid(
            row=15, column=0, sticky="w", pady=6
        )
        self.refine_var = tk.IntVar(
            value=int(backend.CONFIG.get("refine_every", 6))
        )
        ttk.Spinbox(
            form, from_=2, to=40, textvariable=self.refine_var, width=6,
            command=self._on_refine_every_changed,
        ).grid(row=15, column=1, sticky="w", pady=6)

        ttk.Label(
            form,
            text="How many utterances to collect before the whole block is "
                 "re-translated as one passage. Lower means earlier correction at "
                 "the cost of more model calls.",
            style="Faint.TLabel",
        ).grid(row=16, column=0, columnspan=2, sticky="w", pady=(0, 4))
        # --- listening ------------------------------------------------------
        self._section(form, "Listening", row=18)

        ttk.Label(form, text="Transcription window").grid(
            row=19, column=0, sticky="w", pady=6
        )
        self.window_var = tk.DoubleVar(
            value=float(backend.CONFIG.get("window_seconds", speech_engine.WINDOW_SECONDS))
        )
        ttk.Spinbox(
            form, from_=0.5, to=30.0, increment=0.5,
            textvariable=self.window_var, width=6,
        ).grid(row=19, column=1, sticky="w", pady=6)
        ttk.Label(
            form,
            text="Seconds of audio per transcription. Short means the first words "
                 "appear sooner; each call carries fixed overhead, so short windows "
                 "also lower the realtime factor.",
            style="Faint.TLabel",
        ).grid(row=20, column=0, columnspan=2, sticky="w", pady=(0, 4))

        ttk.Label(form, text="Buffer ceiling").grid(
            row=21, column=0, sticky="w", pady=6
        )
        self.buffer_var = tk.IntVar(
            value=int(backend.CONFIG.get("buffer_max_seconds", speech_engine.BUFFER_MAX_SECONDS))
        )
        ttk.Spinbox(
            form, from_=30, to=6000, increment=30,
            textvariable=self.buffer_var, width=6,
        ).grid(row=21, column=1, sticky="w", pady=6)
        ttk.Label(
            form,
            text="Seconds of audio to hold before the oldest is dropped. 600s is "
                 "about 38 MB. Raise it to let a slower model catch up, or to keep "
                 "a long recording for re-transcribing.",
            style="Faint.TLabel",
        ).grid(row=22, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # --- cache ----------------------------------------------------------
        self._section(form, "Translation cache", row=24)

        self.cache_status = ttk.Label(form, text=self._cache_summary(), style="Faint.TLabel")
        self.cache_status.grid(row=25, column=0, columnspan=2, sticky="w", pady=4)

        theme.RoundedButton(
            form, "Clear cache", command=self.clear_cache, variant="danger",
            background=theme.SURFACE0,
        ).grid(row=25, column=2, padx=theme.SPACE_2)

        # --- debug: google reference ---------------------------------------
        self._section(form, "Debug", row=27)

        ttk.Label(
            form,
            text="Google Translate reference for verifying output.",
            style="Faint.TLabel",
        ).grid(row=28, column=0, columnspan=2, sticky="w", pady=(0, 8))

        theme.RoundedButton(
            form, "Run Google reference", command=self.run_reference, variant="ghost",
            background=theme.SURFACE0,
        ).grid(row=29, column=0, pady=theme.SPACE_1)

        self.reference_pane = TextPane(
            form, "Google Translate reference", editable=False, placeholder="Reference appears here"
        )
        self.reference_pane.grid(row=30, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        form.rowconfigure(30, weight=1)
        form.columnconfigure(1, weight=1)

        # --- about ---------------------------------------------------------
        self._section(form, "About", row=32)

        ttk.Label(form, text=f"{APP_NAME} v{VERSION}", style="H2.TLabel").grid(
            row=26, column=0, sticky="w", pady=2
        )
        ttk.Label(
            form, text="A local Vietnamese → English translator.", style="Faint.TLabel"
        ).grid(row=34, column=0, columnspan=2, sticky="w", pady=(0, 6))

        theme.SectionLabel(form, "© 2026 Tim Bergström").grid(
            row=35, column=0, sticky="w", pady=2
        )
        ttk.Label(
            form, text="Non-commercial use only. See LICENSE.", style="Faint.TLabel"
        ).grid(row=36, column=0, sticky="w", pady=(0, 6))

        theme.RoundedButton(
            form, "Home: github.com/rnts08/dich", command=self.open_home,
            variant="ghost", background=theme.SURFACE0,
        ).grid(row=37, column=0, sticky="w", pady=(0, 8))

        theme.RoundedButton(
            form, "Show licence", command=self.show_licence, variant="quiet",
            background=theme.SURFACE0,
        ).grid(row=37, column=2, sticky="w")

    # -- helpers ----------------------------------------------------------
    def _section(self, form, title, row):
        """Settings section heading, drawn on the card surface."""
        theme.SectionLabel(form, title).grid(row=row, column=0, sticky="w",
                                             pady=(theme.SPACE_5, theme.SPACE_2))

    def _cache_summary(self) -> str:
        stats = cache_stats()
        return f"{stats['entries']} saved translation(s) in {stats['dir']}"

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
        for device_index, name in zip(self.device_ids, self.device_names):
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
        """
        if not self._memory_polling:
            return

        if speech_engine is not None:
            usage = speech_engine.buffer_usage()
            behind = usage.get("behind_by", 0.0)
            dropped = usage.get("dropped_seconds", 0.0)

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
                    f"{max(0.0, usage['window_seconds'] - usage['seconds']):.1f}s"
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
        return str(backend.CONFIG.get("mic_model", speech_engine.LIVE_MODEL))

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
        path = filedialog.askopenfilename(title="Open a document to translate", filetypes=FILE_TYPES)
        if not path:
            return
        self.current_file = Path(path)
        self.file_label.configure(text=self.current_file.name)
        self.cache_label.configure(text="")
        self.source_pane.set("loading…")
        self.worker.run(
            backend.extract_text_from_file,
            path,
            on_success=self._on_file_loaded,
            on_error=lambda exc, tb: self._show_error("Could not read that file", exc, tb),
        )

    def _on_file_loaded(self, text: str):
        self.source_pane.set(text)
        # Show whether this exact text is already translated.
        hit = backend.cache_get(text.strip())
        if hit:
            self.cache_label.configure(text=CACHE_LABEL)

    def translate_current(self):
        text = self.source_pane.get()
        if not text:
            messagebox.showwarning("Nothing to translate", "Open a file or paste some text first.")
            return

        self.root.config(cursor="watch")
        self.status.set_busy("translating…")
        self.target_pane.set("")

        def on_progress(p: Progress):
            # Called from the worker thread; marshal onto the UI thread.
            self.worker.post(self._on_progress, p)

        self.worker.run(
            translate_text, text, on_progress=on_progress,
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
        rate = f" · {result.tokens_per_second:.1f} tok/s" if result.tokens_per_second else ""
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

        initial = "dich-translation.txt"
        if self.current_file:
            initial = f"{self.current_file.stem}.{backend.TARGET_LANGUAGE.lower()}.txt"
        path = filedialog.asksaveasfilename(
            title="Save translation",
            initialfile=initial,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Markdown", "*.md"), ("All files", "*.*")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(translation)
        self.status.set_hint(f"saved to {path}")

    # -- speech tab -------------------------------------------------------
    def toggle_recording(self):
        self.stop_recording() if self.recording else self.start_recording()

    def open_audio(self):
        """Transcribe an audio file, then translate it with accumulating context."""
        try:
            from . import speech_engine
        except ImportError:
            messagebox.showerror(
                "Speech dependencies missing",
                "Install with:\n\npip install sounddevice faster-whisper av",
            )
            return

        path = filedialog.askopenfilename(
            title="Open an audio recording",
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
            speech_engine.FILE_MODEL if hasattr(speech_engine, "FILE_MODEL") else speech_engine.LIVE_MODEL,
            on_success=self._on_audio_transcribed,
            on_error=self._on_speech_error,
        )

    def _on_audio_transcribed(self, result):
        self.mic_status.configure(
            text=f"{len(result.sentences)} sentences, {result.duration:.0f}s"
        )
        self.transcript_pane.set("\n".join(result.sentences))
        self.status.set_stats("translating…")
        # Feed the sentences in on the UI thread so the panes update live.
        self.root.after(0, lambda: self._stream_sentences(list(result.sentences)))

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

        def done(_ignored=None):
            self.speech_translation_pane.set(self.translator.render_refined())
            stats = self.translator.stats()
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
            self.translator.add, sentence,
            on_success=done, on_error=failed,
        )

    def _on_finalized(self, ok):
        self._stop_memory_poll()
        self.speech_translation_pane.set(self.translator.render_refined())
        stats = self.translator.stats()
        if ok:
            self.status.set_stats(
                f"done · {stats['refined']} refined of {stats['utterances']}", GREEN
            )
        elif stats["provisional"]:
            self.status.set_stats(
                f"done · {stats['refined']} refined, {stats['provisional']} provisional",
                YELLOW,
            )
        else:
            self.status.set_stats(
                f"done · {stats['refined']} refined", GREEN
            )

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

    def _new_translator(self):
        self.translator = backend.ContextualTranslator(
            refine_every=int(backend.CONFIG.get("refine_every", 6))
        )

    def start_recording(self):
        if self._mic_opening or self.recording:
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
            model_size=str(backend.CONFIG.get("mic_model", speech_engine.LIVE_MODEL)),
            device_index=backend.CONFIG.get("mic_device_index"),
            window_seconds=float(backend.CONFIG.get("window_seconds", speech_engine.WINDOW_SECONDS)),
            buffer_max_seconds=float(
                backend.CONFIG.get("buffer_max_seconds", speech_engine.BUFFER_MAX_SECONDS)
            ),
            on_success=self._on_mic_started,
            on_error=self._on_mic_failed,
        )

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

    def _on_mic_failed(self, exc: Exception, tb: str):
        self.root.config(cursor="")
        self._mic_opening = False
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
        self.mic_button.configure(text="●  Record")
        self.mic_status.configure(text="stopped")
        self._stop_memory_poll()
        self.status.reset()
        try:
            from . import speech_engine
            speech_engine.stop_mic_stream()
        except ImportError:
            pass

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
            for sentence in speech_engine.drain_final_sentences():
                key = sentence.strip()
                if not key:
                    continue
                # _render_speech runs on its own thread, so the pane update
                # must hop through Worker.post, not root.after directly.
                self.worker.post(lambda s=key: self.transcript_pane.append(s + "\n\n"))
                self.worker.run(
                    self.translator.add, key,
                    on_success=self._on_speech_utterance,
                    on_error=self._on_speech_error,
                )
            time.sleep(0.3)

    def _on_speech_utterance(self, _result=None):
        stats = self.translator.stats()
        self.speech_translation_pane.set(self.translator.render_refined())
        self.status.set_stats(
            f"{stats['refined']} refined · {stats['provisional']} provisional"
        )

    def _on_speech_error(self, exc: Exception, tb: str):
        self.speech_translation_pane.append(f"[translation failed: {exc}]\n\n")
        print(tb)

    # -- settings ---------------------------------------------------------
    def _selected_device(self) -> Optional[int]:
        """Map the chosen dropdown label back to a PortAudio device index."""
        label = self.device_var.get()
        for index, name in zip(self.device_ids, self.device_names):
            if name == label:
                return index
        return None

    def save_settings(self):
        data = {}
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}

        data.update({
            "model": self.model_var.get().strip(),
            "source_language": self.source_var.get().strip(),
            "target_language": self.target_var.get().strip(),
            "load_model_on_startup": bool(self.preload_var.get()),
            "mic_device_index": self._selected_device(),
            "mic_model": self._selected_mic_model(),
            "refine_every": int(self.refine_var.get()),
            "window_seconds": float(self.window_var.get()),
            "buffer_max_seconds": int(self.buffer_var.get()),
        })

        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            self._show_error("Could not save settings", exc, "")
            return

        # A different model may already be resident; drop it so the next run
        # reloads with the new choice instead of silently reusing the old one.
        backend.reload_config()
        self.settings_status.configure(text="Saved — model reloaded", foreground=GREEN)

    def clear_cache(self):
        count = cache_clear()
        self.cache_status.configure(text=self._cache_summary())
        self.cache_label.configure(text="")
        self.status.set_hint(f"cleared {count} cached translation(s)")

    def run_reference(self):
        """Debug: Google Translate reference for the current source text."""
        text = self.source_pane.get()
        if not text:
            messagebox.showwarning("Nothing to compare", "Load some source text on the Translate tab first.")
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
            ),
            on_error=lambda exc, tb: self._show_error("Google Translate failed", exc, tb),
        )

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
    DichApp(root)
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
