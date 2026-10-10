"""
Dịch — Vietnamese translator (Tkinter UI).

Tabs:
  1. Translate  - open a file (pdf/docx/docm/doc/rtf/md/txt) or paste text, see
                  source and translation side by side
  2. Speech     - microphone -> Vietnamese speech-to-text -> English translation
  3. Settings   - model / languages / port, cache controls, and a debug panel
                  with the Google Translate reference translation

The model loads in-process (backend.py), so there is no separate server to start.
Just run:  python ui.py
"""
import json
import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import backend
from backend import (
    CONFIG_PATH,
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    Progress,
    TranslationResult,
    cache_clear,
    cache_stats,
    google_reference_translation,
    is_server_ready,
    translate_text,
)

APP_NAME = "Dịch"
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

BG = "#1e1e2e"
SURFACE = "#313244"
FG = "#cdd6f4"
SUBTLE = "#6c7086"
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
BLUE = "#89b4fa"

CACHE_LABEL = "loaded from cache"


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
    def __init__(self, master, label: str, editable: bool = True, placeholder: str = ""):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.placeholder = placeholder

        ttk.Label(self, text=label).grid(row=0, column=0, sticky="w", pady=(0, 4))

        self.text = tk.Text(
            self, wrap="word", bd=0, relief="flat",
            bg=SURFACE, fg=FG, insertbackground=FG,
            font=("SF Pro Text", 12), padx=10, pady=8,
            state="normal" if editable else "disabled",
        )
        self.text.grid(row=1, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(self, command=self.text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.text.configure(yscrollcommand=scrollbar.set)

        self.editable = editable
        self._has_placeholder = False
        if not editable and placeholder:
            self._show_placeholder()
            self.text.bind("<FocusIn>", self._on_focus_in)
            self.text.bind("<Button-1>", self._on_focus_in)

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
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", content)
        if not self.editable:
            self.text.configure(state="disabled")
        self._has_placeholder = False

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
    def __init__(self, master):
        super().__init__(master)
        self.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.stats = ttk.Label(
            self, text="idle", anchor="e",
            font=("SF Mono", 11), foreground=SUBTLE,
        )
        self.stats.grid(row=0, column=1, sticky="e")

        self.hint = ttk.Label(
            self, text="", anchor="w", font=("SF Pro Text", 11), foreground=SUBTLE,
        )
        self.hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def set_progress(self, percent: int):
        self.progress["value"] = max(0, min(100, percent))

    def reset(self):
        self.progress["value"] = 0
        self.stats.configure(text="idle")

    def update(self, percent: int, stats: str):
        self.set_progress(percent)
        self.stats.configure(text=stats)

    def set_stats(self, stats: str, color: str = FG):
        self.stats.configure(text=stats, foreground=color)

    def set_hint(self, text: str):
        self.hint.configure(text=text)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class DichApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{APP_NAME} {VERSION} · {APP_TAGLINE}")
        self.root.geometry("1280x820")
        self.root.configure(bg=BG)

        self.worker = Worker(root)
        self.current_file: Path | None = None
        self.recording = False

        self._build_style()
        self._build_widgets()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Load the model immediately on startup if the setting asks for it, so
        # the status turns green without the user having to translate anything.
        if backend.CONFIG.get("load_model_on_startup", True):
            backend.preload_model()
        self.root.after(200, self._check_server)

    # -- style ------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG, font=("SF Pro Text", 12))
        style.configure("Header.TLabel", background=BG, foreground=FG, font=("SF Pro Text", 15, "bold"))
        style.configure("App.TLabel", background=BG, foreground=FG, font=("SF Pro Text", 22, "bold"))
        style.configure("Tagline.TLabel", background=BG, foreground=SUBTLE, font=("SF Pro Text", 11))
        style.configure("Status.TLabel", background=BG, foreground=FG, font=("SF Pro Text", 11))
        style.configure("TButton", font=("SF Pro Text", 12), padding=8)
        style.configure("Primary.TButton", font=("SF Pro Text", 12, "bold"), padding=8)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG, foreground=SUBTLE, padding=[18, 8])
        style.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", FG)])
        style.configure("TEntry", fieldbackground=SURFACE, foreground=FG, padding=6)
        style.configure("TCombobox", fieldbackground=SURFACE, foreground=FG, padding=6)
        style.configure("Horizontal.TProgressbar", background=BLUE, troughcolor=SURFACE, borderwidth=0)

    # -- widgets -----------------------------------------------------------
    def _build_widgets(self):
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=20, pady=(16, 8))

        titles = ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles, text=APP_NAME, style="App.TLabel").pack(anchor="w")
        ttk.Label(
            titles, text=f"{APP_TAGLINE} · v{VERSION}", style="Tagline.TLabel"
        ).pack(anchor="w")

        self.health_dot = ttk.Label(header, text="●", style="Status.TLabel", foreground=YELLOW)
        self.health_dot.pack(side="right")
        self.health_text = ttk.Label(header, text="checking…", style="Tagline.TLabel")
        self.health_text.pack(side="right", padx=(0, 6))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(4, 8))

        self._build_translate_tab(self.notebook)
        self._build_speech_tab(self.notebook)
        self._build_settings_tab(self.notebook)

        footer = ttk.Frame(self.root)
        footer.pack(fill="x", padx=20, pady=(0, 16))

        self.status = StatusBar(footer)
        self.status.pack(fill="x")

    def _build_translate_tab(self, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="  Translate  ")

        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", padx=12, pady=10)

        ttk.Button(toolbar, text="Open File…", command=self.open_file).pack(side="left", padx=(0, 8))
        ttk.Button(
            toolbar, text="Translate", style="Primary.TButton", command=self.translate_current
        ).pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Save…", command=self.save_translation).pack(side="left")

        self.file_label = ttk.Label(toolbar, text="nothing loaded", style="Tagline.TLabel")
        self.file_label.pack(side="right")

        self.cache_label = ttk.Label(toolbar, text="", style="Tagline.TLabel", foreground=GREEN)
        self.cache_label.pack(side="right", padx=(0, 14))

        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.source_pane = TextPane(
            body, f"Source — {SOURCE_LANGUAGE}", placeholder=f"Paste {SOURCE_LANGUAGE} text here, or open a file…"
        )
        self.source_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.target_pane = TextPane(
            body, f"Translation — {TARGET_LANGUAGE}", editable=False, placeholder="Translation appears here"
        )
        self.target_pane.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    def _build_speech_tab(self, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="  Speech  ")

        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", padx=12, pady=10)

        self.mic_button = ttk.Button(toolbar, text="●  Record", command=self.toggle_recording)
        self.mic_button.pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Clear", command=self.clear_speech).pack(side="left")

        self.mic_status = ttk.Label(toolbar, text="idle", style="Tagline.TLabel")
        self.mic_status.pack(side="left", padx=(14, 0))

        ttk.Label(
            tab,
            text=f"Speak {SOURCE_LANGUAGE}; the transcript fills in on the left and its "
                 f"{TARGET_LANGUAGE} translation on the right.",
            style="Tagline.TLabel",
        ).pack(fill="x", padx=12)

        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True, padx=12, pady=10)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.transcript_pane = TextPane(
            body, f"Transcript — {SOURCE_LANGUAGE}", editable=False, placeholder="Transcript appears here"
        )
        self.transcript_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.speech_translation_pane = TextPane(
            body, f"Translation — {TARGET_LANGUAGE}", editable=False, placeholder="Translation appears here"
        )
        self.speech_translation_pane.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    def _build_settings_tab(self, notebook: ttk.Notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="  Settings  ")

        canvas = tk.Canvas(tab, bg=BG, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        form = ttk.Frame(canvas, padding=24)
        window = canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # --- model settings ------------------------------------------------
        ttk.Label(form, text="Model", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.model_var = tk.StringVar(value=backend.MODEL_NAME)
        self.source_var = tk.StringVar(value=SOURCE_LANGUAGE)
        self.target_var = tk.StringVar(value=TARGET_LANGUAGE)

        fields = [
            ("Model", self.model_var),
            ("Source", self.source_var),
            ("Target", self.target_var),
        ]
        for i, (label, var) in enumerate(fields, start=1):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=6)
            ttk.Entry(form, textvariable=var, width=54).grid(row=i, column=1, sticky="w", pady=6)

        ttk.Button(form, text="Save", command=self.save_settings).grid(row=5, column=0, pady=(16, 0), padx=(0, 8))
        self.settings_status = ttk.Label(form, text="", style="Tagline.TLabel")
        self.settings_status.grid(row=5, column=1, sticky="w", pady=(16, 0))

        # --- startup -------------------------------------------------------
        ttk.Label(form, text="Startup", style="Header.TLabel").grid(
            row=7, column=0, sticky="w", pady=(24, 10)
        )

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
            style="Tagline.TLabel",
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # --- cache ----------------------------------------------------------
        ttk.Label(form, text="Translation cache", style="Header.TLabel").grid(
            row=11, column=0, sticky="w", pady=(24, 10)
        )

        self.cache_status = ttk.Label(form, text=self._cache_summary(), style="Tagline.TLabel")
        self.cache_status.grid(row=12, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Button(form, text="Clear cache", command=self.clear_cache).grid(row=13, column=0, pady=4)

        # --- debug: google reference ---------------------------------------
        ttk.Label(form, text="Debug", style="Header.TLabel").grid(
            row=15, column=0, sticky="w", pady=(24, 10)
        )

        ttk.Label(
            form,
            text="Google Translate reference for verifying output.",
            style="Tagline.TLabel",
        ).grid(row=16, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Button(
            form, text="Run Google reference", command=self.run_reference
        ).grid(row=17, column=0, pady=4)

        self.reference_pane = TextPane(
            form, "Google Translate reference", editable=False, placeholder="Reference appears here"
        )
        self.reference_pane.grid(row=18, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        form.rowconfigure(18, weight=1)
        form.columnconfigure(1, weight=1)

    # -- helpers ----------------------------------------------------------
    def _cache_summary(self) -> str:
        stats = cache_stats()
        return f"{stats['entries']} saved translation(s) in {stats['dir']}"

    def _on_preload_toggled(self):
        """Turning the option on should load the model now, not just persist it."""
        if self.preload_var.get() and not backend.preload_model():
            self.health_dot.configure(foreground=YELLOW)
            self.health_text.configure(text="loading…")
            self.status.set_stats("loading model…", YELLOW)
            self.status.set_hint("started loading — status turns green when ready")
            # Nudge the poll so the state is picked up promptly.
            self.root.after(400, self._check_server)

    # -- server status ----------------------------------------------------
    def _check_server(self):
        self.health_dot.configure(foreground=YELLOW)
        self.health_text.configure(text="checking…")
        self.worker.run(is_server_ready, on_success=self._on_server_checked)

    def _on_server_checked(self, info: dict):
        if info.get("ready"):
            self.health_dot.configure(foreground=GREEN)
            self.health_text.configure(text="ready")
            self.status.set_stats("ready", GREEN)
            if self.status.hint.cget("text").startswith("Loading"):
                self.status.set_hint("")
            # Nothing is in flight, so a slow poll is enough.
            self.root.after(15000, self._check_server)
            return

        if info.get("loading"):
            self.health_dot.configure(foreground=YELLOW)
            self.health_text.configure(text="loading…")
            self.status.set_stats("loading model…", YELLOW)
            self.status.set_hint(f"{info.get('model', '').split('/')[-1]} — first load takes a few seconds")
            # Poll quickly so the dot turns green the moment it is ready.
            self.root.after(600, self._check_server)
            return

        self.health_dot.configure(foreground=RED)
        self.health_text.configure(text="not loaded")
        self.status.set_stats("not loaded", RED)
        self.status.set_hint(
            f"{info.get('model', '')} · loads on first translation "
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
        self.status.set_stats("translating…")
        self.status.set_progress(0)
        self.status.set_hint("")
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
        translation = self.target_pane.get()
        if not translation or translation == self.target_pane.placeholder:
            messagebox.showwarning("Nothing to save", "Translate something first.")
            return

        initial = "dich-translation.txt"
        if self.current_file:
            initial = f"{self.current_file.stem}.{TARGET_LANGUAGE.lower()}.txt"
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

    def start_recording(self):
        try:
            import speech_engine
        except ImportError:
            messagebox.showerror(
                "Speech dependencies missing",
                "Install them with:\n\npip install sounddevice faster-whisper",
            )
            return
        try:
            speech_engine.start_mic_stream()
        except Exception as exc:  # noqa: BLE001
            self._show_error("Could not start the microphone", exc, traceback.format_exc())
            return

        self.recording = True
        self.mic_button.configure(text="■  Stop")
        self.mic_status.configure(text="listening…", foreground=GREEN)
        self.status.set_stats("listening…", GREEN)
        threading.Thread(target=self._render_speech, daemon=True).start()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.mic_button.configure(text="●  Record")
        self.mic_status.configure(text="stopped")
        self.status.reset()
        try:
            import speech_engine
            speech_engine.stop_mic_stream()
        except ImportError:
            pass

    def clear_speech(self):
        self.transcript_pane.set("")
        self.speech_translation_pane.set("")

    def _render_speech(self):
        """
        Pull completed sentences from the speech engine, show them, and
        translate each one as a whole unit.
        """
        import time
        try:
            import speech_engine
        except ImportError:
            return

        seen = set()
        while self.recording:
            for sentence in speech_engine.drain_final_sentences():
                key = sentence.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                self.root.after(0, lambda s=key: self.transcript_pane.append(s + "\n\n"))
                self.worker.run(
                    translate_text,
                    key,
                    on_success=self._on_speech_translated,
                    on_error=self._on_speech_error,
                )
            time.sleep(0.3)

    def _on_speech_translated(self, result: TranslationResult):
        self.speech_translation_pane.append(result.text.strip() + "\n\n")

    def _on_speech_error(self, exc: Exception, tb: str):
        self.speech_translation_pane.append(f"[translation failed: {exc}]\n\n")
        print(tb)

    # -- settings ---------------------------------------------------------
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
            import speech_engine
            speech_engine.stop_mic_stream()
        except Exception:  # noqa: BLE001
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    DichApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
