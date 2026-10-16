"""
Backend for the translation tool.
Loads models in-process with mlx_lm and renders each model's own chat template.

NOTE ON MODELS:
    mlx-lm only supports causal language models (llama, qwen, gemma, plamo2, ...).
    Encoder-decoder models such as VTSNLP/trans_model_vi_en (T5) cannot be loaded:
    "ValueError: Model type t5 not supported."
    Dedicated MLX translation models (plamo-2-translate) are English<->Japanese only.
    Models are configured in config.json, with fallback_models tried in order
    when the primary model fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

logging.basicConfig(
    level=os.environ.get("DICH_LOG_LEVEL", "WARNING").upper(),
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)
# HuggingFace and faster-whisper chat at INFO on every request and every file,
# which buries the app's own output. Quiet unless explicitly asked for.
for noisy in ("httpx", "httpcore", "huggingface_hub", "faster_whisper", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Locations - everything the app owns lives under ~/.config/dich
# ---------------------------------------------------------------------------
# The model cache is redirected before huggingface_hub is first imported, which
# is why it happens at module import rather than in a function: the library
# reads these variables once at import time. Keeping models here puts every
# weight in one place instead of splitting them between HuggingFace's default
# cache and the working directory.
CONFIG_ROOT = Path(os.environ.get("DICH_HOME", Path.home() / ".config" / "dich"))
MODELS_DIR = CONFIG_ROOT / "models"
CONFIG_PATH = CONFIG_ROOT / "config.json"

# PyInstaller freezes the app into a temp extraction dir where __file__ points
# at the packed module, not the source tree. Bundled data files (VERSION, the
# seed config.json) sit in the dich/ dir under sys._MEIPASS there and beside
# the source in a normal checkout, so every data lookup goes through
# resource_path().
IS_FROZEN = bool(getattr(sys, "frozen", False))


def resource_path(name: str) -> Path:
    """
    Resolve a file bundled with the app, frozen or not.

    The package keeps its name inside the frozen extraction dir, so data
    collected as package data lands in <_MEIPASS>/dich/ exactly where it
    sits (dich/) in a checkout.
    """
    base = (Path(sys._MEIPASS) / "dich") if IS_FROZEN else Path(__file__).parent
    return base / name


LEGACY_CONFIG_PATH = resource_path("config.json")

os.environ.setdefault("HF_HOME", str(CONFIG_ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))
# HF prints a tqdm bar for its "is the snapshot current" check on every load,
# which is not logging and therefore not covered by the level below.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _migrate_legacy_config() -> None:
    """
    Copy the in-repo config.json to ~/.config/dich on first run.

    Without this, an install that already has a working config.json beside the
    code would silently fall back to defaults and lose the chosen model.
    """
    if CONFIG_PATH.exists() or not LEGACY_CONFIG_PATH.exists():
        return
    try:
        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LEGACY_CONFIG_PATH, CONFIG_PATH)
        logger.info(f"Migrated config to {CONFIG_PATH}")
    except OSError as e:
        logger.warning(f"Could not migrate config to {CONFIG_PATH}: {e}")


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "mlx-community/translategemma-4b-it-4bit",
    "source_language": "Vietnamese",
    "target_language": "English",
    "load_model_on_startup": True,
    "fallback_models": ["mlx-community/Qwen3-4B-Instruct-2507-4bit"],
    # --- microphone ---
    "mic_device_index": None,  # null = the system default microphone
    "mic_model": "small",  # keeps up with a live speaker at 2s windows
    # auto -> GPU (mlx) when mlx-audio can reach the GPU, else CPU. Measured on
    # this machine: large-v3-turbo is 3.1x realtime on the CPU and 31x on the
    # GPU, so "auto" removes the CPU's accuracy/latency tradeoff.
    "stt_backend": "auto",
    "refine_every": 6,  # utterances between block re-translations
    # --- app lifecycle (Milestone 0.5) ---
    "window_geometry": "",  # "WxH+X+Y", remembered between runs
    "recent_files": [],  # most-recent first, capped on write
    "idle_unload_minutes": 15,  # free model RAM after this long idle; 0 = never
    "glossary": {},  # source term -> preferred translation, kept verbatim
    # --- speech behaviour (Milestone 0.2.5) ---
    "push_to_talk": False,  # hold to talk instead of always-on listening
    "combine_short_utterances": False,  # merge short segments before translating
    "ocr_enabled": True,  # OCR image-based PDFs via tesseract when pypdf yields nothing
    # --- microphone archive (0.2.5) -------------------------------------
    "archive_audio": False,  # save raw mic input for later re-transcription
    "audio_archive_path": "",  # defaults to <config>/audio when empty
    "last_document": "",  # path reopened on launch
}


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load config.json, merged over defaults.

    Reads from ~/.config/dich/config.json, migrating an in-repo copy on first
    run so an existing configuration is not thrown away.
    """
    config = dict(DEFAULT_CONFIG)
    if path is None:
        _migrate_legacy_config()
        path = CONFIG_PATH
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                config.update(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read {path}: {e}. Using defaults.")
    return config


def ensure_dirs() -> None:
    """Create ~/.config/dich and its models directory."""
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"Could not create {MODELS_DIR}: {e}")


VERSION_FILE = resource_path("VERSION")


def get_version() -> str:
    """
    Read the app version from the VERSION file.

    Returns "dev" when the file is missing or unparseable, so a fresh checkout
    still reports something useful rather than crashing the UI.
    """
    try:
        text = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "dev"
    if re.fullmatch(r"\d+\.\d+\.\d+", text):
        return text
    return "dev"


VERSION = get_version()


# Importing this module must not touch the filesystem: tests, embedding and
# PyInstaller all care, and a read-only checkout must still import. The
# globals start at the documented defaults; init_config() brings them up to
# date from disk on first real use.
CONFIG: Dict[str, Any] = dict(DEFAULT_CONFIG)
MODEL_NAME: str = CONFIG["model"]
SOURCE_LANGUAGE: str = CONFIG["source_language"]
TARGET_LANGUAGE: str = CONFIG["target_language"]
_CONFIG_LOADED = False


def init_config() -> Dict[str, Any]:
    """
    Create the config directory if needed, load config.json, refresh globals.

    Idempotent and a flag-check after the first call. ui.py calls it at
    startup; the public translation entry points call it defensively so a
    headless user of backend.py gets the configured model without having to
    remember.
    """
    global _CONFIG_LOADED, CONFIG, MODEL_NAME, SOURCE_LANGUAGE, TARGET_LANGUAGE
    if _CONFIG_LOADED:
        return CONFIG
    ensure_dirs()
    CONFIG = load_config()
    MODEL_NAME = CONFIG["model"]
    SOURCE_LANGUAGE = CONFIG["source_language"]
    TARGET_LANGUAGE = CONFIG["target_language"]
    _CONFIG_LOADED = True
    return CONFIG


def reload_config() -> Dict[str, Any]:
    """
    Re-read config.json, refresh the module-level settings and drop any cached
    engines so the next translation loads the model actually configured.

    Language-name changes only fully apply to labels on restart, since the UI
    builds them at startup.
    """
    global _CONFIG_LOADED
    _CONFIG_LOADED = False
    init_config()

    # Drop resident engines so a model switch takes effect immediately rather
    # than silently reusing the previously loaded one.
    unload_models()
    logger.info(f"Config reloaded; active model is now {MODEL_NAME}")
    return CONFIG


# Defaults for models with no dedicated profile. Models with a small context
# window override these via MODEL_PROFILES.
BATCH_SIZE = 24  # paragraphs per batch
BATCH_CHAR_LIMIT = 8000  # characters per batch, keeps prompts under context limits

# Cache: identical input + model + language pair never needs re-translating.
CACHE_DIR = Path("/tmp/dich-cache")

# ---------------------------------------------------------------------------
# Per-model prompt strategies
# ---------------------------------------------------------------------------
# Different models expect fundamentally different request shapes. A generic
# causal LLM takes a system prompt plus a plain string; TranslateGemma has its
# own chat template that only accepts user/assistant roles, requires the text
# inside a content list tagged with source/target language codes, and runs on a
# much smaller context window.

LANG_CODES = {
    "Vietnamese": "vi",
    "English": "en",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Russian": "ru",
    "Thai": "th",
}

MODEL_PROFILES = {
    "mlx-community/translategemma-4b-it-4bit": {
        "prompt_style": "translategemma",
        "batch_char_limit": 1400,  # model context is only ~2K tokens
        "batch_size": 8,
    },
    "mlx-community/translategemma-12b-it-4bit": {
        "prompt_style": "translategemma",
        "batch_char_limit": 1400,
        "batch_size": 8,
    },
}

DEFAULT_PROFILE = {
    "prompt_style": "chat",
    "batch_char_limit": BATCH_CHAR_LIMIT,
    "batch_size": BATCH_SIZE,
}


def _lang_code(language: str) -> str:
    return LANG_CODES.get(language, language.lower()[:2])


def get_profile(model: str) -> dict:
    """Resolve the prompt strategy for a model, letting config.json override."""
    profile = dict(DEFAULT_PROFILE)
    profile.update(MODEL_PROFILES.get(model, {}))
    # Explicit config values win over the built-in profile.
    if model == MODEL_NAME and "prompt_style" in CONFIG:
        profile["prompt_style"] = CONFIG["prompt_style"]
    return profile


def _cache_key(text: str) -> str:
    """Hash of everything that affects the output for a given input."""
    basis = f"{MODEL_NAME}|{SOURCE_LANGUAGE}|{TARGET_LANGUAGE}|{text}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def cache_get(text: str) -> Optional[str]:
    key = _cache_key(text)
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("translation")
    except (json.JSONDecodeError, OSError):
        return None


def cache_put(text: str, translation: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(text)
        path = CACHE_DIR / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"translation": translation, "model": MODEL_NAME, "key": key},
                f,
                ensure_ascii=False,
            )
        tmp.replace(path)  # atomic, avoids half-written cache entries
    except OSError as e:
        logger.warning(f"Could not write translation cache: {e}")


def cache_clear() -> int:
    """Remove all cached translations. Returns how many were removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for p in CACHE_DIR.glob("*.json"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def cache_stats() -> Dict[str, Any]:
    count = len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0
    return {"entries": count, "dir": str(CACHE_DIR)}


# ---------------------------------------------------------------------------
# Model management: what is on disk, how big it is, and freeing it
# ---------------------------------------------------------------------------
# The picker used to be free text, so a typo like "…-4bit-" silently started a
# multi-gigabyte download. Everything here exists so the UI can show what is
# already cached, its size, and roughly how much RAM it will want before the
# user commits to it.


def human_bytes(num: float) -> str:
    """Format a byte count the way a person reads it."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def memory_pressure() -> Optional[float]:
    """
    Return system memory pressure as a 0..1 fraction, or None when unreadable.

    On Apple Silicon there is no separate VRAM: the GPU shares unified memory,
    so low free RAM is the signal that we should unload resident models.
    Parses `vm_stat`, which is always available on macOS.
    """
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        pages: Dict[str, int] = {}
        for line in result.stdout.splitlines():
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip().replace(" ", "_").lower()
                value = parts[1].strip().rstrip(".")
                if value.isdigit():
                    pages[key] = int(value)
        if not pages:
            return None
        total = sum(pages.values())
        if total == 0:
            return None
        # "pages free" plus "pages speculative" are immediately reusable.
        available = pages.get("pages_free", 0) + pages.get("pages_speculative", 0)
        return 1.0 - (available / total)
    except (OSError, subprocess.SubprocessError, ValueError):  # noqa: BLE001
        return None


#: Pressure above which idle models are unloaded to free unified memory (VRAM).
MEMORY_PRESSURE_THRESHOLD = 0.85


def _dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:  # pragma: no cover - a racing delete is not fatal
            continue
    return total


def repo_cache_dir(repo: str) -> Path:
    """Where huggingface_hub stores `repo` under MODELS_DIR."""
    return MODELS_DIR / f"models--{repo.replace('/', '--')}"


def is_model_cached(repo: str) -> bool:
    """True when the repo has at least one snapshot on disk."""
    snapshots = repo_cache_dir(repo) / "snapshots"
    return snapshots.is_dir() and any(snapshots.glob("*"))


def cached_models() -> List[Dict[str, Any]]:
    """
    Every model repo already on disk, with its size.

    Ordered by size descending, so the heavy models — the ones worth thinking
    about before switching to — come first.
    """
    models: List[Dict[str, Any]] = []
    if not MODELS_DIR.exists():
        return models
    for entry in MODELS_DIR.glob("models--*"):
        if not entry.is_dir():
            continue
        repo = entry.name[len("models--") :].replace("--", "/")
        size = _dir_size(entry)
        models.append({"model": repo, "bytes": size, "size": human_bytes(size)})
    models.sort(key=lambda m: m["bytes"], reverse=True)
    return models


def model_size_bytes(repo: str) -> Optional[int]:
    """On-disk size of a cached model, or None when it is not cached."""
    if not is_model_cached(repo):
        return None
    return _dir_size(repo_cache_dir(repo))


def estimated_download_bytes(repo: str) -> Optional[int]:
    """
    Download size of a repo that is *not* cached, from the Hub API.

    Returns None when the lookup fails (offline, renamed repo). The UI must not
    block on this; it is a courtesy warning, not a gate.
    """
    try:
        response = httpx.get(
            f"https://huggingface.co/api/models/{repo}",
            params={"blobs": "true"},
            timeout=10,
        )
        response.raise_for_status()
        siblings = response.json().get("siblings", [])
    except (httpx.HTTPError, ValueError):
        return None
    total = sum(s.get("size") or 0 for s in siblings if isinstance(s, dict))
    return total or None


# 4-bit MLX weights inflate a little once the tokenizer and a KV cache are
# resident. This is an estimate for the picker, not a promise.
RAM_OVERHEAD = 1.25


def estimated_ram_bytes(repo: str) -> Optional[int]:
    """Rough resident RAM for a model, from disk when cached, else the Hub."""
    size = model_size_bytes(repo)
    if size is None:
        size = estimated_download_bytes(repo)
    return int(size * RAM_OVERHEAD) if size else None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def _pdftotext_text(file_path: str) -> str:
    """
    Extract a PDF's text with layout preserved via poppler's `pdftotext -layout`.

    Returns empty text when poppler is not installed or produces nothing.
    Layout-preserving extraction keeps columns and spacing closer to what the
    page looks like, rather than pypdf's flat reading-order stream.
    """
    if shutil.which("pdftotext") is None:
        return ""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", file_path, "-"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            return ""
        return result.stdout or ""
    except (OSError, subprocess.SubprocessError):  # noqa: BLE001
        return ""


def _read_pdf(file_path: str) -> str:
    """Extract text from a PDF file.

    pypdf is tried first because it preserves diacritics and inline text; if it
    yields nothing, poppler's `pdftotext -layout` is used to keep columns and
    spacing closer to the page; otherwise OCR is the last resort for scanned
    pages. When a text layer exists, embedded images (diagrams, photos, scans)
    are OCR'd and appended so nothing is lost.
    """
    text = _pypdf_text(file_path)
    if text.strip():
        if CONFIG.get("ocr_enabled") and tesseract_available():
            image_text = ocr_embedded_images(file_path)
            if image_text.strip():
                text = text + "\n\n" + image_text
        return text
    text = _pdftotext_text(file_path)
    if text.strip():
        if CONFIG.get("ocr_enabled") and tesseract_available():
            image_text = ocr_embedded_images(file_path)
            if image_text.strip():
                text = text + "\n\n" + image_text
        return text
    # Scanned / image-only PDF: OCR is the only way to get the text out.
    if CONFIG.get("ocr_enabled") and tesseract_available():
        return ocr_pdf(file_path)
    return text


# ---------------------------------------------------------------------------
# OCR for embedded images inside text PDFs
# ---------------------------------------------------------------------------


def ocr_embedded_images(pdf_path: str, lang: Optional[str] = None) -> str:
    """
    OCR images embedded inside a PDF's pages and return their text.

    Useful for text PDFs that carry diagrams, photos or scans as images:
    the text layer is kept, and each embedded image is OCR'd with a marker.
    Returns empty text when tesseract is absent or there are no images.
    """
    if not tesseract_available():
        return ""
    lang = lang or _tesseract_langs()
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(pdf_path)
    except Exception:  # noqa: BLE001 - unreadable PDF, skip images
        return ""
    parts: List[str] = []
    for pageno, page in enumerate(reader.pages, 1):
        try:
            images = page.images or []
        except Exception:  # noqa: BLE001 - a page without image support
            images = []
        for imgno, image in enumerate(images, 1):
            data = getattr(image, "data", None)
            if not data:
                continue
            suffix = _image_suffix(data)
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            try:
                with os.fdopen(fd, "wb") as tf:
                    tf.write(data)
                text = ocr_image(tmp, lang)
                if text.strip():
                    parts.append(f"--- image {imgno} on page {pageno} ---\n{text}")
            finally:
                try:
                    os.unlink(tmp)
                except OSError:  # noqa: BLE001 - best effort cleanup
                    pass
    return "\n\n".join(parts)


def _image_suffix(data: bytes) -> str:
    """Guess an image file extension from magic bytes so tesseract can read it."""
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if data.startswith(b"BM"):
        return ".bmp"
    if data[:2] in (b"II", b"MM"):
        return ".tif"
    return ".png"


# ---------------------------------------------------------------------------
# OCR for image-based (scanned) PDFs
# ---------------------------------------------------------------------------
# tesseract and pdftoppm are shell commands, not Python packages, so this
# shells out to them rather than adding a heavy OCR dependency. Only English
# trained data is guaranteed present ("vie" for Vietnamese is missing on this
# machine), so Vietnamese diacritics come through imperfectly unless the user
# installs tesseract's vie lang data.


def tesseract_available() -> bool:
    """True when the tesseract binary is on PATH."""
    return shutil.which("tesseract") is not None


def _tesseract_langs() -> str:
    """Prefer Vietnamese+English; fall back to English when vie is absent."""
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        langs = result.stdout.split()
    except (OSError, subprocess.SubprocessError):  # noqa: BLE001
        langs = []
    return "vie+eng" if "vie" in langs else "eng"


def ocr_image(image_path: str, lang: Optional[str] = None) -> str:
    """OCR a single image with tesseract; returns the recognized text."""
    if not tesseract_available():
        raise RuntimeError(
            "tesseract is not installed; install it with: brew install tesseract"
        )
    lang = lang or _tesseract_langs()
    result = subprocess.run(
        ["tesseract", image_path, "stdout", "-l", lang],
        capture_output=True,
        text=True,
        timeout=300,
    )
    result.check_returncode()
    return result.stdout


def render_pdf_pages(pdf_path: str, out_dir: str, dpi: int = 200) -> List[str]:
    """Render each PDF page to a PNG via pdftoppm; returns the image paths."""
    if shutil.which("pdftoppm") is None:
        raise RuntimeError(
            "pdftoppm is not installed; install it with: brew install poppler"
        )
    prefix = Path(out_dir) / "page"
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), pdf_path, str(prefix)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    return sorted(Path(out_dir).glob("page-*.png"))


def ocr_pdf(pdf_path: str, lang: Optional[str] = None) -> str:
    """OCR every page of a PDF and join the results with page markers."""
    lang = lang or _tesseract_langs()
    with tempfile.TemporaryDirectory() as td:
        pages = render_pdf_pages(pdf_path, td)
        parts = []
        for page in pages:
            page = Path(page)
            parts.append(f"--- page {page.stem} ---\n{ocr_image(str(page), lang)}")
    return "\n\n".join(parts)


def _pypdf_text(file_path: str) -> str:
    """Text layer of a PDF, or empty string when there is none."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            import PyPDF2

            with open(file_path, "rb") as f:
                return "\n".join(
                    (p.extract_text() or "") for p in PyPDF2.PdfReader(f).pages
                )
        except ImportError:
            # Both pypdf and PyPDF2 missing: replace the incidental ImportError
            # with the actionable message rather than chaining it.
            raise ImportError("PDF support needs pypdf: pip install pypdf") from None
    reader = PdfReader(file_path)
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _read_docx(file_path: str) -> str:
    """Extract text from a .docx / .docm file, including tables."""
    import docx

    document = docx.Document(file_path)
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append("\t".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def _read_doc(file_path: str) -> str:
    """Extract text from a legacy binary .doc file using macOS textutil."""
    import shutil
    import subprocess

    if shutil.which("textutil") is None:
        raise ImportError("Legacy .doc needs macOS 'textutil' (or convert to .docx).")
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", file_path],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return result.stdout


def _read_rtf(file_path: str) -> str:
    """Extract plain text from an RTF file by stripping control words."""
    from striprtf.striprtf import rtf_to_text

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return rtf_to_text(f.read())


def _read_markdown(file_path: str) -> str:
    """
    Read a Markdown file as its source text.

    It used to run the file through `markdown.markdown()` to HTML, which meant
    the model translated HTML while the output pane renders Markdown — so a
    translated `.md` arrived showing literal `<h1>` and `<strong>` tags. The
    Markdown source is what the model should see, because that is what the
    output is asked to mirror.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


_EXTRACTORS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".docm": _read_docx,
    ".doc": _read_doc,
    ".rtf": _read_rtf,
    ".md": _read_markdown,
    ".markdown": _read_markdown,
    ".txt": _read_text,
    ".text": _read_text,
    "": _read_text,
}


def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file, dispatching on extension."""
    ext = Path(file_path).suffix.lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(e for e in _EXTRACTORS if e))}"
        )
    return extractor(file_path)


def export_text(text: str, path: str, source_path: Optional[str] = None) -> str:
    """
    Write a translation to `path`, keeping the source format where possible.

    A `.docx` in should be a `.docx` out: when the source file is available the
    structure (paragraphs, headings, lists and tables) is preserved block by
    block rather than flattened. Formats that cannot be re-emitted faithfully
    (PDF) fall back to plain UTF-8 text, which is why the suffix is honoured
    rather than assumed.
    """
    target = Path(path)
    suffix = target.suffix.lower()

    if suffix in (".docx", ".docm") and source_path:
        source_suffix = Path(source_path).suffix.lower()
        if source_suffix in (".docx", ".docm"):
            return _export_docx_structured(source_path, str(target))

    if suffix in (".docx", ".docm"):
        import docx

        document = docx.Document()
        for block in text.split("\n\n"):
            # Word has no "#", so a heading line becomes a real heading; the
            # rest becomes body text. This is best-effort structure, not a
            # lossless round trip.
            lines = block.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if i == 0 and stripped.startswith("#"):
                    level = min(len(stripped) - len(stripped.lstrip("#")), 4)
                    document.add_heading(stripped.lstrip("# ").strip(), level=level)
                else:
                    document.add_paragraph(stripped)
        document.save(str(target))
        return str(target)

    target.write_text(text, encoding="utf-8")
    return str(target)


def _export_docx_structured(source_path: str, target_path: str) -> str:
    """
    Translate a .docx/.docm preserving paragraphs, headings, lists and tables.

    Each paragraph and table cell is translated in place, so the output keeps
    the same block order, table dimensions and paragraph styles as the source.
    """
    import docx
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    source = docx.Document(source_path)
    target = docx.Document()

    def paragraph_style(paragraph: Paragraph) -> str:
        try:
            return paragraph.style.name
        except Exception:  # noqa: BLE001 - style may be unset
            return "Normal"

    def apply_style(new_paragraph, name: str) -> None:
        try:
            new_paragraph.style = name
        except (KeyError, ValueError):  # style not in target template
            pass

    for element in source.element.body.iterchildren():
        if isinstance(element, CT_P):
            paragraph = Paragraph(element, source)
            if not paragraph.text.strip():
                target.add_paragraph("")
                continue
            translated = translate_text(paragraph.text).text
            new_paragraph = target.add_paragraph(translated)
            apply_style(new_paragraph, paragraph_style(paragraph))
        elif isinstance(element, CT_Tbl):
            table = Table(element, source)
            rows = len(table.rows)
            cols = len(table.columns)
            new_table = target.add_table(rows=rows, cols=cols)
            try:
                new_table.style = table.style.name or "Table Grid"
            except (KeyError, ValueError):
                new_table.style = "Table Grid"
            for r_i, row in enumerate(table.rows):
                for c_i, cell in enumerate(row.cells):
                    new_cell = new_table.cell(r_i, c_i)
                    if cell.text.strip():
                        new_cell.text = translate_text(cell.text).text

    target.save(target_path)
    return str(target_path)


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------


def glossary_directive() -> str:
    """
    Instruction pinning configured terms to a preferred translation.

    Empty when nothing is configured, so the prompt is byte-identical to the
    un-glossaried one in the common case — the glossary must never change
    ordinary output.
    """
    glossary = CONFIG.get("glossary") or {}
    if not isinstance(glossary, dict):
        return ""
    pairs = [
        (str(src).strip(), str(dst).strip())
        for src, dst in glossary.items()
        if str(src).strip() and str(dst).strip()
    ]
    if not pairs:
        return ""
    lines = "\n".join(f"- {src} -> {dst}" for src, dst in pairs)
    return (
        "\n\nUse these exact translations for these terms wherever they "
        "appear:\n" + lines
    )


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a professional {source} ({source_native}) to {target} translator. "
    "Translate the user's text into natural, idiomatic {target}.\n\n"
    "Rules:\n"
    "1. Translate by MEANING and CONCEPT, not word for word. Render idioms, "
    "proverbs, honorifics and figurative language as a native {target} writer would.\n"
    "2. Preserve formatting. Output as **Markdown**: keep headings, lists and "
    "emphasis where the source has them, and use them where the meaning benefits.\n"
    "3. Do NOT leave any {source} untranslated. Do NOT add explanations, "
    "notes, transliterations or a preamble.\n"
    "4. Output ONLY the {target} translation, as Markdown."
)

# Appended for models whose own chat template supplies the translation
# instruction (TranslateGemma), so the same Markdown behaviour is requested
# without a system turn to put it in.
MARKDOWN_INSTRUCTION = (
    "\n\nFormat the translation as Markdown: preserve headings, lists and "
    "emphasis, and use them where the meaning benefits."
)


def _split_oversized(part: str, char_limit: int) -> List[str]:
    """Split one paragraph longer than the limit, cutting on sentence boundaries.

    Keeps every character: the pieces join back to the exact input. A single
    sentence that is itself over the limit is hard-split, since there is no
    softer place to cut.
    """
    if len(part) <= char_limit:
        return [part]
    pieces = re.split(r"(?<=[.!?…])(\s+)", part)  # separators kept in the list
    out: List[str] = []
    current = ""
    for piece in pieces:
        if piece is None:
            continue
        while len(piece) > char_limit:
            if current:
                out.append(current)
                current = ""
            out.append(piece[:char_limit])
            piece = piece[char_limit:]
        if current and len(current) + len(piece) > char_limit:
            out.append(current)
            current = piece
        else:
            current += piece
    if current:
        out.append(current)
    return out


def _split_batches(
    text: str, size: int = BATCH_SIZE, char_limit: int = BATCH_CHAR_LIMIT
) -> List[str]:
    """Split text into paragraph-aligned batches, keeping each under a size cap."""
    paragraphs = re.split(r"(\n\s*\n)", text)  # separators are kept in the list
    if len(paragraphs) <= size and len(text) <= char_limit:
        return [text]
    chunks, current = [], []
    count, current_len = 0, 0
    for part in paragraphs:
        # An oversized paragraph used to ride through whole, because neither
        # loop condition could split inside a part.
        for piece in _split_oversized(part, char_limit):
            if count >= size or (current_len + len(piece) > char_limit and current):
                chunks.append("".join(current))
                current, count, current_len = [], 0, 0
            current.append(piece)
            if piece.strip():
                count += 1
                current_len += len(piece)
    if current:
        chunks.append("".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


@dataclass
class Progress:
    """Streamed back to the UI while a translation runs."""

    done_batches: int = 0
    total_batches: int = 0
    words_translated: int = 0
    words_total: int = 0
    tokens_out: int = 0
    tokens_per_second: float = 0.0
    elapsed: float = 0.0
    started_at: float = 0.0
    events: List[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        if self.total_batches <= 0:
            return 0.0
        return self.done_batches / self.total_batches

    @property
    def percent(self) -> int:
        """
        Percentage of the way through the model work. Caps at 99 so the bar
        never sits at "full" while output is still being stitched together;
        the UI jumps to 100 when the final result arrives.
        """
        if self.total_batches <= 0:
            return 0
        return int(min(99, round(self.fraction * 95)))


@dataclass
class TranslationResult:
    text: str
    cached: bool = False
    model: str = ""
    fell_back: bool = False
    words_in: int = 0
    words_out: int = 0
    tokens_out: int = 0
    tokens_per_second: float = 0.0
    elapsed: float = 0.0
    batches: int = 1


def count_words(text: str) -> int:
    """Word count that works reasonably for Vietnamese (space-separated runs)."""
    return len(re.findall(r"[^\s]+", text))


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
# NOTE ON WHY THIS IS NOT AN HTTP SERVER:
#   mlx_lm.server cannot carry every model's prompt format. Its
#   process_message_content() collapses any `content` list into a plain string,
#   which throws away the source_lang_code/target_lang_code that TranslateGemma's
#   chat template requires. Loading models in-process with mlx_lm.load() keeps
#   the full message structure intact (and gives streaming plus exact token
#   counts), so translation runs in this process rather than over HTTP.

_ENGINE_LOCK = threading.Lock()
_ENGINES: Dict[str, "LocalEngine"] = {}


class LocalEngine:
    """Loads one model in-process and renders prompts with its own chat template."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Load the model once and reuse it across requests."""
        if self._model is not None:
            return self._model, self._tokenizer
        with self._load_lock:
            if self._model is None:
                from mlx_lm import load as mlx_load

                logger.info(f"Loading model {self.model_name} ...")
                started = time.time()
                self._model, self._tokenizer = mlx_load(self.model_name)
                self._align_stop_tokens()
                logger.info(f"Loaded {self.model_name} in {time.time() - started:.1f}s")
        return self._model, self._tokenizer

    def _align_stop_tokens(self):
        """
        Teach the tokenizer this model's real stop token.

        mlx-lm stops generation on `tokenizer.eos_token_ids`. Gemma-family chat
        models actually end a turn with `<end_of_turn>`, which is NOT in that
        set by default -- so generation would run to max_tokens, repeating the
        marker forever. Without this, a short sentence cost ~10s and produced
        thousands of junk characters.
        """
        try:
            token_id = self._tokenizer.convert_tokens_to_ids("<end_of_turn>")
        except Exception:  # noqa: BLE001 - tokenizer without that helper
            return
        if token_id and token_id not in self._tokenizer.eos_token_ids:
            self._tokenizer.eos_token_ids.add(token_id)
            logger.info(f"Added <end_of_turn> (id {token_id}) to the stop tokens")

    def unload(self):
        with self._load_lock:
            self._model = None
            self._tokenizer = None

    # -- prompting --------------------------------------------------------
    def render_prompt(self, messages: list) -> str:
        """Apply this model's own chat template to the messages."""
        _, tokenizer = self.load()
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def build_messages(self, chunk: str) -> list:
        """Shape the request for this model's expected format."""
        glossary = glossary_directive()
        if get_profile(self.model_name)["prompt_style"] == "translategemma":
            # The template injects the translator instruction itself, so only
            # the language tags are supplied. The Markdown request rides along
            # with the source text, since there is no system turn to hold it.
            text = chunk
            if MARKDOWN_INSTRUCTION.strip() not in chunk:
                text = chunk + MARKDOWN_INSTRUCTION + glossary
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": _lang_code(SOURCE_LANGUAGE),
                            "target_lang_code": _lang_code(TARGET_LANGUAGE),
                            "text": text,
                        }
                    ],
                }
            ]
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    source=SOURCE_LANGUAGE,
                    source_native="tiếng Việt"
                    if SOURCE_LANGUAGE == "Vietnamese"
                    else SOURCE_LANGUAGE,
                    target=TARGET_LANGUAGE,
                )
                + glossary,
            },
            {"role": "user", "content": chunk},
        ]

    # -- generation -------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        _, tokenizer = self.load()
        return len(tokenizer.encode(text))

    def run(self, chunk: str, max_tokens: int):
        """Translate one chunk and return the text plus its token count."""
        return self.run_messages(self.build_messages(chunk), max_tokens)

    def run_messages(self, messages: list, max_tokens: int):
        """
        Translate a prepared message list and return (text, token count).

        Accepting pre-built messages lets a caller supply prior conversation
        turns as context instead of a single isolated paragraph.
        """
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self.load()
        prompt = self.render_prompt(messages)
        raw = mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),  # greedy: translation must be stable
            verbose=False,
        )
        text = _trim_stop_tokens(raw)
        return text, self.count_tokens(text)

    def run_block(self, text: str, model: Optional[str] = None):
        """
        Translate a whole passage as one unit, returning a TranslationResult.

        Kept separate from translate_text() so it is never fed through
        batching or the cache: a refinement must reflect the current model, not
        a stored earlier answer.
        """
        started = time.time()
        translation, tokens = self.run_messages(
            self.build_messages(text), max(512, min(8192, len(text) * 3))
        )
        elapsed = time.time() - started
        return TranslationResult(
            text=translation,
            model=model or self.model_name,
            words_in=count_words(text),
            words_out=count_words(translation),
            tokens_out=tokens,
            tokens_per_second=tokens / max(elapsed, 1e-6),
            elapsed=elapsed,
        )


def _trim_stop_tokens(output: str) -> str:
    """Cut the completion at the first end-of-turn marker."""
    for marker in ("<end_of_turn>", "<eos>", "<end_of_text>"):
        idx = output.find(marker)
        if idx != -1:
            output = output[:idx]
            break
    return output.strip()


def get_engine(model_name: str) -> "LocalEngine":
    """Get (and cache) the engine for a model."""
    with _ENGINE_LOCK:
        engine = _ENGINES.get(model_name)
        if engine is None:
            engine = LocalEngine(model_name)
            _ENGINES[model_name] = engine
        return engine


def unload_models(keep: Optional[str] = None) -> List[str]:
    """
    Drop resident engines, freeing their RAM. Returns the names unloaded.

    `keep` names a model to leave loaded, which is how the idle sweep frees
    everything except the one the user is actively using.
    """
    with _ENGINE_LOCK:
        names = [name for name in _ENGINES if name != keep]
        engines = [_ENGINES.pop(name) for name in names]
    for engine in engines:
        try:
            engine.unload()
        except Exception as e:  # noqa: BLE001 - freeing memory must not raise
            logger.warning(f"Unload of {engine.model_name} failed: {e}")
    if names:
        logger.info(f"Unloaded models: {', '.join(names)}")
    return names


def _translate_chunks(
    chunks: List[str],
    engine: "LocalEngine",
    on_progress: Optional[Callable[[Progress], None]],
    progress: Progress,
) -> List[str]:
    """Translate every chunk with one model, reporting progress as it goes."""
    translated: List[str] = []
    for i, chunk in enumerate(chunks):
        chunk_started = time.time()
        text, tokens = engine.run(chunk, max_tokens=max(512, min(8192, len(chunk) * 3)))
        chunk_seconds = max(time.time() - chunk_started, 1e-6)

        translated.append(text)
        progress.done_batches = i + 1
        progress.words_translated += count_words(chunk)
        progress.tokens_out += tokens
        progress.elapsed = time.time() - progress.started_at
        # Rolling average throughput across the whole run.
        progress.tokens_per_second = progress.tokens_out / max(progress.elapsed, 1e-6)
        progress.events.append(f"batch {i + 1}/{len(chunks)}: {chunk_seconds:.1f}s")

        if on_progress:
            on_progress(progress)

    return translated


def translate_text(
    text: str,
    model: Optional[str] = None,
    on_progress: Optional[Callable[[Progress], None]] = None,
    use_cache: bool = True,
) -> TranslationResult:
    """
    Translate text from the configured source language to the target language.

    Args:
        text: Input text (usually Vietnamese).
        model: Optional model override. Defaults to the configured model.
        on_progress: Called with a Progress snapshot as batches complete.
        use_cache: Return a stored translation when the input is unchanged.

    Returns:
        TranslationResult with the translated text plus timing/counter stats.

    Raises:
        RuntimeError: if no model in the chain (configured model plus
            fallback_models) succeeds.
    """
    init_config()
    started = time.time()
    source = text.strip()
    if not source:
        return TranslationResult(text="")

    resolved_model = model or MODEL_NAME
    if not isinstance(resolved_model, str):
        # Almost always a positional-argument mix-up at the call site: the
        # signature is (text, model, on_progress), so passing a callback second
        # lands here. Fail loudly rather than silently "falling back".
        raise TypeError(
            f"model must be a string model id, got {type(resolved_model).__name__}. "
            "Call as translate_text(text, on_progress=fn)."
        )
    if on_progress is not None and not callable(on_progress):
        raise TypeError("on_progress must be callable; use on_progress=fn.")

    # Already translated this exact input? Skip the model entirely.
    if use_cache:
        hit = cache_get(source)
        if hit is not None:
            logger.info("Cache hit, skipping translation")
            return TranslationResult(
                text=hit,
                cached=True,
                words_in=count_words(source),
                words_out=count_words(hit),
                elapsed=time.time() - started,
            )

    primary = resolved_model
    chain = [primary] + [m for m in CONFIG.get("fallback_models", []) if m != primary]

    last_error: Optional[Exception] = None
    for attempt, current in enumerate(chain):
        profile = get_profile(current)
        chunks = _split_batches(
            source, profile["batch_size"], profile["batch_char_limit"]
        )

        progress = Progress(
            total_batches=len(chunks),
            words_total=count_words(source),
            started_at=started,
        )
        if on_progress:
            on_progress(progress)

        try:
            logger.info(
                f"Translating with {current} (style={profile['prompt_style']}, "
                f"{len(chunks)} batch(es), limit={profile['batch_char_limit']} chars)"
            )
            engine = get_engine(current)
            translated = _translate_chunks(chunks, engine, on_progress, progress)
        except Exception as e:  # noqa: BLE001 - try the fallback model, then report
            last_error = e
            logger.warning(
                f"Translation with {current} failed: {type(e).__name__}: {e}"
            )
            if attempt == len(chain) - 1:
                break
            logger.info(f"Falling back to {chain[attempt + 1]}")
            continue

        out = (
            translated[0]
            if len(translated) == 1
            else "".join(p if p.endswith("\n") else p + "\n" for p in translated)
        )
        if use_cache:
            cache_put(source, out)

        return TranslationResult(
            text=out,
            model=current,
            fell_back=(current != primary),
            words_in=count_words(source),
            words_out=count_words(out),
            tokens_out=progress.tokens_out,
            tokens_per_second=progress.tokens_out / max(time.time() - started, 1e-6),
            elapsed=time.time() - started,
            batches=len(chunks),
        )

    raise RuntimeError(f"All candidate models failed. Last error: {last_error}")


def translate_file(
    file_path: str,
    model: Optional[str] = None,
    on_progress: Optional[Callable[[Progress], None]] = None,
) -> TranslationResult:
    """Extract text from a file and translate it."""
    text = extract_text_from_file(file_path)
    logger.info(f"Extracted {len(text)} chars from {file_path}")
    return translate_text(text, model=model, on_progress=on_progress)


def translate_files(
    file_paths: List[str],
    model: Optional[str] = None,
    on_progress: Optional[Callable[[Progress], None]] = None,
) -> Dict[str, Any]:
    """Translate several files; one failure does not abort the rest."""
    results = []
    for path in file_paths:
        try:
            results.append(
                {
                    "file": path,
                    "ok": True,
                    "result": translate_file(
                        path, model=model, on_progress=on_progress
                    ),
                }
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to translate {path}: {e}")
            results.append({"file": path, "ok": False, "error": str(e)})
    return {"results": results}


def folder_files(folder: str) -> List[str]:
    """Supported files directly inside `folder`, sorted, non-recursive."""
    root = Path(folder)
    if not root.is_dir():
        return []
    return sorted(
        str(p)
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in _EXTRACTORS
    )


def export_suffix_for(file_path: str) -> str:
    """
    Extension to give an exported translation of `file_path`.

    A format that can be rebuilt is kept (`.docx` -> `.docx`); a PDF's flat text
    has no faithful container, so it becomes `.txt`.
    """
    suffix = Path(file_path).suffix.lower()
    return (
        suffix if suffix in (".docx", ".docm", ".md", ".markdown", ".txt") else ".txt"
    )


def translate_folder(
    folder: str,
    model: Optional[str] = None,
    on_progress: Optional[Callable[[Progress], None]] = None,
    on_file: Optional[Callable[[str, dict], None]] = None,
) -> Dict[str, Any]:
    """
    Translate every supported file in a folder and write each result beside it.

    Output names carry the target language, e.g. `notes.docx` ->
    `notes.en.docx`, so a batch never overwrites its source. One file failing
    does not stop the rest.
    """
    files = folder_files(folder)
    results: List[dict] = []
    for path in files:
        try:
            result = translate_file(path, model=model, on_progress=on_progress)
            out_path = str(Path(path).with_suffix("")) + (
                f".{_lang_code(TARGET_LANGUAGE)}{export_suffix_for(path)}"
            )
            export_text(result.text, out_path)
            results.append(
                {"file": path, "ok": True, "out": out_path, "result": result}
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to translate {path}: {e}")
            results.append({"file": path, "ok": False, "error": str(e)})
        if on_file is not None:
            on_file(path, results[-1])
    return {"folder": folder, "results": results}


# ---------------------------------------------------------------------------
# Microphone archive: raw input saved for later re-transcription / refinement
# ---------------------------------------------------------------------------


def audio_archive_dir() -> Path:
    """
    Where raw microphone recordings are stored.

    Defaults to `<config>/audio`; the user can override it in Settings. The
    directory is created on demand.
    """
    configured = str(CONFIG.get("audio_archive_path") or "").strip()
    target = Path(configured) if configured else CONFIG_ROOT / "audio"
    target.mkdir(parents=True, exist_ok=True)
    return target


def archive_files(archive_dir: Optional[str | Path] = None) -> List[Path]:
    """Archived WAV recordings, newest first."""
    directory = Path(archive_dir) if archive_dir else audio_archive_dir()
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def archive_storage_bytes(archive_dir: Optional[str | Path] = None) -> int:
    """Total bytes held by archived recordings."""
    directory = Path(archive_dir) if archive_dir else audio_archive_dir()
    if not directory.is_dir():
        return 0
    return sum(
        p.stat().st_size
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".wav"
    )


def clear_audio_archive(archive_dir: Optional[str | Path] = None) -> int:
    """Delete every archived recording; returns how many were removed."""
    directory = Path(archive_dir) if archive_dir else audio_archive_dir()
    if not directory.is_dir():
        return 0
    removed = 0
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() == ".wav":
            try:
                p.unlink()
                removed += 1
            except OSError:  # noqa: BLE001 - a locked file is skipped
                pass
    return removed


# ---------------------------------------------------------------------------
# Context-aware speech translation
# ---------------------------------------------------------------------------
# Whisper segments on acoustic boundaries, so a "sentence" often ends mid-clause
# ("nhau một"). Translating each in isolation produces fragments that read
# wrong, and Vietnamese honorifics/pronouns only resolve with what came before.
#
# Two stages fix this:
#   1. translate one utterance *with* the preceding (Vietnamese, English) pairs
#      supplied as prior conversation turns, so context is available from the
#      first line;
#   2. every `refine_every` utterances, re-translate the whole accumulated block
#      as a single passage and replace the provisional line-by-line output.
# Stage 2 is what makes the result improve as more speech is understood.

CONTEXT_TURNS = 3  # prior utterances supplied as context
REFINE_EVERY = 6  # utterances between whole-block re-translations

# Utterances shorter than this (in characters) are combined with the next one
# into a single model call when combining is enabled, cutting overhead and
# giving fragments ("vâng", "được", "cảm ơn") more context.
COMBINE_MIN_CHARS = 30


def _context_messages(sentence: str, history: List[tuple], style: str) -> list:
    """
    Build a multi-turn conversation: prior Vietnamese with its translation, then
    the new utterance. Both prompt styles accept this shape.
    """
    messages: List[dict] = []

    def user_turn(text: str) -> dict:
        if style == "translategemma":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": _lang_code(SOURCE_LANGUAGE),
                        "target_lang_code": _lang_code(TARGET_LANGUAGE),
                        "text": text,
                    }
                ],
            }
        return {"role": "user", "content": text}

    for prior_vi, prior_en in history[-CONTEXT_TURNS:]:
        messages.append(user_turn(prior_vi))
        messages.append({"role": "assistant", "content": prior_en})
    messages.append(user_turn(sentence))
    return messages


class ContextualTranslator:
    """
    Translates an incoming stream of Vietnamese utterances, refining earlier
    translations as context accumulates.

    `translation[i]` is the best-known English for `transcript[i]` and may be
    replaced later by a more coherent block translation, so consumers should
    read the list rather than cache individual lines.
    """

    def __init__(
        self,
        refine_every: int = REFINE_EVERY,
        on_refine=None,
        on_update=None,
        combine_short_utterances: bool = False,
    ):
        self.transcript: List[str] = []
        self.translation: List[str] = []
        self.elaborate: List[bool] = []  # True once a line has been refined
        self.refined_blocks: List[dict] = []  # {start, end, text}
        self.refine_every = max(1, refine_every)
        self.combine_short = combine_short_utterances
        # Short utterances held here until they can be combined with the next
        # one into a single model call.
        self._combine_buffer: List[str] = []
        self._on_refine = on_refine
        self._on_update = on_update
        self._last_error: Optional[str] = None

    # -- state ------------------------------------------------------------
    @property
    def refined_count(self) -> int:
        return sum(1 for e in self.elaborate if e)

    @property
    def pending_count(self) -> int:
        return len(self.transcript) - self.refined_count

    def add(self, sentence: str) -> str:
        """
        Add a Vietnamese utterance and return its current translation.

        Every `refine_every` utterances, only the newly accumulated ones are
        re-translated as a passage; earlier output is left alone because it has
        already been through refinement.

        When combining short utterances is enabled, fragments shorter than
        `COMBINE_MIN_CHARS` characters are buffered and translated together with
        the next utterance, cutting model calls and giving fragments more
        context.
        """
        sentence = sentence.strip()
        if not sentence:
            return ""

        if self.combine_short and len(sentence) < COMBINE_MIN_CHARS:
            self._combine_buffer.append(sentence)
            return ""  # shown once combined with the following utterance

        if self._combine_buffer:
            sentence = " ".join(self._combine_buffer + [sentence])
            self._combine_buffer = []

        index = len(self.transcript)
        history = list(zip(self.transcript, self.translation, strict=False))
        english = translate_utterance(sentence, history)
        self.transcript.append(sentence)
        self.translation.append(english)
        self.elaborate.append(False)

        self._notify(self._on_update, index, english)

        if (index + 1) % self.refine_every == 0:
            self.refine()
        return english

    @property
    def block_start(self) -> int:
        """Index of the first utterance not yet refined."""
        for i, done in enumerate(self.elaborate):
            if not done:
                return i
        return len(self.transcript)

    def refine(self, start: Optional[int] = None) -> bool:
        """
        Translate `transcript[start:]` as one passage, replacing those lines
        with the coherent result.

        `start` defaults to the first unrefined index, so a refinement never
        re-translates sentences already refined. Earlier blocks only change when
        their own sentences do, and re-reading them cost 1.6x the necessary
        model calls for no gain in output. The earlier conversation is supplied
        as context turns instead, so the new block still reads as a continuation
        without re-reading what came before.

        The model may return a different number of lines than it was given, so
        the merged text is kept verbatim and per-line guesses are only
        overwritten where the mapping is unambiguous. Every index in the block
        is marked refined, because the block really was re-translated;
        `refined_blocks` holds the authoritative text for display.
        """
        if start is None:
            start = self.block_start
        block = self.transcript[start:]
        if not block:
            return False

        history = list(
            zip(self.transcript[:start], self.translation[:start], strict=False)
        )
        try:
            merged = translate_block(block, history=history)
        except Exception as e:  # noqa: BLE001 - keep earlier results usable
            self._last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"Block refinement failed: {self._last_error}")
            return False

        lines = _split_translation(merged, len(block))
        for i, line in enumerate(lines, start=start):
            if line and line.strip():
                self.translation[i] = line.strip()

        # A refinement re-reads from `start` to the end, so any earlier block it
        # overlaps has been superseded. Keeping both would render the same
        # sentences twice, once per block.
        end = len(self.transcript) - 1
        self.refined_blocks = [
            b for b in self.refined_blocks if b["end"] < start or b["start"] > end
        ]
        self.refined_blocks.append(
            {
                "start": start,
                "end": end,
                "text": merged.strip(),
            }
        )
        for i in range(start, len(self.transcript)):
            self.elaborate[i] = True

        self._notify(self._on_refine, start)
        return True

    def render_refined(self) -> str:
        """
        Best display text: refined passages in place, with any utterances that
        sit outside a refined block shown individually.
        """
        covered = [False for _ in range(len(self.transcript))]
        out: List[str] = []
        for block in self.refined_blocks:
            for i in range(block["start"], block["end"] + 1):
                covered[i] = True
            out.append((block["start"], block["text"]))

        for i, en in enumerate(self.translation):
            if not covered[i]:
                out.append((i, en or "(pending)"))

        out.sort(key=lambda pair: pair[0])
        return "\n".join(text for _index, text in out if text)

    # -- output -----------------------------------------------------------
    def finalize(self) -> bool:
        """
        Refine whatever is left once no more speech is coming.

        Without this, a partial trailing block -- four utterances after the last
        `refine_every` boundary -- would stay as provisional line-by-line guesses
        even though nothing further will give them context.

        Any buffered short utterances are flushed as one final translation so
        they are not dropped when recording stops.
        """
        if self.combine_short and self._combine_buffer:
            sentence = " ".join(self._combine_buffer)
            self._combine_buffer = []
            # Force-translate the buffered fragment as one final utterance so it
            # is not re-buffered (it is already short) and not re-refined.
            index = len(self.transcript)
            history = list(zip(self.transcript, self.translation, strict=False))
            english = translate_utterance(sentence, history)
            self.transcript.append(sentence)
            self.translation.append(english)
            self.elaborate.append(True)
            self._notify(self._on_update, index, english)
        if self.block_start >= len(self.transcript):
            return False
        return self.refine(self.block_start)

    def render(self) -> str:
        return "\n".join(en if en else "(pending)" for en in self.translation)

    def stats(self) -> dict:
        return {
            "utterances": len(self.transcript),
            "refined": self.refined_count,
            "provisional": self.pending_count,
            "refined_blocks": len(self.refined_blocks),
            "context_turns": CONTEXT_TURNS,
            "refine_every": self.refine_every,
            "last_error": self._last_error,
        }

    @staticmethod
    def _notify(callback, *args):
        if callback:
            try:
                callback(*args)
            except Exception as e:  # noqa: BLE001 - a listener must not break flow
                logger.warning(f"ContextualTranslator callback failed: {e}")


def translate_utterance(
    sentence: str, history: List[tuple], model: Optional[str] = None
) -> str:
    """
    Translate a single utterance with the preceding conversation as context.

    Args:
        sentence: The Vietnamese utterance to translate.
        history: Prior (vietnamese, english) pairs, most recent last.
        model: Optional model override.
    """
    init_config()
    name = model or MODEL_NAME
    style = get_profile(name)["prompt_style"]
    messages = _context_messages(sentence, history, style)
    engine = get_engine(name)
    text, _tokens = engine.run_messages(messages, max_tokens=256)
    return text


def translate_block(
    sentences: List[str],
    model: Optional[str] = None,
    history: Optional[List[tuple]] = None,
) -> str:
    """
    Translate several sentences as one passage, preserving line structure.

    Args:
        sentences: The new sentences to translate together.
        model: Optional model override.
        history: Prior (vietnamese, english) pairs supplied as context turns, so
            a partial block reads as a continuation without re-translating what
            came before.
    """
    init_config()
    name = model or MODEL_NAME
    engine = get_engine(name)
    style = get_profile(name)["prompt_style"]
    messages = _context_messages("\n".join(sentences), history or [], style)
    text, _tokens = engine.run_messages(messages, max_tokens=1024)
    return text


def _split_translation(text: str, expected: int) -> List[str]:
    """
    Split a block translation back into per-line pieces.

    A block translation should contain one line per input utterance. When the
    model returns fewer or more lines than expected, the surplus is folded into
    the last line rather than silently dropping or inventing content.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == expected:
        return lines
    if len(lines) > expected:
        return lines[: expected - 1] + [" ".join(lines[expected - 1 :])]
    return lines + [""] * (expected - len(lines))


def align_paragraphs(source: str, translation: str) -> List[tuple]:
    """
    Pair source and translation paragraphs side by side.

    Splits both on blank lines and pairs them by index. When one side has more
    paragraphs than the other, the missing side is padded with an empty string,
    so every source paragraph has a translation slot and vice versa.
    """
    src = [p.strip() for p in source.split("\n\n") if p.strip()]
    trg = [p.strip() for p in translation.split("\n\n") if p.strip()]
    n = max(len(src), len(trg))
    src += [""] * (n - len(src))
    trg += [""] * (n - len(trg))
    return list(zip(src, trg, strict=True))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

# Models whose load has been started but has not finished yet. Lets health
# reporting distinguish "loading" from "never asked to load".
_LOADING: set = set()

# Set when preload_model() fails, so the UI can surface why instead of
# showing an indefinite loading state.
_LOAD_ERROR: Dict[str, str] = {}


def preload_model(model: Optional[str] = None) -> bool:
    """
    Start loading a model on a background thread.

    Called at startup when "load model on startup" is enabled, so the UI turns
    green without the user having to translate anything first. Safe to call
    repeatedly; returns True if the model is already resident.

    Args:
        model: Model to load. Defaults to the configured model.

    Returns:
        True if already loaded, False if a load was just started (or is running).
    """
    init_config()
    name = model or MODEL_NAME
    if not isinstance(name, str):
        return False
    engine = get_engine(name)
    if engine.loaded:
        return True

    with _ENGINE_LOCK:
        if name in _LOADING:
            return False
        _LOADING.add(name)
        _LOAD_ERROR.pop(name, None)

    def task():
        try:
            engine.load()
            logger.info(f"Preload of {name} finished")
        except Exception as e:  # noqa: BLE001 - report through health_check
            message = f"{type(e).__name__}: {e}"
            logger.warning(f"Preload of {name} failed: {message}")
            _LOAD_ERROR[name] = message
        finally:
            with _ENGINE_LOCK:
                _LOADING.discard(name)

    threading.Thread(target=task, daemon=True, name=f"preload-{name}").start()
    return False


def health_check() -> Dict[str, Any]:
    """
    Report the state of the configured model.

    ready    - model is resident and can translate
    'loading' - a preload is in progress
    idle     - nothing has asked for the model yet
    """
    chain = [MODEL_NAME] + [
        m for m in CONFIG.get("fallback_models", []) if m != MODEL_NAME
    ]
    engine = _ENGINES.get(MODEL_NAME)

    if engine is not None and engine.loaded:
        return {"ready": True, "model": MODEL_NAME, "chain": chain}

    if MODEL_NAME in _LOADING:
        return {"ready": False, "loading": True, "model": MODEL_NAME, "chain": chain}

    info: Dict[str, Any] = {"ready": False, "model": MODEL_NAME, "chain": chain}
    if MODEL_NAME in _LOAD_ERROR:
        info["reason"] = _LOAD_ERROR[MODEL_NAME]
    else:
        info["reason"] = "not loaded yet (loads on first translation)"
    return info


# ---------------------------------------------------------------------------
# Google Translate reference (debug / verification only)
# ---------------------------------------------------------------------------


def _google_translate_chunk(chunk: str, source: str, target: str) -> str:
    response = httpx.get(
        "https://translate.googleapis.com/translate_a/t",
        params={"client": "gtx", "sl": source, "tl": target, "q": chunk},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and data and isinstance(data[0], str):
        return data[0]
    if isinstance(data, dict) and "sentences" in data:
        return "".join(s.get("trans", "") for s in data["sentences"])
    return str(data)


def google_reference_translation(text: str, retries: int = 3) -> str:
    """
    Translate text with Google Translate as an independent reference to
    compare against the local model's output. Debug/verification only.
    """
    init_config()
    if not text.strip():
        return ""

    source = "vi" if SOURCE_LANGUAGE == "Vietnamese" else "auto"
    target = {"English": "en"}.get(TARGET_LANGUAGE, TARGET_LANGUAGE.lower()[:2])

    chunks = _split_batches(text, 20)
    parts = []
    delay = 2.0
    for chunk in chunks:
        for attempt in range(retries):
            try:
                parts.append(_google_translate_chunk(chunk, source, target))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    raise
                logger.warning(
                    f"Google reference attempt {attempt + 1}/{retries} failed "
                    f"({type(e).__name__}); retrying in {delay:.0f}s"
                )
                time.sleep(delay)
                delay *= 2
    return "\n".join(p for p in parts if p)


if __name__ == "__main__":
    init_config()
    status = health_check()
    print(
        json.dumps(
            {
                "server": status,
                "cache": cache_stats(),
                "config": {
                    "model": MODEL_NAME,
                    "source": SOURCE_LANGUAGE,
                    "target": TARGET_LANGUAGE,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
