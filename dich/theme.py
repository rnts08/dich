"""
Design tokens and the ttk style setup for Dịch.

Everything visual lives here so the look of the app can be changed in one
place. The palette is Catppuccin Mocha, chosen because it is a complete,
labelled colour system rather than a handful of ad-hoc hex values: accent
colours are already balanced against the backgrounds they sit on.

Widgets follow a small custom set rather than fighting ttk's defaults:
    RoundedButton  - flat rounded primary/ghost buttons with hover states
    Card           - a section container with a hairline border
    SectionLabel   - the small uppercase label that heads a card
Tk's `clam` theme is the base because it ignores most platform chrome, which
otherwise leaks through on Aqua and breaks the flat look.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

# ---------------------------------------------------------------------------
# Palette - Catppuccin Mocha
# ---------------------------------------------------------------------------
BASE = "#1e1e2e"  # deepest background
MANTLE = "#181825"  # app chrome / menu
SURFACE0 = "#313244"  # raised surface: inputs, panes
SURFACE1 = "#45475a"  # borders, dividers
SURFACE2 = "#585b70"  # stronger dividers, disabled text
OVERLAY = "#6c7086"  # placeholder / hint
TEXT = "#cdd6f4"  # primary text
SUBTEXT = "#a6adc8"  # secondary text
ROSEWATER = "#f5e0dc"

LAVENDER = "#b4befe"
BLUE = "#89b4fa"
SKY = "#89dceb"
TEAL = "#94e2d5"
GREEN = "#a6e3a1"
YELLOW = "#f9e2af"
PEACH = "#fab387"
RED = "#f38ba8"
MAUVE = "#cba6f7"
PINK = "#f5c2e7"

# ---------------------------------------------------------------------------
# Type scale
# ---------------------------------------------------------------------------
FONT_FAMILY = "SF Pro Text"
MONO_FAMILY = "SF Mono"

FONT_APP = (FONT_FAMILY, 22, "bold")  # app name
FONT_H1 = (FONT_FAMILY, 16, "bold")
FONT_H2 = (FONT_FAMILY, 13, "bold")
FONT_BODY = (FONT_FAMILY, 12)
FONT_SMALL = (FONT_FAMILY, 11)
FONT_FAINT = (FONT_FAMILY, 10)
FONT_MONO = (MONO_FAMILY, 11)
FONT_MONO_LARGE = (MONO_FAMILY, 12)

# ---------------------------------------------------------------------------
# Spacing scale - a 4pt rhythm keeps gaps consistent
# ---------------------------------------------------------------------------
SPACE_1 = 4
SPACE_2 = 8
SPACE_3 = 12
SPACE_4 = 16
SPACE_5 = 24
SPACE_6 = 32

RADIUS = 8  # rounded corners
RADIUS_SMALL = 5


# ---------------------------------------------------------------------------
# Markdown tag styling for the translation panes
# ---------------------------------------------------------------------------
MD_TAGS = {
    "h1": dict(font=(FONT_FAMILY, 20, "bold"), spacing1=12, spacing3=8),
    "h2": dict(font=(FONT_FAMILY, 16, "bold"), spacing1=10, spacing3=6),
    "h3": dict(font=(FONT_FAMILY, 14, "bold"), spacing1=8, spacing3=4),
    "bold": dict(font=(FONT_FAMILY, 12, "bold")),
    "italic": dict(font=(FONT_FAMILY, 12, "italic")),
    "bold_italic": dict(font=(FONT_FAMILY, 12, "bold italic")),
    "code": dict(font=(MONO_FAMILY, 11), background=SURFACE0),
    "code_block": dict(
        font=(MONO_FAMILY, 11), background=SURFACE0, spacing1=6, spacing3=6
    ),
    "list": dict(spacing1=2, spacing3=2, lmargin1=18, lmargin2=18),
    "quote": dict(foreground=OVERLAY, lmargin1=18, lmargin2=18, spacing1=6),
}


# ---------------------------------------------------------------------------
# Theming
# ---------------------------------------------------------------------------
def apply(root: tk.Tk) -> None:
    """Apply the full visual system to a Tk root window."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:  # pragma: no cover - clam ships with Tk
        pass

    _style_frames(style)
    _style_labels(style)
    _style_buttons(style)
    _style_inputs(style)
    _style_notebook(style)
    _style_progressbar(style)
    _style_scrollbar(style)

    try:
        root.configure(bg=BASE)
    except tk.TclError:
        pass


def _style_frames(style: ttk.Style) -> None:
    style.configure("TFrame", background=BASE)
    style.configure("Surface.TFrame", background=SURFACE0)
    style.configure("Card.TFrame", background=SURFACE0)


def _style_labels(style: ttk.Style) -> None:
    style.configure("TLabel", background=BASE, foreground=TEXT, font=FONT_BODY)
    style.configure("App.TLabel", background=BASE, foreground=TEXT, font=FONT_APP)
    style.configure("H1.TLabel", background=BASE, foreground=TEXT, font=FONT_H1)
    style.configure("H2.TLabel", background=BASE, foreground=TEXT, font=FONT_H2)
    style.configure("Body.TLabel", background=BASE, foreground=TEXT, font=FONT_BODY)
    style.configure(
        "Faint.TLabel", background=BASE, foreground=OVERLAY, font=FONT_SMALL
    )
    style.configure(
        "Section.TLabel", background=SURFACE0, foreground=SUBTEXT, font=FONT_FAINT
    )
    style.configure("Mono.TLabel", background=BASE, foreground=TEXT, font=FONT_MONO)
    style.configure("Status.TLabel", background=BASE, foreground=TEXT, font=FONT_SMALL)
    style.configure("Stat.TLabel", background=BASE, foreground=SUBTEXT, font=FONT_MONO)
    style.configure("Good.TLabel", background=BASE, foreground=GREEN, font=FONT_SMALL)
    style.configure("Bad.TLabel", background=BASE, foreground=RED, font=FONT_SMALL)
    style.configure("Busy.TLabel", background=BASE, foreground=YELLOW, font=FONT_SMALL)


def _style_buttons(style: ttk.Style) -> None:
    style.configure(
        "TButton",
        background=SURFACE0,
        foreground=TEXT,
        font=FONT_BODY,
        borderwidth=0,
        focuscolor=BASE,
        padding=(SPACE_3, SPACE_1 + 2),
    )
    style.map(
        "TButton",
        background=[("active", SURFACE1), ("disabled", SURFACE0)],
        foreground=[("disabled", SURFACE2)],
    )

    # The one loud control on a screen, e.g. Translate.
    style.configure(
        "Primary.TButton",
        background=BLUE,
        foreground=BASE,
        font=(FONT_FAMILY, 12, "bold"),
        borderwidth=0,
        focuscolor=BASE,
        padding=(SPACE_4, SPACE_1 + 2),
    )
    style.map(
        "Primary.TButton",
        background=[("active", SKY), ("disabled", SURFACE1)],
        foreground=[("active", BASE), ("disabled", SURFACE2)],
    )

    style.configure(
        "Danger.TButton",
        background=RED,
        foreground=BASE,
        font=FONT_BODY,
        borderwidth=0,
        focuscolor=BASE,
        padding=(SPACE_3, SPACE_1 + 2),
    )
    style.map("Danger.TButton", background=[("active", PINK)])

    style.configure(
        "Ghost.TButton",
        background=BASE,
        foreground=SUBTEXT,
        font=FONT_SMALL,
        borderwidth=0,
        focuscolor=BASE,
        padding=(SPACE_2, SPACE_1),
    )
    style.map("Ghost.TButton", foreground=[("active", TEXT)])


def _style_inputs(style: ttk.Style) -> None:
    style.configure(
        "TEntry",
        fieldbackground=SURFACE0,
        foreground=TEXT,
        insertcolor=TEXT,
        borderwidth=0,
        padding=(SPACE_2, SPACE_1),
    )
    style.map("TEntry", fieldbackground=[("focus", SURFACE1)])

    style.configure(
        "TCombobox",
        fieldbackground=SURFACE0,
        background=SURFACE0,
        foreground=TEXT,
        arrowcolor=TEXT,
        borderwidth=0,
        padding=(SPACE_2, SPACE_1),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", SURFACE0), ("focus", SURFACE1)],
        background=[("readonly", SURFACE0)],
        foreground=[("readonly", TEXT)],
    )

    style.configure(
        "TCheckbutton",
        background=BASE,
        foreground=TEXT,
        font=FONT_BODY,
        focuscolor=BASE,
        borderwidth=0,
    )
    style.map(
        "TCheckbutton",
        background=[("active", BASE)],
        foreground=[("disabled", SURFACE2)],
        indicatorcolor=[("selected", BLUE), ("!selected", SURFACE1)],
        indicatorbackground=[("selected", BLUE)],
    )

    style.configure(
        "TSpinbox",
        fieldbackground=SURFACE0,
        foreground=TEXT,
        arrowcolor=TEXT,
        borderwidth=0,
        padding=(SPACE_2, SPACE_1),
    )

    style.configure(
        "Vertical.TScrollbar",
        background=SURFACE1,
        troughcolor=BASE,
        borderwidth=0,
        arrowcolor=OVERLAY,
    )
    style.configure(
        "Horizontal.TScrollbar",
        background=SURFACE1,
        troughcolor=BASE,
        borderwidth=0,
        arrowcolor=OVERLAY,
    )


def _style_notebook(style: ttk.Style) -> None:
    """Underlined active tab rather than the boxed default."""
    style.configure("TNotebook", background=BASE, borderwidth=0, tabmargins=0)
    style.configure(
        "TNotebook.Tab",
        background=BASE,
        foreground=OVERLAY,
        font=FONT_BODY,
        borderwidth=0,
        padding=(SPACE_4, SPACE_2),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", BASE)],
        foreground=[("selected", TEXT)],
    )
    # clam draws a box around the selected tab; flatten it and rely on the
    # foreground change plus the status bar for the active state.
    try:
        style.layout(
            "TNotebook.Tab",
            [
                (
                    "Notebook.tab",
                    {
                        "sticky": "nswe",
                        "children": [
                            ("Notebook.label", {"side": "left", "sticky": ""}),
                        ],
                    },
                )
            ],
        )
    except tk.TclError:  # pragma: no cover - layout change is cosmetic
        pass


def _style_progressbar(style: ttk.Style) -> None:
    style.configure(
        "Horizontal.TProgressbar",
        background=BLUE,
        troughcolor=SURFACE0,
        borderwidth=0,
        thickness=6,
    )
    style.configure(
        "Horizontal.Busy.TProgressbar",
        background=MAUVE,
        troughcolor=SURFACE0,
        borderwidth=0,
        thickness=6,
    )


def _style_scrollbar(style: ttk.Style) -> None:
    style.configure("Treeview", background=SURFACE0, fieldbackground=SURFACE0)


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------
class RoundedButton(tk.Canvas):
    """
    A flat, rounded button with a hover state.

    ttk buttons cannot be given rounded corners on the clam theme, so this is a
    small canvas that draws a pill and tracks the pointer.

    IMPORTANT: the items are created once and never deleted. Recreating them on
    hover deletes the item currently under the pointer, which makes Tk
    re-evaluate the pointer's position, fire <Enter> again, and recreate again
    -- an infinite loop inside Tk's event dispatch that pegs the UI thread and
    spins the beachball. Hover therefore only recolours existing items.
    """

    def __init__(self, master, text, command=None, *, variant="ghost", **kwargs):
        self._bg_root = kwargs.pop("background", BASE)
        palette = {
            "primary": (BLUE, BASE, SKY),
            "ghost": (SURFACE0, TEXT, SURFACE1),
            "quiet": (self._bg_root, SUBTEXT, SURFACE0),
            "danger": (RED, BASE, PINK),
        }
        self._fill, self._fg, self._hover = palette.get(variant, palette["ghost"])
        self._original_fg = self._fg

        self._font = kwargs.pop("font", FONT_BODY)
        self._height = kwargs.pop("height", 30)
        self._pad_x = kwargs.pop("padx", SPACE_4)
        self.command = command
        self._text = text
        self._hovering = False
        self._enabled = True

        font = tkfont.Font(font=self._font)
        self._text_width = int(font.measure(text))
        self._width = self._text_width + self._pad_x * 2

        super().__init__(
            master,
            bg=self._bg_root,
            highlightthickness=0,
            bd=0,
            width=self._width,
            height=self._height,
            scrollregion=(0, 0, self._width, self._height),
            **kwargs,
        )

        self._build_items()

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    # -- construction -----------------------------------------------------
    def _build_items(self) -> None:
        """Create the pill and its label exactly once."""
        height = self._height
        radius = height // 2
        self._shape = self.create_rectangle(
            0, 0, self._width, height, fill=self._fill, outline=""
        )
        self._cap_left = self.create_arc(
            -radius,
            0,
            radius,
            height,
            start=90,
            extent=180,
            fill=self._fill,
            outline="",
        )
        self._cap_right = self.create_arc(
            self._width - radius,
            0,
            self._width + radius,
            height,
            start=270,
            extent=180,
            fill=self._fill,
            outline="",
        )
        self._label = self.create_text(
            self._width // 2,
            height // 2,
            text=self._text,
            fill=self._fg,
            font=self._font,
        )

    # -- painting ---------------------------------------------------------
    def _paint(self, fill: str) -> None:
        """Recolour the existing items. Never deletes or creates anything."""
        for item in (self._shape, self._cap_left, self._cap_right):
            self.itemconfigure(item, fill=fill, outline="")

    def _on_enter(self, _event=None):
        if self._enabled and not self._hovering:
            self._hovering = True
            self._paint(self._hover)

    def _on_leave(self, _event=None):
        if self._hovering:
            self._hovering = False
            self._paint(self._fill)

    def _on_click(self, _event=None):
        if self._enabled and self.command:
            self.command()

    # -- api --------------------------------------------------------------
    def set_text(self, text: str) -> None:
        """Relabel and resize the button. Not safe to call while hovered."""
        self._text = text
        font = tkfont.Font(font=self._font)
        self._text_width = int(font.measure(text))
        self._width = self._text_width + self._pad_x * 2
        self.configure(
            width=self._width,
            scrollregion=(0, 0, self._width, self._height),
        )
        self.itemconfigure(self._label, text=text)
        self.coords(self._label, self._width // 2, self._height // 2)
        radius = self._height // 2
        self.coords(self._shape, 0, 0, self._width, self._height)
        self.coords(self._cap_left, -radius, 0, radius, self._height)
        self.coords(
            self._cap_right,
            self._width - radius,
            0,
            self._width + radius,
            self._height,
        )

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._fg = SURFACE2 if not enabled else self._original_fg
        self._hovering = False
        self._paint(self._fill)
        self.itemconfigure(self._label, fill=self._fg)


class SectionLabel(ttk.Label):
    """Small uppercase label heading a settings section."""

    def __init__(self, master, text, **kwargs):
        super().__init__(master, text=text.upper(), style="Section.TLabel", **kwargs)


def hrule(master, color: str = SURFACE1) -> tk.Canvas:
    """A one-pixel horizontal rule."""
    line = tk.Canvas(master, height=1, bg=color, highlightthickness=0, bd=0)
    return line
