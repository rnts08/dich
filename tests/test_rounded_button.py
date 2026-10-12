"""RoundedButton must not recreate canvas items on hover.

Deleting the item under the pointer made Tk re-evaluate the pointer position,
re-fire <Enter>, and recreate again — an infinite loop inside Tk's event
dispatch that pegged the UI thread (the beachball freeze). The invariant is:
hover recolours, never rebuilds.
"""
import pytest

pytestmark = pytest.mark.display


def _button(root, clicks=None):
    import theme
    recorded = [] if clicks is None else clicks
    button = theme.RoundedButton(
        root, text="Open", command=lambda: recorded.append(1))
    button.pack(padx=20, pady=20)
    root.update()
    return button


def test_hover_never_mutates_canvas_items(requires_display):
    root = requires_display
    root.deiconify()
    button = _button(root)
    width = button.winfo_width()

    baseline = len(button.find_all())
    assert baseline > 0

    # The exact sequence that used to beachball: enter, sweep, leave, repeat.
    for _ in range(6):
        button.event_generate("<Enter>", x=5, y=5)
        for x in range(0, width, 10):
            button.event_generate("<Motion>", x=x, y=5)
        button.event_generate("<Leave>")
        root.update()

    assert len(button.find_all()) == baseline


def test_hover_recolours_and_leaves_cleanly(requires_display):
    root = requires_display
    root.deiconify()
    button = _button(root)

    idle = button.itemcget(button._shape, "fill")
    button.event_generate("<Enter>", x=5, y=5)
    root.update()
    hovered = button.itemcget(button._shape, "fill")
    assert hovered != idle

    button.event_generate("<Leave>")
    root.update()
    assert button.itemcget(button._shape, "fill") == idle


def test_disabled_button_ignores_clicks(requires_display):
    root = requires_display
    root.deiconify()
    clicks = []
    button = _button(root, clicks)

    button.set_enabled(False)
    button.event_generate("<Button-1>", x=5, y=5)
    root.update()
    assert clicks == []

    button.set_enabled(True)
    button.event_generate("<Button-1>", x=5, y=5)
    root.update()
    assert clicks == [1]


def test_double_enter_does_not_reenter(requires_display):
    root = requires_display
    root.deiconify()
    button = _button(root)

    button.event_generate("<Enter>", x=5, y=5)
    button.event_generate("<Enter>", x=5, y=5)   # second enter while hovered
    root.update()
    assert button._hovering is True
    assert len(button.find_all()) == 4           # shape, two caps, label
