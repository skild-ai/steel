"""The shelf's side bar: one imgui window on the left, in place of Newton's panel.

In tune mode it has two tabs, Tuner and Scene; in play mode it shows playback and the Scene
controls (see modes.Tuner and modes.PlayHost, the ``host`` it draws for). The panel only draws
and edits the host's state (``edits``, ``pending``, ``overrides``, ``rebuild``, ``save``,
recording); the host's loop applies the changes between frames.

Every setting gets a control from :mod:`common.ranges`: a slider with a typed field (linear or
logarithmic), a stepper for small integers, segmented buttons for options, a switch for booleans.
A faint tick on each slider marks the value saved in the preset file; a gold dot marks an edit that
is not applied yet. Themes (colours, button shape and size) live in :data:`THEMES`; pick one with
``SHELF_PANEL_THEME``, ``SHELF_PANEL_BUTTONS`` and ``SHELF_PANEL_SIZE`` while trying them out.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from common import defaults, gallery, live, looks, presets, ranges

WIDTH = 460  # panel width in unscaled pixels
STATS_WIDTH = 168  # Newton's FPS / model stats overlay, top-right


def _hex(h: str, a: float = 1.0) -> tuple[float, float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


def _rgba(c, alpha: float | None = None):
    r, g, b, a = c
    return (r, g, b, a if alpha is None else alpha)


# --------------------------------------------------------------------------- themes
# "bg" is the bar's colour; its alpha eases between "bg_idle" when the cursor is elsewhere and
# "bg_hover" when the cursor is on the bar. "on_accent" is text drawn on the accent colour. Optional:
# "border" outlines controls, "value" colours the numbers in the stats card, "font" is "sans" (Roboto) or
# "mono" (Inconsolata), "rounding" is the bar's corner radius.
THEMES: dict[str, dict[str, Any]] = {
    # bright cream, sky-blue accent, sun-yellow primary, saturated friendly tier colours
    "sunny": {
        "bg": _hex("FFFBF2"),
        "bg_idle": 0.72,
        "bg_hover": 0.98,
        "card": (0, 0, 0, 0.05),
        "text": _hex("2B2D42"),
        "muted": _hex("7D7F8E"),
        "faint": (0, 0, 0, 0.10),
        "frame": (0, 0, 0, 0.06),
        "button": (0, 0, 0, 0.07),
        "popup": (1.0, 0.99, 0.97, 1.0),
        "accent": _hex("3DA5D9"),
        "on_accent": (1, 1, 1, 1),
        "primary": _hex("FFC53D"),
        "primary_text": _hex("2B2D42"),
        "tiers": {"live": _hex("3FA35B"), "model": _hex("3DA5D9"), "swap": _hex("F28C28"), "rebuild": _hex("EF476F")},
    },
    # warm charcoal, teal accent, gold primary: sits with the studio look
    "studio": {
        "bg": (0.102, 0.102, 0.114, 0.94),
        "bg_idle": 0.74,
        "bg_hover": 0.97,
        "card": (1, 1, 1, 0.045),
        "text": _hex("EDE9E2"),
        "muted": _hex("9A958E"),
        "faint": (1, 1, 1, 0.08),
        "frame": (1, 1, 1, 0.07),
        "button": (1, 1, 1, 0.08),
        "popup": (0.13, 0.13, 0.145, 0.98),
        "accent": _hex("6FB7C9"),
        "primary": _hex("F2C14E"),
        "primary_text": (0.1, 0.1, 0.1, 1.0),
        "tiers": {"live": _hex("6FA86B"), "model": _hex("7495C9"), "swap": _hex("F2C14E"), "rebuild": _hex("E66145")},
    },
    # cool graphite, one coral accent for everything interactive
    "graphite": {
        "bg": (0.082, 0.090, 0.110, 0.95),
        "on_accent": (1, 1, 1, 1),
        "card": (1, 1, 1, 0.04),
        "text": _hex("E6EAF0"),
        "muted": _hex("8B93A1"),
        "faint": (1, 1, 1, 0.08),
        "frame": (1, 1, 1, 0.065),
        "button": (1, 1, 1, 0.075),
        "popup": (0.11, 0.12, 0.145, 0.98),
        "accent": _hex("E66145"),
        "primary": _hex("E66145"),
        "primary_text": (1, 1, 1, 1),
        "tiers": {"live": _hex("6FA86B"), "model": _hex("7495C9"), "swap": _hex("F2C14E"), "rebuild": _hex("E66145")},
    },
    # light paper over the bright scene, deep teal accent
    "paper": {
        "bg": (0.957, 0.945, 0.925, 0.96),
        "on_accent": (1, 1, 1, 1),
        "card": (0, 0, 0, 0.045),
        "text": _hex("24221F"),
        "muted": _hex("736D66"),
        "faint": (0, 0, 0, 0.10),
        "frame": (0, 0, 0, 0.06),
        "button": (0, 0, 0, 0.06),
        "popup": (0.985, 0.98, 0.97, 1.0),
        "accent": _hex("2B998F"),
        "primary": _hex("2B998F"),
        "primary_text": (1, 1, 1, 1),
        "tiers": {"live": _hex("4E8A49"), "model": _hex("3A61A0"), "swap": _hex("B8860B"), "rebuild": _hex("C94A32")},
    },
    # midnight blue, amber primary, teal secondary
    "midnight": {
        "bg": (0.059, 0.090, 0.149, 0.95),
        "card": (1, 1, 1, 0.05),
        "text": _hex("E6ECF5"),
        "muted": _hex("8C99AD"),
        "faint": (1, 1, 1, 0.09),
        "frame": (1, 1, 1, 0.07),
        "button": (1, 1, 1, 0.08),
        "popup": (0.09, 0.125, 0.19, 0.98),
        "accent": _hex("6FB7C9"),
        "primary": _hex("F2C14E"),
        "primary_text": (0.1, 0.1, 0.1, 1.0),
        "tiers": {"live": _hex("6FA86B"), "model": _hex("7495C9"), "swap": _hex("F2C14E"), "rebuild": _hex("E66145")},
    },
    # steel-webgpu's HUD (apps/viewer/src/steel/styles.ts): blue-black panel, outlined controls, one
    # light-blue accent, monospace text; sits with the steel look
    "steel": {
        "bg": _hex("14181C"),
        "bg_idle": 0.82,
        "bg_hover": 0.96,
        "card": (1, 1, 1, 0.05),
        "text": _hex("E6EBEF"),
        "muted": _hex("9AA7B2"),
        "faint": (1, 1, 1, 0.10),
        "frame": (1, 1, 1, 0.08),
        "button": (1, 1, 1, 0.10),
        "border": _hex("45525E"),
        "popup": _hex("1D2125", 0.98),
        "accent": _hex("9FD3FF"),
        "on_accent": _hex("14181C"),
        "primary": _hex("9FD3FF"),
        "primary_text": _hex("14181C"),
        "value": _hex("9FD3FF"),
        "font": "mono",
        "rounding": 8.0,
        "tiers": {"live": _hex("A3D9A5"), "model": _hex("9FD3FF"), "swap": _hex("FFD54F"), "rebuild": _hex("FFB4A8")},
    },
}
THEME_CHOICES = {"steel": "steel", "studio": "dark", "sunny": "bright"}  # in the bar's drop-down; the rest by env
DEFAULT_THEME = "steel"
BUTTON_SHAPES = {"pill": 99.0, "soft": 8.0, "sharp": 3.0}  # rounding, unscaled px
BUTTON_SIZES = {"compact": 28.0, "regular": 34.0, "large": 40.0}  # height, unscaled px

TIER_HELP = {
    "live": "Apply sets it on the running solver; the scene keeps going.",
    "model": "Apply writes it into the model arrays and notifies the solver.",
    "swap": "Apply builds a new solver on the running scene; the state is kept.",
    "rebuild": "Apply builds the scene again.",
}
SECTION_ORDER = ("sim", "collision", "solver", "model", "entries")


_examples: dict[str, list[str]] = {}


def _catalog(path: Path) -> dict[str, list[str]]:
    """Section -> example names on the shelf that holds ``path`` (an example's ``example.py``)."""
    if not _examples:
        for found in sorted(path.parents[2].glob("*/*/example.py")):
            _examples.setdefault(found.parent.parent.name, []).append(found.parent.name)
    return _examples


class Panel:
    def __init__(self, host, mode: str = "tune"):
        self.host = host
        self.mode = mode  # "tune": tabs with the tuner; "play": playback and the scene controls only
        self.theme_name = os.environ.get("SHELF_PANEL_THEME", DEFAULT_THEME)
        if self.theme_name not in THEMES:
            self.theme_name = DEFAULT_THEME
        self.theme = THEMES[self.theme_name]
        self.opacity = 1.0  # eases toward 1 while the cursor is on the bar, 0 when it is elsewhere
        self.stats_opacity = 0.0  # the same for the stats card on the right
        self.last_draw: dict[str, float] = {}  # opacity attribute -> wall-clock time of its last update
        self.save_popup_open = False
        self.save_fields: list[str] | None = None  # [file name, folder] typed in the save dialog
        self.native_dialog = None  # a portable_file_dialogs save_file in progress
        self.button_rounding = BUTTON_SHAPES.get(os.environ.get("SHELF_PANEL_BUTTONS", "soft"), 8.0)
        self.button_height = BUTTON_SIZES.get(os.environ.get("SHELF_PANEL_SIZE", "regular"), 34.0)
        self.fonts: dict[str, tuple] | None = None  # family -> (regular, bold) once loaded
        self.font_error = False
        self.range_overrides: dict[str, ranges.Control] = {}  # widened by typed values
        self.default_filters: dict[str, str] = {}  # section -> filter text of its "at their defaults" list
        self.select_tab: str | None = os.environ.get("SHELF_PANEL_TAB")
        self.minimized = False  # True: the bar is tucked into a pill in the top-left corner (key B)
        self.example_filter = ""  # text typed in the example picker
        self.gallery = gallery.Gallery()  # the example browser overlay (the bar's "browse" button)

    def adopt(self, host, mode: str | None = None) -> None:
        """Draw for ``host`` from now on: the example picked in the bar, running in this window;
        ``mode`` switches the bar between play and the tuner's tabs (the browser opens into either)."""
        self.host = host
        if mode is not None:
            self.mode = mode
        self.range_overrides.clear()
        self.default_filters.clear()
        self.example_filter = ""

    # ------------------------------------------------------------------ fonts and colours
    def _load_fonts(self, imgui) -> None:
        if self.fonts is not None or self.font_error:
            return
        try:
            import imgui_bundle

            fonts = Path(imgui_bundle.__file__).parent / "assets" / "fonts" / "Roboto"
            atlas = imgui.get_io().fonts
            regular = atlas.add_font_from_file_ttf(str(fonts / "Roboto-Regular.ttf"), 16.0)
            bold = atlas.add_font_from_file_ttf(str(fonts / "Roboto-Bold.ttf"), 16.0)
            mono = atlas.add_font_from_file_ttf(str(fonts.parent / "Inconsolata-Medium.ttf"), 16.0)
            self.fonts = {"sans": (regular, bold), "mono": (mono, mono)}
        except Exception:  # noqa: BLE001 (keep Newton's font)
            self.font_error = True

    def _font(self, imgui, bold: bool, size: float) -> bool:
        if self.fonts is None:
            return False
        family = self.fonts.get(self.theme.get("font", "sans"), self.fonts["sans"])
        imgui.push_font(family[1] if bold else family[0], size)
        return True

    def _scale(self) -> float:
        gui = getattr(self.host.viewer, "gui", None)
        ui = getattr(gui, "ui", None)
        return float(getattr(ui, "dpi_scale", 1.0) or 1.0)

    def _u32(self, imgui, color, alpha: float | None = None) -> int:
        return imgui.get_color_u32(imgui.ImVec4(*_rgba(color, alpha)))

    def _vec(self, imgui, color, alpha: float | None = None):
        return imgui.ImVec4(*_rgba(color, alpha))

    # ------------------------------------------------------------------ small drawings
    def _pill(self, imgui, text: str, color, s: float, filled: bool = True) -> None:
        """A small rounded label, drawn inline."""
        pad_x, pad_y = 7 * s, 2 * s
        pushed = self._font(imgui, False, 12.5 * s)
        size = imgui.calc_text_size(text)
        pos = imgui.get_cursor_screen_pos()
        w, h = size.x + 2 * pad_x, size.y + 2 * pad_y
        draw = imgui.get_window_draw_list()
        end = imgui.ImVec2(pos.x + w, pos.y + h)
        if filled:
            draw.add_rect_filled(pos, end, self._u32(imgui, color, 0.22), h / 2)
        else:
            draw.add_rect(pos, end, self._u32(imgui, color, 0.6), h / 2)
        draw.add_text(imgui.ImVec2(pos.x + pad_x, pos.y + pad_y), self._u32(imgui, color), text)
        imgui.dummy(imgui.ImVec2(w, h))
        if pushed:
            imgui.pop_font()

    def _icon_button(self, imgui, ident: str, icon: str, size: float, s: float, fill=None, tooltip: str = "") -> bool:
        """A square button with a drawn icon: ``minimize`` («), ``maximize`` (»), ``play``, ``pause``,
        ``record`` (a red dot) or ``stop`` (a white square). ``fill`` overrides the button colour."""
        th = self.theme
        pos = imgui.get_cursor_screen_pos()
        clicked = imgui.invisible_button(ident, imgui.ImVec2(size, size))
        hovered, held = imgui.is_item_hovered(), imgui.is_item_active()
        draw = imgui.get_window_draw_list()
        end = imgui.ImVec2(pos.x + size, pos.y + size)
        if fill is not None:
            bg = self._u32(imgui, fill, 1.0 if hovered else 0.85)
        elif held:
            bg = self._u32(imgui, th["accent"], 0.55)
        elif hovered:
            bg = self._u32(imgui, th["accent"], 0.35)
        else:
            bg = self._u32(imgui, th["button"])
        draw.add_rect_filled(pos, end, bg, min(self.button_rounding * s, size / 2))
        cx, cy, u = pos.x + size / 2, pos.y + size / 2, size / 34.0  # u: one unit of a 34 px button
        ink = self._u32(imgui, (1, 1, 1, 1) if fill is not None else th["text"])
        if icon in ("minimize", "maximize"):
            d = -1.0 if icon == "minimize" else 1.0  # « points left (tuck the bar away), » right
            for off in (-4.0, 4.0):
                x = cx + off * u
                points = [
                    imgui.ImVec2(x - 3.5 * d * u, cy - 7 * u),
                    imgui.ImVec2(x + 3.5 * d * u, cy),
                    imgui.ImVec2(x - 3.5 * d * u, cy + 7 * u),
                ]
                draw.add_polyline(points, col=ink, thickness=2.2 * u, flags=imgui.ImDrawFlags_.none)
        elif icon == "play":
            draw.add_triangle_filled(
                imgui.ImVec2(cx - 5 * u, cy - 7.5 * u),
                imgui.ImVec2(cx + 8 * u, cy),
                imgui.ImVec2(cx - 5 * u, cy + 7.5 * u),
                ink,
            )
        elif icon == "pause":
            for x in (cx - 5.5 * u, cx + 2 * u):
                draw.add_rect_filled(imgui.ImVec2(x, cy - 7 * u), imgui.ImVec2(x + 3.5 * u, cy + 7 * u), ink, 1.0 * u)
        elif icon == "record":
            draw.add_circle_filled(imgui.ImVec2(cx, cy), 6 * u, self._u32(imgui, th["tiers"]["rebuild"]))
        elif icon == "stop":
            draw.add_rect_filled(
                imgui.ImVec2(cx - 5.5 * u, cy - 5.5 * u), imgui.ImVec2(cx + 5.5 * u, cy + 5.5 * u), ink, 1.5 * u
            )
        if tooltip:
            imgui.set_item_tooltip(tooltip)
        return clicked

    def _fit_label(self, imgui, name: str, width: float, s: float) -> None:
        """A setting's name; long names break at an underscore onto a second line instead of clipping."""
        if imgui.calc_text_size(name).x <= width or "_" not in name:
            imgui.text(name)
            return
        cuts = [i for i, ch in enumerate(name) if ch == "_"]
        cut = min(cuts, key=lambda i: abs(i - len(name) / 2)) + 1
        first, second = name[:cut], name[cut:]
        pushed = (
            self._font(imgui, False, 13.5 * s) if imgui.calc_text_size(max(first, second, key=len)).x > width else False
        )
        imgui.begin_group()
        imgui.text(first)
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() - 4 * s)
        imgui.text(second)
        imgui.end_group()
        if pushed:
            imgui.pop_font()

    def _label(self, imgui, text: str, s: float) -> None:
        """A section label: small muted capitals with a rule."""
        th = self.theme
        imgui.dummy(imgui.ImVec2(0, 4 * s))
        pushed = self._font(imgui, True, 12 * s)
        imgui.text_colored(self._vec(imgui, th["muted"]), text.upper())
        if pushed:
            imgui.pop_font()
        imgui.same_line()
        pos = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail().x
        y = pos.y + imgui.get_text_line_height() / 2
        imgui.get_window_draw_list().add_line(
            imgui.ImVec2(pos.x + 6 * s, y), imgui.ImVec2(pos.x + avail, y), self._u32(imgui, th["faint"])
        )
        imgui.new_line()

    # ------------------------------------------------------------------ controls
    def _slider(self, imgui, value, ctrl: ranges.Control, saved, color, s: float, width: float) -> tuple[bool, Any]:
        """A thin track with a round grab; drawn by hand so it can carry the tier colour and a saved tick."""
        th = self.theme
        h = 24 * s
        r = 7 * s
        imgui.invisible_button("##slider", imgui.ImVec2(width, h))
        active, hovered = imgui.is_item_active(), imgui.is_item_hovered()
        p0, p1 = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        x0, x1 = p0.x + r, p1.x - r
        cy = (p0.y + p1.y) / 2
        new = value
        if active:
            t = (imgui.get_io().mouse_pos.x - x0) / max(x1 - x0, 1.0)
            new = ranges.from_unit(ctrl, t, value)
        draw = imgui.get_window_draw_list()
        track = 4 * s
        draw.add_rect_filled(
            imgui.ImVec2(x0, cy - track / 2), imgui.ImVec2(x1, cy + track / 2), self._u32(imgui, th["faint"]), track
        )
        if ctrl.kind == "log":  # a dot per decade
            decade = 10.0 ** np.ceil(np.log10(max(ctrl.lo, 1e-30)))
            while decade < ctrl.hi:
                tx = x0 + ranges.to_unit(ctrl, decade) * (x1 - x0)
                draw.add_circle_filled(imgui.ImVec2(tx, cy), 1.5 * s, self._u32(imgui, th["muted"], 0.5))
                decade *= 10.0
        gx = x0 + ranges.to_unit(ctrl, new) * (x1 - x0)
        draw.add_rect_filled(
            imgui.ImVec2(x0, cy - track / 2),
            imgui.ImVec2(gx, cy + track / 2),
            self._u32(imgui, color, 0.9 if (hovered or active) else 0.7),
            track,
        )
        if saved is not None and isinstance(saved, (int, float)) and not isinstance(saved, bool):
            sx = x0 + ranges.to_unit(ctrl, saved) * (x1 - x0)
            draw.add_line(
                imgui.ImVec2(sx, cy - 7 * s), imgui.ImVec2(sx, cy + 7 * s), self._u32(imgui, th["muted"], 0.6), 1.5 * s
            )
        grab = r if (hovered or active) else r - 1.5 * s
        draw.add_circle_filled(imgui.ImVec2(gx, cy), grab, self._u32(imgui, color))
        draw.add_circle_filled(imgui.ImVec2(gx, cy), grab * 0.45, self._u32(imgui, th["bg"], 1.0))
        return (new != value), new

    def _field(self, imgui, value, width: float) -> tuple[bool, Any]:
        imgui.set_next_item_width(width)
        if isinstance(value, int):
            return imgui.input_int("##field", value, 0, 0)
        return imgui.input_double("##field", value, 0.0, 0.0, "%.5g")

    def _stepper(self, imgui, value: int, ctrl: ranges.Control, s: float, width: float) -> tuple[bool, int]:
        """[−] value [+], compact; the value is a field you can type into."""
        th = self.theme
        size = 26 * s
        field_w = 52 * s
        changed, new = False, value
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, size / 2)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0, 0))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(4 * s, 0))
        imgui.push_style_color(imgui.Col_.button, self._vec(imgui, th["frame"], th["frame"][3] * 1.8))
        pushed = self._font(imgui, True, 18 * s)
        if imgui.button("−", imgui.ImVec2(size, size)) and value > ctrl.lo:
            changed, new = True, value - 1
        imgui.same_line()
        imgui.set_next_item_width(field_w)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(8 * s, 3 * s))
        imgui.push_style_color(imgui.Col_.frame_bg, self._vec(imgui, (0, 0, 0, 0)))
        if pushed:
            imgui.pop_font()
        typed, v = imgui.input_int("##step", new, 0, 0)
        pushed = self._font(imgui, True, 18 * s)
        imgui.pop_style_color()
        imgui.pop_style_var()
        if typed:
            changed, new = True, v
        imgui.same_line()
        if imgui.button("+", imgui.ImVec2(size, size)) and value < ctrl.hi:
            changed, new = True, value + 1
        if pushed:
            imgui.pop_font()
        imgui.pop_style_color()
        imgui.pop_style_var(3)
        imgui.set_item_tooltip(f"{int(ctrl.lo)} to {int(ctrl.hi)}; type a value to go beyond")
        del width
        return changed, new

    def _segmented(self, imgui, value: str, options: list[str], s: float, width: float) -> tuple[bool, str]:
        th = self.theme
        changed, new = False, value
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(3 * s, 3 * s))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 99.0)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(10 * s, 4 * s))
        pushed = self._font(imgui, False, 13.5 * s)
        x_start = imgui.get_cursor_pos_x()
        used = 0.0
        for opt in options:
            w = imgui.calc_text_size(opt).x + 20 * s
            if used > 0 and used + w > width:
                imgui.new_line()
                imgui.set_cursor_pos_x(x_start)
                used = 0.0
            elif used > 0:
                imgui.same_line()
            selected = opt == value
            if selected:
                imgui.push_style_color(imgui.Col_.button, self._vec(imgui, th["accent"], 0.9))
                imgui.push_style_color(imgui.Col_.button_hovered, self._vec(imgui, th["accent"]))
                imgui.push_style_color(imgui.Col_.button_active, self._vec(imgui, th["accent"]))
                imgui.push_style_color(
                    imgui.Col_.text,
                    self._vec(imgui, th.get("on_accent", (0.08, 0.08, 0.09, 1.0))),
                )
            if imgui.button(opt) and not selected:
                changed, new = True, opt
            if selected:
                imgui.pop_style_color(4)
            used += w + 3 * s
        if pushed:
            imgui.pop_font()
        imgui.pop_style_var(3)
        return changed, new

    def _switch(self, imgui, value: bool, s: float) -> tuple[bool, bool]:
        th = self.theme
        w, h = 40 * s, 22 * s
        imgui.invisible_button("##switch", imgui.ImVec2(w, h))
        changed = imgui.is_item_clicked()
        new = (not value) if changed else value
        p0 = imgui.get_item_rect_min()
        draw = imgui.get_window_draw_list()
        on = new
        draw.add_rect_filled(
            p0,
            imgui.ImVec2(p0.x + w, p0.y + h),
            self._u32(imgui, th["accent"] if on else th["faint"], 0.9 if on else 1.0),
            h / 2,
        )
        kx = p0.x + (w - h / 2) if on else p0.x + h / 2
        draw.add_circle_filled(
            imgui.ImVec2(kx, p0.y + h / 2), h / 2 - 3 * s, self._u32(imgui, (1, 1, 1, 1) if on else th["muted"])
        )
        imgui.same_line()
        imgui.align_text_to_frame_padding()
        imgui.text_colored(self._vec(imgui, th["muted"]), "on" if on else "off")
        return changed, new

    def _text_field(self, imgui, value: Any, width: float) -> tuple[bool, Any]:
        imgui.set_next_item_width(width)
        text = value if isinstance(value, str) else json.dumps(value)
        changed, new_text = imgui.input_text("##text", text)
        if not changed:
            return False, value
        return True, (new_text if isinstance(value, str) else _parse_literal(new_text, value))

    def _control(self, imgui, key: str, shown: Any, saved: Any, tier: str, s: float, width: float) -> tuple[bool, Any]:
        """Draw the right control for ``shown`` and return (changed, new value)."""
        t = self.host
        ctrl = self.range_overrides.get(key) or ranges.control(key, shown, t.preset)
        color = self.theme["tiers"][tier]
        if ctrl.kind == "bool":
            return self._switch(imgui, bool(shown), s)
        if ctrl.kind == "enum":
            if shown not in ctrl.options:
                ctrl.options = [str(shown), *ctrl.options]
            return self._segmented(imgui, str(shown), ctrl.options, s, width)
        if ctrl.kind == "text":
            return self._text_field(imgui, shown, width)
        if isinstance(shown, int) and ctrl.kind == "int" and ctrl.hi - ctrl.lo <= 64:
            changed, new = self._stepper(imgui, shown, ctrl, s, width)
            if changed and not (ctrl.lo <= new <= ctrl.hi):
                self.range_overrides[key] = ranges.widen(ctrl, new)
            return changed, new
        if not (ctrl.lo <= float(shown) <= ctrl.hi):
            ctrl = self.range_overrides[key] = ranges.widen(ctrl, shown)
        field_w = 68 * s
        changed, new = self._slider(
            imgui, shown, ctrl, saved, color, s, width - field_w - imgui.get_style().item_spacing.x
        )
        imgui.same_line()
        typed, typed_value = self._field(imgui, new, field_w)
        if typed:
            changed, new = True, typed_value
            if not (ctrl.lo <= float(new) <= ctrl.hi):
                self.range_overrides[key] = ranges.widen(ctrl, new)
        return changed, new

    # ------------------------------------------------------------------ the window
    def draw(self, imgui) -> None:
        self._load_fonts(imgui)
        s = self._scale()
        io = imgui.get_io()
        if self.gallery.open:  # the browser covers the window; the bar waits underneath
            pushed = self._push_theme(imgui, s, 1.0)
            self.gallery.draw(imgui, self, s, io)
            self._pop_theme(imgui, pushed)
            return
        # one bar on the left, where Newton's panel was (the bar turns Newton's own panel off)
        self._toggle_key(imgui, io)
        if not self.minimized:  # the pill sizes itself to its buttons
            imgui.set_next_window_pos(imgui.ImVec2(10 * s, 10 * s), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size(imgui.ImVec2(WIDTH * s, io.display_size.y - 20 * s), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size_constraints(
                imgui.ImVec2(320 * s, 200 * s), imgui.ImVec2(io.display_size.x, io.display_size.y)
            )

        pushed = self._push_theme(imgui, s, self.opacity)

        if self.minimized:
            self._minimized_bar(imgui, s, io)
            self._pop_theme(imgui, pushed)
            return

        flags = imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings
        opened = imgui.begin("Shelf", None, flags)[0]
        if opened:
            self._track_hover(imgui, io)
        if opened and self.mode == "play":
            self._play_body(imgui, s)
        elif opened:
            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(18 * s, 7 * s))
            tab_row = imgui.get_cursor_pos()
            tab_height = imgui.get_frame_height()
            if imgui.begin_tab_bar("##tabs", imgui.TabBarFlags_.fitting_policy_scroll):
                select = self.select_tab  # SHELF_PANEL_TAB opens a tab once (for screenshots and tests)
                self.select_tab = None
                tuner_flags = imgui.TabItemFlags_.set_selected if select == "tuner" else 0
                scene_flags = imgui.TabItemFlags_.set_selected if select == "scene" else 0
                tuner_open = imgui.begin_tab_item("Tuner", None, tuner_flags)[0]
                if tuner_open:
                    imgui.end_tab_item()
                scene_open = imgui.begin_tab_item("Scene", None, scene_flags)[0]
                if scene_open:
                    imgui.end_tab_item()
                imgui.end_tab_bar()
            imgui.pop_style_var()
            # the minimize button sits at the right end of the tab row, where the two tabs leave room
            below = imgui.get_cursor_pos()
            size = tab_height - 2 * s
            right = below.x + imgui.get_content_region_avail().x
            imgui.set_cursor_pos(imgui.ImVec2(right - size, tab_row.y + 1 * s))
            self._minimize_button(imgui, size, s)
            imgui.set_cursor_pos(below)
            imgui.dummy(imgui.ImVec2(0, 6 * s))
            if tuner_open:
                self._tuner_tab(imgui, s)
            elif scene_open:
                self._scene_tab(imgui, s)
        if opened:
            self._save_dialog(imgui, s)
        imgui.end()
        self._pop_theme(imgui, pushed)

    # ------------------------------------------------------------------ stats card
    def draw_stats(self, gui) -> None:
        """Newton's FPS and model-size overlay in the top-right corner, drawn as a card in the bar's
        theme: rounded, translucent until the cursor is on it, labels left and counts right."""
        if not gui.is_available or self.gallery.open:  # the browser covers the card's corner
            return
        imgui = gui.ui.imgui
        self._load_fonts(imgui)
        th, s, io = self.theme, self._scale(), imgui.get_io()
        viewer = gui._viewer
        model = viewer.model
        imgui.set_next_window_pos(imgui.ImVec2(io.display_size.x - 10 * s, 10 * s), pivot=imgui.ImVec2(1.0, 0.0))
        flags = (
            imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.no_focus_on_appearing
            | imgui.WindowFlags_.no_nav
            | imgui.WindowFlags_.no_move
        )
        pushed = self._push_theme(imgui, s, self.stats_opacity)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(14 * s, 12 * s))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8 * s, 3 * s))
        width = 150 * s
        if imgui.begin("Performance Stats", None, flags)[0]:
            self._track_hover(imgui, io, "stats_opacity")
            font = self._font(imgui, True, 22 * s)
            imgui.text_colored(self._vec(imgui, th.get("value", th["text"])), f"{gui._current_fps:.1f}")
            if font:
                imgui.pop_font()
            imgui.same_line(0, 6 * s)
            font = self._font(imgui, False, 12.5 * s)
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 7 * s)
            imgui.text_colored(self._vec(imgui, th["muted"]), "fps")
            if font:
                imgui.pop_font()
            imgui.dummy(imgui.ImVec2(width, 2 * s))

            def rule():
                pos = imgui.get_cursor_screen_pos()
                imgui.get_window_draw_list().add_line(
                    imgui.ImVec2(pos.x, pos.y + 2 * s),
                    imgui.ImVec2(pos.x + width, pos.y + 2 * s),
                    self._u32(imgui, th["faint"]),
                    1.0 * s,
                )
                imgui.dummy(imgui.ImVec2(width, 5 * s))

            def row(label: str, value: int, indent: float = 0.0, size: float = 14 * s):
                font = self._font(imgui, False, size)
                start = imgui.get_cursor_pos_x()
                imgui.set_cursor_pos_x(start + indent)
                imgui.text_colored(self._vec(imgui, th["muted"]), label)
                text = f"{value:,}"
                imgui.same_line(start + width - imgui.calc_text_size(text).x)
                imgui.text_colored(self._vec(imgui, th.get("value", th["text"])), text)
                if font:
                    imgui.pop_font()

            if model is not None:
                rule()
                row("worlds", model.world_count)
                row("bodies", model.body_count)
                row("shapes", model.shape_count)
                if getattr(gui, "_shape_visual_count", None) is not None:
                    row("visual", gui._shape_visual_count, 10 * s, 12.5 * s)
                    row("collision", gui._shape_collision_count, 10 * s, 12.5 * s)
                row("joints", model.joint_count)
                row("particles", model.particle_count)
                row("springs", model.spring_count)
                row("triangles", model.tri_count)
                row("edges", model.edge_count)
                row("tetrahedra", model.tet_count)
            objects = getattr(viewer, "objects", None)
            if objects is not None:
                rule()
                row("unique objects", len(objects))
            for callback in gui._ui_callbacks["stats"]:
                callback(imgui)
        imgui.end()
        imgui.pop_style_var(2)
        self._pop_theme(imgui, pushed)

    # ------------------------------------------------------------------ minimized
    def _toggle_key(self, imgui, io) -> None:
        """B minimizes the bar or brings it back (Newton binds no B; H still hides all of the UI)."""
        if io.want_text_input or imgui.is_popup_open("", imgui.PopupFlags_.any_popup):
            return
        if imgui.is_key_pressed(imgui.Key.b, False):
            self.minimized = not self.minimized

    def _minimize_button(self, imgui, size: float, s: float) -> None:
        if self._icon_button(imgui, "##minimize", "minimize", size, s, tooltip="Minimize the side bar  ( B )"):
            self.minimized = True

    def _minimized_bar(self, imgui, s: float, io) -> None:
        """The bar tucked away: a pill in the top-left corner with the button that brings the bar
        back, the example's name, play / pause and record / stop."""
        t, th = self.host, self.theme
        viewer = t.viewer
        imgui.set_next_window_pos(imgui.ImVec2(10 * s, 10 * s), imgui.Cond_.always)
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_saved_settings
        )
        size = self.button_height * s
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(7 * s, 7 * s))
        imgui.push_style_var(imgui.StyleVar_.window_rounding, (size + 14 * s) / 2)
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(6 * s, 0))
        if imgui.begin("Shelf (minimized)", None, flags)[0]:
            self._track_hover(imgui, io)
            if self._icon_button(imgui, "##maximize", "maximize", size, s, tooltip="Show the side bar  ( B )"):
                self.minimized = False
            imgui.same_line(0, 10 * s)
            pushed = self._font(imgui, True, 16 * s)
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + (size - imgui.get_text_line_height()) / 2)
            imgui.text(t.example_name)
            if pushed:
                imgui.pop_font()
            imgui.same_line(0, 10 * s)
            paused = bool(getattr(viewer, "_paused", False))
            imgui.set_cursor_pos_y(7 * s)
            if self._icon_button(
                imgui,
                "##play",
                "play" if paused else "pause",
                size,
                s,
                fill=th["accent"] if paused else None,
                tooltip="Play  ( Space )" if paused else "Pause  ( Space )",
            ):
                viewer._paused = not paused
            imgui.same_line()
            recording = t.recorder is not None
            if self._icon_button(
                imgui,
                "##record",
                "stop" if recording else "record",
                size,
                s,
                fill=th["tiers"]["rebuild"] if recording else None,
                tooltip="Stop recording; then save or discard it" if recording else "Record the viewport",
            ):
                t.stop_recording() if recording else t.start_recording()
            if t.recorder is not None:  # read again: a click on Stop just ended it
                imgui.same_line(0, 8 * s)
                imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + (size - imgui.get_text_line_height()) / 2)
                imgui.text_colored(self._vec(imgui, th["tiers"]["rebuild"]), f"{t.recorder.count}")
                imgui.same_line(0, 4 * s)
            self._save_dialog(imgui, s)
        imgui.end()
        imgui.pop_style_var(3)

    def _push_theme(self, imgui, s: float, opacity: float) -> tuple[int, int, bool]:
        """Push the theme's style and colours (window background at ``opacity`` between idle and
        hover) and the bar's font; :meth:`_pop_theme` undoes it."""
        th = self.theme
        sv, col = imgui.StyleVar_, imgui.Col_
        styles = [
            (sv.window_rounding, th.get("rounding", 12.0) * s),
            (sv.tab_rounding, 8.0 * s),
            (sv.tab_bar_border_size, 0.0),
            (sv.tab_bar_overline_size, 0.0),
            (sv.window_border_size, 0.0),
            (sv.frame_border_size, 1.0 if "border" in th else 0.0),
            (sv.window_padding, imgui.ImVec2(18 * s, 16 * s)),
            (sv.frame_rounding, 6.0 * s),
            (sv.frame_padding, imgui.ImVec2(10 * s, 5 * s)),
            (sv.item_spacing, imgui.ImVec2(8 * s, 7 * s)),
            (sv.item_inner_spacing, imgui.ImVec2(8 * s, 6 * s)),
            (sv.grab_rounding, 6.0 * s),
            (sv.child_rounding, 10.0 * s),
            (sv.scrollbar_size, 8.0 * s),
            (sv.scrollbar_rounding, 4.0 * s),
            (sv.cell_padding, imgui.ImVec2(6 * s, 6 * s)),
        ]
        colors = [
            (
                col.window_bg,
                _rgba(
                    th["bg"],
                    th.get("bg_idle", 0.6) + (th.get("bg_hover", 0.96) - th.get("bg_idle", 0.6)) * opacity,
                ),
            ),
            (col.child_bg, th["card"]),
            (col.text, th["text"]),
            (col.text_disabled, th["muted"]),
            (col.border, th.get("border", (0, 0, 0, 0))),
            (col.frame_bg, th["frame"]),
            (col.frame_bg_hovered, _rgba(th["frame"], th["frame"][3] * 1.6)),
            (col.frame_bg_active, _rgba(th["frame"], th["frame"][3] * 2.0)),
            (col.button, th["button"]),
            (col.button_hovered, _rgba(th["accent"], 0.35)),
            (col.button_active, _rgba(th["accent"], 0.55)),
            (col.check_mark, th["accent"]),
            (col.slider_grab, th["accent"]),
            (col.slider_grab_active, th["accent"]),
            (col.header, _rgba(th["button"], th["button"][3] * 0.8)),
            (col.header_hovered, _rgba(th["button"], th["button"][3] * 1.4)),
            (col.header_active, _rgba(th["button"], th["button"][3] * 1.8)),
            (col.separator, th["faint"]),
            (col.plot_lines, th["accent"]),
            (col.plot_histogram, th["accent"]),
            (col.scrollbar_bg, (0, 0, 0, 0)),
            (col.scrollbar_grab, _rgba(th["muted"], 0.35)),
            (col.popup_bg, th["popup"]),
            (col.table_row_bg, (0, 0, 0, 0)),
            (col.table_row_bg_alt, _rgba(th["card"], th["card"][3] * 0.5)),
            (col.table_border_light, (0, 0, 0, 0)),
            (col.table_border_strong, (0, 0, 0, 0)),
            (col.text_selected_bg, _rgba(th["accent"], 0.35)),
            (col.tab, (0, 0, 0, 0)),
            (col.tab_hovered, _rgba(th["button"], th["button"][3] * 1.5)),
            (col.tab_selected, th["button"]),
            (col.tab_dimmed, (0, 0, 0, 0)),
            (col.tab_dimmed_selected, th["button"]),
            (col.tab_selected_overline, th["accent"]),
        ]
        for var, value in styles:
            imgui.push_style_var(var, value)
        for which, value in colors:
            imgui.push_style_color(which, imgui.ImVec4(*value))
        return len(styles), len(colors), self._font(imgui, False, 15.5 * s)

    @staticmethod
    def _pop_theme(imgui, pushed: tuple[int, int, bool]) -> None:
        n_styles, n_colors, font = pushed
        if font:
            imgui.pop_font()
        imgui.pop_style_color(n_colors)
        imgui.pop_style_var(n_styles)

    def set_theme(self, name: str) -> None:
        if name in THEMES:
            self.theme_name = name
            self.theme = THEMES[name]

    def _look_and_theme(self, imgui, s: float) -> None:
        """Right-aligned drop-downs: the scene's look and the bar's theme (bright or dark)."""
        t = self.host
        imgui.same_line()
        avail = imgui.get_content_region_avail().x
        spacing = imgui.get_style().item_spacing.x
        has_look = hasattr(t.viewer, "renderer")
        w_look, w_theme = 104 * s, 92 * s
        total = w_theme + (w_look + spacing if has_look else 0.0)
        if avail < total + 8 * s:  # the row is full (e.g. a "tuned" pill is showing): drop to the next line
            imgui.new_line()
            avail = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, avail - total))
        if has_look:
            look_names = looks.names()
            look_index = look_names.index(looks.current(t.viewer) or look_names[0])
            imgui.set_next_item_width(w_look)
            changed, look_index = imgui.combo("##look", look_index, look_names)
            if changed:
                looks.switch(t.viewer, look_names[look_index])
            imgui.set_item_tooltip("look: the scene's palette and lighting")
            imgui.same_line()
        names = list(THEME_CHOICES)
        if self.theme_name not in names:
            names.append(self.theme_name)
        labels = [THEME_CHOICES.get(n, n) for n in names]
        index = names.index(self.theme_name)
        imgui.set_next_item_width(w_theme)
        changed, index = imgui.combo("##theme", index, labels)
        if changed:
            self.set_theme(names[index])
        imgui.set_item_tooltip("theme: the bar's colours")

    def _save_dialog(self, imgui, s: float) -> None:
        """After Stop: a modal to save the recording under a name and folder, or discard it."""
        t, th = self.host, self.theme
        pending = getattr(t, "pending_recording", None)
        if pending is None:
            self.save_popup_open = False
            self.save_fields = None
            return
        if not self.save_popup_open:
            imgui.open_popup("Save recording")
            self.save_popup_open = True
            self.save_fields = [pending.name, str(pending.folder)]
        io = imgui.get_io()
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size.x / 2, io.display_size.y / 2), imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5)
        )
        imgui.set_next_window_size(imgui.ImVec2(540 * s, 0))
        flags = (
            imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.no_title_bar
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(20 * s, 18 * s))
        imgui.push_style_var(imgui.StyleVar_.popup_rounding, 12.0 * s)
        if imgui.begin_popup_modal("Save recording", None, flags)[0]:
            pushed = self._font(imgui, True, 20 * s)
            imgui.text("Recording stopped")
            if pushed:
                imgui.pop_font()
            imgui.text_colored(
                self._vec(imgui, th["muted"]),
                f"{pending.frames} frames  ·  {pending.seconds:.1f} s "
                + ("of simulation" if pending.mode == "sim" else "real time")
                + f"  ·  {pending.size_mb():.1f} MB",
            )
            imgui.dummy(imgui.ImVec2(0, 6 * s))
            fields = self.save_fields
            imgui.text_colored(self._vec(imgui, th["muted"]), "file name")
            imgui.set_next_item_width(-1)
            _, fields[0] = imgui.input_text("##name", fields[0])
            imgui.text_colored(self._vec(imgui, th["muted"]), "folder")
            imgui.set_next_item_width(-1)
            _, fields[1] = imgui.input_text("##folder", fields[1])
            imgui.dummy(imgui.ImVec2(0, 8 * s))

            avail = imgui.get_content_region_avail().x
            spacing = imgui.get_style().item_spacing.x
            w = (avail - 2 * spacing) / 3
            h = self.button_height * s
            imgui.push_style_var(imgui.StyleVar_.frame_rounding, self.button_rounding * s)
            imgui.push_style_color(imgui.Col_.button, self._vec(imgui, th["primary"], 0.9))
            imgui.push_style_color(imgui.Col_.button_hovered, self._vec(imgui, th["primary"]))
            imgui.push_style_color(imgui.Col_.button_active, self._vec(imgui, th["primary"], 0.75))
            imgui.push_style_color(imgui.Col_.text, self._vec(imgui, th["primary_text"]))
            pushed = self._font(imgui, True, 15.5 * s)
            if imgui.button("Save", imgui.ImVec2(w, h)):
                t.save_recording(fields[1], fields[0])
                imgui.close_current_popup()
            if pushed:
                imgui.pop_font()
            imgui.pop_style_color(4)
            imgui.same_line()
            if imgui.button("Save as…", imgui.ImVec2(w, h)):
                self._open_native_save(fields)
            imgui.set_item_tooltip("Pick the file in a system dialog")
            imgui.same_line()
            imgui.push_style_color(imgui.Col_.button_hovered, self._vec(imgui, th["tiers"]["rebuild"], 0.6))
            imgui.push_style_color(imgui.Col_.button_active, self._vec(imgui, th["tiers"]["rebuild"], 0.8))
            if imgui.button("Discard", imgui.ImVec2(w, h)):
                t.discard_recording()
                imgui.close_current_popup()
            imgui.pop_style_color(2)
            imgui.pop_style_var()
            if self.native_dialog is not None:
                self._poll_native_save(imgui)
            imgui.end_popup()
        imgui.pop_style_var(2)

    def _open_native_save(self, fields: list[str]) -> None:
        try:
            from imgui_bundle import portable_file_dialogs as pfd

            default = str(Path(fields[1]).expanduser() / fields[0])
            self.native_dialog = pfd.save_file("Save recording", default, ["MP4 video", "*.mp4", "All files", "*"])
        except Exception as e:  # noqa: BLE001 (no dialog backend: the fields above still work)
            self.native_dialog = None
            self.host.message = f"no system dialog available ({e}); use the fields"

    def _poll_native_save(self, imgui) -> None:
        dialog = self.native_dialog
        try:
            ready = dialog.ready(0)
        except TypeError:
            ready = dialog.ready()
        if not ready:
            return
        result = dialog.result()
        self.native_dialog = None
        if result:
            chosen = Path(result)
            self.host.save_recording(chosen.parent, chosen.name)
            imgui.close_current_popup()

    def _track_hover(self, imgui, io, attr: str = "opacity") -> None:
        """Ease ``self.<attr>`` toward 1 while the cursor is on the current window (or a control of it
        is in use), toward 0 otherwise."""
        flags = (
            imgui.HoveredFlags_.root_and_child_windows
            | imgui.HoveredFlags_.allow_when_blocked_by_active_item
            | imgui.HoveredFlags_.allow_when_blocked_by_popup
        )
        pos, size, m = imgui.get_window_pos(), imgui.get_window_size(), io.mouse_pos
        inside = pos.x <= m.x <= pos.x + size.x and pos.y <= m.y <= pos.y + size.y
        engaged = imgui.is_window_hovered(flags) or inside or imgui.is_any_item_active()
        engaged = engaged or imgui.is_popup_open("", imgui.PopupFlags_.any_popup)
        target = 1.0 if engaged else 0.0
        now = time.perf_counter()  # wall clock: io.delta_time is not reliable under Newton's loop
        last = self.last_draw.get(attr)
        dt = min(0.1, now - last) if last else 0.016
        self.last_draw[attr] = now
        value = getattr(self, attr)
        setattr(self, attr, value + (target - value) * min(1.0, dt * 10.0))  # ~0.3 s to settle

    # ------------------------------------------------------------------ play mode
    def _play_body(self, imgui, s: float) -> None:
        """Play mode: playback, the example's name and look, then Newton's scene controls."""
        t, th = self.host, self.theme
        self._playback(imgui, s, minimize=True)
        self._title(imgui, s)
        self._look_and_theme(imgui, s)
        preset = getattr(t.example, "preset", None)
        if preset is not None:
            imgui.text_colored(
                self._vec(imgui, th["muted"]),
                f"preset {preset.name}  ·  tune it with  python -m shelf tune {t.example_name}",
            )
        if t.message:
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(self._vec(imgui, th["muted"]), t.message)
            imgui.pop_text_wrap_pos()
        imgui.dummy(imgui.ImVec2(0, 6 * s))
        self._scene_tab(imgui, s)

    def _title(self, imgui, s: float) -> None:
        """The example's name and section, and the picker that switches the window to another example."""
        t, th = self.host, self.theme
        pushed = self._font(imgui, True, 24 * s)
        if t.path is None:  # the browser-only window, while the example picked there loads
            imgui.text("examples")
            if pushed:
                imgui.pop_font()
            return
        imgui.text(t.example_name)
        if pushed:
            imgui.pop_font()
        imgui.same_line()
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + 4 * s)
        imgui.align_text_to_frame_padding()
        pushed = self._font(imgui, False, 12.5 * s)
        imgui.text_colored(self._vec(imgui, th["muted"]), t.path.parent.parent.name)
        if pushed:
            imgui.pop_font()

        # a recording belongs to this example: finish it before leaving
        busy = t.recorder is not None or t.pending_recording is not None or t.next_example is not None
        browse_w = imgui.calc_text_size("browse").x + 2 * imgui.get_style().frame_padding.x + 8 * s
        imgui.set_next_item_width(imgui.get_content_region_avail().x - browse_w - imgui.get_style().item_spacing.x)
        imgui.begin_disabled(busy)
        label = f"loading {t.next_example}…" if t.next_example else "switch example…"
        if imgui.begin_combo("##example", label, imgui.ComboFlags_.height_largest):
            if imgui.is_window_appearing():
                imgui.set_keyboard_focus_here()
            imgui.set_next_item_width(-1)
            _, self.example_filter = imgui.input_text_with_hint("##example_filter", "filter", self.example_filter)
            word = self.example_filter.strip().lower()
            for section, names in _catalog(t.path).items():
                names = [n for n in names if word in n.lower() or word in section.lower()]
                if not names:
                    continue
                imgui.separator_text(section)
                for name in names:
                    if imgui.selectable(name, name == t.example_name)[0] and name != t.example_name:
                        t.next_example = name
                        t.message = f"loading {name}…"
            imgui.end_combo()
        if busy and t.next_example is None:
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip("Stop the recording and save or discard it first")
        else:
            imgui.set_item_tooltip("Open another example in this window, at its default preset")
        imgui.same_line()
        if imgui.button("browse", imgui.ImVec2(browse_w, 0)):
            self.gallery.show(t)
        imgui.set_item_tooltip("Every example with a preview, by section; a chip opens it with that preset")
        imgui.end_disabled()

    # ------------------------------------------------------------------ tabs
    def _tuner_tab(self, imgui, s: float) -> None:
        t, th = self.host, self.theme
        self._playback(imgui, s)
        self._header(imgui, s)
        if t.example is None:
            imgui.dummy(imgui.ImVec2(0, 4 * s))
            building = not t.ran or t.rebuild or t.next_example is not None
            color = th["muted"] if building else th["tiers"]["rebuild"]
            text = t.message if t.next_example else "building the scene…" if building else (t.message or "no scene")
            imgui.text_colored(self._vec(imgui, color), text)
            if building:
                return
        self._metrics(imgui, s)
        self._settings(imgui, s)
        self._actions(imgui, s)
        self._extras(imgui, s)

    def _playback(self, imgui, s: float, minimize: bool = False) -> None:
        """Play / pause, step, reset, record: the controls Newton's panel had, minus 'pause rendering'.

        ``minimize`` ends the row with the button that tucks the bar away (play mode, which has no
        tab row for it)."""
        t, th = self.host, self.theme
        viewer = t.viewer
        avail = imgui.get_content_region_avail().x
        spacing = imgui.get_style().item_spacing.x
        h = self.button_height * s
        if minimize:
            avail -= h + spacing
        w = (avail - 3 * spacing) / 4
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, self.button_rounding * s)
        paused = bool(getattr(viewer, "_paused", False))
        pushed = self._font(imgui, True, 15.5 * s)
        if paused:
            imgui.push_style_color(imgui.Col_.button, self._vec(imgui, th["accent"], 0.85))
            imgui.push_style_color(imgui.Col_.button_hovered, self._vec(imgui, th["accent"]))
            imgui.push_style_color(imgui.Col_.text, self._vec(imgui, th.get("on_accent", (0.08, 0.08, 0.09, 1.0))))
        if imgui.button(("Play" if paused else "Pause") + "###play", imgui.ImVec2(w, h)):
            viewer._paused = not paused
        if paused:
            imgui.pop_style_color(3)
        if pushed:
            imgui.pop_font()
        imgui.set_item_tooltip("Space")
        imgui.same_line()
        imgui.begin_disabled(not paused)
        if imgui.button("Step", imgui.ImVec2(w, h)):
            viewer._step_requested = True
        imgui.end_disabled()
        imgui.set_item_tooltip("One frame while paused  ( . )")
        imgui.same_line()
        can_reset = getattr(t, "can_reset", lambda: True)()
        imgui.begin_disabled(not can_reset)
        if imgui.button("Reset", imgui.ImVec2(w, h)):
            t.reset()
        imgui.end_disabled()
        imgui.set_item_tooltip("Start the scene over" if can_reset else "This example has no reset")
        imgui.same_line()
        recording = t.recorder is not None
        if recording:
            imgui.push_style_color(imgui.Col_.button, self._vec(imgui, th["tiers"]["rebuild"], 0.85))
            imgui.push_style_color(imgui.Col_.button_hovered, self._vec(imgui, th["tiers"]["rebuild"]))
            imgui.push_style_color(imgui.Col_.text, self._vec(imgui, (1, 1, 1, 1)))
        label = f"Stop  {t.recorder.count}" if recording else "Record"
        pos = imgui.get_cursor_screen_pos()
        # a fixed id ("###record"): the label changes every frame while recording, and a button whose id
        # changes between press and release never reports a click
        if imgui.button(label + "###record", imgui.ImVec2(w, h)):
            t.stop_recording() if recording else t.start_recording()
        if not recording:  # a red dot before the word
            text_w = imgui.calc_text_size(label).x
            cx = pos.x + (w - text_w) / 2 - 9 * s
            imgui.get_window_draw_list().add_circle_filled(
                imgui.ImVec2(cx, pos.y + h / 2), 4 * s, self._u32(imgui, th["tiers"]["rebuild"])
            )
        if recording:
            imgui.pop_style_color(3)
        imgui.set_item_tooltip("Record the viewport; Stop asks where to save the mp4")
        imgui.pop_style_var()
        if minimize:
            imgui.same_line()
            self._minimize_button(imgui, h, s)
        # what a video second means: simulation time or the wall clock
        imgui.begin_disabled(recording)
        imgui.align_text_to_frame_padding()
        pushed = self._font(imgui, False, 12.5 * s)
        imgui.text_colored(self._vec(imgui, th["muted"]), "record as")
        if pushed:
            imgui.pop_font()
        imgui.same_line()
        modes_ = {"sim": "sim time", "real": "real time"}
        current = getattr(t, "record_mode", "sim")
        changed, label = self._segmented(imgui, modes_[current], list(modes_.values()), s, 220 * s)
        if changed:
            t.record_mode = next(k for k, v in modes_.items() if v == label)
        imgui.end_disabled()
        fps = getattr(t.example, "fps", None)
        hint = (
            f"sim time: one video frame per simulation frame, played at {fps:g} fps; the video runs at simulation speed"
            if current == "sim" and fps
            else "sim time: one video frame per simulation frame, played at the example's frame rate"
            if current == "sim"
            else "real time: a 60 fps video as long as the recording took; slow frames are held, fast ones dropped"
        )
        imgui.set_item_tooltip(hint)
        imgui.dummy(imgui.ImVec2(0, 4 * s))

    def _scene_tab(self, imgui, s: float) -> None:
        """Newton's viewer panel: model info, visualization toggles, example and rendering options."""
        t, th = self.host, self.theme
        viewer = t.viewer
        gui = getattr(viewer, "gui", None)
        callbacks = getattr(gui, "_ui_callbacks", {}) if gui is not None else {}
        renderer = getattr(viewer, "renderer", None)
        model = getattr(viewer, "model", None)
        flags = imgui.TreeNodeFlags_.default_open

        for callback in callbacks.get("panel", []):
            # Newton's example browser loads Newton's own modules, outside the shelf and its presets.
            if not getattr(callback, "__qualname__", "").startswith("_ExampleBrowser."):
                callback(imgui)

        if model is not None and imgui.collapsing_header("Model", flags):
            axis = ["X", "Y", "Z"][model.up_axis]
            gravity = model.gravity.numpy()
            g0, gn = gravity[0], gravity[-1]
            imgui.text_colored(self._vec(imgui, th["muted"]), f"up axis {axis}")
            imgui.text_colored(self._vec(imgui, th["muted"]), f"gravity ({g0[0]:.2f}, {g0[1]:.2f}, {g0[2]:.2f})")
            if (gn != g0).any():
                imgui.text_colored(
                    self._vec(imgui, th["muted"]), f"global gravity ({gn[0]:.2f}, {gn[1]:.2f}, {gn[2]:.2f})"
                )
            for name, count in (
                ("bodies", "body_count"),
                ("shapes", "shape_count"),
                ("joints", "joint_count"),
                ("particles", "particle_count"),
            ):
                n = getattr(model, count, 0)
                if n:
                    imgui.text_colored(self._vec(imgui, th["muted"]), f"{n:,} {name}")

        if model is not None and imgui.collapsing_header("Visualization", flags):
            _, viewer.show_visual = imgui.checkbox("Visual shapes", viewer.show_visual)
            _, viewer.show_collision = imgui.checkbox("Collision shapes", viewer.show_collision)
            _, viewer.show_ground = imgui.checkbox("Ground", viewer.show_ground)
            _, viewer.show_joints = imgui.checkbox("Joints", viewer.show_joints)
            if viewer.show_joints and renderer is not None and hasattr(renderer, "joint_scale"):
                _, renderer.joint_scale = imgui.slider_float("joint scale", renderer.joint_scale, 0.25, 5.0)
            _, viewer.show_contacts = imgui.checkbox("Contacts", viewer.show_contacts)
            if viewer.show_contacts and renderer is not None:
                imgui.indent(16 * s)
                _, viewer.show_contact_normals = imgui.checkbox("normals", bool(viewer.show_contact_normals))
                _, viewer.show_contact_disks = imgui.checkbox("contact mode", bool(viewer.show_contact_disks))
                _, viewer.show_contact_forces = imgui.checkbox("forces", bool(viewer.show_contact_forces))
                log_flag = imgui.SliderFlags_.logarithmic.value
                base = float(viewer._contact_viz_scale_default) or 1.0
                _, viewer.contact_viz_scale = imgui.slider_float(
                    "scale", float(viewer.contact_viz_scale), base * 0.01, base * 100.0, "%.4g", log_flag
                )
                if viewer.show_contact_forces:
                    base = float(viewer._contact_force_scale_default) or 0.5
                    _, viewer.contact_force_scale = imgui.slider_float(
                        "force scale", float(viewer.contact_force_scale), base * 0.01, base * 100.0, "%.4g", log_flag
                    )
                if hasattr(renderer, "arrow_scale") and (viewer.show_contact_normals or viewer.show_contact_forces):
                    _, renderer.arrow_scale = imgui.slider_float("arrow thickness", renderer.arrow_scale, 0.25, 5.0)
                imgui.unindent(16 * s)
            _, viewer.show_particles = imgui.checkbox("Particles", viewer.show_particles)
            _, viewer.show_springs = imgui.checkbox("Springs", viewer.show_springs)
            _, viewer.show_triangles = imgui.checkbox("Cloth", viewer.show_triangles)
            if renderer is not None and hasattr(renderer, "draw_edges"):
                _, renderer.draw_edges = imgui.checkbox("Edges", renderer.draw_edges)
            _, viewer.show_com = imgui.checkbox("Centers of mass", viewer.show_com)
            if viewer.show_com and renderer is not None and hasattr(renderer, "com_scale"):
                _, renderer.com_scale = imgui.slider_float("COM scale", renderer.com_scale, 0.25, 5.0)
            _, viewer.show_inertia_boxes = imgui.checkbox("Inertia boxes", viewer.show_inertia_boxes)
            sdf_mode = getattr(viewer, "sdf_margin_mode", None)
            sdf_type = getattr(type(viewer), "SDFMarginMode", None)
            if sdf_mode is not None and sdf_type is not None:
                imgui.set_next_item_width(160 * s)
                _, idx = imgui.combo("gap + margin", int(sdf_mode), ["Off", "Margin", "Margin + Gap"])
                viewer.sdf_margin_mode = sdf_type(idx)
                if viewer.sdf_margin_mode != sdf_type.OFF and renderer is not None:
                    _, renderer.wireframe_line_width = imgui.slider_float(
                        "wireframe width", renderer.wireframe_line_width, 0.5, 5.0
                    )

        if callbacks.get("side") and imgui.collapsing_header("Example options", flags):
            for callback in callbacks["side"]:
                callback(imgui)

        if imgui.collapsing_header("Rendering"):
            _, viewer.vsync = imgui.checkbox("VSync", viewer.vsync)
            for callback in callbacks.get("rendering", []):
                callback(imgui)
            image_logger = getattr(viewer, "_image_logger", None)
            if image_logger is not None:
                image_logger.draw_controls(imgui)

        wind = getattr(viewer, "wind", None)
        if wind is not None and imgui.collapsing_header("Wind"):
            _, wind.amplitude = imgui.slider_float("amplitude", wind.amplitude, -2.0, 2.0, "%.2f")
            _, wind.period = imgui.slider_float("period", wind.period, 1.0, 30.0, "%.2f")
            _, wind.frequency = imgui.slider_float("frequency", wind.frequency, 0.1, 5.0, "%.2f")
            changed, direction = imgui.slider_float3(
                "direction", [wind.direction[0], wind.direction[1], wind.direction[2]], -1.0, 1.0, "%.2f"
            )
            if changed:
                wind.direction = direction

        if imgui.collapsing_header("Camera and keys"):
            if gui is not None and hasattr(gui, "_render_camera_info"):
                gui._render_camera_info()
            imgui.dummy(imgui.ImVec2(0, 4 * s))
            for keys, what in (
                ("W A S D", "move"),
                ("Q E", "pan up / down"),
                ("left drag", "look"),
                ("right click", "pick objects"),
                ("middle drag", "orbit"),
                ("shift + middle", "pan"),
                ("ctrl + middle", "dolly"),
                ("scroll", "dolly"),
                ("ctrl + scroll", "FOV"),
                ("space", "pause / resume"),
                (".", "step one frame"),
                ("H", "toggle UI"),
                ("F", "frame the model"),
            ):
                imgui.text_colored(self._vec(imgui, th["muted"]), f"{keys:>14}")
                imgui.same_line()
                imgui.text(what)

        if gui is not None and hasattr(gui, "_render_selection_panel"):
            gui._render_selection_panel()

    # ------------------------------------------------------------------ sections
    def _header(self, imgui, s: float) -> None:
        t, th = self.host, self.theme
        data = presets.read(t.path)
        names = [n for n in presets.names(t.path) if data[n].get("status") != "unsupported"]
        placeholders = [n for n in names if presets.is_placeholder(n, data[n])]  # no preset for that solver yet
        names = [n for n in names if n not in placeholders]
        current = t.preset_name

        self._title(imgui, s)

        imgui.set_next_item_width(170 * s)
        chosen = None
        if imgui.begin_combo("##preset", current):
            for n in names:
                if imgui.selectable(n, n == current)[0] and n != current:
                    chosen = n
            if placeholders:  # listed so it is clear they exist, but there is nothing to run yet
                imgui.separator()
                imgui.begin_disabled()
                for n in placeholders:
                    imgui.selectable(f"{n}  ·  no preset yet", False)
                imgui.end_disabled()
            imgui.end_combo()
        if placeholders:
            imgui.set_item_tooltip(
                "Greyed-out solvers have no preset for this example yet: fill its table in presets.toml "
                "and run `python -m shelf check`."
            )
        if chosen is not None:
            t.switch_to = chosen
            t.overrides.clear()
            t.edits.clear()
            t.rebuild = True
        imgui.same_line()
        status = data.get(current, {}).get("status", "untested")
        status_colors = {
            "works": th["tiers"]["live"],
            "unstable": th["tiers"]["swap"],
            "untested": th["muted"],
            "unsupported": th["tiers"]["rebuild"],
        }
        self._pill(imgui, status, status_colors.get(status, th["muted"]), s)
        if t.overrides:
            imgui.same_line()
            self._pill(imgui, "tuned, unchecked", th["tiers"]["swap"], s, filled=False)
        self._look_and_theme(imgui, s)

    def _metrics(self, imgui, s: float) -> None:
        t, th = self.host, self.theme
        from common.modes import _mem_mb

        times = np.asarray(t.step_times, dtype=np.float32) * 1e3
        mean_ms = float(times.mean()) if len(times) else 0.0
        mem = _mem_mb(high=False)
        total_mb = None
        with contextlib.suppress(Exception):
            import warp as wp

            total_mb = wp.get_device().total_memory / 2**20

        imgui.dummy(imgui.ImVec2(0, 2 * s))
        h = 108 * s
        imgui.begin_child("metrics", imgui.ImVec2(0, h), imgui.ChildFlags_.always_use_window_padding)
        half = imgui.get_content_region_avail().x / 2
        imgui.begin_group()
        pushed = self._font(imgui, True, 26 * s)
        imgui.text(f"{mean_ms:.2f}" if len(times) else "–")
        if pushed:
            imgui.pop_font()
        imgui.same_line()
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 7 * s)  # sit the unit on the number's baseline
        imgui.text_colored(self._vec(imgui, th["muted"]), "ms / frame")
        if len(times) > 1:
            imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0, 0, 0, 0))
            imgui.plot_lines(
                "##steps", times, 0, None, 0.0, float(times.max()) * 1.15, imgui.ImVec2(half - 12 * s, 34 * s)
            )
            imgui.pop_style_color()
        else:
            imgui.text_colored(self._vec(imgui, th["muted"]), "waiting for frames")
        imgui.end_group()
        imgui.same_line()
        imgui.set_cursor_pos_x(half + 8 * s)
        imgui.begin_group()
        if mem is not None:
            pushed = self._font(imgui, True, 26 * s)
            imgui.text(f"{mem:,.0f}" if mem < 10000 else f"{mem / 1024:.1f}")
            if pushed:
                imgui.pop_font()
            imgui.same_line()
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 7 * s)
            imgui.text_colored(self._vec(imgui, th["muted"]), "MB" if mem < 10000 else "GB")
            if total_mb:
                frac = min(1.0, mem / total_mb)
                imgui.text_colored(
                    self._vec(imgui, th["muted"]), f"GPU memory, {100 * frac:.0f}% of {total_mb / 1024:.0f} GB"
                )
                imgui.push_style_var(imgui.StyleVar_.frame_rounding, 3.0 * s)
                imgui.progress_bar(frac, imgui.ImVec2(-1, 5 * s), "")
                imgui.pop_style_var()
            else:
                imgui.text_colored(self._vec(imgui, th["muted"]), "GPU memory")
        else:
            imgui.text_colored(self._vec(imgui, th["muted"]), "GPU memory unavailable")
        imgui.end_group()
        imgui.end_child()

    def _settings(self, imgui, s: float) -> None:
        t = self.host
        settings = t.current_settings()
        data = presets.read(t.path)
        saved = _flatten({k: v for k, v in data.get(t.preset_name, {}).items() if k in (*presets.SECTIONS, "entries")})
        defs = defaults.solver_defaults(t.example, t.preset) if t.example is not None else {}
        # the solver settings the preset leaves at their defaults, listed under its own ones
        unset: dict[str, list[tuple[str, Any]]] = {}
        for key, d in defs.items():
            if key not in settings:
                unset.setdefault(key.split(".", 1)[0], []).append((key, d.value))
        by_section: dict[str, list[tuple[str, Any]]] = {}
        for key, value in settings.items():
            if value == {}:
                continue  # an entry with no settings of its own (``entries.xpbd = {}``): nothing to edit
            by_section.setdefault(key.split(".", 1)[0], []).append((key, value))
        for section in (*SECTION_ORDER, *sorted((set(by_section) | set(unset)) - set(SECTION_ORDER))):
            rows, more = by_section.get(section), unset.get(section)
            if not rows and not more:
                continue
            self._label(imgui, section, s)
            if rows:
                self._rows(imgui, section, rows, saved, defs, s)
            if more:
                self._default_rows(imgui, section, more, saved, defs, s)

    def _default_rows(self, imgui, section: str, rows: list[tuple[str, Any]], saved, defs, s: float) -> None:
        """The section's settings the preset leaves out, in a closed group with a filter."""
        th = self.theme
        pushed = self._font(imgui, False, 13 * s)
        imgui.push_style_color(imgui.Col_.text, self._vec(imgui, th["muted"]))
        is_open = imgui.tree_node_ex(
            f"{len(rows)} more at their defaults###defaults_{section}", imgui.TreeNodeFlags_.span_avail_width
        )
        imgui.pop_style_color()
        if pushed:
            imgui.pop_font()
        imgui.set_item_tooltip(
            "Settings the preset does not set, at the value the solver uses without them.\n"
            "auto: the solver works the value out itself (from the model or other settings).\n"
            "Change one and Apply: it moves up with the preset's settings and is saved with them."
        )
        if not is_open:
            return
        word = self.default_filters.get(section, "")
        imgui.set_next_item_width(-1)
        _, word = imgui.input_text_with_hint(f"##filter_{section}", "filter", word)
        self.default_filters[section] = word
        if word.strip():
            rows = [(k, v) for k, v in rows if word.strip().lower() in k.lower()]
        if rows:
            self._rows(imgui, f"{section}_defaults", rows, saved, defs, s, at_default=True)
        imgui.tree_pop()

    def _rows(self, imgui, table_id: str, rows, saved, defs, s: float, at_default: bool = False) -> None:
        t, th = self.host, self.theme
        control_w = 200 * s
        # rows are grouped by the path between the section and the leaf (kamino's dvi.* / padmm.*,
        # an entry's mujoco.solver.*), so each row shows only its leaf
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
        if not imgui.begin_table(f"##{table_id}", 3, flags):
            return
        imgui.table_setup_column("key", imgui.TableColumnFlags_.width_stretch, 1.0)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_fixed, control_w)
        imgui.table_setup_column("tier", imgui.TableColumnFlags_.width_fixed, 58 * s)
        group = None
        for key, value in rows:
            parts = key.split(".")
            this_group = " · ".join(parts[1:-1])
            if this_group != group:
                group = this_group
                if group:
                    imgui.table_next_row()
                    imgui.table_set_column_index(0)
                    pushed = self._font(imgui, True, 12 * s)
                    imgui.text_colored(self._vec(imgui, th["muted"]), group)
                    if pushed:
                        imgui.pop_font()
            d = defs.get(key)
            shown = t.edits.get(key, value)
            pending = key in t.edits and t.overrides.get(key, value) != shown
            tier = live.tier(t.example, key) if t.example is not None else "rebuild"
            imgui.push_id(key)
            imgui.table_next_row()
            imgui.table_set_column_index(0)
            imgui.align_text_to_frame_padding()
            name = parts[-1]
            if group:
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + 10 * s)
            if pending:
                imgui.text_colored(self._vec(imgui, th["tiers"]["swap"]), "•")
                imgui.same_line()
            if at_default and not pending:
                imgui.push_style_color(imgui.Col_.text, self._vec(imgui, th["muted"]))
            short, text = ranges.note(key, t.preset)
            imgui.begin_group()
            self._fit_label(imgui, name, imgui.get_content_region_avail().x - (40 * s if pending else 0), s)
            if short:  # what the name leaves out (see the [[note]] tables in ranges.toml)
                imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() - 4 * s)
                pushed = self._font(imgui, False, 12 * s)
                imgui.text_colored(self._vec(imgui, th["muted"]), short)
                if pushed:
                    imgui.pop_font()
            imgui.end_group()
            if at_default and not pending:
                imgui.pop_style_color()
            if imgui.is_item_hovered():
                origin = f"saved in the preset: {saved[key]}" if key in saved else "not in the preset"
                default = "" if d is None else f"\nsolver default: {_describe(d.value, d.effective)}"
                about = f"\n\n{text}" if text else ""
                imgui.set_tooltip(f"{key}\n{origin}{default}\n{tier}: {TIER_HELP[tier]}{about}")
            back = saved.get(key, d.value if d is not None else _NONE)
            if back is not _NONE and shown != back:
                imgui.same_line()
                pushed = self._font(imgui, False, 12 * s)
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(6 * s, 1 * s))
                if imgui.small_button("reset"):
                    if at_default and key not in t.overrides:
                        t.edits.pop(key, None)
                    else:
                        t.edits[key] = back
                imgui.pop_style_var()
                if pushed:
                    imgui.pop_font()
                where = "saved" if key in saved else "default"
                imgui.set_item_tooltip(f"back to the {where} value {_describe(back, None)}")
            imgui.table_set_column_index(1)
            if shown is None:
                changed, new = self._auto(imgui, key, d, s)
            else:
                changed, new = self._control(imgui, key, shown, saved.get(key), tier, s, control_w)
            if changed:
                if at_default and d is not None and new == d.value and key not in t.overrides:
                    t.edits.pop(key, None)  # moved back onto the default: nothing to apply
                else:
                    t.edits[key] = new
            imgui.table_set_column_index(2)
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 3 * s)
            self._pill(imgui, tier, th["tiers"][tier], s)
            if imgui.is_item_hovered():
                imgui.set_tooltip(TIER_HELP[tier])
            imgui.pop_id()
        imgui.end_table()

    def _auto(self, imgui, key: str, d: defaults.Default | None, s: float) -> tuple[bool, Any]:
        """An auto setting: the value the solver worked out, if known, and a button to set one."""
        th = self.theme
        imgui.align_text_to_frame_padding()
        known = d is not None and d.effective is not None
        imgui.text_colored(self._vec(imgui, th["muted"]), f"auto · {_describe(d.effective, None)}" if known else "auto")
        if d is None:
            return False, None
        imgui.same_line()
        avail = imgui.get_content_region_avail().x
        w = 44 * s
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, avail - w))
        pushed = self._font(imgui, False, 12.5 * s)
        clicked = imgui.button("set", imgui.ImVec2(w, 0))
        if pushed:
            imgui.pop_font()
        imgui.set_item_tooltip("give it a value instead of letting the solver choose")
        if not clicked:
            return False, None
        ctrl = self.range_overrides.get(key) or ranges.control(key, d.effective if known else 0.0, self.host.preset)
        if d.types == ("bool",):
            ctrl = ranges.Control("bool")
        return True, d.start(ctrl)

    def _actions(self, imgui, s: float) -> None:
        t, th = self.host, self.theme
        pending = {k: v for k, v in t.edits.items() if t.overrides.get(k, None) != v}
        imgui.dummy(imgui.ImVec2(0, 6 * s))
        avail = imgui.get_content_region_avail().x
        spacing = imgui.get_style().item_spacing.x
        w_apply = (avail - 2 * spacing) * 0.5
        w_other = (avail - 2 * spacing) * 0.25
        height = self.button_height * s
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, self.button_rounding * s)
        if pending:
            imgui.push_style_color(imgui.Col_.button, self._vec(imgui, th["primary"], 0.9))
            imgui.push_style_color(imgui.Col_.button_hovered, self._vec(imgui, th["primary"]))
            imgui.push_style_color(imgui.Col_.button_active, self._vec(imgui, th["primary"], 0.75))
            imgui.push_style_color(imgui.Col_.text, self._vec(imgui, th["primary_text"]))
        else:
            imgui.push_style_color(imgui.Col_.button, self._vec(imgui, th["button"]))
            imgui.push_style_color(imgui.Col_.button_hovered, self._vec(imgui, th["button"]))
            imgui.push_style_color(imgui.Col_.button_active, self._vec(imgui, th["button"]))
            imgui.push_style_color(imgui.Col_.text, self._vec(imgui, th["muted"]))
        label = f"Apply {len(pending)} change{'s' if len(pending) != 1 else ''}" if pending else "Apply"
        pushed = self._font(imgui, True, 15.5 * s)
        if imgui.button(label + "###apply", imgui.ImVec2(w_apply, height)) and pending:
            t.pending = dict(pending)
            t.edits.clear()
            t.switch_to = t.preset_name
        if pushed:
            imgui.pop_font()
        imgui.pop_style_color(4)
        imgui.same_line()
        if imgui.button("Rebuild", imgui.ImVec2(w_other, height)):
            t.overrides.update(t.edits)
            t.edits.clear()
            t.switch_to = t.preset_name
            t.rebuild = True
        imgui.set_item_tooltip("Build the scene again with the applied settings")
        imgui.same_line()
        if imgui.button("Revert", imgui.ImVec2(w_other, height)):
            t.overrides.clear()
            t.edits.clear()
            t.switch_to = t.preset_name
            t.rebuild = True
        imgui.set_item_tooltip("Drop every applied change and rebuild the preset as saved")
        imgui.pop_style_var()

        if t.message:
            m = t.message
            tiers = th["tiers"]
            color = (
                tiers["live"]
                if m.startswith("applied")
                else tiers["rebuild"]
                if ("failed" in m or "invalid" in m)
                else tiers["swap"]
                if m.startswith("rebuild")
                else th["muted"]
            )
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(self._vec(imgui, color), m)
            imgui.pop_text_wrap_pos()

    def _extras(self, imgui, s: float) -> None:
        t, th = self.host, self.theme
        imgui.dummy(imgui.ImVec2(0, 6 * s))
        if imgui.collapsing_header("Add a setting"):
            imgui.set_next_item_width(-70 * s)
            _, t.new_key = imgui.input_text_with_hint("##new", "section.key = value", t.new_key)
            imgui.same_line()
            if imgui.button("Add", imgui.ImVec2(-1, 0)) and t.new_key.strip():
                try:
                    key, value = presets.parse_override(t.new_key)
                    t.edits[key] = value
                    t.new_key = ""
                except presets.PresetError as e:
                    t.message = str(e)
        if imgui.collapsing_header("Save as"):
            current = t.preset_name
            if not t.save_name:
                t.save_name = f"{current}_tuned.toml"
            imgui.set_next_item_width(-1)
            _, t.save_name = imgui.input_text("##file", t.save_name)
            _, t.overwrite = imgui.checkbox("overwrite if it exists", t.overwrite)
            imgui.same_line()
            avail = imgui.get_content_region_avail().x
            w = 90 * s
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, avail - w))
            imgui.push_style_var(imgui.StyleVar_.frame_rounding, self.button_rounding * s)
            if imgui.button("Save", imgui.ImVec2(w, 0)):
                t.message = t.save(current, t.save_name, t.overwrite)
            imgui.pop_style_var()
            imgui.text_colored(self._vec(imgui, th["muted"]), "a new file next to presets.toml; load it with --presets")


def _flatten(table: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in table.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


_NONE = object()  # "no value to go back to" (None is a value: auto)


def _describe(value: Any, effective: Any) -> str:
    if value is None:
        return (
            "auto"
            if effective is None
            else f"auto ({effective:.6g})"
            if isinstance(effective, float)
            else f"auto ({effective})"
        )
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _parse_literal(text: str, like: Any) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return like
