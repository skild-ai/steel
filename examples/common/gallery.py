# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""The example browser: every example as a card under its section, with a preview, a one-line
description and a chip per preset that runs it.

``python -m shelf play`` with no example opens it in an otherwise empty window; the bar's **browse**
button opens it over a running example (the simulation pauses meanwhile). A card opens the example
at its default preset, a chip at that preset; Escape or Close returns to the running example.
Previews come from the cache :mod:`common.thumbs` fills; a card without one says why.

The card's text comes from the example's ``presets.toml`` ``[gallery]`` table (``short``, optional
``title`` and ``frame``); a file without one gets a warning on stderr and a card without a description.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

from common import presets, thumbs

ROOT = Path(__file__).resolve().parents[1]
SHELF = ROOT / "shelf"
CARD_MIN = 330.0  # unscaled px: cards are at least this wide; the grid takes as many as fit
GAP = 14.0
TEXTURE_CAP = 96  # previews kept on the GPU at once; the least recently drawn go first


@dataclasses.dataclass
class Card:
    name: str
    section: str
    path: Path  # example.py
    title: str
    short: str | None  # None: no [gallery] table
    default: str | None
    chips: list[tuple[str, str]]  # (preset, status) for every preset that runs
    warning: str | None = None


def catalog() -> tuple[list[Card], list[str]]:
    """Every example on the shelf as a card, in shelf order, and the warnings for missing entries."""
    cards, warnings = [], []
    for path in sorted(SHELF.glob("*/*/example.py")):
        name = path.parent.name
        try:
            data = presets.read(path)
        except presets.PresetError as e:
            warnings.append(str(e))
            continue
        short = title = None
        warning = None
        try:
            entry = presets.gallery(path)
        except presets.PresetError as e:
            entry, warning = None, str(e)
        if entry is None and warning is None:
            rel = presets.presets_path(path).relative_to(ROOT)
            warning = f"{rel} has no [gallery] table; add one with short = \"...\" for the example browser"
        if entry is not None:
            short, title = entry["short"], entry.get("title")
        if warning:
            warnings.append(warning)
        chips = [(n, data[n].get("status", "untested")) for n in presets.ready(path)]
        cards.append(Card(name, path.parent.parent.name, path, title or name, short, data.get("default"), chips, warning))
    return cards, warnings


def section_label(section: str) -> str:
    number, _, rest = section.partition("_")
    return f"{number} · {rest.replace('_', ' ')}" if number.isdigit() else section.replace("_", " ")


class Gallery:
    """The overlay's state: open or not, the filter, the cards and the previews on the GPU."""

    def __init__(self):
        self.open = False
        self.filter = ""
        self.cards: list[Card] | None = None
        self.textures: dict[str, tuple[int, float]] = {}  # png path -> (GL texture, last drawn)
        self.missing_png: dict[str, float] = {}  # png path -> when it was last found absent
        self.building = 0  # previews being rendered since the browser opened (0: none started here)
        self.total = 0
        self.status: dict[str, Any] | None = None
        self.status_read = 0.0
        self.missing = 0  # previews absent, counted with the status
        self.failed = 0
        self._focus = False
        self._paused_by_us = False
        self.host = None

    # ---------------------------------------------------------------- open / close
    def show(self, host, building: int = 0, total: int = 0) -> None:
        self.host = host
        self.open = True
        self._focus = True
        self.building, self.total = building, total
        if self.cards is None:
            self.cards, warnings = catalog()
            for w in warnings:
                print(f"[shelf] warning: {w}", file=sys.stderr)
        viewer = getattr(host, "viewer", None)
        if getattr(host, "example", None) is not None and hasattr(viewer, "_paused") and not viewer._paused:
            viewer._paused = True  # the simulation waits while the browser covers it
            self._paused_by_us = True

    def hide(self) -> None:
        self.open = False
        viewer = getattr(self.host, "viewer", None)
        if self._paused_by_us and hasattr(viewer, "_paused"):
            viewer._paused = False
        self._paused_by_us = False
        self.release()

    def pick(self, name: str, preset: str | None) -> None:
        host = self.host
        if host is None:
            return
        if name == getattr(host, "example_name", None) and preset is None:
            self.hide()
            return
        host.next_example = name
        host.next_solver = preset
        host.message = f"loading {name}…"
        self.hide()

    # ---------------------------------------------------------------- previews on the GPU
    def _texture(self, path: Path) -> int | None:
        key = str(path)
        now = time.monotonic()
        found = self.textures.get(key)
        if found is not None:
            self.textures[key] = (found[0], now)
            return found[0]
        if now - self.missing_png.get(key, -10.0) < 1.0:
            return None  # asked a moment ago; the builder may still be at it
        if not path.exists():
            self.missing_png[key] = now
            return None
        try:
            from PIL import Image
            from pyglet import gl

            with Image.open(path) as image:
                rgba = image.convert("RGBA")
                w, h = rgba.size
                data = rgba.tobytes()
        except (OSError, ValueError):
            self.missing_png[key] = now  # half-written: try again shortly
            return None
        tex = (gl.GLuint * 1)()
        gl.glGenTextures(1, tex)
        tid = int(tex[0])
        gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self.textures[key] = (tid, now)
        if len(self.textures) > TEXTURE_CAP:
            oldest = min(self.textures, key=lambda k: self.textures[k][1])
            self._delete(self.textures.pop(oldest)[0])
        return tid

    @staticmethod
    def _delete(tid: int) -> None:
        from pyglet import gl

        gl.glDeleteTextures(1, (gl.GLuint * 1)(tid))

    def release(self) -> None:
        for tid, _ in self.textures.values():
            self._delete(tid)
        self.textures.clear()
        self.missing_png.clear()

    # ---------------------------------------------------------------- the build's progress
    def _status(self, look: str) -> dict[str, Any] | None:
        now = time.monotonic()
        if now - self.status_read > 1.0:
            self.status = thumbs.status(look)
            self.status_read = now
            cards = self.cards or []
            self.missing = sum(1 for c in cards if not thumbs.png(look, c.name).exists())
            self.failed = sum(1 for c in cards if thumbs.error_file(look, c.name).exists())
        return self.status

    # ---------------------------------------------------------------- drawing
    def draw(self, imgui, panel, s: float, io) -> None:
        """The overlay, drawn inside the bar's theme (the caller pushed it)."""
        from common import looks

        host, th = self.host, panel.theme
        viewer = getattr(host, "viewer", None)
        look = looks.current(viewer) or "steel"
        has_example = getattr(host, "example", None) is not None or getattr(host, "path", None) is not None
        imgui.set_next_window_pos(imgui.ImVec2(0, 0), imgui.Cond_.always)
        imgui.set_next_window_size(imgui.ImVec2(io.display_size.x, io.display_size.y), imgui.Cond_.always)
        if self._focus:
            imgui.set_next_window_focus()
            self._focus = False
        flags = (
            imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.no_bring_to_front_on_focus
        )
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(26 * s, 20 * s))
        imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(th["bg"][0], th["bg"][1], th["bg"][2], 0.97))
        if imgui.begin("##gallery", None, flags)[0]:
            if has_example and imgui.is_key_pressed(imgui.Key.escape, False) and not io.want_text_input:
                self.hide()
            self._header(imgui, panel, s, has_example, look)
            if self.open:
                self._cards(imgui, panel, s, look)
        imgui.end()
        imgui.pop_style_color()
        imgui.pop_style_var(2)

    def _header(self, imgui, panel, s: float, has_example: bool, look: str) -> None:
        th = panel.theme
        pushed = panel._font(imgui, True, 24 * s)
        imgui.text("Examples")
        if pushed:
            imgui.pop_font()
        imgui.same_line(0, 10 * s)
        imgui.align_text_to_frame_padding()
        imgui.text_colored(panel._vec(imgui, th["muted"]), f"{len(self.cards or [])}")
        # right-aligned: the filter and, over a running example, Close
        close_w = (imgui.calc_text_size("Close").x + 2 * imgui.get_style().frame_padding.x + 8 * s) if has_example else 0.0
        filter_w = 260 * s
        spacing = imgui.get_style().item_spacing.x
        imgui.same_line()
        avail = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, avail - filter_w - (close_w + spacing if close_w else 0.0)))
        imgui.set_next_item_width(filter_w)
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here()
        _, self.filter = imgui.input_text_with_hint("##gallery_filter", "filter by name or section", self.filter)
        if has_example:
            imgui.same_line()
            if imgui.button("Close", imgui.ImVec2(close_w, 0)):
                self.hide()
            imgui.set_item_tooltip("Back to the running example  ( Esc )")
        self._build_line(imgui, panel, s, look)
        imgui.dummy(imgui.ImVec2(0, 4 * s))

    def _build_line(self, imgui, panel, s: float, look: str) -> None:
        """What is happening to the previews: the first build, a build of new ones, or nothing."""
        th = panel.theme
        state = self._status(look)
        alive = bool(state and state.get("alive"))
        text = ""
        if alive:
            done, total = state.get("done", 0), state.get("total", 0)
            eta = state.get("eta_s")
            left = f", about {max(1, round(eta / 60))} min left" if eta else ""
            if self.building and self.building >= self.total > 0:
                text = (
                    f"Rendering the previews for the first time: one frame of each example, in the background. "
                    f"The whole shelf takes about {thumbs.FIRST_BUILD_MINUTES} min on an RTX 3090; {done} of {total} done{left}. "
                    f"Cards fill in as they finish; pick any example meanwhile."
                )
            else:
                text = f"Rendering {total} previews in the background: {done} done{left}."
        elif self.cards and self.missing:
            n, failed = self.missing, self.failed
            if n > failed:
                text = (
                    f"{n - failed} examples have no preview yet: `python -m shelf play` with no example renders them "
                    f"in the background, or `python -m shelf thumbs --missing` now."
                )
            if failed:
                text += f"{' ' if text else ''}{failed} could not be rendered; hover a card for the reason."
        if text:
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(panel._vec(imgui, th["muted"]), text)
            imgui.pop_text_wrap_pos()

    def _cards(self, imgui, panel, s: float, look: str) -> None:
        th = panel.theme
        word = self.filter.strip().lower()
        cards = [c for c in self.cards or [] if not word or word in c.name.lower() or word in c.section.lower() or word in c.title.lower()]
        imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(0, 0, 0, 0))
        if imgui.begin_child("##cards", imgui.ImVec2(0, 0), 0, imgui.WindowFlags_.no_background):
            avail = imgui.get_content_region_avail().x
            gap = GAP * s
            cols = max(1, int((avail + gap) // (CARD_MIN * s + gap)))
            card_w = (avail - (cols - 1) * gap) / cols
            img_h = (card_w - 2 * 8 * s) * 9 / 16
            card_h = img_h + 112 * s
            if not cards:
                imgui.text_colored(panel._vec(imgui, th["muted"]), "nothing matches the filter")
            section = None
            i = 0
            for card in cards:
                if card.section != section:
                    section = card.section
                    i = 0
                    imgui.dummy(imgui.ImVec2(0, 6 * s))
                    pushed = panel._font(imgui, True, 17 * s)
                    imgui.separator_text(section_label(section))
                    if pushed:
                        imgui.pop_font()
                if i % cols:
                    imgui.same_line(0, gap)
                self._card(imgui, panel, s, look, card, card_w, card_h, img_h)
                i += 1
            imgui.dummy(imgui.ImVec2(0, 20 * s))
        imgui.end_child()
        imgui.pop_style_color()

    def _card(self, imgui, panel, s: float, look: str, card: Card, w: float, h: float, img_h: float) -> None:
        th = panel.theme
        current = card.name == getattr(self.host, "example_name", None)
        pad = 8 * s
        imgui.push_style_var(imgui.StyleVar_.child_rounding, 8 * s)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(pad, pad))
        imgui.push_style_var(imgui.StyleVar_.child_border_size, 2.0 if current else 1.0)
        imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(*th["card"]))
        border = th["accent"] if current else th.get("border", (1, 1, 1, 0.12))
        imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(*border))
        visible = imgui.is_rect_visible(imgui.ImVec2(w, h))
        if imgui.begin_child(f"##card_{card.name}", imgui.ImVec2(w, h), imgui.ChildFlags_.borders,
                             imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse):  # fmt: skip
            inner_w = w - 2 * pad
            top = imgui.get_cursor_screen_pos()
            # the preview and the name open the example at its default preset
            imgui.set_next_item_allow_overlap()
            clicked = imgui.invisible_button(f"##open_{card.name}", imgui.ImVec2(inner_w, img_h + 24 * s))
            hovered = imgui.is_item_hovered()
            if hovered:
                imgui.set_tooltip(
                    (card.short or card.title) + f"\n\npython -m shelf play {card.name}"
                    + (f" --solver {card.default}" if card.default else "")
                )
            if clicked:
                self.pick(card.name, None)
            imgui.set_cursor_screen_pos(top)
            tex = self._texture(thumbs.png(look, card.name)) if visible else None
            draw = imgui.get_window_draw_list()
            end = imgui.ImVec2(top.x + inner_w, top.y + img_h)
            if tex is not None:
                draw.add_image_rounded(imgui.ImTextureRef(tex), top, end, imgui.ImVec2(0, 0), imgui.ImVec2(1, 1),
                                       imgui.get_color_u32(imgui.ImVec4(1, 1, 1, 1)), 5 * s)  # fmt: skip
            else:
                draw.add_rect_filled(top, end, panel._u32(imgui, th["faint"]), 5 * s)
                self._placeholder(imgui, panel, s, look, card, top, end)
            if hovered:
                draw.add_rect(top, end, panel._u32(imgui, th["accent"], 0.9), rounding=5 * s, thickness=2.0)
            imgui.dummy(imgui.ImVec2(inner_w, img_h + 4 * s))
            pushed = panel._font(imgui, True, 15.5 * s)
            imgui.text(card.title)
            if pushed:
                imgui.pop_font()
            # the description, clipped to two lines
            pushed = panel._font(imgui, False, 13 * s)
            line_h = imgui.get_text_line_height()
            if imgui.begin_child(f"##short_{card.name}", imgui.ImVec2(inner_w, 2 * line_h + 2 * s), 0,
                                 imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse | imgui.WindowFlags_.no_background):  # fmt: skip
                imgui.push_text_wrap_pos(0.0)
                imgui.text_colored(panel._vec(imgui, th["muted"]), card.short or "no description: add [gallery] short to presets.toml")
                imgui.pop_text_wrap_pos()
            imgui.end_child()
            if pushed:
                imgui.pop_font()
            self._chips(imgui, panel, s, card)
        imgui.end_child()
        imgui.pop_style_color(2)
        imgui.pop_style_var(3)

    def _placeholder(self, imgui, panel, s: float, look: str, card: Card, top, end) -> None:
        th = panel.theme
        state = self._status(look)
        error = thumbs.error_file(look, card.name)
        if error.exists():
            text, color = "no preview", th["tiers"]["rebuild"]
        elif state and state.get("alive") and card.name in state.get("running", []):
            text, color = "rendering…", th["accent"]
        elif state and state.get("alive"):
            text, color = "preview pending", th["muted"]
        else:
            text, color = "no preview yet", th["muted"]
        pushed = panel._font(imgui, False, 13 * s)
        size = imgui.calc_text_size(text)
        pos = imgui.ImVec2((top.x + end.x - size.x) / 2, (top.y + end.y - size.y) / 2)
        imgui.get_window_draw_list().add_text(pos, panel._u32(imgui, color), text)
        if pushed:
            imgui.pop_font()
        if error.exists() and imgui.is_mouse_hovering_rect(top, end):
            try:
                reason = error.read_text().strip()
            except OSError:
                reason = ""
            imgui.set_tooltip(f"the preview could not be rendered:\n{reason}\n\nretry with  python -m shelf thumbs {card.name} --force")

    def _chips(self, imgui, panel, s: float, card: Card) -> None:
        """One small button per preset that runs; the default is filled. A click opens with that preset."""
        th = panel.theme
        colors = {"works": th["tiers"]["live"], "unstable": th["tiers"]["swap"], "untested": th["muted"]}
        pushed = panel._font(imgui, False, 12 * s)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(7 * s, 2 * s))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 99.0)
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(5 * s, 4 * s))
        avail = imgui.get_content_region_avail().x
        used = 0.0
        for preset, status in card.chips:
            color = colors.get(status, th["muted"])
            is_default = preset == card.default
            width = imgui.calc_text_size(preset).x + 14 * s
            if used and used + width > avail:
                break  # keep the row to one line; the rest are in the combo
            if used:
                imgui.same_line()
            imgui.push_style_color(imgui.Col_.button, panel._vec(imgui, color, 0.45 if is_default else 0.16))
            imgui.push_style_color(imgui.Col_.button_hovered, panel._vec(imgui, color, 0.6))
            imgui.push_style_color(imgui.Col_.button_active, panel._vec(imgui, color, 0.8))
            imgui.push_style_color(imgui.Col_.text, panel._vec(imgui, th["text"] if is_default else color))
            if imgui.button(f"{preset}##{card.name}"):
                self.pick(card.name, preset)
            imgui.pop_style_color(4)
            imgui.set_item_tooltip(f"open with the {preset} preset ({status}{', the default' if is_default else ''})")
            used += width + 5 * s
        if not card.chips:
            imgui.text_colored(panel._vec(imgui, th["muted"]), "no preset runs this example yet")
        imgui.pop_style_var(3)
        if pushed:
            imgui.pop_font()
