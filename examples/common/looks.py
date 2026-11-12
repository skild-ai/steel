"""Looks: palettes and lighting for the OpenGL viewer, read from ``looks.toml``.

A look changes how a scene is drawn, never what is simulated. It recolors the shapes
that use Newton's default palette, sets the floor, and sets the renderer's sky, ambient
fill, light, exposure, roughness and shadow softness. Shapes whose color the example chose
keep it. :func:`attach` applies the look after every ``set_model`` so resets and rebuilds
keep it, and the tuner switches looks live with :func:`switch`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import tomllib

FILE = Path(__file__).with_name("looks.toml")
# Renderer attributes a look may set, with the key used in looks.toml.
_RENDERER_KEYS = {
    "sky_upper": "sky_upper",
    "sky_lower": "sky_lower",
    "ambient_sky": "ambient_sky",
    "ambient_ground": "ambient_ground",
    "light_color": "_light_color",
    "exposure": "exposure",
    "shadow_radius": "shadow_radius",
    "spotlight": "spotlight_enabled",
    "sun_direction": "_sun_direction",
}
_KEYS = {*_RENDERER_KEYS, "note", "palette", "ground", "checker", "checker_dark", "roughness", "fog"}
NEWTON_FOG = (20.0, 200.0)  # the range compiled into Newton's shape shader
NEWTON_CHECKER = 0.7  # the factor by which that shader darkens the dark checker tiles


class LookError(ValueError):
    """looks.toml is malformed or names an unknown look."""


def _rgb(value: Any, where: str) -> tuple[float, float, float]:
    """A color as floats in [0, 1] from ``"#rrggbb"`` or ``[r, g, b]``."""
    if isinstance(value, str):
        text = value.lstrip("#")
        if len(text) != 6:
            raise LookError(f"{where}: {value!r} is not a #rrggbb color")
        return tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    if isinstance(value, list | tuple) and len(value) == 3:
        return tuple(float(c) for c in value)
    raise LookError(f"{where}: {value!r} is not a color")


@dataclasses.dataclass
class Look:
    name: str
    note: str = ""
    palette: list[tuple[float, float, float]] | None = None
    ground: tuple[float, float, float] | None = None
    checker: bool | None = None
    checker_dark: tuple[float, float, float] | None = None  # the dark tiles' colour; None keeps Newton's darkening
    roughness: float | None = None
    fog: tuple[float, float] | None = None  # (start, end) in metres; None keeps Newton's
    renderer: dict[str, Any] = dataclasses.field(default_factory=dict)  # renderer attribute -> value


def read() -> dict[str, dict[str, Any]]:
    with FILE.open("rb") as f:
        return tomllib.load(f)


def names() -> list[str]:
    return list(read())


def load(name: str) -> Look:
    data = read()
    if name not in data:
        raise LookError(f"{FILE} has no look {name!r}; available: {names()}")
    block = data[name]
    unknown = [k for k in block if k not in _KEYS]
    if unknown:
        raise LookError(f"{FILE} [{name}]: unknown key(s) {unknown}; expected {sorted(_KEYS)}")
    where = f"{FILE} [{name}]"
    look = Look(name=name, note=block.get("note", ""))
    if "palette" in block:
        look.palette = [_rgb(c, where) for c in block["palette"]]
    if "ground" in block:
        look.ground = _rgb(block["ground"], where)
    look.checker = block.get("checker")
    if "checker_dark" in block:
        if "ground" not in block:
            raise LookError(f"{where}: checker_dark needs ground, the colour of the light tiles")
        look.checker_dark = _rgb(block["checker_dark"], where)
    look.roughness = block.get("roughness")
    if "fog" in block:
        fog = block["fog"]
        if not (isinstance(fog, list) and len(fog) == 2 and 0 <= float(fog[0]) < float(fog[1])):
            raise LookError(f"{where}: fog must be [start, end] in metres with start < end, got {fog!r}")
        look.fog = (float(fog[0]), float(fog[1]))
    for key, attr in _RENDERER_KEYS.items():
        if key in block:
            look.renderer[attr] = _renderer_value(key, block[key], where)
    return look


def _renderer_value(key: str, value: Any, where: str) -> Any:
    """A renderer attribute's value from its looks.toml entry."""
    if key == "sun_direction":
        if not (isinstance(value, list) and len(value) == 3):
            raise LookError(f"{where}: sun_direction must be [x, y, z] toward the sun, got {value!r}")
        d = np.asarray(value, dtype=np.float32)
        if not np.linalg.norm(d) > 0.0:
            raise LookError(f"{where}: sun_direction must not be zero")
        return d / np.linalg.norm(d)
    if key == "spotlight":
        if not isinstance(value, bool):
            raise LookError(f"{where}: spotlight must be true or false, got {value!r}")
        return value
    return _rgb(value, where) if isinstance(value, list | str) else float(value)


_compiled: dict[int, tuple] = {}  # id(renderer) -> (fog range, checker factor) its shape shader has
_NEWTON_SHADER = (NEWTON_FOG, None)


def _patch_shader(renderer, fog: tuple[float, float], checker: tuple[float, float, float] | None) -> None:
    """Compile the renderer's shape shader with ``fog`` = (start, end) and the dark checker tiles at
    ``checker`` times the floor's linear albedo (None: Newton's 0.7).

    Newton writes both as literals in the shader source, so the source is rewritten and the two
    shape shaders are built again (a few milliseconds); nothing happens when they are already the
    ones compiled.
    """
    import re

    from newton._src.viewer.gl import shaders
    from newton._src.viewer.gl.opengl import RendererGL

    if _compiled.get(id(renderer), _NEWTON_SHADER) == (fog, checker) or not hasattr(renderer, "_shape_shader"):
        return
    dark = f"vec3({checker[0]:.4f}, {checker[1]:.4f}, {checker[2]:.4f})" if checker else f"{NEWTON_CHECKER}"
    source = shaders.shape_fragment_shader
    source = re.sub(r"float fog_start = [0-9.]+;", f"float fog_start = {fog[0]:.1f};", source)
    source = re.sub(r"float fog_end\s*= [0-9.]+;", f"float fog_end = {fog[1]:.1f};", source)
    source, n = re.subn(r"vec3 albedo2 = albedo\*[^;]+;", f"vec3 albedo2 = albedo*{dark};", source)
    if n != 1:
        raise LookError("Newton's shape shader has no checker line to rewrite; update looks._patch_shader")
    shaders.shape_fragment_shader = source
    renderer._shape_shader = shaders.ShaderShape(RendererGL.gl)
    if getattr(renderer, "_shape_transparent_shader", None) is not None:
        renderer._shape_transparent_shader = shaders.ShaderShape(RendererGL.gl, enable_transparency=True)
    _compiled[id(renderer)] = (fog, checker)


class _State:
    """What the look changed on a viewer, so another look can start from the original."""

    def __init__(self):
        self.look: Look | None = None
        self.colors: np.ndarray | None = None  # model.shape_color before the look
        self.materials: dict[int, np.ndarray] = {}  # id(batch) -> batch.materials before the look
        self.renderer: dict[str, Any] = {}  # renderer attribute -> value before the look


def attach(viewer, name: str) -> None:
    """Apply the look now and after every later ``set_model`` on ``viewer``."""
    look = load(name)
    if not hasattr(viewer, "renderer"):
        return  # not the OpenGL viewer
    state = _State()
    viewer._shelf_look = state
    original_set_model = viewer.set_model

    def set_model(model, *args, **kwargs):
        result = original_set_model(model, *args, **kwargs)
        state.colors = None
        state.materials = {}
        _apply(viewer, state, state.look or look)
        return result

    viewer.set_model = set_model
    _apply(viewer, state, look)


def switch(viewer, name: str) -> None:
    """Change the look on a viewer that :func:`attach` set up."""
    state = getattr(viewer, "_shelf_look", None)
    if state is None:
        attach(viewer, name)
    else:
        _apply(viewer, state, load(name))


def current(viewer) -> str | None:
    state = getattr(viewer, "_shelf_look", None)
    return state.look.name if state is not None and state.look is not None else None


def _apply(viewer, state: _State, look: Look) -> None:
    import newton
    import warp as wp

    state.look = look
    renderer = viewer.renderer
    for attr, value in state.renderer.items():  # undo the previous look
        setattr(renderer, attr, value)
    state.renderer = {attr: getattr(renderer, attr) for attr in look.renderer}
    for attr, value in look.renderer.items():
        setattr(renderer, attr, value)
    checker = None
    if look.checker_dark is not None and look.ground is not None:
        # the shader works in linear light with a 2.2 gamma; the dark tiles scale the floor's albedo
        checker = tuple((d / g) ** 2.2 if g > 0.0 else 0.0 for d, g in zip(look.checker_dark, look.ground, strict=True))
    _patch_shader(renderer, look.fog or NEWTON_FOG, checker)

    model = viewer.model
    if model is None or model.shape_color is None:
        return
    if state.colors is None:
        state.colors = model.shape_color.numpy().copy()
    colors = state.colors.copy()
    shape_type = model.shape_type.numpy()
    is_plane = shape_type == int(newton.GeoType.PLANE)
    if look.palette:
        default = np.array(newton.ModelBuilder._SHAPE_COLOR_PALETTE, dtype=np.float32) / 255.0
        # A shape colored by the default palette maps to the same slot of the look's palette.
        dist = np.abs(colors[:, None, :] - default[None, :, :]).max(axis=2)
        slot = dist.argmin(axis=1)
        from_palette = (dist.min(axis=1) < 2e-3) & ~is_plane
        for i in np.flatnonzero(from_palette):
            colors[i] = look.palette[slot[i] % len(look.palette)]
    if look.ground is not None:
        colors[is_plane] = look.ground
    model.shape_color.assign(wp.array(colors, dtype=wp.vec3, device=model.shape_color.device))

    # Materials live per viewer batch: (roughness, metallic, checker, texture) per instance.
    for batch in viewer._shape_instances.values():
        if not isinstance(batch.materials, wp.array) or not len(batch.model_shapes):
            continue
        if id(batch) not in state.materials:
            state.materials[id(batch)] = batch.materials.numpy().copy()
        materials = state.materials[id(batch)].copy()
        shapes = np.asarray(batch.model_shapes)
        if look.roughness is not None:
            dielectric = materials[:, 1] == 0.0
            materials[dielectric, 0] = look.roughness
        if look.checker is not None:
            materials[is_plane[shapes], 2] = 1.0 if look.checker else 0.0
        batch.materials.assign(wp.array(materials, dtype=wp.vec4, device=batch.materials.device))
    viewer.model_changed = True  # re-upload materials on the next frame
