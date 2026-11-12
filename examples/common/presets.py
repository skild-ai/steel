# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Solver presets: per-example, per-solver settings stored in ``presets.toml``.

Each example folder holds an ``example.py`` and a ``presets.toml``. The TOML has
one table per solver (``[xpbd]``, ``[mujoco]``, ``[steel]``, ...). A table holds
the solver's status plus four groups of settings:

* ``sim``       -- frame rate and substeps (``fps``, ``substeps``) and other timing.
* ``collision`` -- collision pipeline settings.
* ``solver``    -- keyword arguments for the solver constructor, or fields of its
  ``Config`` for config-driven solvers (Kamino, implicit MPM). Nested tables set
  fields of nested configs.
* ``model``     -- per-solver model or builder tweaks that the example applies.

Coupled examples put one sub-table per entry under the preset, e.g.
``[mujoco_vbd.entries.mujoco]``.

Examples call :func:`load` with ``__file__``. The runner (``python -m shelf run``)
picks the preset with ``--solver`` and overrides values with ``--set``; when an
example is run directly, ``SHELF_SOLVER`` and ``SHELF_SET`` do the same.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import os
import tomllib
from pathlib import Path
from typing import Any

# Preset name -> solver class name in ``newton.solvers``. These are the matrix columns.
SOLVERS: dict[str, str] = {
    "xpbd": "SolverXPBD",
    "vbd": "SolverVBD",
    "mujoco": "SolverMuJoCo",
    "featherstone": "SolverFeatherstone",
    "semi_implicit": "SolverSemiImplicit",
    "feather_pgs": "SolverFeatherPGS",
    "steel": "SolverSteel",
    "kamino": "SolverKamino",
    "implicit_mpm": "SolverImplicitMPM",
    "style3d": "SolverStyle3D",
}

STATUSES = ("works", "unstable", "unsupported", "untested")
# Preset name for examples that build no physics solver (viewers, IK-only, camera demos).
NO_SOLVER = "none"
SECTIONS = ("sim", "collision", "solver", "model")
# `kind` names the solver a variant table builds, e.g. [kamino_dvi] with kind = "kamino".
_META_KEYS = {"status", "checked", "note", "reason", "check", "entries", "kind"}
# The example browser's table (`python -m shelf play`): `short`, one line; `frame`, the preview's frame; `title`.
GALLERY_KEYS = ("short", "frame", "title")
DEFAULT_THUMB_FRAME = 60

_selected: str | None = None
_overrides: dict[str, Any] = {}
_file: str | None = None  # preset file name to use instead of presets.toml (e.g. one saved by `shelf tune`)


class PresetError(ValueError):
    """A preset file is missing, malformed, or names a setting the solver does not have."""


def select(name: str | None, overrides: dict[str, Any] | None = None, file: str | None = None) -> None:
    """Choose the preset that the next :func:`load` returns (used by the runner).

    ``file`` reads presets from that file in the example's folder (or that path) instead of presets.toml.
    """
    global _selected, _overrides, _file
    _selected = name
    _overrides = dict(overrides or {})
    _file = file


def selected() -> str | None:
    """The preset the runner selected (``--solver`` or ``$SHELF_SOLVER``), None for the file's default."""
    return _selected or os.environ.get("SHELF_SOLVER") or None


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``section.key=value`` where value is a TOML literal (bare words become strings)."""
    if "=" not in text:
        raise PresetError(f"--set expects section.key=value, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    if key.split(".", 1)[0] not in (*SECTIONS, "entries"):
        raise PresetError(f"--set key must start with one of {SECTIONS}, got {key!r}")
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    return key, value


class Settings(dict):
    """A settings group (``sim``, ``solver``, ...) with attribute access and required lookups."""

    def __init__(self, data: dict[str, Any], where: str):
        super().__init__(data)
        self._where = where

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        return self.need(key)

    def need(self, key: str) -> Any:
        if key not in self:
            raise PresetError(f"{self._where} has no {key!r}; add it to presets.toml")
        return self[key]


@dataclasses.dataclass
class Preset:
    """One solver's settings for one example."""

    example: str
    name: str
    path: Path
    status: str
    sim: Settings
    collision: Settings
    solver: Settings
    model: Settings
    entries: dict[str, Preset]
    raw: dict[str, Any]
    kind: str  # the solver key this preset builds; an entry's name for coupled entries

    @property
    def fps(self) -> float:
        return self.sim.need("fps")

    @property
    def substeps(self) -> int:
        return self.sim.need("substeps")

    def entry(self, name: str) -> Preset:
        """The sub-preset for one entry of a coupled solver."""
        if name not in self.entries:
            raise PresetError(f"{self.path} [{self.name}] has no entry {name!r}; entries: {sorted(self.entries)}")
        return self.entries[name]

    def solver_class(self, solver: str | None = None) -> type:
        key = solver or self.kind
        if key not in SOLVERS:
            raise PresetError(f"{self.path} [{self.name}]: {key!r} is not a known solver; use one of {sorted(SOLVERS)}")
        newton = importlib.import_module("newton")
        return getattr(newton.solvers, SOLVERS[key])

    def make_solver(self, model, solver: str | None = None, **kwargs):
        """Build the solver from the ``solver`` settings; ``kwargs`` add arguments that are not settings."""
        cls = self.solver_class(solver)
        settings = dict(self.solver)
        params = inspect.signature(cls.__init__).parameters
        if "config" in params and hasattr(cls, "Config") and "config" not in kwargs:
            return cls(model, config=self.make_config(model, cls), **kwargs)
        accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        unknown = [k for k in settings if k not in params and not accepts_any]
        if unknown:
            raise PresetError(
                f"{self.path} [{self.name}.solver]: {cls.__name__} has no argument(s) {unknown}; "
                f"valid: {sorted(p for p in params if p not in ('self', 'model'))}"
            )
        return cls(model, **settings, **kwargs)

    def make_config(self, model, cls: type | None = None):
        """Build ``cls.Config`` for config-driven solvers and apply the ``solver`` settings to it."""
        cls = cls or self.solver_class()
        config_cls = cls.Config
        where = f"{self.path} [{self.name}.solver]"
        # Top-level fields go to the constructor (they decide how nested configs are derived);
        # nested tables are applied afterwards to the nested configs it created.
        scalars = {k: v for k, v in self.solver.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in self.solver.items() if isinstance(v, dict)}
        fields = {f.name for f in dataclasses.fields(config_cls)} if dataclasses.is_dataclass(config_cls) else None
        unknown = [k for k in scalars if fields is not None and k not in fields]
        if unknown:
            raise PresetError(f"{where}: {config_cls.__qualname__} has no field(s) {unknown}; valid: {sorted(fields)}")
        if hasattr(config_cls, "from_model"):
            config = config_cls.from_model(model, **scalars)
        else:
            config = config_cls(**scalars)
        _apply(config, nested, where)
        return config

    def apply_model(self, target) -> None:
        """Set each ``model`` setting as an attribute of ``target`` (a Model or ModelBuilder)."""
        _apply(target, dict(self.model), f"{self.path} [{self.name}.model]")


def _apply(obj, settings: dict[str, Any], where: str) -> None:
    for key, value in settings.items():
        if not hasattr(obj, key):
            raise PresetError(f"{where}: {type(obj).__name__} has no field {key!r}")
        current = getattr(obj, key)
        if isinstance(value, dict) and current is not None and not isinstance(current, dict):
            _apply(current, value, f"{where}.{key}")
        else:
            setattr(obj, key, value)


def presets_path(example_file: str | os.PathLike, file: str | None = None) -> Path:
    """The preset file in effect: ``file``, else the runner's ``--presets``, else presets.toml."""
    folder = Path(example_file).resolve().parent
    chosen = file or _file or os.environ.get("SHELF_PRESETS") or "presets.toml"
    return Path(chosen) if Path(chosen).is_absolute() else folder / chosen


def read(example_file: str | os.PathLike, file: str | None = None) -> dict[str, Any]:
    path = presets_path(example_file, file)
    if not path.exists():
        raise PresetError(f"{path} does not exist")
    with path.open("rb") as f:
        return tomllib.load(f)


def names(example_file: str | os.PathLike, file: str | None = None) -> list[str]:
    """Preset names in the file, in file order."""
    return [k for k, v in read(example_file, file).items() if isinstance(v, dict) and k not in ("check", "gallery")]


def is_placeholder(name: str, table: dict[str, Any]) -> bool:
    """True for a table that only says a solver has no preset yet: ``untested`` with no settings.

    (A ``[none]`` table may have no settings and still run: the example builds no solver.)
    """
    return (
        name != NO_SOLVER
        and table.get("status", "untested") == "untested"
        and not any(k in table for k in (*SECTIONS, "entries", "kind"))
    )


def ready(example_file: str | os.PathLike, file: str | None = None) -> list[str]:
    """Preset names that can be run: not unsupported and not a placeholder, in file order."""
    data = read(example_file, file)
    return [
        n
        for n in names(example_file, file)
        if data[n].get("status") != "unsupported" and not is_placeholder(n, data[n])
    ]


def gallery(example_file: str | os.PathLike, file: str | None = None) -> dict[str, Any] | None:
    """The ``[gallery]`` table for the example browser, or None when the file has none.

    ``short`` is the card's one-line description, ``frame`` the frame its preview is rendered at
    (default :data:`DEFAULT_THUMB_FRAME`), ``title`` an optional display name.
    """
    data = read(example_file, file)
    table = data.get("gallery")
    if table is None:
        return None
    path = presets_path(example_file, file)
    if not isinstance(table, dict):
        raise PresetError(f'{path}: [gallery] must be a table with short = "..."')
    unknown = [k for k in table if k not in GALLERY_KEYS]
    if unknown:
        raise PresetError(f"{path} [gallery]: unknown key(s) {unknown}; expected {GALLERY_KEYS}")
    short = table.get("short")
    if not isinstance(short, str) or not short.strip():
        raise PresetError(f"{path} [gallery]: short must be a one-line description")
    frame = table.get("frame", DEFAULT_THUMB_FRAME)
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise PresetError(f"{path} [gallery]: frame must be a non-negative integer, got {frame!r}")
    title = table.get("title")
    if title is not None and not isinstance(title, str):
        raise PresetError(f"{path} [gallery]: title must be a string")
    return {"short": " ".join(short.split()), "frame": frame, "title": title}


def load(example_file: str | os.PathLike, name: str | None = None) -> Preset:
    """Load one preset for the example at ``example_file`` (pass ``__file__``).

    The preset is, in order of precedence: ``name``, the runner's ``--solver``,
    ``$SHELF_SOLVER``, then the file's ``default``.
    """
    data = read(example_file)
    path = presets_path(example_file)
    name = name or _selected or os.environ.get("SHELF_SOLVER") or data.get("default")
    if not name:
        raise PresetError(f"{path} has no default preset; pass --solver")
    if name not in data or not isinstance(data[name], dict):
        raise PresetError(f"{path} has no preset {name!r}; available: {names(example_file)}")
    overrides = dict(_overrides)
    if not overrides and os.environ.get("SHELF_SET"):
        overrides = json.loads(os.environ["SHELF_SET"])
    block = _with_overrides(data[name], overrides)
    if is_placeholder(name, block):
        raise PresetError(
            f"{data.get('example', path.parent.name)} has no {name} preset yet: [{name}] in {path} has no "
            f"settings. Presets that run: {', '.join(ready(example_file)) or 'none'}. To add one, fill "
            f"[{name}] (see AGENTS.md) and run `python -m shelf check`."
        )
    preset = _build(data.get("example", path.parent.name), name, path, block)
    if preset.status == "unsupported":
        raise PresetError(f"{path} [{name}] is unsupported: {block.get('reason', 'no reason given')}")
    return preset


def _with_overrides(block: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    block = json.loads(json.dumps(block, default=str))
    for dotted, value in overrides.items():
        node = block
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return block


def _build(example: str, name: str, path: Path, block: dict[str, Any], kind: str | None = None) -> Preset:
    unknown = [k for k in block if k not in (*SECTIONS, *_META_KEYS)]
    if unknown:
        raise PresetError(f"{path} [{name}]: unknown key(s) {unknown}; expected {SECTIONS} or {sorted(_META_KEYS)}")
    status = block.get("status", "untested")
    if status not in STATUSES:
        raise PresetError(f"{path} [{name}]: status {status!r} is not one of {STATUSES}")
    if "kind" in block:
        if block["kind"] not in SOLVERS:
            raise PresetError(f"{path} [{name}]: kind {block['kind']!r} is not one of {sorted(SOLVERS)}")
        kind = block["kind"]
    entries = {
        entry: _build(example, f"{name}.entries.{entry}", path, {"status": status, **sub}, kind=entry)
        for entry, sub in block.get("entries", {}).items()
    }
    return Preset(
        example=example,
        name=name,
        path=path,
        status=status,
        sim=Settings(block.get("sim", {}), f"{path} [{name}.sim]"),
        collision=Settings(block.get("collision", {}), f"{path} [{name}.collision]"),
        solver=Settings(block.get("solver", {}), f"{path} [{name}.solver]"),
        model=Settings(block.get("model", {}), f"{path} [{name}.model]"),
        entries=entries,
        raw=block,
        kind=kind or name,
    )
