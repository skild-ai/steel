# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Which control the tuner draws for a setting, and its range or options.

Ranges come from ``common/ranges.toml`` (the first entry whose pattern matches the setting's key,
optionally restricted to a solver kind), then from Newton's own type hints (``Literal`` options of
the solver constructor or Config), and finally from the current value: a window around it that the
panel widens when you type past it.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import math
import re
import typing
from pathlib import Path
from typing import Any

import tomllib

from common import presets

FILE = Path(__file__).with_name("ranges.toml")


@dataclasses.dataclass
class Control:
    kind: str  # int | float | log | enum | bool | text
    lo: float = 0.0
    hi: float = 1.0
    options: list[str] = dataclasses.field(default_factory=list)
    declared: bool = False  # from ranges.toml or a type hint, not inferred


_rules: list[dict[str, Any]] | None = None
_notes: list[dict[str, Any]] | None = None


def rules() -> list[dict[str, Any]]:
    global _rules, _notes
    if _rules is None:
        with open(FILE, "rb") as f:
            data = tomllib.load(f)
        _rules, _notes = data.get("control", []), data.get("note", [])
    return _rules


def note(key: str, preset: presets.Preset | None) -> tuple[str, str]:
    """``(short, text)`` from the first ``[[note]]`` in ranges.toml matching ``key``; empty if none."""
    rules()
    kind = _solver_kind(preset, key)
    for rule in _notes or []:
        solvers = rule.get("solver")
        if solvers and kind not in solvers:
            continue
        if any(fnmatch.fnmatchcase(key, pat) for pat in rule["match"]):
            return rule.get("short", ""), rule.get("text", "").strip()
    return "", ""


def control(key: str, value: Any, preset: presets.Preset | None) -> Control:
    """The control for ``key`` (``section.name``) holding ``value`` under ``preset``."""
    if isinstance(value, bool):
        return Control("bool", declared=True)
    if isinstance(value, (list, dict)):
        return Control("text")
    kind = _solver_kind(preset, key)
    for rule in rules():
        solvers = rule.get("solver")
        if solvers and kind not in solvers:
            continue
        if any(fnmatch.fnmatchcase(key, pat) for pat in rule["match"]):
            if rule["kind"] == "enum":
                return Control("enum", options=[str(o) for o in rule["options"]], declared=True)
            return Control(rule["kind"], float(rule["min"]), float(rule["max"]), declared=True)
    options = _literal_options(key, preset)
    if options:
        return Control("enum", options=options, declared=True)
    if isinstance(value, str):
        return Control("text")
    return _inferred(value)


def _inferred(value: float) -> Control:
    v = float(value)
    if isinstance(value, int):
        return Control("int", 0.0, float(max(16, 4 * abs(value))))
    if 0.0 <= v <= 1.0:
        return Control("float", 0.0, 1.0)
    if v > 0.0:
        return Control("log", v / 100.0, v * 100.0)
    if v < 0.0:
        return Control("float", 4.0 * v, 0.0)
    return Control("float", 0.0, 1.0)


def widen(ctrl: Control, value: float) -> Control:
    """Stretch ``ctrl`` so ``value`` fits (used when a typed value falls outside the slider)."""
    if ctrl.kind not in ("int", "float", "log"):
        return ctrl
    v = float(value)
    lo, hi = ctrl.lo, ctrl.hi
    if ctrl.kind == "log":
        if v <= 0.0:
            return Control("float", min(v * 2, 0.0), max(hi, 1.0), ctrl.options, ctrl.declared)
        while v < lo:
            lo /= 10.0
        while v > hi:
            hi *= 10.0
    else:
        span = max(hi - lo, 1.0)
        while v < lo:
            lo -= span
        while v > hi:
            hi += span
    return Control(ctrl.kind, lo, hi, ctrl.options, ctrl.declared)


# --------------------------------------------------------------------------- helpers
def _solver_kind(preset: presets.Preset | None, key: str) -> str | None:
    if preset is None:
        return None
    if key.startswith("entries."):
        entry = key.split(".")[1]
        sub = preset.entries.get(entry)
        return sub.kind if sub is not None else entry
    return preset.kind if preset.kind in presets.SOLVERS else "coupled"


def _literal_options(key: str, preset: presets.Preset | None) -> list[str]:
    """Options from a ``Literal`` type hint of the solver's constructor or Config, if any."""
    if preset is None:
        return []
    parts = key.split(".")
    if parts[0] == "entries" and len(parts) >= 4 and parts[2] == "solver":
        target, path = preset.entries.get(parts[1]), parts[3:]
    elif parts[0] == "solver":
        target, path = preset, parts[1:]
    else:
        return []
    if target is None or target.kind not in presets.SOLVERS:
        return []
    try:
        cls = target.solver_class()
    except Exception:  # noqa: BLE001
        return []
    config = getattr(cls, "Config", None)
    owner: Any = config if config is not None and dataclasses.is_dataclass(config) else cls.__init__
    for part in path[:-1]:  # nested configs (kamino: dvi.*, padmm.*)
        hints = _hints(owner)
        owner = _config_class(hints.get(part))
        if owner is None:
            return []
    hint = _hints(owner).get(path[-1])
    return _options_of(hint)


def _config_class(hint: Any) -> type | None:
    """The dataclass in a nested-config hint: ``SomeConfig``, ``SomeConfig | None``, or that as a string."""
    if isinstance(hint, str):  # an annotation that did not resolve: find the class by name in Newton
        import sys

        name = hint.replace("| None", "").replace("Optional[", "").strip(" ]'\"")
        for module_name, module in list(sys.modules.items()):
            found = getattr(module, name, None) if module_name.startswith("newton") else None
            if isinstance(found, type) and dataclasses.is_dataclass(found):
                return found
        return None
    for arg in (hint, *typing.get_args(hint)):
        if isinstance(arg, type) and dataclasses.is_dataclass(arg):
            return arg
    return None


def _hints(owner: Any) -> dict[str, Any]:
    try:
        return typing.get_type_hints(owner, include_extras=True)
    except Exception:  # noqa: BLE001 (forward references that do not resolve: fall back to raw annotations)
        return dict(getattr(owner, "__annotations__", {}))


def _options_of(hint: Any) -> list[str]:
    if hint is None:
        return []
    if isinstance(hint, str):  # an unresolved annotation such as "Literal['a', 'b'] | None"
        m = re.search(r"Literal\[(.*?)\]", hint)
        return [s.strip("'\" ") for s in m.group(1).split(",")] if m else []
    if typing.get_origin(hint) is typing.Literal:
        return [str(a) for a in typing.get_args(hint)]
    for arg in typing.get_args(hint):  # Optional[Literal[...]]
        if typing.get_origin(arg) is typing.Literal:
            return [str(a) for a in typing.get_args(arg)]
    return []


def to_unit(ctrl: Control, value: float) -> float:
    """Map a value to 0..1 along the control's scale."""
    if ctrl.kind == "log":
        lo, hi = math.log10(max(ctrl.lo, 1e-30)), math.log10(max(ctrl.hi, 1e-30))
        v = math.log10(max(float(value), 1e-30))
    else:
        lo, hi, v = ctrl.lo, ctrl.hi, float(value)
    if hi <= lo:
        return 0.0
    return min(1.0, max(0.0, (v - lo) / (hi - lo)))


def from_unit(ctrl: Control, t: float, like: Any) -> Any:
    """Map 0..1 back to a value of the same type as ``like``."""
    t = min(1.0, max(0.0, t))
    if ctrl.kind == "log":
        lo, hi = math.log10(max(ctrl.lo, 1e-30)), math.log10(max(ctrl.hi, 1e-30))
        v = 10.0 ** (lo + t * (hi - lo))
    else:
        v = ctrl.lo + t * (ctrl.hi - ctrl.lo)
    if isinstance(like, int):
        return round(v)
    # keep a sensible number of digits so the field shows a clean number
    return float(f"{v:.4g}")
