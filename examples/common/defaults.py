# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Every setting a preset's solver accepts, with the value it runs at when the preset leaves it out.

The tuner lists the preset's own settings first and these below them, so any solver setting can be
tuned without typing its name. A setting comes from the solver's constructor arguments, or for
config-driven solvers (Kamino, implicit MPM) from the fields of its ``Config`` and nested configs.

Only settings a control can edit are listed: numbers, switches and strings. ``None`` defaults mean
the solver derives the value (from the model, an MJCF file, or other settings); the tuner shows
them as *auto*, with the running solver's value when it can be read, and a button to set one.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
import types
import typing
from typing import Any

import tomllib

from common import live, presets, ranges

# Settings that are not tuning knobs: debugging output, file dumps, or switches that change what
# the example has to build (CPU MuJoCo, separate worlds, an external rigid solver).
HIDDEN = {
    "deterministic",
    "save_to_mjcf",
    "warn_constraint_overflow",
    "row_watermark",
    "collect_solver_info",
    "compute_solution_metrics",
    "enable_mouse_dragging",
    "use_mujoco_cpu",
    "separate_worlds",
    "integrate_with_external_rigid_solver",
}

_SCALARS = {"int": int, "float": float, "bool": bool, "str": str}


@dataclasses.dataclass
class Default:
    value: Any  # what the solver runs at without the setting; None: it derives the value ("auto")
    types: tuple[str, ...]  # the plain types the setting takes: "int", "float", "bool", "str"
    effective: Any = None  # for an auto value, what the running solver uses, when it can be read

    def start(self, ctrl: ranges.Control) -> Any:
        """The first value to put in the control when an auto setting is given one."""
        if self.effective is not None:
            value = self.effective
        elif ctrl.kind == "enum" and ctrl.options:
            return ctrl.options[0]
        elif "bool" in self.types and not ({"int", "float"} & set(self.types)):
            return False
        elif ctrl.declared and ctrl.kind == "log":
            value = 10 ** ((math.log10(max(ctrl.lo, 1e-30)) + math.log10(max(ctrl.hi, 1e-30))) / 2)
        elif ctrl.declared:
            value = (ctrl.lo + ctrl.hi) / 2
        else:
            value = 1.0
        if ctrl.kind == "enum" and str(value) not in ctrl.options:
            return ctrl.options[0] if ctrl.options else str(value)
        if "int" in self.types and "float" not in self.types:
            return round(float(value))
        if "float" in self.types and not isinstance(value, bool):
            return float(value)
        return value


_cache: dict[str, Any] = {"key": None, "value": {}}


def solver_defaults(example, preset: presets.Preset | None) -> dict[str, Default]:
    """``key -> Default`` for every solver setting of ``preset`` (keys as in the tuner: ``solver.x``,
    ``solver.padmm.x``, ``entries.<entry>.solver.x``), including the ones the preset sets."""
    if preset is None:
        return {}
    key = (id(example), str(preset.path), preset.name)
    if _cache["key"] == key:
        return _cache["value"]
    # defaults are taken against the preset as saved: values the tuner changed are not defaults, but
    # the saved settings can change what a config derives (Kamino's dvi settings, for one)
    preset = _saved(preset)
    out: dict[str, Default] = {}
    model = getattr(example, "model", None)
    if preset.entries:
        for name, entry in preset.entries.items():
            out.update(_for(entry, f"entries.{name}.solver.", model, None))
    elif preset.kind in presets.SOLVERS:
        out.update(_for(preset, "solver.", model, getattr(example, "solver", None)))
    # arguments the example passes to make_solver itself cannot be set from the preset
    passed = {k for call in live._record for k in call["kwargs"]}
    out = {k: v for k, v in out.items() if k.rsplit(".", 1)[-1] not in passed}
    _cache.update(key=key, value=out)
    return out


def _saved(preset: presets.Preset) -> presets.Preset:
    """``preset`` as its file has it, without the tuner's overrides."""
    try:
        with open(preset.path, "rb") as f:
            block = tomllib.load(f).get(preset.name)
        if isinstance(block, dict):
            return presets._build(preset.example, preset.name, preset.path, block)
    except (OSError, tomllib.TOMLDecodeError, presets.PresetError):
        pass
    return preset


def _for(preset: presets.Preset, prefix: str, model, solver) -> dict[str, Default]:
    try:
        cls = preset.solver_class()
    except Exception:  # noqa: BLE001 (a solver this Newton does not have)
        return {}
    config_cls = getattr(cls, "Config", None)
    if config_cls is not None and dataclasses.is_dataclass(config_cls):
        return _config(preset, cls, config_cls, prefix, model)
    out = {}
    hints = ranges._hints(cls.__init__)
    for p in list(inspect.signature(cls.__init__).parameters.values())[2:]:
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) or p.name in HIDDEN or p.default is p.empty:
            continue
        d = _default(p.default, hints.get(p.name, p.annotation))
        if d is None:
            continue
        if d.value is None and solver is not None:
            d.effective = _running_value(solver, p.name, d.types)
        out[prefix + p.name] = d
    return out


def _config(preset: presets.Preset, cls, config_cls, prefix: str, model) -> dict[str, Default]:
    """Fields of the effective Config (preset applied, nested configs derived from the model)."""
    config = None
    if model is not None:
        try:
            config = preset.make_config(model, cls)
        except Exception:  # noqa: BLE001 (fall back to the class defaults)
            config = None
    if config is None:
        try:
            config = config_cls()
        except Exception:  # noqa: BLE001
            return {}
    out: dict[str, Default] = {}
    _walk(config, prefix, out)
    return out


def _walk(config, prefix: str, out: dict[str, Default]) -> None:
    hints = ranges._hints(type(config))
    for f in dataclasses.fields(config):
        if f.name in HIDDEN or f.name.startswith("_"):
            continue
        value = getattr(config, f.name, None)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            _walk(value, f"{prefix}{f.name}.", out)
            continue
        hint = hints.get(f.name, f.type)
        if value is None and ranges._config_class(hint) is not None:
            try:
                _walk(ranges._config_class(hint)(), f"{prefix}{f.name}.", out)
            except Exception:  # noqa: BLE001, S110 (a nested config that needs arguments: leave it out)
                pass
            continue
        d = _default(value, hint)
        if d is not None:
            out[prefix + f.name] = d


def _default(value: Any, hint: Any) -> Default | None:
    """A Default for a setting holding ``value`` with type ``hint``, or None if no control fits."""
    kinds = _types(hint)
    if value is None:
        return Default(None, kinds) if kinds else None
    if isinstance(value, bool):
        return Default(value, kinds or ("bool",))
    if isinstance(value, (int, float)) and math.isfinite(value):
        return Default(value, kinds or (type(value).__name__,))
    if isinstance(value, str):
        return Default(value, kinds or ("str",))
    return None


def _types(hint: Any) -> tuple[str, ...]:
    """The plain scalar types in ``hint`` (``float | None`` -> ("float",)); () if it allows others."""
    if hint is None or hint is inspect.Parameter.empty:
        return ()
    if isinstance(hint, str):
        if "Literal[" in hint:
            return ("str",)
        words = [w for w in re.split(r"[\s|]+", hint.replace("Optional[", "").rstrip("]")) if w]
        words = [w for w in words if w != "None"]
        return tuple(words) if words and all(w in _SCALARS for w in words) else ()
    if typing.get_origin(hint) is typing.Literal:
        return ("str",)
    args = typing.get_args(hint) if typing.get_origin(hint) in (typing.Union, types.UnionType) else (hint,)
    names = []
    for arg in args:
        if arg is type(None):
            continue
        if typing.get_origin(arg) is typing.Literal:
            names.append("str")
        elif arg in _SCALARS.values():
            names.append(arg.__name__)
        else:
            return ()
    return tuple(names)


# MuJoCo stores these options as enum integers; the preset takes their names (see ranges.toml)
_MUJOCO_ENUMS = {
    "solver": ("pgs", "cg", "newton"),
    "integrator": ("euler", "rk4", "implicit", "implicitfast"),
    "cone": ("pyramidal", "elliptic"),
    "jacobian": ("dense", "sparse", "auto"),
}


def _running_value(solver, name: str, kinds: tuple[str, ...]) -> Any:
    """What the running ``solver`` uses for ``name``, read from its attributes, or None."""
    candidates = [getattr(solver, name, None), getattr(solver, "_" + name, None)]
    opt = getattr(getattr(solver, "mjw_model", None), "opt", None)  # MuJoCo's per-model options
    if opt is not None:
        value = getattr(opt, name, None)
        if name in _MUJOCO_ENUMS:
            try:
                return _MUJOCO_ENUMS[name][int(value)]
            except (TypeError, ValueError, IndexError):
                return None
        candidates.append(value)
    data = getattr(solver, "mjw_data", None)  # MuJoCo's buffer sizes (njmax, nconmax)
    if data is not None:
        candidates.append(getattr(data, name, None))
    for value in candidates:
        if value is None or callable(value):
            continue
        if hasattr(value, "numpy"):
            try:
                value = value.numpy().reshape(-1)[0].item()
            except Exception:  # noqa: BLE001, S112 (not a readable array: try the next place)
                continue
        if isinstance(value, bool) and "bool" in kinds:
            return value
        if isinstance(value, float) and {"int", "float"} & set(kinds):
            return float(f"{value:.6g}")  # float32 storage: 1e-06, not 9.99999997e-07
        if isinstance(value, int) and not isinstance(value, bool) and {"int", "float"} & set(kinds):
            return value
        if isinstance(value, str) and "str" in kinds:
            return value
    return None
