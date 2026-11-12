"""Apply changed preset settings to a running example without rebuilding the scene.

The tuner calls :func:`apply` with the settings the user changed. Each setting has a *tier*,
the cheapest way to make a new value take effect (see :func:`tier`):

``live``
    The solver or the example reads the value on every step: set the attribute.
``model``
    The value lives in a Model array or scalar: write it and call ``notify_model_changed``.
``swap``
    The solver consumes the value at construction: build a new solver on the same model and
    states. The simulation keeps its current state; only the solver's warm start is lost.
``rebuild``
    The example consumes the value while building the scene: construct the example again.
    :func:`apply` raises :class:`NeedsRebuild` and the tuner falls back to a full rebuild.

After any tier the example's CUDA graph, if it has one, is captured again so the new values
are what the graph runs. Nothing here edits Newton; everything goes through public attributes,
``notify_model_changed`` and the preset's ``make_solver``.

Unknown ``solver.*`` keys default to ``swap`` (always correct for constructor arguments) and
unknown ``model.*`` keys default to ``rebuild``; only verified keys are listed as ``live`` or
``model``. See ``README.md`` ("Tune") for the per-solver list.
"""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np

from common import presets

TIERS = ("live", "model", "swap", "rebuild")


class NeedsRebuild(Exception):
    """The change cannot take effect without constructing the example again."""


# --------------------------------------------------------------------------- live solver settings
# ``solver.<key>`` entries the running solver reads every step, per solver kind:
# key -> (where, attribute). ``where`` is "solver" (attribute of the solver instance),
# "coupling" (field of the coupled solver's stored Config), "mjw_opt" (int field of the
# mujoco_warp model options) or "mjw_opt_fill" (per-world array field, filled with the value).
LIVE_SOLVER: dict[str, dict[str, tuple[str, str]]] = {
    "xpbd": {
        "iterations": ("solver", "iterations"),
        "rigid_contact_relaxation": ("solver", "rigid_contact_relaxation"),
        "soft_contact_relaxation": ("solver", "soft_contact_relaxation"),
        "joint_linear_relaxation": ("solver", "joint_linear_relaxation"),
    },
    "vbd": {
        "iterations": ("solver", "iterations"),
        "friction_epsilon": ("solver", "friction_epsilon"),
    },
    "semi_implicit": {
        "angular_damping": ("solver", "angular_damping"),
        "friction_smoothing": ("solver", "friction_smoothing"),
        "enable_tri_contact": ("solver", "enable_tri_contact"),
        "joint_mimic_ke": ("solver", "joint_mimic_ke"),
        "joint_mimic_kd": ("solver", "joint_mimic_kd"),
    },
    "featherstone": {
        # angular_damping is stored by SolverFeatherstone but not read in the pinned Newton
        "angular_damping": ("solver", "angular_damping"),
        "update_mass_matrix_interval": ("solver", "update_mass_matrix_interval"),
    },
    "style3d": {
        "iterations": ("solver", "nonlinear_iterations"),
        "linear_iterations": ("solver", "linear_iterations"),
    },
    "steel": {
        # friction_anchor_beta stays a swap: Steel validates it against contact_compliance at build
        "iterations": ("solver", "_iterations"),
        "enable_restitution": ("solver", "_enable_restitution"),
    },
    "implicit_mpm": {
        "max_iterations": ("solver", "max_iterations"),
        "tolerance": ("solver", "tolerance"),
    },
    "mujoco": {
        "update_data_interval": ("solver", "update_data_interval"),
        "iterations": ("mjw_opt", "iterations"),
        "ls_iterations": ("mjw_opt", "ls_iterations"),
        "tolerance": ("mjw_opt_fill", "tolerance"),
        "ls_tolerance": ("mjw_opt_fill", "ls_tolerance"),
    },
    "kamino": {},
    "feather_pgs": {},
    # coupled solvers (preset kinds like "mujoco_vbd") keep their Config and read these from it
    "coupled": {
        "rho": ("coupling", "rho"),
        "gamma": ("coupling", "gamma"),
        "iterations": ("coupling", "iterations"),
        "contact_matching_force_scale": ("coupling", "contact_matching_force_scale"),
    },
}

# --------------------------------------------------------------------------- model settings
# ``model.<key>`` entries that map to a Model array or scalar: key -> (array attribute, column,
# ModelFlags name, default-source). Arrays are remapped: entries still holding the previous value
# take the new one, so shapes or joints an example configured on purpose keep theirs.
# default-source names the builder config whose default applies when the preset had no value.
MODEL_ARRAYS: dict[str, tuple[str, int | None, str, str | None]] = {
    "default_shape_cfg.mu": ("shape_material_mu", None, "SHAPE_PROPERTIES", "ShapeConfig"),
    "default_shape_cfg.ke": ("shape_material_ke", None, "SHAPE_PROPERTIES", "ShapeConfig"),
    "default_shape_cfg.kd": ("shape_material_kd", None, "SHAPE_PROPERTIES", "ShapeConfig"),
    "default_shape_cfg.kf": ("shape_material_kf", None, "SHAPE_PROPERTIES", "ShapeConfig"),
    "default_shape_cfg.ka": ("shape_material_ka", None, "SHAPE_PROPERTIES", "ShapeConfig"),
    "default_shape_cfg.restitution": ("shape_material_restitution", None, "SHAPE_PROPERTIES", "ShapeConfig"),
    "default_shape_cfg.mu_rolling": ("shape_material_mu_rolling", None, "SHAPE_PROPERTIES", "ShapeConfig"),
    "default_shape_cfg.mu_torsional": ("shape_material_mu_torsional", None, "SHAPE_PROPERTIES", "ShapeConfig"),
    "default_shape_cfg.margin": ("shape_margin", None, "SHAPE_PROPERTIES", "ShapeConfig"),
    "default_joint_cfg.target_ke": ("joint_target_ke", None, "JOINT_DOF_PROPERTIES", "JointDofConfig"),
    "default_joint_cfg.target_kd": ("joint_target_kd", None, "JOINT_DOF_PROPERTIES", "JointDofConfig"),
    "joint_target_ke": ("joint_target_ke", None, "JOINT_DOF_PROPERTIES", None),
    "joint_target_kd": ("joint_target_kd", None, "JOINT_DOF_PROPERTIES", None),
    "cloth.tri_ke": ("tri_materials", 0, "MODEL_PROPERTIES", None),
    "cloth.tri_ka": ("tri_materials", 1, "MODEL_PROPERTIES", None),
    "cloth.tri_kd": ("tri_materials", 2, "MODEL_PROPERTIES", None),
    "cloth.spring_ke": ("spring_stiffness", None, "MODEL_PROPERTIES", None),
    "cloth.spring_kd": ("spring_damping", None, "MODEL_PROPERTIES", None),
}
# ``model.<key>`` entries that are plain scalars on the Model, passed to kernels every step.
MODEL_SCALARS = {"soft_contact_ke", "soft_contact_kd", "soft_contact_kf", "soft_contact_mu", "soft_contact_restitution"}

# example attributes that hold the timing values Newton's examples derive from ``sim``
_TIMING_ATTRS = ("frame_dt", "sim_dt", "sim_substeps")


def _solver_kind(preset: presets.Preset) -> str:
    kind = preset.kind
    if kind in presets.SOLVERS:
        return kind
    return "coupled" if preset.entries else kind


def tier(example, key: str, preset: presets.Preset | None = None) -> str:
    """The cheapest tier that makes a new value of ``key`` (``section.name``) take effect."""
    preset = preset or getattr(example, "preset", None)
    section, _, rest = key.partition(".")
    if section == "sim":
        if rest in ("fps", "substeps"):
            # diffsim examples allocate one State per substep up front (``self.states``), so their
            # timing is fixed at build time
            if isinstance(getattr(example, "states", None), list):
                return "rebuild"
            return "live" if all(hasattr(example, a) for a in _TIMING_ATTRS) else "rebuild"
        return "live" if rest and hasattr(example, rest) else "rebuild"
    if section == "collision":
        return "live" if rest == "interval" and hasattr(example, "collision_interval") else "rebuild"
    if section == "solver":
        # live setters address ``example.solver``; examples that name their solvers differently swap
        if preset is not None and hasattr(example, "solver") and rest in LIVE_SOLVER.get(_solver_kind(preset), {}):
            return "live"
        return "swap"
    if section == "entries":
        parts = rest.split(".")
        return "swap" if len(parts) >= 3 and parts[1] == "solver" else "rebuild"
    if section == "model":
        if rest in MODEL_ARRAYS or rest in MODEL_SCALARS:
            return "model"
        return "rebuild"
    return "rebuild"


def apply(example, changes: dict[str, Any], preset: presets.Preset) -> str:
    """Make ``changes`` (``key -> value``) take effect on the running ``example``.

    ``preset`` is the preset with the changes already applied; it replaces ``example.preset``.
    Returns the most expensive tier that was needed. Raises :class:`NeedsRebuild` before touching
    anything when a change needs a full rebuild, and after a failed attempt otherwise.
    """
    if example is None:
        raise NeedsRebuild("the example is not built")
    old = getattr(example, "preset", None)
    tiers = {key: tier(example, key, old) for key in changes}
    needs = [k for k, t in tiers.items() if t == "rebuild"]
    if needs:
        raise NeedsRebuild(f"{', '.join(needs)} is read while the scene is built")

    used = "live"
    try:
        if any(t == "swap" for t in tiers.values()):
            # the new solver is built from ``preset``, so its kwargs already carry the changes;
            # live solver keys of the same solver are covered by the swap as well
            _swap(example, preset)
            used = "swap"
        for key, value in changes.items():
            section, _, rest = key.partition(".")
            if tiers[key] == "live" and section in ("sim", "collision"):
                _set_timing(example, rest, value, preset)
            elif tiers[key] == "live" and used != "swap":
                _set_solver(example, rest, value, old)
            elif tiers[key] == "model":
                _set_model(example, rest, value, old)
                used = used if used == "swap" else "model"
        example.preset = preset
        recapture(example)
    except NeedsRebuild:
        raise
    except Exception as e:  # anything the live path could not do: let the tuner rebuild instead
        raise NeedsRebuild(f"{type(e).__name__}: {e}") from e
    return used


# --------------------------------------------------------------------------- tiers
def _set_timing(example, name: str, value: Any, preset: presets.Preset) -> None:
    if name == "fps":
        if hasattr(example, "fps"):
            example.fps = value
        example.frame_dt = 1.0 / value
        example.sim_dt = example.frame_dt / example.sim_substeps
    elif name == "substeps":
        example.sim_substeps = int(value)
        example.sim_dt = example.frame_dt / example.sim_substeps
    elif name == "interval":
        example.collision_interval = value
    else:
        setattr(example, name, value)


def _set_solver(example, name: str, value: Any, preset: presets.Preset | None) -> None:
    where, attr = LIVE_SOLVER[_solver_kind(preset)][name]
    solver = example.solver
    if where == "solver":
        if not hasattr(solver, attr):
            raise NeedsRebuild(f"{type(solver).__name__} has no attribute {attr!r}")
        setattr(solver, attr, value)
    elif where == "coupling":
        # the coupled solvers keep a frozen Config dataclass and read it every step
        import dataclasses

        if dataclasses.is_dataclass(solver._coupling):
            solver._coupling = dataclasses.replace(solver._coupling, **{attr: value})
        else:
            setattr(solver._coupling, attr, value)
    elif where == "mjw_opt":
        setattr(solver.mjw_model.opt, attr, value)
    elif where == "mjw_opt_fill":
        getattr(solver.mjw_model.opt, attr).fill_(value)


def _set_model(example, name: str, value: Any, old: presets.Preset | None) -> None:
    import newton

    model = example.model
    if name in MODEL_SCALARS:
        setattr(model, name, value)
        _notify(example, newton.ModelFlags.MODEL_PROPERTIES)
        return
    array_name, column, flag, default_source = MODEL_ARRAYS[name]
    array = getattr(model, array_name, None)
    if array is None or array.shape[0] == 0:
        raise NeedsRebuild(f"the model has no {array_name}")
    previous = _previous_value(old, name, default_source)
    data = array.numpy()
    view = data if column is None else data[:, column]
    mask = np.isclose(view, previous, rtol=1e-6, atol=0.0)
    if not mask.any():
        raise NeedsRebuild(f"no entry of {array_name} holds the previous value {previous!r}")
    view[mask] = value
    array.assign(data)
    _notify(example, getattr(newton.ModelFlags, flag))


def _previous_value(old: presets.Preset | None, name: str, default_source: str | None) -> Any:
    import newton

    if old is not None:
        node: Any = old.model
        for part in name.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is not None:
            return node
    if default_source == "ShapeConfig":
        return getattr(newton.ModelBuilder.ShapeConfig(), name.split(".")[-1])
    if default_source == "JointDofConfig":
        return getattr(newton.ModelBuilder.JointDofConfig(), name.split(".")[-1])
    raise NeedsRebuild(f"model.{name} had no value before, so the entries to change are unknown")


def _notify(example, flags) -> None:
    import newton

    for value in list(vars(example).values()):
        if isinstance(value, newton.solvers.SolverBase):
            value.notify_model_changed(flags)


def _swap(example, preset: presets.Preset) -> None:
    """Build the example's solver again from ``preset`` on the same model and states."""
    example.preset = preset
    if hasattr(example, "build_solver"):
        example.build_solver()
        return
    reason = _not_plain(example)
    if reason:
        raise NeedsRebuild(f"{reason}; give the example a build_solver() method to swap its solver live")
    example.solver = preset.make_solver(example.model)


# --------------------------------------------------------------------------- construction record
# ``Preset.make_solver`` calls made while an example is built, so the swap tier knows whether the
# example's solver was the plain ``self.solver = self.preset.make_solver(self.model)`` it can redo.
_record: list[dict[str, Any]] = []
_installed = False


def install_recorder() -> None:
    """Record every ``Preset.make_solver`` call (idempotent)."""
    global _installed
    if _installed:
        return
    original = presets.Preset.make_solver

    def recording(self, model, solver=None, **kwargs):
        result = original(self, model, solver, **kwargs)
        try:
            ref = weakref.ref(result)
        except TypeError:
            ref = lambda r=result: r
        _record.append({"solver": solver, "kwargs": sorted(kwargs), "result": ref})
        return result

    presets.Preset.make_solver = recording
    _installed = True


def reset_record() -> None:
    _record.clear()


def _not_plain(example) -> str | None:
    """Why the recorded construction is not the plain one-liner, or None when it is."""
    if not _installed:
        return "solver construction was not recorded"
    solver = getattr(example, "solver", None)
    if solver is None:
        return "the example has no self.solver"
    preset = getattr(example, "preset", None)
    if preset is not None and (preset.entries or preset.kind not in presets.SOLVERS):
        return "the preset builds a coupled solver"
    calls = [c for c in _record if c["result"]() is not None]
    if len(calls) != 1:
        return f"the example built {len(calls) or len(_record)} solvers"
    call = calls[0]
    if call["kwargs"] or call["solver"] is not None:
        return f"the solver was built with extra arguments {call['kwargs']}"
    if call["result"]() is not solver:
        return "the built solver is not self.solver"
    for name, value in vars(example).items():
        if value is not solver and getattr(value, "_solver", None) is solver:
            return f"self.{name} belongs to the solver instance"
    return None


# --------------------------------------------------------------------------- graphs
def graphs(example) -> dict[str, Any]:
    """The CUDA graphs the example holds, by attribute name (``graph``, ``_graph``, ``sand_graph``, ...)."""
    import warp as wp

    return {n: v for n, v in vars(example).items() if isinstance(v, wp.Graph)}


def recapture(example) -> None:
    """Capture the example's CUDA graphs again, or drop them, so new values take effect.

    ``capture()`` is called again when the example has one. Any graph it did not re-create (one
    stored under another name, or an example without ``capture()``) still runs the old solver and
    the old values, so it is dropped: Newton's examples guard every launch with ``if self.<graph>:``
    and step eagerly without one.
    """
    before = graphs(example)
    if hasattr(example, "capture") and (before or getattr(example, "use_graph", False)):
        example.graph = None
        example.capture()
    for name, graph in before.items():
        if getattr(example, name, None) is graph:
            setattr(example, name, None)
