# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Stability rules: turn episode facts (``episode.py``) into ordered Steel fixes, each with a reason, cost and check.

Pure Python on the facts dict, so it runs offline on a recorded ``trial.json``. The order is the tuning guide's
decision procedure: construction errors, then capacity (dropped rows make every other metric meaningless), then
scene bugs no setting can fix (start state, overlapping shapes), then physics accuracy (joint limits, mass ratios,
grasps, warm start), then cost. ``STABILITY_CALIBRATION`` holds the measured thresholds and costs.

An action carries ``settings`` as ``{"section.key": value}`` with section ``sim`` or ``solver``; a fix that needs a
scene change carries none and is reported as a flag.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

STABILITY_CALIBRATION = {
    # Row capacity: the requested high-water mark undercounts while rows are dropped (dropped contacts change the
    # dynamics), so an overflowing family is sized from 2x its requested mark and re-measured; a healthy one keeps
    # 1.3x headroom over its requested mark.
    "cap_overflow_factor": 2.0,
    "cap_headroom": 1.3,
    "cap_round": 32,
    # Start state: body arrays that differ from FK of the joint coordinates by more than these are lost on step 0
    # (float32 FK of an exported pose differs by ~2 mrad; the dropped starts measured 0.8 rad and 20 rad/s).
    "start_pos_m": 5e-3,
    "start_rot_rad": 0.05,
    "start_vel": 0.05,
    # Rows a world can hold under the immediate response before its kernels stop building; past it, colored
    # propagation (a 20k-row pour built nothing at 41,760 free-body rows; colored ran it at 16,384 propagation rows).
    "immediate_rows_max": 8192,
    # Mass ratio of a contact island (heaviest / lightest dynamic body).
    "mass_ratio_stack": 100.0,  # a static heavy-on-light stack: 3x per layer, ~270:1 overall, toppled
    "mass_ratio_impact": 1000.0,  # a heavy impactor: ~6,500:1 sank light cubes into the ground
    "impact_cost": (2.6, 2.8),  # 2x substeps x 2x iterations + baumgarte 0.5, measured frame-time ratio
    "baumgarte_max": 0.5,  # 0.8 launched bodies out of the scene
    # Penetration of an object into a support, as a fraction of its thinnest half-extent.
    "sink_half_extent": 0.5,
    # Grasp slip over a hold (health.py thresholds).
    "slip_mm": 10.0,
    "slip_deg": 10.0,
    # Joint-limit excess with limits enabled [rad or m].
    "limit_excess": 0.05,
    # Initial overlap between shapes of one articulation [m].
    "initial_overlap_m": 1e-3,
    "loop_gap_m": 1e-3,  # loop-closure anchor separation that counts as an open loop
    "iterations_max": 48,
    "cost_keep_speedup": 0.97,  # a cheaper candidate must be at least 3% faster to be kept
    "pile_free_bodies": 20,  # free bodies per world from which colored propagation is worth a trial
    "frozen_frac": 0.9,  # share of frames repeating the previous poses exactly, from the start, that means no motion
    "substeps_max": 32,
}

# Construction errors SolverSteel raises, and the settings that resolve them.
CONSTRUCTION_FIXES = (
    (
        r"Loop-closing joints are not supported with articulated_contact_response",
        {"solver.articulated_contact_response": "immediate", "solver.solve_mode": "serial"},
        "Loop-closing joints (four-bars, closed chains) are solved only with the immediate response.",
    ),
    (
        r"contact torsion with articulated_contact_response=.* requires contact_torsion_device=True",
        {"solver.contact_torsion_device": True},
        "Shapes author a torsional patch radius; with the propagation response its rows are prepared on the device.",
    ),
    (
        r"Contact torsion graph capture requires contact_torsion_device=True",
        {"solver.contact_torsion_device": True},
        "Shapes author a torsional patch radius; capturing the step in a CUDA graph needs the device torsion path.",
    ),
    (
        r"requires velocity_iterations=0",
        {"solver.velocity_iterations": 0},
        "Contact torsion with the propagation response needs velocity_iterations=0.",
    ),
    (
        r"nesterov_momentum requires solve_mode='serial'",
        {"solver.solve_mode": "serial", "solver.articulated_contact_response": "immediate"},
        "Nesterov momentum runs only in the serial immediate sweep.",
    ),
    (
        r"solve_mode='colored' requires articulated_contact_response='propagation'|colored.*requires.*propagation",
        {"solver.articulated_contact_response": "propagation"},
        "The colored sweep colors propagation rows; it needs the propagation response.",
    ),
    (
        r"propagation_constraint_max requires articulated_contact_response|propagation_constraint_max.*immediate",
        {"solver.propagation_constraint_max": None},
        "propagation_constraint_max applies only with the propagation response.",
    ),
)


@dataclass
class Action:
    id: str
    settings: dict
    reason: str
    evidence: list[str] = field(default_factory=list)
    cost: str = ""
    verify: str = ""
    level: str = "critical"  # critical: the result is wrong without it; warn: likely wrong; info: optional
    metric: str = ""  # summary key the fix must lower; a fix that does not is reverted by the tune loop


def recommend_stability(facts: dict) -> dict:
    """Ordered fixes for one trial's facts; ``actions`` change settings, ``flags`` need a scene change or a look."""
    actions: list[Action] = []
    flags: list[tuple[str, str]] = []
    err = facts.get("build_error") or facts.get("step_error")
    if err and _kernel_build_fix(err, facts.get("settings") or {}, actions):
        return _out(actions, flags)
    if err and _missing_setting_fix(err, facts.get("solver_defaults") or {}, actions):
        return _out(actions, flags)
    if err and _construction_fix(err, actions):
        return _out(actions, flags)  # nothing else ran
    ep = facts.get("episode") or {}
    kw = facts.get("solver_kwargs") or {}
    settings = facts.get("settings") or {}
    _rule_capacity(ep, kw, actions, flags)
    _rule_start_state(facts.get("start_state") or {}, flags)
    _rule_initial_overlap(ep, flags)
    _rule_joint_limits(facts, ep, kw, actions, flags)
    _rule_loop_gap(facts, ep, kw, actions)
    capacity_fix = any(a.id.endswith("_capacity") for a in actions)
    if not capacity_fix:
        # Accuracy rules read contacts and penetration, which dropped rows distort; size the rows first.
        _rule_warmstart(ep, kw, actions, flags)
        _rule_grasp(ep, kw, actions, flags)
        _rule_mass_ratio(facts, ep, kw, settings, actions, flags)
        _rule_escalate(facts, ep, kw, settings, actions, flags)
    _rule_spellings(settings, flags)
    _rule_episode_length(facts, flags)
    if err:
        flags.append(("critical", f"The episode raised: {err}"))
    if facts.get("run_error"):
        flags.append(("critical", f"The example raised after its episode: {facts['run_error']}"))
    _rule_invalid_contacts(ep, flags)
    frames = ep.get("frames") or 0
    frozen = (ep.get("unchanged_frames") or 0) > STABILITY_CALIBRATION["frozen_frac"] * frames
    if frames >= 20 and frozen and not kw.get("enable_sleeping"):
        # Poses repeating from the start for most of the episode: either nothing was meant to move, or the state the
        # example exposes is not advancing. A start velocity says something should have moved.
        moving = (ep.get("initial_speed_max") or 0.0) > 1e-3
        flags.append(
            (
                "critical" if moving else "warn",
                f"Body poses repeated exactly from frame to frame from the start, for {ep['unchanged_frames']} of "
                f"{frames} frames"
                + (" although bodies start moving" if moving else "")
                + ": the state the example exposes may "
                "not be advancing (examples that swap two states inside a captured CUDA graph freeze at an odd "
                "substep count; keep it even). A frozen scene passes every other metric.",
            )
        )
    if (ep.get("bodies_inside_static") or 0) > 0:
        flags.append(
            (
                "critical",
                f"{ep['bodies_inside_static']} bodies' centres of mass entered a colliding static box "
                f"(up to {1e3 * (ep.get('inside_static_depth_max_m') or 0):.0f} mm deep, first at frame "
                f"{ep.get('inside_static_first_frame')}): they sank or tunnelled through it.",
            )
        )
    _drop_unapplied(facts, actions, flags)
    if (facts.get("check") or {}).get("ok") is False:
        flags.append(("critical", f"The example's own check failed: {(facts['check'].get('note') or '')[:300]}"))
    return _out(actions, flags)


def summarize(facts: dict) -> dict:
    """The handful of numbers a tuning table needs."""
    ep = facts.get("episode") or {}
    h = ep.get("health") or {}
    wm = ep.get("row_watermarks") if isinstance(ep.get("row_watermarks"), dict) else {}
    return {
        "frames": ep.get("frames"),
        "frames_planned": facts.get("frames_planned"),
        "frames_done": facts.get("frames_done"),
        "nonfinite_frames": h.get("nonfinite_frames"),
        "invalid_contact_frames": h.get("invalid_contact_frames"),
        "overflow_world_steps": {
            f: wm.get(f"{f}_overflow_world_steps") for f in ("articulation", "free_body", "propagation")
        },
        "capacity_check": ep.get("capacity_check"),
        "frame_ms": ep.get("frame_ms_median"),
        "penetration_peak_m": ep.get("penetration_peak_m"),
        "penetration_p95_m": ep.get("penetration_p95_m"),
        "sink_over_half_extent": h.get("object_support_penetration_over_half_thickness"),
        "speed_max": ep.get("body_speed_max"),
        "bursts": h.get("bursts"),
        "island_mass_ratio": ep.get("island_mass_ratio_max"),
        "grasp_slip_max_mm": max(
            (g["slip_mm"] for g in h.get("grasp_holds") or [] if g.get("object_free", True)), default=None
        ),
        "grasp_slip_max_deg": max(
            (g["slip_deg"] for g in h.get("grasp_holds") or [] if g.get("object_free", True)), default=None
        ),
        "joint_limit_excess": ep.get("joint_limit_excess_max"),
        "loop_gap_m": ep.get("loop_gap_max_m"),
        "bodies_inside_static": ep.get("bodies_inside_static"),
        "frozen_frac": (ep.get("unchanged_frames") or 0) / max(ep.get("frames") or 1, 1),
    }


def healthy(result: dict, rejected: set[str] | frozenset = frozenset()) -> bool:
    """A trial with no fix left (other than ``rejected`` warn-level ones), no critical flag and a passing (or
    skipped) example check. Non-finite state or contact geometry, a truncated episode and a failed capacity check
    always block it."""
    advice = result.get("advice") or {}
    summary = result.get("summary") or {}
    left = [a for a in advice.get("actions") or [] if not (a.get("id") in rejected and a.get("level") != "critical")]
    critical = [
        m
        for m in advice.get("critical") or []
        if not any(m == a.get("reason") for a in advice.get("actions") or [] if a.get("id") in rejected)
    ]
    valid = (
        not summary.get("nonfinite_frames")
        and not summary.get("invalid_contact_frames")
        and summary.get("capacity_check") == "ok"
        and summary.get("frames_planned") is not None
        and summary.get("frames_done") == summary.get("frames_planned")
    )
    return result.get("status") == "ok" and valid and not left and not critical and summary.get("check_ok") is not False


def improved(before: dict, after: dict, metric: str, factor: float = 0.75) -> bool:
    """Whether any of the ``|``-separated metrics dropped to at most ``factor`` of its value. A metric missing after
    the fix is not evidence of improvement; one missing before it cannot be compared and is skipped."""
    for key in metric.split("|"):
        b, a = (before or {}).get(key), (after or {}).get(key)
        if a is None or b is None or b <= 0:
            continue
        if a <= factor * b:
            return True
    return False


def no_worse(base: dict, cand: dict, best: dict | None = None) -> bool:
    """A cheaper trial is kept only if it is faster than ``best`` (default ``base``) and its health numbers stay
    near those of ``base``, the first healthy settings."""

    def g(d, k, default=0.0):
        v = d.get(k)
        return default if v is None else v

    c = STABILITY_CALIBRATION
    best = base if best is None else best
    if any(base.get(k) is not None and cand.get(k) is None for k in _HEALTH_KEYS):
        return False  # a number the first healthy settings measured is no longer measured: unknown, not zero
    return (
        g(cand, "frame_ms", math.inf) < c["cost_keep_speedup"] * g(best, "frame_ms", math.inf)
        and g(cand, "penetration_p95_m") <= 1.5 * g(base, "penetration_p95_m") + 1e-3
        and g(cand, "speed_max") <= 1.5 * g(base, "speed_max") + 1.0
        and g(cand, "sink_over_half_extent") <= max(c["sink_half_extent"], 1.5 * g(base, "sink_over_half_extent"))
        and g(cand, "bursts") <= 1.5 * g(base, "bursts") + 2  # a chaotic scene's burst count varies run to run
        and g(cand, "joint_limit_excess") <= g(base, "joint_limit_excess") + 0.02
        and g(cand, "loop_gap_m") <= max(c["loop_gap_m"], 1.5 * g(base, "loop_gap_m"))
        and g(cand, "bodies_inside_static") <= g(base, "bodies_inside_static")
        and g(cand, "frozen_frac") <= g(base, "frozen_frac") + 0.05
        and g(cand, "grasp_slip_max_mm") <= max(c["slip_mm"], 1.5 * g(base, "grasp_slip_max_mm"))
        and g(cand, "grasp_slip_max_deg") <= max(c["slip_deg"], 1.5 * g(base, "grasp_slip_max_deg"))
    )


_HEALTH_KEYS = (
    "penetration_p95_m",
    "speed_max",
    "sink_over_half_extent",
    "bursts",
    "joint_limit_excess",
    "loop_gap_m",
    "bodies_inside_static",
    "grasp_slip_max_mm",
    "grasp_slip_max_deg",
)


def cost_candidates(facts: dict, settings: dict) -> list[dict]:
    """Cheaper settings to time once a trial is healthy, most promising first (``Action`` dicts)."""
    c = STABILITY_CALIBRATION
    kw = facts.get("solver_kwargs") or {}
    sim = settings.get("sim") or {}
    solver = settings.get("solver") or {}
    sub = int(sim.get("substeps") or facts.get("substeps") or 1)
    its = int(solver.get("iterations") or kw.get("iterations") or 12)
    ep = facts.get("episode") or {}
    h = ep.get("health") or {}
    out: list[Action] = []
    model = facts.get("model") or {}
    loops = (ep.get("structure") or {}).get("connect_rows_joints")
    if (
        (model.get("free_bodies_world0") or 0) >= c["pile_free_bodies"]
        and not loops
        and not kw.get("nesterov_momentum")
        and kw.get("articulated_contact_response") in (None, "immediate")
    ):
        colored = {"solver.solve_mode": "colored", "solver.articulated_contact_response": "propagation"}
        if (model.get("torsion") or {}).get("shapes_with_radius_world0"):
            colored.update({"solver.contact_torsion_device": True, "solver.velocity_iterations": 0})
        out.append(
            Action(
                "colored_propagation",
                colored,
                f"{model['free_bodies_world0']} free bodies per world: colored propagation solves independent "
                "contacts in parallel instead of one world's rows in series.",
                [
                    "Colored won on every large free-body pile timed (card house, jenga, dominoes, a bolt sweep, a "
                    "pour, a drawer of loose objects, a toy train) and lost on robots and small scenes."
                ],
                "0.4-0.9x frame time on piles",
                "health, the check, and capacities re-measured",
                "info",
            )
        )
    if sub >= 2:
        out.append(
            Action(
                "fewer_substeps",
                {"sim.substeps": sub // 2},
                "Half the substeps at the same iterations: the step cost is mostly per substep.",
                [
                    "Winning cells on 18 contact scenes used 1-2 substeps in 13; one substep's fixed cost equals "
                    "7-60 iteration sweeps (median 18)."
                ],
                "about 0.5x frame time",
                "health and the check",
                "info",
            )
        )
        out.append(
            Action(
                "fewer_substeps_same_sweeps",
                {"sim.substeps": sub // 2, "solver.iterations": min(2 * its, c["iterations_max"])},
                "Half the substeps, twice the iterations: a convergence-limited scene needs a sweep count, "
                "and sweeps are cheaper as iterations than as substeps.",
                [
                    "Six-cube 3x stack: every cell with about 96 sweeps per frame stood (momentum on); 1 substep "
                    "x 88 iterations was 2.0x the 8 x 16 preset. A time-step-limited rope (4 x 6 robust, 2 x 12 "
                    "not) rejects it."
                ],
                "0.55-0.8x frame time",
                "health and the check",
                "info",
            )
        )
    if kw.get("enable_joint_limits") and solver.get("joint_limit_activation_gap") is None:
        out.append(
            Action(
                "limit_gap",
                {"solver.joint_limit_activation_gap": 0.1},
                "A fixed 0.1 activation gap builds limit rows only near a limit.",
                ["8 scene presets: 8-40% faster with the same result."],
                "0.7-0.92x",
                "joint-limit excess",
                "info",
            )
        )
    held = [g for g in h.get("grasp_holds") or [] if g.get("frames", 0) >= 30]
    beta = kw.get("friction_anchor_beta")
    if not held and (beta is None or beta > 0):
        out.append(
            Action(
                "point_friction",
                {"solver.friction_anchor_beta": 0.0},
                "Nothing is held: point friction drops the friction-patch bookkeeping. It is a different "
                "friction model (one point per contact, as in MuJoCo); keep anchors for grasps and pinches.",
                ["17-25% faster on legged and sensor scenes; patches kept where objects are pinched or carried."],
                "0.75-0.85x",
                "creep of resting bodies, the check",
                "info",
            )
        )
    if its >= 8:
        out.append(
            Action(
                "fewer_iterations",
                {"solver.iterations": max(4, its // 2)},
                "Half the iterations: scenes with few coupled rows converge well below Steel's default 12.",
                [
                    "One substep's fixed cost equals 7-60 iteration sweeps (median 18) on 24 contact scenes, so "
                    "iterations are the smaller lever."
                ],
                "less than the substep candidates; time it",
                "health and the check",
                "info",
            )
        )
    return [asdict(a) for a in out]


# ------------------------------------------------------------------------------------------------
# Rules
# ------------------------------------------------------------------------------------------------


def _construction_fix(err: str, actions: list[Action]) -> bool:
    for pattern, settings, why in CONSTRUCTION_FIXES:
        if re.search(pattern, err):
            actions.append(
                Action(
                    "construction_" + next(iter(settings)).split(".")[1],
                    dict(settings),
                    why,
                    [err[:300]],
                    "none",
                    "the solver builds",
                )
            )
            return True
    m = re.search(r"SolverSteel has no argument\(s\) \[([^\]]*)\]", err)
    if m:
        names = [n.strip(" '\"") for n in m.group(1).split(",")]
        actions.append(
            Action(
                "construction_unknown_kwargs",
                {f"solver.{n}": None for n in names},
                "The installed Newton has no such option (renamed or removed); drop it and re-check.",
                [err[:300]],
                "none",
                "the solver builds",
            )
        )
        return True
    return False


def _missing_setting_fix(err: str, defaults: dict, actions: list[Action]) -> bool:
    """The example reads a preset setting the settings leave out: write Steel's default for it."""
    m = re.search(r"\[steel\.(sim|solver)\] has no '(\w+)'", err)
    if not m:
        return False
    section, key = m.groups()
    value = defaults.get(key) if section == "solver" else {"substeps": 1, "fps": 60}.get(key)
    if value is None:
        return False
    settings = {f"{section}.{key}": value}
    if section == "solver":
        # Examples that read one of these usually read all three.
        for k in ("iterations", "solve_mode", "articulated_contact_response"):
            if defaults.get(k) is not None:
                settings.setdefault(f"solver.{k}", defaults[k])
    actions.append(
        Action(
            f"missing_{key}",
            settings,
            f"The example reads {section}.{key} from its preset; write Steel's defaults for it and the "
            "other settings examples read.",
            [err[:300]],
            "none",
            "the example builds",
            "critical",
        )
    )
    return True


def _kernel_build_fix(err: str, settings: dict, actions: list[Action]) -> bool:
    """A kernel that fails to build under oversized immediate-response capacities: switch to colored propagation."""
    c = STABILITY_CALIBRATION
    solver = settings.get("solver") or {}
    big = {
        k: v
        for k, v in solver.items()
        if k == "free_body_constraint_max" and isinstance(v, int) and v > c["immediate_rows_max"]
    }
    if (
        "kernel build failed" not in err.lower()
        or not big
        or solver.get("articulated_contact_response") not in (None, "immediate")
    ):
        return False
    target = _round_up(sum(big.values()) / c["cap_overflow_factor"] * c["cap_headroom"], c["cap_round"])
    actions.append(
        Action(
            "colored_for_rows",
            {
                "solver.solve_mode": "colored",
                "solver.articulated_contact_response": "propagation",
                "solver.propagation_constraint_max": max(target, c["immediate_rows_max"]),
                **{f"solver.{k}": None for k in big},
            },
            f"Kernels failed to build with {big}: too many rows for one world under the immediate response. Colored "
            "propagation holds them in the propagation family.",
            [err[:300]],
            "see the next trial",
            "the solver builds; overflow_world_steps == 0",
        )
    )
    return True


def _rule_capacity(ep: dict, kw: dict, actions: list[Action], flags: list) -> None:
    """Dropped rows: size each overflowing family from its requested mark; any overflow is a failed run."""
    wm = ep.get("row_watermarks")
    if not isinstance(wm, dict):
        flags.append(
            ("warn", f"No row watermarks ({wm}); construct SolverSteel(row_watermark=True) to size capacities.")
        )
        if ep.get("capacity_check") not in (None, "ok"):
            flags.append(("critical", f"check_constraint_capacity() failed: {ep['capacity_check']}"))
        return
    caps = ep.get("capacities") or {}
    propagation = (ep.get("structure") or {}).get("propagation") or kw.get(
        "articulated_contact_response"
    ) == "propagation"
    c = STABILITY_CALIBRATION
    for fam, key in (
        ("articulation", "articulation_constraint_max"),
        ("free_body", "free_body_constraint_max"),
        ("propagation", "propagation_constraint_max"),
    ):
        over = wm.get(f"{fam}_overflow_world_steps", 0) or 0
        raw = wm.get(f"{fam}_raw_high_water", 0) or 0
        cap = caps.get(fam)
        if not over:
            continue
        if fam == "propagation" and not propagation:
            continue
        target = _round_up(max(raw * c["cap_overflow_factor"], (cap or 0) + c["cap_round"]), c["cap_round"])
        dropped = wm.get(f"{fam}_dropped_contact_rows_high_water", 0)
        actions.append(
            Action(
                f"{fam}_capacity",
                {f"solver.{key}": target},
                f"Steel dropped {fam} rows in {over} world-steps (requested up to {raw} rows, capacity {cap}, up to "
                f"{dropped} dropped contact rows in one world-step). Rows past the capacity are skipped, so contacts, limits "
                "or loop rows silently stop acting; the result can look normal on video.",
                [
                    f"The requested mark undercounts while rows are dropped (a 20-box overlap scene read 1009 rows at the "
                    f"default 544 and 1038 with room), so the capacity goes to {c['cap_overflow_factor']}x it and the next "
                    "trial re-measures."
                ],
                "rows that were dropped are now solved: +0-22% frame time measured (overlap +22%, linkages +0%)",
                f"next trial: {fam}_overflow_world_steps == 0 and check_constraint_capacity() passes",
            )
        )
    _switch_to_colored(ep, kw, actions)
    if not any(a.id.endswith("_capacity") for a in actions) and ep.get("capacity_check") not in (None, "ok"):
        flags.append(("critical", f"check_constraint_capacity() failed without a row overflow: {ep['capacity_check']}"))
    hw, cmax = ep.get("contact_count_max") or 0, ep.get("rigid_contact_max") or 0
    if cmax and hw >= cmax:
        flags.append(
            (
                "critical",
                f"The contact buffer filled ({hw} of {cmax} contacts): contacts past it are lost. Raise "
                "the collision pipeline's rigid_contact_max.",
            )
        )


def _switch_to_colored(ep: dict, kw: dict, actions: list[Action]) -> None:
    """Immediate-response rows past what one world's kernels hold: move the contacts to colored propagation."""
    c = STABILITY_CALIBRATION
    if kw.get("articulated_contact_response") not in (None, "immediate") or kw.get("nesterov_momentum"):
        return
    if (ep.get("structure") or {}).get("connect_rows_joints"):
        return  # loop closures need the immediate response
    # Only free-body contact rows move to the propagation family; joint limits, mimic, loop and drive rows stay dense,
    # so an articulation raise is kept as it is.
    fb = next((a for a in actions if a.id == "free_body_capacity"), None)
    art = next((a for a in actions if a.id == "articulation_capacity"), None)
    if fb is None or next(iter(fb.settings.values())) <= c["immediate_rows_max"]:
        return
    actions.remove(fb)
    need = next(iter(fb.settings.values())) + (
        next(iter(art.settings.values())) if art else (ep.get("capacities") or {}).get("articulation", 32)
    )
    target = _round_up(need, c["cap_round"])
    actions.append(
        Action(
            "propagation_capacity",
            {
                "solver.solve_mode": "colored",
                "solver.articulated_contact_response": "propagation",
                "solver.propagation_constraint_max": target,
                "solver.free_body_constraint_max": None,
            },
            f"The scene asks for about {need} rows per world, past the {c['immediate_rows_max']} the immediate response's "
            "per-world kernels hold. Colored propagation routes contacts, free bodies included, to the propagation family "
            "and solves independent contacts in parallel.",
            [
                "A 168-piece pour asked for ~20,000 free-body rows; at 41,760 the immediate kernels failed to build, and the "
                "colored preset runs it at 16,384 propagation rows. Colored also won on every large free-body pile timed "
                "(card house, jenga, dominoes, a bolt sweep, the pour)."
            ],
            "colored is bound by the busiest world and its most-contacted body; often faster on piles",
            "the next trial re-measures every family; overflow_world_steps == 0",
        )
    )


def _rule_start_state(st: dict, flags: list) -> None:
    if not st.get("checked") or st.get("nonfinite"):
        return
    c = STABILITY_CALIBRATION
    big = (
        (st.get("pose_mismatch_max_m") or 0.0) > c["start_pos_m"]
        or (st.get("pose_mismatch_max_rad") or 0.0) > c["start_rot_rad"]
        or (st.get("velocity_mismatch_max") or 0.0) > c["start_vel"]
    )
    if (st.get("pose_mismatch_bodies") or st.get("velocity_mismatch_bodies")) and not big:
        flags.append(
            (
                "info",
                f"Start state: body arrays differ from FK of the joint coordinates by at most "
                f"{1e3 * (st.get('pose_mismatch_max_m') or 0):.2f} mm / {st.get('pose_mismatch_max_rad') or 0:.4f} rad / "
                f"{st.get('velocity_mismatch_max') or 0:.3g} (rounding of an exported pose).",
            )
        )
        return
    if big:
        what = []
        if st.get("pose_mismatch_bodies"):
            what.append(
                f"{st['pose_mismatch_bodies']} body poses differ from FK of joint_q by up to "
                f"{1e3 * st['pose_mismatch_max_m']:.1f} mm / {st['pose_mismatch_max_rad']:.3f} rad"
            )
        if st.get("velocity_mismatch_bodies"):
            what.append(
                f"{st['velocity_mismatch_bodies']} body velocities differ from FK of joint_qd by up to "
                f"{st['velocity_mismatch_max']:.3g}"
                + (
                    f" ({st['velocity_in_body_qd_only_bodies']} have joint_qd = 0)"
                    if st.get("velocity_in_body_qd_only_bodies")
                    else ""
                )
            )
        flags.append(
            (
                "critical",
                "Start state: " + "; ".join(what) + f" (e.g. {st.get('examples')}). Steel integrates "
                "State.joint_q / joint_qd; a pose or velocity written only to body_q / body_qd is "
                "overwritten on the first step. Write the start into the state's joint coordinates "
                "(builder joint_q / joint_qd before finalize, or eval_ik into the state, then eval_fk).",
            )
        )


def _rule_initial_overlap(ep: dict, flags: list) -> None:
    first = (ep.get("health") or {}).get("first_frame_penetration_by_class") or {}
    rr = first.get("robot_robot", 0.0)
    if rr > STABILITY_CALIBRATION["initial_overlap_m"]:
        flags.append(
            (
                "warn",
                f"Robot links overlap by {1e3 * rr:.1f} mm after the first frame. Shapes of one mechanism "
                "that overlap at rest fight the drives (a wrist capsule inside its hand put 65 J into a "
                "0.4 kg box; filtered, 1.3 J). Add collision-filter pairs for shapes that should never "
                "touch; no solver setting fixes it.",
            )
        )


def _rule_joint_limits(facts: dict, ep: dict, kw: dict, actions: list[Action], flags: list) -> None:
    limited = (facts.get("model") or {}).get("limited_dofs_world0", 0)
    excess = ep.get("joint_limit_excess_max") or 0.0
    if limited and not kw.get("enable_joint_limits"):
        actions.append(
            Action(
                "enable_joint_limits",
                {"solver.enable_joint_limits": True},
                f"{limited} DOFs have finite limits but SolverSteel's default is enable_joint_limits=False: the limits are "
                f"not simulated (largest excess this episode {excess:.3f}).",
                ["Other solvers enforce authored limits; with them off, joints travel freely past them."],
                "a few rows per limited DOF near its limit",
                "joint_limit_excess_max small; capacities re-measured",
                "warn",
            )
        )
    elif excess > STABILITY_CALIBRATION["limit_excess"]:
        flags.append(
            (
                "warn",
                f"Joint {ep.get('joint_limit_excess_joint')} went {excess:.3f} past its limit with limits on. "
                "Impacts can do this on any solver; compare with a reference solver before tuning. More "
                "substeps halve it at 2x cost.",
            )
        )


def _rule_loop_gap(facts: dict, ep: dict, kw: dict, actions: list[Action]) -> None:
    """Loop-closing joints that open: the closure rows have not converged within the step."""
    gap = ep.get("loop_gap_max_m")
    c = STABILITY_CALIBRATION
    if gap is None or gap <= c["loop_gap_m"]:
        return
    its = int(kw.get("iterations") or 12)
    sub = int(((facts.get("settings") or {}).get("sim") or {}).get("substeps") or facts.get("substeps") or 1)
    new = (
        {"solver.iterations": min(2 * its, c["iterations_max"])}
        if its < c["iterations_max"]
        else {"sim.substeps": min(2 * sub, c["substeps_max"])}
    )
    actions.append(
        Action(
            "loop_gap",
            new,
            f"Loop joint {ep.get('loop_gap_joint')} opened {1e3 * gap:.2f} mm: the loop-closure rows did not converge.",
            [
                "A closed leg linkage: 24 iterations held a 0.10 mm loop gap, 50 held 0.052 mm; a four-bar at 12-16 "
                "iterations let out-of-plane tilt grow until it fell (32 held)."
            ],
            "about linear in iterations for the sweep share",
            f"loop_gap_max_m below {1e3 * c['loop_gap_m']:.1f} mm",
            "warn",
            "loop_gap_m",
        )
    )


def _rule_warmstart(ep: dict, kw: dict, actions: list[Action], flags: list) -> None:
    if not kw.get("contact_warmstart"):
        return
    h = ep.get("health") or {}
    ro = (h.get("warmstart_match_fast") or {}).get("robot_object") or {}
    if h.get("bursts"):
        actions.append(
            Action(
                "warmstart_bursts",
                {"solver.contact_warmstart": False},
                f"{h['bursts']} velocity bursts with contact warm start on. Warm-started impulses on deeply wedged resting "
                "contacts occasionally launch a body.",
                [
                    "Pour scene: 14 of 256 worlds burst (peak 70 m/s) with warm start; 0 with it off, 0 with "
                    "contact_regularization=0.01 and warm start kept."
                ],
                "none",
                "bursts == 0 over the episode, at the worlds you run",
                "warn",
            )
        )
        return
    if ro.get("contacts") and (ro.get("matched_frac") or 0.0) < 0.5:
        actions.append(
            Action(
                "warmstart_unmatched",
                {"solver.contact_warmstart": False},
                f"Warm start is on but only {100 * ro['matched_frac']:.0f}% of moving robot-object contacts matched their "
                "previous step. Contact matching compares world-space midpoints within a small threshold, so carried "
                "contacts often start cold; matching also forces a sorted contact order, which biases a pinch.",
                [
                    "Pinch repro: 100% matched at 0.05 m/s, 28% at 0.1 m/s (24.5 deg twist); static hold 4.5 deg unsorted vs "
                    "27.8 deg sorted. Raising the match threshold to 2-10 mm did not save the grasp."
                ],
                "saves the sort and the match",
                "grasp slip and task check",
                "warn",
            )
        )


def _rule_grasp(ep: dict, kw: dict, actions: list[Action], flags: list) -> None:
    holds = (ep.get("health") or {}).get("grasp_holds") or []
    c = STABILITY_CALIBRATION
    slipping = [g for g in holds if g["slip_mm"] > c["slip_mm"] or g["slip_deg"] > c["slip_deg"]]
    jointed = [g for g in slipping if g.get("object_free") is False]
    if jointed:
        g = jointed[0]
        flags.append(
            (
                "info",
                f"{g['object']} moved {g['slip_mm']} mm / {g['slip_deg']} deg against {g['robot']} while "
                "touching it, but it is on a joint (a handle, a knob): its joint motion counts here, so this "
                "is not read as grasp slip.",
            )
        )
    slipping = [g for g in slipping if g.get("object_free") is not False]
    if not slipping:
        return
    g = slipping[0]
    beta = kw.get("friction_anchor_beta")
    ev = [
        f"{g['object']} drifted {g['slip_mm']} mm / {g['slip_deg']} deg against {g['robot']} over a {g['frames']}-frame hold."
    ]
    if beta is not None and beta <= 0:
        settings = {"solver.friction_anchor_beta": 0.2}
        if kw.get("contact_warmstart"):
            settings["solver.contact_warmstart"] = False
        actions.append(
            Action(
                "grasp_anchors",
                settings,
                "A carried object slips under point friction. Friction anchors carry the tangential displacement of each "
                "contact region across steps and are matched by body pair and geometry, so they hold a moving grasp that "
                "point friction plus warm start loses.",
                ev
                + [
                    "Pinch-and-pour: point friction + warm start 0/64 worlds; anchors 0.2 + warm start off 64/64 (x4) and "
                    "256/256, 1.10x faster. Anchors 0.1: 34/64. 32 iterations: 53/64 at 2x cost; 8 substeps x 16: 63/64 "
                    "at 3x."
                ],
                "none measured (faster on the pour scene)",
                "grasp slip < 10 mm / 10 deg; the task's own check at >= 64 worlds",
                "warn",
                "grasp_slip_max_mm|grasp_slip_max_deg",
            )
        )
        return
    its = int(kw.get("iterations") or 12)
    if its < c["iterations_max"]:
        actions.append(
            Action(
                "grasp_iterations",
                {"solver.iterations": min(2 * its, c["iterations_max"])},
                "A carried object slips with friction anchors on; the friction rows have not converged.",
                ev
                + [
                    "In-hand manipulation moves the object against the fingers by design: the tune loop keeps this only "
                    "if the slip drops."
                ],
                "about linear in iterations for the sweep share of the step",
                "grasp slip and the task check",
                "warn",
                "grasp_slip_max_mm|grasp_slip_max_deg",
            )
        )
    else:
        flags.append(
            (
                "warn",
                f"Unresolved grasp slip at {its} iterations with friction anchors on: {ev[0]} No further setting is "
                "tried; check that the motion is intended (in-hand manipulation) or fix the grasp in the scene.",
            )
        )


def _rule_mass_ratio(facts: dict, ep: dict, kw: dict, settings: dict, actions: list[Action], flags: list) -> None:
    """Heavy bodies on light ones: Gauss-Seidel moves the load through each contact row in turn, so it converges
    slowly. An impactor needs more substeps and iterations; a resting stack needs momentum."""
    c = STABILITY_CALIBRATION
    ratio = ep.get("island_mass_ratio_max") or 1.0
    h = ep.get("health") or {}
    sink = h.get("object_support_penetration_over_half_thickness") or 0.0
    check_failed = (facts.get("check") or {}).get("ok") is False
    if ratio < c["mass_ratio_stack"]:
        return
    pair = ep.get("island_mass_ratio_bodies")
    sinking = sink > c["sink_half_extent"]
    impact = ratio >= c["mass_ratio_impact"]
    # A heavy impactor shows as light bodies sinking; a resting stack as a failed check (thin pieces wedged in a hull
    # support also sink past half their thickness, with the check passing).
    if not ((impact and sinking) or check_failed or h.get("nonfinite_frames")):
        flags.append(
            (
                "info",
                f"Contact island mass ratio {ratio:.0f}:1 ({pair}) with no sinking seen. Keep the episode "
                "long enough to include the heavy body's load (impact, stacking) before trusting it.",
            )
        )
        return
    sim = settings.get("sim") or {}
    sub = int(sim.get("substeps") or facts.get("substeps") or 1)
    its = int(kw.get("iterations") or 12)
    if impact:
        new = {"sim.substeps": min(2 * sub, c["substeps_max"]), "solver.iterations": min(2 * its, c["iterations_max"])}
        if (kw.get("baumgarte") or 0.2) < c["baumgarte_max"]:
            new["solver.baumgarte"] = c["baumgarte_max"]
        if new == {"sim.substeps": sub, "solver.iterations": its}:
            flags.append(("warn", f"Mass ratio {ratio:.0f}:1 still sinks at {sub} substeps x {its} iterations."))
            return
        actions.append(
            Action(
                "mass_ratio_impact",
                new,
                f"Contact island mass ratio {ratio:.0f}:1 ({pair}) and a light body sank {sink:.1f}x its half-extent into "
                "a support. Gauss-Seidel moves a heavy body's load through each light body's rows in turn and needs "
                "more substeps and iterations; a stronger position correction pulls the bodies it pressed in back out.",
                [
                    "Ball on a cube pyramid (6,545:1): 2 substeps x 12 iterations sank cubes 0.8 m; 4 x 24 with baumgarte 0.5 "
                    "kept every cube within 0.2 m of rest (p95 penetration 3.5-10 cm) at 2.6-2.8x frame time. Iterations alone "
                    "(24/48), warm start, velocity iterations and relaxation 1.3 did not fix it (relaxation made it worse); "
                    "baumgarte 0.8 launched bodies; 16 substeps cost 7.4-8.5x."
                ],
                f"{c['impact_cost'][0]}-{c['impact_cost'][1]}x frame time per doubling of both",
                "no light body below its rest height by more than half its size; peak speeds against a reference solver",
                "critical" if sink > 1.0 else "warn",
            )
        )
        return
    # A resting heavy-on-light stack: momentum on the serial immediate sweep, unless the scene is a pile that the
    # colored sweep serves better, where the fix is more sweeps.
    has_momentum = "nesterov_momentum" in (facts.get("solver_params") or ["nesterov_momentum"])
    pile = ((facts.get("model") or {}).get("free_bodies_world0") or 0) >= c["pile_free_bodies"]
    colored = kw.get("articulated_contact_response") == "propagation"
    new = {
        "solver.nesterov_momentum": True,
        "solver.solve_mode": "serial",
        "solver.articulated_contact_response": "immediate",
    }
    if kw.get("nesterov_momentum") or not has_momentum or pile or colored:
        # Momentum on (or not in this Newton): spend more sweeps, as iterations while they last.
        new = (
            {"solver.iterations": min(2 * its, c["iterations_max"])}
            if its < c["iterations_max"]
            else {"sim.substeps": min(2 * sub, c["substeps_max"])}
        )
    actions.append(
        Action(
            "mass_ratio_stack",
            new,
            f"Contact island mass ratio {ratio:.0f}:1 ({pair}) in a resting stack that fails. Nesterov momentum "
            "extrapolates the impulses between sweeps and carries the load down the stack in fewer iterations.",
            [
                "Six cubes, each 3x the one below (~270:1): 8 substeps x 16 iterations stands with momentum (10/12 one-minute "
                "repeats) and topples without (top cube ends 0.35 m vs 0.85 m). On an impact scene momentum cost 5-29x."
            ],
            "serial immediate sweep; momentum itself is cheap on small stacks",
            "the stack stands for the whole check",
            "warn",
        )
    )


def _rule_escalate(facts: dict, ep: dict, kw: dict, settings: dict, actions: list[Action], flags: list) -> None:
    """Last resort: the example's check failed (or the state went non-finite) and no rule named a cause."""
    if actions or any(lv == "critical" and m.startswith("Start state") for lv, m in flags):
        return  # a scene bug no setting fixes
    h = ep.get("health") or {}
    check = facts.get("check") or {}
    sank = ep.get("bodies_inside_static") or 0
    if check.get("ok") is not False and not h.get("nonfinite_frames") and not sank:
        return
    c = STABILITY_CALIBRATION
    sub = int(((settings.get("sim") or {}).get("substeps")) or facts.get("substeps") or 1)
    its = int(kw.get("iterations") or 12)
    if sub < c["substeps_max"]:
        new, what = {"sim.substeps": min(2 * sub, c["substeps_max"])}, "substeps"
    elif its < c["iterations_max"]:
        new, what = {"solver.iterations": min(2 * its, c["iterations_max"])}, "iterations"
    else:
        return
    why = ((check.get("note") or "").strip()[:200] if check.get("ok") is False else "") or (
        f"{sank} bodies sank into static geometry" if sank else "non-finite state"
    )
    actions.append(
        Action(
            "escalate_" + what,
            new,
            f"The run failed ({why}) and no rule found a cause: double the {what}. Substeps help both a "
            "time-step-limited scene and a convergence-limited one; the cost phase then tries to give them back as "
            "iterations.",
            [
                "Rope winch: 1 substep went non-finite at any iteration count, 4 x 6 passed. A 100 m/s projectile scene "
                "passes only at 20 substeps."
            ],
            "about 2x frame time",
            "the check passes",
            "critical",
        )
    )


def _rule_invalid_contacts(ep: dict, flags: list) -> None:
    """Contacts with non-finite geometry are left out of every contact metric, so those metrics look cleaner."""
    h = ep.get("health") or {}
    if not h.get("invalid_contact_frames"):
        return
    first = h.get("invalid_contact_first") or {}
    flags.append(
        (
            "critical",
            f"{h.get('invalid_contacts')} contacts with non-finite geometry in {h['invalid_contact_frames']} frames, "
            f"first at frame {first.get('frame')} (shapes {first.get('shape0')}-{first.get('shape1')}, body "
            f"{first.get('body')}): penetration and grasp metrics leave them out and are incomplete.",
        )
    )


def _rule_spellings(settings: dict, flags: list) -> None:
    s = (settings or {}).get("solver") or {}
    old = []
    if s.get("solve_mode") in ("matrix_free", "split"):
        old.append(f"solve_mode={s['solve_mode']!r}")
    if s.get("articulated_contact_response") == "propagation_colored":
        old.append("articulated_contact_response='propagation_colored'")
    if old:
        flags.append(
            (
                "info",
                f"{', '.join(old)}: Newton builds whose SolverSteel takes solve_mode='serial'|'colored' map "
                "these with a deprecation warning (matrix_free -> serial, propagation_colored -> colored + "
                "propagation; split runs serial, so results differ). Builds without them still need the "
                "old names: match the spelling to the installed signature.",
            )
        )


def _rule_episode_length(facts: dict, flags: list) -> None:
    done, planned = facts.get("frames_done"), facts.get("frames_planned")
    if done is not None and planned and done < planned:
        flags.append(
            ("critical", f"The episode stopped at frame {done} of {planned}: everything after it is unobserved.")
        )
    if planned and facts.get("frame_dt") and planned * facts["frame_dt"] < 5.0:
        flags.append(
            (
                "info",
                f"The episode is {planned * facts['frame_dt']:.1f} s. Instability often shows only at the "
                "load (an impact, a grasp, a pour): run the whole task at the settings you ship.",
            )
        )


def _round_up(x: float, step: int) -> int:
    return int(math.ceil(x / step) * step)


def _drop_unapplied(facts: dict, actions: list[Action], flags: list) -> None:
    """Remove fixes the example already asked for but its solver does not run with (it builds its own solver)."""
    requested = (facts.get("settings") or {}).get("solver") or {}
    kw = facts.get("solver_kwargs") or {}
    for a in list(actions):
        keys = [k.split(".", 1)[1] for k in a.settings if k.startswith("solver.")]
        ignored = [
            k
            for k in keys
            if k in requested
            and k in kw
            and requested[k] is not None
            and kw[k] != requested[k]
            and a.settings[f"solver.{k}"] == requested[k]
        ]
        if ignored:
            actions.remove(a)
            # The fix was never tested: the problem it answers is still there, so the run cannot pass.
            flags.append(
                (
                    "critical",
                    f"{a.id}: the settings ask for {', '.join(f'{k}={requested[k]!r}' for k in ignored)} but "
                    f"the solver runs with {', '.join(f'{k}={kw[k]!r}' for k in ignored)}: the example sets "
                    "these itself. Change them in the example's code, then re-run.",
                )
            )


def _out(actions: list[Action], flags: list) -> dict:
    return {
        "actions": [asdict(a) for a in actions],
        "flags": [list(f) for f in flags],
        "fixes_left": bool(actions),
        "critical": [f[1] for f in flags if f[0] == "critical"] + [a.reason for a in actions if a.level == "critical"],
    }
