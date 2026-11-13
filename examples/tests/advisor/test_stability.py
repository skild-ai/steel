# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Stability rules, the tune loop's acceptance logic and the trial preset writer, on synthetic facts."""

import math
import pathlib

import numpy as np
import tomllib

from common.advisor.episode import _island_ratio
from common.advisor.loop import adapt_spellings, apply_actions, check_contract, episode_frames, trial_toml
from common.advisor.stability import (
    STABILITY_CALIBRATION,
    cost_candidates,
    healthy,
    improved,
    no_worse,
    recommend_stability,
)


def _wm(**over):
    out = {}
    for fam in ("articulation", "free_body", "propagation"):
        out.update(
            {
                f"{fam}_high_water": 0,
                f"{fam}_raw_high_water": 0,
                f"{fam}_dropped_contact_rows_high_water": 0,
                f"{fam}_overflow_excess_high_water": 0,
                f"{fam}_overflow_world_steps": 0,
            }
        )
    out["contact_high_water"] = 0
    out.update(over)
    return out


def _facts(ep=None, kw=None, settings=None, **top):
    episode = {
        "row_watermarks": _wm(),
        "capacities": {"articulation": 32, "free_body": 512, "propagation": 544},
        "structure": {"propagation": False},
        "capacity_check": "ok",
        "health": {},
        "frames": 600,
        "island_mass_ratio_max": 1.0,
    }
    episode.update(ep or {})
    f = {
        "episode": episode,
        "solver_kwargs": {"iterations": 12, "baumgarte": 0.2, **(kw or {})},
        "settings": settings or {"sim": {"fps": 60, "substeps": 4}, "solver": {}},
        "frames_planned": 600,
        "frames_done": 600,
        "frame_dt": 1 / 60,
        "substeps": 4,
        "start_state": {"checked": True},
    }
    f.update(top)
    return f


def _ids(advice):
    return [a["id"] for a in advice["actions"]]


def _flags(advice, level):
    return [m for lv, m in advice["flags"] if lv == level]


def test_clean_episode_has_no_fix():
    advice = recommend_stability(_facts())
    assert advice["actions"] == [] and advice["critical"] == []


def test_overflowing_family_is_sized_from_twice_its_requested_mark():
    ep = {
        "row_watermarks": _wm(propagation_raw_high_water=1009, propagation_overflow_world_steps=2400),
        "structure": {"propagation": True},
        "capacity_check": "Steel dropped contacts",
    }
    advice = recommend_stability(_facts(ep))
    assert _ids(advice) == ["propagation_capacity"]
    target = advice["actions"][0]["settings"]["solver.propagation_constraint_max"]
    assert target == 2048 and target % STABILITY_CALIBRATION["cap_round"] == 0


def test_capacity_never_shrinks_below_current_plus_a_step():
    ep = {"row_watermarks": _wm(articulation_raw_high_water=10, articulation_overflow_world_steps=3)}
    advice = recommend_stability(_facts(ep))
    assert advice["actions"][0]["settings"] == {"solver.articulation_constraint_max": 64}


def test_propagation_family_is_ignored_under_immediate_response():
    ep = {"row_watermarks": _wm(propagation_raw_high_water=900, propagation_overflow_world_steps=5)}
    advice = recommend_stability(_facts(ep))
    assert "propagation_capacity" not in _ids(advice)


def test_capacity_fix_defers_accuracy_rules():
    ep = {
        "row_watermarks": _wm(free_body_raw_high_water=4700, free_body_overflow_world_steps=50),
        "island_mass_ratio_max": 6545.0,
        "health": {"object_support_penetration_over_half_thickness": 2.0},
    }
    advice = recommend_stability(_facts(ep))
    assert _ids(advice) == ["propagation_capacity"]  # 9,408 rows: past what immediate holds, so colored


def test_large_immediate_capacity_switches_to_colored_unless_loops():
    ep = {
        "row_watermarks": _wm(
            free_body_raw_high_water=20880,
            free_body_overflow_world_steps=2400,
            articulation_raw_high_water=120,
            articulation_overflow_world_steps=2384,
        )
    }
    advice = recommend_stability(_facts(ep))
    # Free-body contact rows move to propagation; the dense (articulation) raise is kept as it is.
    assert _ids(advice) == ["articulation_capacity", "propagation_capacity"]
    st = advice["actions"][1]["settings"]
    assert st["solver.solve_mode"] == "colored" and st["solver.free_body_constraint_max"] is None
    assert st["solver.propagation_constraint_max"] == 2 * 20880 + 256
    assert advice["actions"][0]["settings"] == {"solver.articulation_constraint_max": 256}
    ep["structure"] = {"propagation": False, "connect_rows_joints": 2}
    assert _ids(recommend_stability(_facts(ep))) == ["articulation_capacity", "free_body_capacity"]


def test_dense_row_overflow_alone_never_moves_to_propagation():
    ep = {"row_watermarks": _wm(articulation_raw_high_water=4700, articulation_overflow_world_steps=1)}
    advice = recommend_stability(_facts(ep))
    assert _ids(advice) == ["articulation_capacity"]
    assert advice["actions"][0]["settings"] == {"solver.articulation_constraint_max": 9408}


def test_kernel_build_failure_under_huge_immediate_caps_switches_to_colored():
    settings = {"sim": {"substeps": 4}, "solver": {"free_body_constraint_max": 41760}}
    advice = recommend_stability(
        {"build_error": "Exception: CUDA kernel build failed with error code 6", "settings": settings}
    )
    a = advice["actions"][0]
    assert a["id"] == "colored_for_rows" and a["settings"]["solver.free_body_constraint_max"] is None
    assert a["settings"]["solver.articulated_contact_response"] == "propagation"


def test_failed_capacity_check_without_overflow_is_critical():
    advice = recommend_stability(_facts({"capacity_check": "Steel contact torsion rejected its input"}))
    assert any("check_constraint_capacity" in m for m in _flags(advice, "critical"))


def test_full_contact_buffer_is_critical():
    advice = recommend_stability(_facts({"contact_count_max": 4096, "rigid_contact_max": 4096}))
    assert any("contact buffer filled" in m for m in _flags(advice, "critical"))


def test_missing_watermarks_warns():
    advice = recommend_stability(_facts({"row_watermarks": "RuntimeError: off"}))
    assert any("row_watermark=True" in m for m in _flags(advice, "warn"))


def test_construction_errors_map_to_fixes():
    cases = {
        "NotImplementedError: Loop-closing joints are not supported with articulated_contact_response='propagation'": {
            "solver.articulated_contact_response": "immediate",
            "solver.solve_mode": "serial",
        },
        "NotImplementedError: contact torsion with articulated_contact_response='propagation' requires "
        "contact_torsion_device=True": {"solver.contact_torsion_device": True},
        "ValueError: nesterov_momentum requires solve_mode='serial', articulated_contact_response='immediate'": {
            "solver.solve_mode": "serial",
            "solver.articulated_contact_response": "immediate",
        },
        "PresetError: x [steel.solver]: SolverSteel has no argument(s) ['enable_torsional_rolling_friction']; valid": {
            "solver.enable_torsional_rolling_friction": None
        },
    }
    for err, settings in cases.items():
        advice = recommend_stability({"build_error": err, "settings": {}})
        assert advice["actions"][0]["settings"] == settings, err


def test_unknown_construction_error_is_reported_not_guessed():
    advice = recommend_stability({"build_error": "RuntimeError: CUDA out of memory", "settings": {}})
    assert advice["actions"] == [] and any("CUDA out of memory" in m for m in _flags(advice, "critical"))


def test_start_state_written_to_body_arrays_is_critical():
    st = {
        "checked": True,
        "pose_mismatch_bodies": 0,
        "velocity_mismatch_bodies": 2,
        "velocity_mismatch_max": 20.0,
        "velocity_in_body_qd_only_bodies": 2,
        "examples": ["dumbbell"],
    }
    advice = recommend_stability(_facts(start_state=st))
    msg = " ".join(_flags(advice, "critical"))
    assert "joint_q / joint_qd" in msg and "joint_qd = 0" in msg


def test_limits_authored_but_off_are_enabled():
    advice = recommend_stability(_facts(model={"limited_dofs_world0": 6}))
    assert advice["actions"][0]["settings"] == {"solver.enable_joint_limits": True}
    advice = recommend_stability(_facts(model={"limited_dofs_world0": 6}, kw={"enable_joint_limits": True}))
    assert advice["actions"] == []


def test_grasp_slip_under_point_friction_gets_anchors_and_cold_start():
    holds = [{"robot": "finger", "object": "carton", "frames": 300, "slip_mm": 57.3, "slip_deg": 42.3}]
    advice = recommend_stability(
        _facts({"health": {"grasp_holds": holds}}, kw={"friction_anchor_beta": 0.0, "contact_warmstart": True})
    )
    assert "grasp_anchors" in _ids(advice)
    a = next(a for a in advice["actions"] if a["id"] == "grasp_anchors")
    assert a["settings"] == {"solver.friction_anchor_beta": 0.2, "solver.contact_warmstart": False}


def test_grasp_slip_with_anchors_doubles_iterations():
    holds = [{"robot": "finger", "object": "carton", "frames": 300, "slip_mm": 20.0, "slip_deg": 1.0}]
    advice = recommend_stability(_facts({"health": {"grasp_holds": holds}}, kw={"friction_anchor_beta": 0.2}))
    assert advice["actions"][0]["settings"] == {"solver.iterations": 24}


def test_small_slip_is_not_a_fix():
    holds = [{"robot": "finger", "object": "carton", "frames": 300, "slip_mm": 1.2, "slip_deg": 2.0}]
    assert (
        recommend_stability(_facts({"health": {"grasp_holds": holds}}, kw={"friction_anchor_beta": 0.0}))["actions"]
        == []
    )


def test_warm_start_bursts_turn_it_off():
    advice = recommend_stability(_facts({"health": {"bursts": 14}}, kw={"contact_warmstart": True}))
    assert advice["actions"][0]["settings"] == {"solver.contact_warmstart": False}


def test_unmatched_carried_contacts_turn_warm_start_off():
    h = {"warmstart_match_fast": {"robot_object": {"matched_frac": 0.13, "contacts": 900}}}
    advice = recommend_stability(_facts({"health": h}, kw={"contact_warmstart": True}))
    assert _ids(advice) == ["warmstart_unmatched"]


def test_heavy_impactor_doubles_substeps_and_iterations_with_baumgarte():
    ep = {
        "island_mass_ratio_max": 6545.0,
        "island_mass_ratio_bodies": ["ball", "cube"],
        "health": {"object_support_penetration_over_half_thickness": 2.0},
    }
    advice = recommend_stability(_facts(ep, settings={"sim": {"fps": 100, "substeps": 2}, "solver": {}}))
    a = advice["actions"][0]
    assert a["id"] == "mass_ratio_impact" and a["level"] == "critical"
    assert a["settings"] == {"sim.substeps": 4, "solver.iterations": 24, "solver.baumgarte": 0.5}


def test_baumgarte_is_not_raised_past_its_ceiling():
    ep = {"island_mass_ratio_max": 6545.0, "health": {"object_support_penetration_over_half_thickness": 0.8}}
    advice = recommend_stability(_facts(ep, kw={"baumgarte": 0.5}))
    assert "solver.baumgarte" not in advice["actions"][0]["settings"]


def test_resting_stack_that_fails_its_check_gets_momentum_then_sweeps():
    ep = {"island_mass_ratio_max": 243.0}
    check = {"ok": False, "note": "cube 1 fell"}
    advice = recommend_stability(_facts(ep, check=check))
    a = next(a for a in advice["actions"] if a["id"] == "mass_ratio_stack")
    assert a["settings"] == {
        "solver.nesterov_momentum": True,
        "solver.solve_mode": "serial",
        "solver.articulated_contact_response": "immediate",
    }
    advice = recommend_stability(_facts(ep, kw={"nesterov_momentum": True}, check=check))
    assert next(a for a in advice["actions"] if a["id"] == "mass_ratio_stack")["settings"] == {"solver.iterations": 24}
    advice = recommend_stability(_facts(ep, kw={"nesterov_momentum": True, "iterations": 48}, check=check))
    assert next(a for a in advice["actions"] if a["id"] == "mass_ratio_stack")["settings"] == {"sim.substeps": 8}
    # A Newton without momentum: more sweeps straight away.
    advice = recommend_stability(_facts(ep, check=check, solver_params=["iterations", "baumgarte"]))
    assert next(a for a in advice["actions"] if a["id"] == "mass_ratio_stack")["settings"] == {"solver.iterations": 24}


def test_mass_ratio_without_symptoms_is_info_only():
    advice = recommend_stability(_facts({"island_mass_ratio_max": 6545.0}))
    assert advice["actions"] == [] and _flags(advice, "info")


def test_deprecated_spellings_are_reported():
    s = {
        "sim": {"fps": 60, "substeps": 4},
        "solver": {"solve_mode": "matrix_free", "articulated_contact_response": "propagation_colored"},
    }
    advice = recommend_stability(_facts(settings=s))
    assert any("matrix_free" in m and "propagation_colored" in m for m in _flags(advice, "info"))


def test_short_or_truncated_episode_is_flagged():
    advice = recommend_stability(_facts(frames_done=142, frames_planned=600))
    assert any("stopped at frame 142" in m for m in _flags(advice, "critical"))
    advice = recommend_stability(_facts(frames_planned=2, frames_done=2))
    assert any("Instability often shows" in m for m in _flags(advice, "info"))


def test_healthy_requires_no_fix_no_critical_and_a_passing_check():
    full = {"capacity_check": "ok", "frames_planned": 600, "frames_done": 600}
    ok = {"status": "ok", "advice": {"actions": [], "critical": []}, "summary": {"check_ok": True, **full}}
    assert healthy(ok)
    assert not healthy({**ok, "summary": {**full, "check_ok": False}})
    assert not healthy({**ok, "summary": {"check_ok": True}})  # unknown capacity and length are not a pass
    assert not healthy({**ok, "advice": {"actions": [{"id": "x"}], "critical": []}})
    assert not healthy({**ok, "status": "step_error"})


def test_no_worse_keeps_only_faster_and_comparably_healthy_trials():
    base = {
        "frame_ms": 10.0,
        "penetration_p95_m": 0.001,
        "speed_max": 2.0,
        "sink_over_half_extent": 0.05,
        "bursts": 0,
        "joint_limit_excess": 0.0,
        "grasp_slip_max_mm": None,
    }
    assert no_worse(base, {**base, "frame_ms": 5.0})
    assert not no_worse(base, {**base, "frame_ms": 9.9})
    assert not no_worse(base, {**base, "frame_ms": 5.0, "penetration_p95_m": 0.0085})
    assert not no_worse(base, {**base, "frame_ms": 5.0, "bursts": 3})
    noisy = {**base, "bursts": 147}
    assert no_worse(noisy, {**noisy, "frame_ms": 5.0, "bursts": 162})  # run-to-run noise of an impact collapse
    assert not no_worse(noisy, {**noisy, "frame_ms": 5.0, "bursts": 240})
    # Faster than the current best, judged against the first healthy settings' health.
    assert not no_worse(base, {**base, "frame_ms": 5.0}, best={**base, "frame_ms": 4.0})


def test_cost_candidates_order_and_conditions():
    f = _facts(kw={"enable_joint_limits": True, "friction_anchor_beta": None})
    ids = [c["id"] for c in cost_candidates(f, f["settings"])]
    assert ids == ["fewer_substeps", "fewer_substeps_same_sweeps", "limit_gap", "point_friction", "fewer_iterations"]
    held = _facts(
        {"health": {"grasp_holds": [{"frames": 300, "slip_mm": 1.0, "slip_deg": 1.0}]}},
        settings={"sim": {"fps": 60, "substeps": 1}, "solver": {"iterations": 4}},
    )
    assert [c["id"] for c in cost_candidates(held, held["settings"])] == []


def test_apply_actions_merges_and_removes():
    s = {"sim": {"fps": 60, "substeps": 4}, "solver": {"enable_torsional_rolling_friction": True}}
    out = apply_actions(
        s,
        [{"settings": {"sim.substeps": 8, "solver.iterations": 24, "solver.enable_torsional_rolling_friction": None}}],
    )
    assert out == {"sim": {"fps": 60, "substeps": 8}, "solver": {"iterations": 24}}
    assert s["sim"]["substeps"] == 4  # the input is not modified


def test_trial_toml_round_trips_settings():
    data = {"example": "cradle", "check": {"args": ["--num-frames", 600]}, "gallery": {"short": 'say "hi"'}}
    settings = {
        "sim": {"fps": 60, "substeps": 8},
        "model": {"default_shape_cfg": {"mu": 0.0}},
        "solver": {"contact_friction_gap_max": -math.inf, "solve_mode": "serial", "nesterov_momentum": True},
    }
    parsed = tomllib.loads(trial_toml(data, settings, row_watermark=True))
    assert parsed["steel"]["solver"]["row_watermark"] is True
    assert parsed["steel"]["solver"]["contact_friction_gap_max"] == -math.inf
    assert parsed["steel"]["model"] == {"default_shape_cfg": {"mu": 0.0}}
    assert parsed["check"]["args"] == ["--num-frames", "600"] and parsed["default"] == "steel"


def test_episode_frames_follow_the_check_up_to_twenty_seconds():
    assert episode_frames(60, []) == 600
    assert episode_frames(60, ["--num-frames", "1200"]) == 1200
    assert episode_frames(60, ["--num-frames", "5000"]) == 1200
    assert episode_frames(100, ["--num-frames", "120"]) == 1000
    assert episode_frames(60, [], seconds=2) == 120


def test_island_ratio_joins_contact_chains():
    mass = np.array([1.0, 3.0, 9.0, 27.0, 5.0, 5.0])
    b0, b1 = np.array([0, 1, 2, 4]), np.array([1, 2, 3, 5])
    ratio, pair = _island_ratio(b0, b1, mass)
    assert ratio == 27.0 and pair == (3, 0)
    assert _island_ratio(np.array([4]), np.array([5]), mass)[0] == 1.0


def test_open_loop_doubles_iterations_then_substeps():
    advice = recommend_stability(_facts({"loop_gap_max_m": 0.0017, "loop_gap_joint": "closure"}))
    assert advice["actions"][0]["id"] == "loop_gap" and advice["actions"][0]["settings"] == {"solver.iterations": 24}
    advice = recommend_stability(_facts({"loop_gap_max_m": 0.0017}, kw={"iterations": 48}))
    assert advice["actions"][0]["settings"] == {"sim.substeps": 8}
    assert recommend_stability(_facts({"loop_gap_max_m": 0.0005}))["actions"] == []


def test_body_points_are_rotated_and_translated():
    from common.advisor.episode import _point

    s = np.sqrt(0.5)
    bq = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, s, s]])  # 90 deg about z
    out = _point(bq, np.array([0]), np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(out, [[1.0, 3.0, 3.0]])


def test_spellings_follow_the_installed_signature():
    s = {"sim": {"substeps": 4}, "solver": {"solve_mode": "colored", "articulated_contact_response": "propagation"}}
    fork = {"solve_mode": ["serial", "colored"], "articulated_contact_response": ["immediate", "propagation"]}
    upstream = {
        "solve_mode": ["matrix_free", "split"],
        "articulated_contact_response": ["immediate", "propagation", "propagation_colored"],
    }
    assert adapt_spellings(s, fork) == s
    assert adapt_spellings(s, upstream)["solver"] == {
        "solve_mode": "matrix_free",
        "articulated_contact_response": "propagation_colored",
    }
    serial = {"solver": {"solve_mode": "serial", "articulated_contact_response": "immediate"}}
    assert adapt_spellings(serial, upstream)["solver"]["solve_mode"] == "matrix_free"
    assert s["solver"]["solve_mode"] == "colored"  # the input is not modified


def test_failed_check_without_a_cause_escalates_substeps_then_iterations():
    check = {"ok": False, "note": "AssertionError: increase SUBSTEPS"}
    advice = recommend_stability(_facts(check=check))
    assert advice["actions"][0]["id"] == "escalate_substeps" and advice["actions"][0]["settings"] == {"sim.substeps": 8}
    s = {"sim": {"fps": 60, "substeps": 32}, "solver": {}}
    assert recommend_stability(_facts(check=check, settings=s))["actions"][0]["settings"] == {"solver.iterations": 24}
    st = {
        "checked": True,
        "pose_mismatch_bodies": 3,
        "pose_mismatch_max_m": 0.5,
        "pose_mismatch_max_rad": 0.0,
        "velocity_mismatch_bodies": 0,
        "examples": [],
    }
    assert recommend_stability(_facts(check=check, start_state=st))["actions"] == []


def test_rounding_sized_start_mismatch_is_info():
    st = {
        "checked": True,
        "pose_mismatch_bodies": 7,
        "pose_mismatch_max_m": 1e-5,
        "pose_mismatch_max_rad": 0.002,
        "velocity_mismatch_bodies": 0,
        "velocity_mismatch_max": 0.0,
        "examples": ["wrist"],
    }
    advice = recommend_stability(_facts(start_state=st))
    assert not _flags(advice, "critical") and any("rounding" in m for m in _flags(advice, "info"))


def test_setting_the_example_reads_gets_steels_default():
    err = "PresetError: /x/presets.toml [steel.solver] has no 'iterations'; add it to presets.toml"
    advice = recommend_stability({"build_error": err, "settings": {}, "solver_defaults": {"iterations": 12}})
    assert advice["actions"][0]["settings"] == {"solver.iterations": 12}


def test_defaults_start_keeps_scene_timing_and_model_but_no_tuning():
    from common.advisor.loop import start_settings

    data = {
        "steel": {
            "sim": {"fps": 100, "substeps": 2},
            "model": {"rigid_gap": 0.02},
            "solver": {"iterations": 24, "baumgarte": 0.5},
        }
    }
    assert start_settings(data, "defaults") == {
        "sim": {"fps": 100, "substeps": 4},
        "solver": {},
        "model": {"rigid_gap": 0.02},
    }
    assert start_settings(data, "preset")["solver"] == {"iterations": 24, "baumgarte": 0.5}


def test_a_fix_the_example_overrides_is_reported_not_repeated():
    holds = [{"robot": "finger", "object": "box", "frames": 300, "slip_mm": 30.0, "slip_deg": 1.0, "object_free": True}]
    s = {"sim": {"substeps": 4}, "solver": {"friction_anchor_beta": 0.2}}
    advice = recommend_stability(
        _facts({"health": {"grasp_holds": holds}}, kw={"friction_anchor_beta": 0.0}, settings=s)
    )
    assert "grasp_anchors" not in _ids(advice)
    assert any("the example sets these itself" in m for m in _flags(advice, "critical"))
    from common.advisor.stability import summarize

    f = _facts({"health": {"grasp_holds": holds}}, kw={"friction_anchor_beta": 0.0}, settings=s)
    res = {"status": "ok", "advice": recommend_stability(f), "summary": summarize(f) | {"check_ok": True}}
    assert not healthy(res)  # an ignored fix was never tested


def test_motion_of_a_jointed_object_is_not_grasp_slip():
    holds = [
        {"robot": "finger", "object": "knob", "frames": 600, "slip_mm": 6.0, "slip_deg": 27.0, "object_free": False}
    ]
    advice = recommend_stability(_facts({"health": {"grasp_holds": holds}}, kw={"friction_anchor_beta": 0.0}))
    assert advice["actions"] == [] and any("on a joint" in m for m in _flags(advice, "info"))


def test_sinking_through_static_geometry_is_critical_and_escalates():
    advice = recommend_stability(
        _facts({"bodies_inside_static": 1, "inside_static_depth_max_m": 0.007, "inside_static_first_frame": 4})
    )
    assert any("static box" in m for m in _flags(advice, "critical"))
    assert advice["actions"][0]["id"] == "escalate_substeps"


def test_improved_reads_any_of_its_metrics():
    assert improved({"slip": 20.0}, {"slip": 10.0}, "slip")
    assert not improved({"slip": 20.0}, {"slip": 19.0}, "slip")
    assert improved({"mm": 6.0, "deg": 27.0}, {"mm": 6.0, "deg": 5.0}, "mm|deg")
    assert not improved({"mm": None}, {"mm": 3.0}, "mm")  # nothing to compare against
    assert not improved({"mm": 20.0}, {"mm": None}, "mm")  # a metric that disappeared is not an improvement


def test_rejected_warn_fixes_do_not_block_health_but_critical_ones_do():
    full = {"check_ok": True, "capacity_check": "ok", "frames_planned": 600, "frames_done": 600}
    res = {
        "status": "ok",
        "summary": full,
        "advice": {"actions": [{"id": "grasp_iterations", "level": "warn", "reason": "r"}], "critical": []},
    }
    assert not healthy(res) and healthy(res, {"grasp_iterations"})
    crit = {
        "status": "ok",
        "summary": full,
        "advice": {
            "actions": [{"id": "mass_ratio_impact", "level": "critical", "reason": "sank"}],
            "critical": ["sank"],
        },
    }
    assert not healthy(crit, {"mass_ratio_impact"})


def test_piles_try_colored_propagation_first():
    f = _facts(model={"free_bodies_world0": 54, "torsion": {"shapes_with_radius_world0": 0}})
    cands = cost_candidates(f, f["settings"])
    assert cands[0]["id"] == "colored_propagation"
    assert cands[0]["settings"] == {
        "solver.solve_mode": "colored",
        "solver.articulated_contact_response": "propagation",
    }
    loops = _facts({"structure": {"connect_rows_joints": 1}}, model={"free_bodies_world0": 54})
    assert "colored_propagation" not in [c["id"] for c in cost_candidates(loops, loops["settings"])]


def test_wedged_thin_pieces_with_a_passing_check_are_not_a_stack_failure():
    ep = {"island_mass_ratio_max": 246.6, "health": {"object_support_penetration_over_half_thickness": 2.7}}
    assert recommend_stability(_facts(ep, check={"ok": True}))["actions"] == []


def test_a_failing_pile_gets_sweeps_not_a_response_switch():
    ep = {"island_mass_ratio_max": 246.6}
    advice = recommend_stability(
        _facts(
            ep,
            kw={"articulated_contact_response": "propagation"},
            check={"ok": False},
            model={"free_bodies_world0": 168},
        )
    )
    assert advice["actions"][0]["settings"] == {"solver.iterations": 24}


def test_propagation_capacity_under_immediate_is_dropped():
    err = "ValueError: propagation_constraint_max requires articulated_contact_response='propagation'"
    advice = recommend_stability({"build_error": err, "settings": {}})
    assert advice["actions"][0]["settings"] == {"solver.propagation_constraint_max": None}


def test_a_frozen_state_is_critical_and_never_kept_as_cheaper():
    moving = {"unchanged_frames": 599, "frames": 600, "initial_speed_max": 2.0}
    assert any("repeated exactly" in m for m in _flags(recommend_stability(_facts(moving)), "critical"))
    still = {"unchanged_frames": 599, "frames": 600, "initial_speed_max": 0.0}
    advice = recommend_stability(_facts(still))
    assert not any("repeated exactly" in m for m in _flags(advice, "critical"))
    assert any("repeated exactly" in m for m in _flags(advice, "warn"))
    settled = {"unchanged_frames": 0, "frames": 600, "initial_speed_max": 2.0}  # settled poses differ from the start
    assert not any("repeated exactly" in m for lv, m in recommend_stability(_facts(settled))["flags"])
    assert not any(
        "repeated exactly" in m
        for m in _flags(recommend_stability(_facts(moving, kw={"enable_sleeping": True})), "critical")
    )
    base = {"frame_ms": 10.0, "frozen_frac": 0.0}
    assert not no_worse(base, {"frame_ms": 1.0, "frozen_frac": 0.99})


def _healthy_result(facts):
    from common.advisor.stability import summarize

    return {"status": "ok", "advice": recommend_stability(facts), "summary": summarize(facts) | {"check_ok": True}}


def test_validity_gates_block_health_with_no_fix_left():
    s = {"sim": {"fps": 60, "substeps": 32}, "solver": {}}
    nan = _facts({"health": {"nonfinite_frames": 1}}, kw={"iterations": 48}, settings=s)
    assert recommend_stability(nan)["actions"] == [] and not healthy(_healthy_result(nan))
    assert not healthy(_healthy_result(_facts(frames_done=142)))
    nowm = _facts({"row_watermarks": "RuntimeError: off", "capacity_check": "Steel dropped contacts"})
    assert not healthy(_healthy_result(nowm))
    unapplied = _facts(
        model={"limited_dofs_world0": 6},
        kw={"enable_joint_limits": False},
        settings={"sim": {"substeps": 4}, "solver": {"enable_joint_limits": True}},
    )
    advice = recommend_stability(unapplied)
    assert advice["actions"] == [] and any("enable_joint_limits" in m for m in _flags(advice, "critical"))
    assert not healthy(_healthy_result(unapplied))
    assert healthy(_healthy_result(_facts()))


def test_check_contract_keeps_the_presets_own_task_arguments_and_drops_settings_overrides():
    data = {
        "check": {"args": ["--num-frames", "120", "--set", "sim.substeps=1"]},
        "steel": {"check": {"args": ["--world-count", "8", "--num-frames", "420", "--set=solver.iterations=1"]}},
    }
    assert check_contract(data) == ["--world-count", "8", "--num-frames", "420"]
    assert check_contract({"check": {"args": ["--num-frames", "120", "--set", "sim.substeps=1"]}}) == [
        "--num-frames",
        "120",
    ]
    parsed = tomllib.loads(trial_toml(data, {"sim": {"substeps": 12}, "solver": {}}, row_watermark=False))
    assert parsed["check"]["args"] == ["--world-count", "8", "--num-frames", "420"]
    assert "check" not in parsed["steel"]


def test_tune_loop_spends_unused_fix_trials_on_cost_and_stops_at_the_total(tmp_path, monkeypatch):
    import json as _json
    from argparse import Namespace

    from common.advisor import loop

    monkeypatch.setattr(loop.cli, "find", lambda name: tmp_path / name)
    monkeypatch.setattr(
        loop.presets, "read", lambda path: {"steel": {"sim": {"fps": 60, "substeps": 8}, "solver": {"iterations": 48}}}
    )
    facts = _facts(settings={"sim": {"fps": 60, "substeps": 8}, "solver": {}}, kw={"iterations": 48})
    summary = {"frame_ms": 10.0, "capacity_check": "ok", "frames_planned": 600, "frames_done": 600, "check_ok": True}

    def fake_spawn(cmd, log, timeout):
        out = pathlib.Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        settings = _json.loads(cmd[cmd.index("--settings") + 1])
        sub = settings["sim"]["substeps"]
        its = settings.get("solver", {}).get("iterations", 48)
        res = {
            "status": "ok",
            "settings": settings,
            "facts": {**facts, "settings": settings},
            "advice": {"actions": [], "flags": [], "critical": []},
            "summary": {**summary, "frame_ms": sub * (1 + its)},
            "attempt": cmd[cmd.index("--attempt") + 1],
        }  # every cheaper cell is healthy and faster
        (out / "trial.json").write_text(_json.dumps(res))
        return 0

    monkeypatch.setattr(loop, "_spawn", fake_spawn)
    args = Namespace(
        example="x",
        start="preset",
        max_trials=6,
        cost_trials=5,
        seconds=None,
        no_check=True,
        timeout=10,
        out=str(tmp_path / "tune"),
    )
    loop.run_tune(args, [])
    t = _json.loads((tmp_path / "tune" / "tune.json").read_text())
    phases = [x["phase"] for x in t["trials"]]
    assert phases[0] == "fix" and phases.count("fix") == 1
    # max_trials + cost_trials in all: the five fix trials left unused go to the cost phase.
    cost = len(phases) - phases.count("fix")
    assert cost > args.cost_trials and len(phases) <= args.max_trials + args.cost_trials
    assert t["first_healthy_trial"] == 0 and t["final_settings"]["sim"]["substeps"] == 1
