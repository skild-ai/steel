// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

/* Stabilization, bounces, drives and limits: one measured Steel run per sub-concept, with the rule it shows. */
SteelViz.register("stab", (root, lib) => {
  const { css } = lib;
  const f = lib.frame(root, `
    <div class="vizbar">
      <span class="seg" data-role="tab"><button type="button" class="on" data-v="pos">position correction</button><button type="button" data-v="spec">speculative contacts</button><button type="button" data-v="rest">restitution</button><button type="button" data-v="drive">drives</button><button type="button" data-v="lim">limits</button></span>
    </div>
    <p class="formula" aria-live="polite"></p>
    <svg class="vizsvg" data-role="a" role="img"></svg><div class="legend" data-role="la"></div>
    <svg class="vizsvg" data-role="b" role="img" hidden></svg><div class="legend" data-role="lb" hidden></div>
    <p class="readout"></p>`);
  const A = f.querySelector("[data-role=a]"), Bc = f.querySelector("[data-role=b]"), la = f.querySelector("[data-role=la]"), lb = f.querySelector("[data-role=lb]");
  const formula = f.querySelector(".formula"), out = f.querySelector(".readout");
  let tab = "pos";
  const TABS = {
    pos: {
      f: "penetrating contact: target speed ≥ β·|φ|/h   ·   velocity_iterations re-solve velocities without β",
      a: () => ({ id: "velocity-iterations", key: "z_mm", ylabel: "box height above rest [mm], spawned 2 cm deep", cols: ["--series-1", "--series-2", "--series-3"] }),
      t: "The bias that removes penetration becomes real velocity: at β = 0.8 a box spawned 2 cm into the ground is thrown up. Velocity iterations keep the correction but remove the throw. steel:rigid_body_max_depenetration_velocity caps the correction speed per free body.",
    },
    spec: {
      f: "open contact within the gap: closing speed ≤ s·φ/h   (s = contact_speculative_scale)   ·   contacts farther than contact_gap_gate get no rows",
      a: () => ({ id: "speculative-contacts", key: "x", ylabel: "ball position [m], wall face at 0.595 m, 40 m/s ball", cols: ["--series-4", "--series-2", "--series-1", "--series-3"], hl: [{ y: 0.595, label: "wall" }], ymax: 1.2 }),
      t: "A ball at 40 m/s moves 17 cm per step. With a 5 mm shape gap the pipeline never reports it near the wall and it tunnels; with a 20 cm gap a speculative row lets it close exactly the gap and stop at the wall. Scale 0 stops it at the gap; a 1 cm gap gate drops the row and it tunnels again.",
    },
    rest: {
      f: "impact faster than restitution_velocity_threshold: separating speed ≥ e·closing speed at the start of the step",
      a: () => ({ id: "restitution-threshold", key: "height", ylabel: "ball height [m], e = 0.8, dropped from 0.5 m", cols: ["--series-2", "--series-1", "--series-3", "--series-4"], ymin: 0 }),
      t: "Only impacts faster than the threshold bounce, so the threshold decides when a bouncing ball comes to rest, and a resting object pressed by gravity does not bounce. enable_restitution=False turns every bounce off.",
    },
    drive: {
      f: "augmented (default): (H + h·kd + h²·ke) Δv = h(τ + clamp(u₀) − C)   ·   rows: one row per DOF, impulse clamped to effort limit × h",
      a: () => ({ id: "drive-saturation", key: "angle_deg", ylabel: "arm angle [°], 12 N·m effort limit, released 60° below target", cols: ["--series-2", "--series-1"] }),
      t: "The default drive folds stiffness and damping into the mass matrix, so stiff gains stay stable, but only its explicit part is clamped: under load the total drive torque can exceed the limit. Drive rows clamp the whole impulse. In this run both reach the target, and the row drive overshoots less (7.5° against 9.2°).",
    },
    lim: {
      f: "joint limits: a unilateral row per finite bound (enable_joint_limits)   ·   velocity limits: a row pair per limited DOF (enable_joint_velocity_limits)",
      a: () => ({ id: "joint-limits", key: "angle_deg", ylabel: "pendulum angle [°], ±30° limits", cols: ["--series-1", "--series-2"], hl: [{ y: 30, label: "+30°" }, { y: -30, label: "−30°" }] }),
      b: () => ({ id: "velocity-limits", key: "omega", ylabel: "flywheel speed [rad/s], 6 rad/s limit", cols: ["--series-1", "--series-2"], hl: [{ y: 6, label: "limit" }], ymax: 60, ymin: 0 }),
      t: "Both are off by default. With them on, a limited joint stops at its bound without overshoot, and a flywheel under constant torque levels off at its velocity limit.",
    },
  };
  function plot(svg, legend, spec) {
    const d = lib.data[spec.id];
    if (!d) return;
    svg.setAttribute("aria-label", spec.ylabel);
    lib.lineChart(svg, {
      h: 200, ylabel: spec.ylabel + " · measured with Steel", legend,
      series: lib.series(lib.data, spec.id, spec.key, spec.cols.map(css)),
      hlines: (spec.hl || []).map((x) => Object.assign({ color: css("--muted") }, x)), ymin: spec.ymin, ymax: spec.ymax,
    });
  }
  function draw() {
    const t = TABS[tab];
    formula.textContent = t.f;
    plot(A, la, t.a());
    Bc.toggleAttribute("hidden", !t.b); lb.hidden = !t.b;
    if (t.b) plot(Bc, lb, t.b());
    out.textContent = t.t;
  }
  lib.tabs(f.querySelector("[data-role=tab]"), (v) => { tab = v; draw(); });
  lib.onTheme(draw);
  draw();
});
