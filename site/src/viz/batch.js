// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

/* Mixed batch: six different worlds routed to dynamics groups and contact-solve kernels, per step. Schematic. */
SteelViz.register("batch", (root, lib) => {
  const { css, el, text } = lib;
  const f = lib.frame(root, `
    <div class="vizbar">
      <span class="seg" data-role="step"><button type="button" class="on" data-v="0">simulation step t</button><button type="button" data-v="1">step t + 1</button></span>
      <span class="stepbtns"><button type="button" class="btn" data-act="prev" aria-label="Earlier">◀</button><button type="button" class="btn" data-act="play">Play</button><button type="button" class="btn" data-act="next" aria-label="Later">▶</button></span>
    </div>
    <svg class="vizsvg" viewBox="0 0 640 360" role="img" aria-label="Six worlds of different robots, grouped by joint count for the dynamics and routed per step to a small-world kernel or the general sweep, with a timeline of the general sweep\'s world warps in one block."></svg>
    <p class="readout" aria-live="polite"></p>`);
  const svg = f.querySelector("svg"), out = f.querySelector(".readout");
  // rows per world at step t and t+1 (schematic counts)
  const W = [
    { id: "A", what: "7-DOF arm, reaching", dof: 7, rows: [2, 30], objs: [0, 1] },
    { id: "B", what: "7-DOF arm + mug", dof: 7, rows: [14, 14], objs: [1, 1] },
    { id: "C", what: "23-DOF hand + cube", dof: 23, rows: [40, 42], objs: [1, 1] },
    { id: "D", what: "18-DOF quadruped", dof: 18, rows: [24, 26], objs: [0, 0] },
    { id: "E", what: "7-DOF arm + 20 parts", dof: 7, rows: [96, 90], objs: [20, 20] },
    { id: "F", what: "7-DOF arm + mug", dof: 7, rows: [14, 14], objs: [1, 1] },
  ];
  const small = (w, s) => w.dof <= 16 && w.objs[s] <= 1 && w.rows[s] <= 20;
  let simStep = 0, t = 0;
  const TIMES = 6;
  function render(ti) {
    t = ti;
    svg.innerHTML = "";
    const fg = css("--fg"), mu = css("--muted"), hot = css("--hot"), s1 = css("--series-1"), s3 = css("--series-3");
    text(svg, 12, 16, "WORLDS", { "letter-spacing": "0.07em" });
    text(svg, 240, 16, "DYNAMICS · BY DOF COUNT", { "letter-spacing": "0.07em" });
    text(svg, 440, 16, "CONTACT SOLVE · THIS STEP", { "letter-spacing": "0.07em" });
    const groups = [{ k: 7, y: 40 }, { k: 18, y: 100 }, { k: 23, y: 160 }];
    groups.forEach((g) => {
      svg.appendChild(el("rect", { x: 240, y: g.y, width: 150, height: 44, rx: 4, fill: css("--panel"), stroke: s1 }));
      text(svg, 250, g.y + 18, `${g.k}-DOF group`, { fill: fg });
      text(svg, 250, g.y + 34, W.filter((w) => w.dof === g.k).map((w) => w.id).join(" "), {});
    });
    svg.appendChild(el("rect", { x: 240, y: 220, width: 150, height: 34, rx: 4, fill: css("--panel"), stroke: css("--static") }));
    text(svg, 250, 241, "free objects: 6×6", {});
    const owners = [{ y: 40, h: 70, label: "small-world kernels", sub: "≤16-DOF arm + 1 object", col: s3 }, { y: 130, h: 90, label: "general sweep", sub: "one warp per world", col: hot }];
    owners.forEach((o) => {
      svg.appendChild(el("rect", { x: 440, y: o.y, width: 190, height: o.h, rx: 4, fill: css("--bg"), stroke: o.col, "stroke-width": 1.5 }));
      text(svg, 450, o.y + 18, o.label, { fill: fg });
      text(svg, 450, o.y + 34, o.sub, {});
    });
    W.forEach((w, i) => {
      const y = 34 + i * 34, sm = small(w, simStep);
      svg.appendChild(el("rect", { x: 10, y, width: 196, height: 28, rx: 3, fill: css("--panel"), stroke: css("--rule") }));
      text(svg, 18, y + 18, `${w.id}  ${w.what}`, { fill: fg, "font-size": 10.5 });
      const g = groups.find((gg) => gg.k === w.dof);
      svg.appendChild(el("path", { d: `M206 ${y + 14} C 222 ${y + 14}, 222 ${g.y + 22}, 240 ${g.y + 22}`, stroke: css("--rule"), fill: "none" }));
      const o = sm ? owners[0] : owners[1];
      const ty = o.y + 44 + (sm ? W.filter((x, j) => j < i && small(x, simStep)).length : W.filter((x, j) => j < i && !small(x, simStep)).length) * 11;
      svg.appendChild(el("path", { d: `M390 ${g.y + 22} C 415 ${g.y + 22}, 415 ${ty}, 440 ${ty}`, stroke: sm ? s3 : hot, fill: "none", opacity: 0.7 }));
      text(svg, 452 + (sm ? W.filter((x, j) => j < i && small(x, simStep)).length * 22 : W.filter((x, j) => j < i && !small(x, simStep)).length * 22), o.y + 58, w.id, { fill: sm ? s3 : hot, "font-weight": 600 });
    });
    // general sweep: the routed worlds' warps inside one thread block
    const gen = W.filter((w) => !small(w, simStep));
    text(svg, 12, 280, "GENERAL SWEEP · ONE WARP PER WORLD IN A SHARED BLOCK · LENGTH ∝ ROWS", { "letter-spacing": "0.07em" });
    const tmax = Math.max(...gen.map((w) => w.rows[simStep])), X = (v) => 90 + (530 * v) / tmax, now = (tmax * ti) / (TIMES - 1);
    const placed = gen.map((w, s) => ({ w, s, t0: 0, t1: w.rows[simStep] }));
    const bh = 14, by = 288;
    svg.appendChild(el("rect", { x: 86, y: by - 4, width: 538, height: placed.length * (bh + 4) + 4, rx: 3, fill: "none", stroke: css("--static"), "stroke-dasharray": "4 3" }));
    text(svg, 12, by + 10, "block", {});
    placed.forEach((p) => {
      const done = now >= p.t1, active = now < p.t1;
      svg.appendChild(el("rect", { x: X(0), y: by + p.s * (bh + 4), width: Math.max(2, X(p.t1) - X(0) - 2), height: bh, rx: 2, fill: active ? hot : css("--panel"), stroke: active ? hot : s1 }));
      text(svg, X(0) + 4, by + p.s * (bh + 4) + 11, `${p.w.id} · ${p.w.rows[simStep]} rows${done ? " · done" : ""}`, { fill: active ? css("--bg") : fg, "font-size": 10 });
    });
    svg.appendChild(el("line", { x1: X(now), x2: X(now), y1: by - 6, y2: by + placed.length * (bh + 4) + 2, stroke: fg, "stroke-dasharray": "3 3" }));
    const running = placed.filter((p) => now < p.t1).map((p) => p.w.id), finished = placed.filter((p) => now >= p.t1).map((p) => p.w.id);
    out.textContent = `Step ${simStep ? "t + 1" : "t"}: ${W.filter((w) => small(w, simStep)).map((w) => w.id).join(", ")} go to small-world kernels, the rest to the general sweep. `
      + (finished.length ? `Finished their own rows: ${finished.join(", ")}. ` : "") + (running.length ? `Still sweeping: ${running.join(", ")}. ` : "") + (running.length ? "" : "The block is done, and its resources go to new work. ")
      + (simStep ? "A pressed into a tray and now has 30 rows, so it moved to the general sweep; worlds are re-routed every step." : "");
  }
  const ctl = lib.stepper(f, TIMES, render);
  lib.tabs(f.querySelector("[data-role=step]"), (v) => { simStep = +v; ctl.redraw(); });
  lib.onTheme(() => ctl.redraw());
});
