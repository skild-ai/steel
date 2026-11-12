// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

/* Torsion, spin and rolling friction: a schematic of the contact and the spin and roll Steel measured. */
SteelViz.register("torsion", (root, lib) => {
  const { css, el, text } = lib;
  const f = lib.frame(root, `
    <div class="vizbar">
      <span class="seg" data-role="tab"><button type="button" class="on" data-v="art">articulated: patch radius</button><button type="button" data-v="spin">free body: spin</button><button type="button" data-v="roll">free body: rolling</button></span>
    </div>
    <div class="vizrow">
      <svg class="vizsvg vizside" viewBox="0 0 200 200" role="img" aria-label="Schematic of a ball touching a plane, with the effective patch radius and the friction row the tab is about."></svg>
      <div class="vizmain"><svg class="vizsvg" data-role="chart" role="img" aria-label="Measured spin or rolling speed over time for the tab's variants."></svg><div class="legend" data-role="legend"></div></div>
    </div>
    <p class="readout" aria-live="polite"></p>`);
  const sch = f.querySelector(".vizside"), chart = f.querySelector("[data-role=chart]"), out = f.querySelector(".readout");
  let tab = "art";
  const TXT = {
    art: "Articulated bodies: a spin row per contact group bounded by r·(μN − sliding), with r the shape's steel:min_torsional_patch_radius. A ball on a slider and hinge spinning at 20 rad/s keeps spinning with r = 0 and stops quickly with a 1 or 5 cm radius.",
    spin: "Free bodies (integration branch): steel:condim 4 adds a spin row bounded by μ_torsional·N. A ball touches at one point, so sliding friction alone never stops it spinning.",
    roll: "Free bodies: steel:condim 6 adds a rolling pair bounded by μ_rolling·N. Without it a rolling ball keeps going once it rolls without slipping.",
  };
  function drawSchematic() {
    sch.innerHTML = "";
    const hot = css("--hot"), s1 = css("--series-1"), mu = css("--muted");
    sch.appendChild(el("path", { d: "M10 160 L80 130 L190 130 L120 160 Z", fill: css("--panel"), stroke: css("--static") }));
    sch.appendChild(el("circle", { cx: 100, cy: 95, r: 42, fill: css("--bg"), stroke: s1, "stroke-width": 2 }));
    if (tab === "art" || tab === "spin") {
      sch.appendChild(el("ellipse", { cx: 100, cy: 145, rx: tab === "art" ? 26 : 4, ry: tab === "art" ? 6 : 1.5, fill: "none", stroke: hot, "stroke-width": 2 }));
      if (tab === "art") { sch.appendChild(el("line", { x1: 100, y1: 145, x2: 126, y2: 145, stroke: hot })); text(sch, 128, 158, "r", { fill: hot }); }
      sch.appendChild(el("path", { d: "M74 56 A 30 8 0 1 0 128 56", fill: "none", stroke: hot, "stroke-width": 2 }));
      sch.appendChild(el("path", { d: "M128 56 l-8 -5 l1 9 z", fill: hot }));
      sch.appendChild(el("line", { x1: 100, y1: 40, x2: 100, y2: 150, stroke: mu, "stroke-dasharray": "3 3" }));
      text(sch, 100, 190, tab === "art" ? "spin row, budget r·(μN − sliding)" : "spin row, μ_torsional·N", { "text-anchor": "middle", "font-size": 10 });
    } else {
      sch.appendChild(el("path", { d: "M60 95 A 40 40 0 0 1 100 55", fill: "none", stroke: hot, "stroke-width": 2 }));
      sch.appendChild(el("path", { d: "M100 55 l-9 -4 l1 9 z", fill: hot }));
      sch.appendChild(el("line", { x1: 100, y1: 145, x2: 150, y2: 145, stroke: s1, "stroke-width": 2 }));
      sch.appendChild(el("path", { d: "M150 145 l-8 -4 v8 z", fill: s1 }));
      text(sch, 100, 190, "rolling pair, μ_rolling·N", { "text-anchor": "middle", "font-size": 10 });
    }
  }
  function drawChart() {
    const d = lib.data;
    const cols = [css("--series-1"), css("--series-3"), css("--series-2")];
    let opt;
    if (tab === "art" && d["articulated-torsion"]) opt = { ylabel: "spin rate [rad/s] · measured with Steel", series: lib.series(d, "articulated-torsion", "spin", cols), ymin: 0 };
    else if (tab === "spin" && d["angular-friction"]) opt = { ylabel: "spin rate [rad/s] · measured with Steel", series: lib.series(d, "angular-friction", "spin", cols), ymin: 0 };
    else if (d["angular-friction"]) opt = { ylabel: "rolling speed [m/s] · measured with Steel", series: lib.series(d, "angular-friction", "roll", cols), ymin: 0 };
    if (!opt) return;
    opt.h = 200; opt.legend = f.querySelector("[data-role=legend]");
    lib.lineChart(chart, opt);
  }
  const draw = () => { drawSchematic(); drawChart(); out.textContent = TXT[tab]; };
  lib.tabs(f.querySelector("[data-role=tab]"), (v) => { tab = v; draw(); });
  lib.onTheme(draw);
  draw();
});
