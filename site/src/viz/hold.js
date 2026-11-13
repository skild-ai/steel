// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

/* Friction patches vs point friction for a box pushed sideways on a table: where the friction acts this step
   (coupled-disk Gauss-Seidel, the same update as the friction demo), and the creep Steel measured over 10 s. */
SteelViz.register("hold", (root, lib) => {
  const { css, el, text } = lib;
  const f = lib.frame(root, `
    <div class="vizbar">
      <label>push direction <output></output><input type="range" min="0" max="180" step="1" value="30" data-k="dir"></label>
      <label>push / μmg <output></output><input type="range" min="0" max="1.3" step="0.01" value="0.9" data-k="mag"></label>
    </div>
    <svg class="vizsvg" viewBox="0 0 640 250" role="img" aria-label="Top views of a box under a side push: point friction at four corners and patch friction at two anchors, with each location's friction force and friction disk; a side view below."></svg>
    <p class="readout" aria-live="polite"></p>
    <svg class="vizsvg" data-role="chart" role="img" aria-label="Measured slip of a box on a slope at 98 percent of its friction limit over 10 seconds, point friction against patch friction."></svg>
    <div class="legend" data-role="legend"></div>`);
  const svg = f.querySelector("svg"), chart = f.querySelector("[data-role=chart]"), out = f.querySelector(".readout");
  const inputs = [...f.querySelectorAll("input")];
  const m = 1, a = 0.2, b = 0.1, mu = 0.5, g = 9.81, h = 1 / 60, I = (m * ((2 * a) ** 2 + (2 * b) ** 2)) / 12;

  function solveStep(pts, F) {
    const load = (m * g) / pts.length, v = [(h * F[0]) / m, (h * F[1]) / m, 0], lam = pts.map(() => [0, 0]);
    for (let it = 0; it < 300; it++) {
      pts.forEach((r, i) => {
        const A = [1 / m + (r[1] * r[1]) / I, (-r[0] * r[1]) / I, (-r[0] * r[1]) / I, 1 / m + (r[0] * r[0]) / I];
        const u = [v[0] - v[2] * r[1], v[1] + v[2] * r[0]];
        const bb = [A[0] * lam[i][0] + A[1] * lam[i][1] - u[0], A[2] * lam[i][0] + A[3] * lam[i][1] - u[1]];
        const x = diskMin(A, bb, mu * load * h), dl = [x[0] - lam[i][0], x[1] - lam[i][1]];
        lam[i] = x;
        v[0] += dl[0] / m; v[1] += dl[1] / m; v[2] += (r[0] * dl[1] - r[1] * dl[0]) / I;
      });
    }
    return { force: lam.map((l) => [l[0] / h, l[1] / h]), load, speed: Math.hypot(v[0], v[1]) };
  }
  function draw() {
    const p = Object.fromEntries(inputs.map((i) => [i.dataset.k, +i.value]));
    inputs[0].previousElementSibling.textContent = p.dir + "°";
    inputs[1].previousElementSibling.textContent = p.mag.toFixed(2);
    const ang = (p.dir * Math.PI) / 180, Fm = p.mag * mu * m * g, F = [Fm * Math.cos(ang), Fm * Math.sin(ang)];
    svg.innerHTML = "";
    const fg = css("--fg"), mu_ = css("--muted"), hot = css("--hot"), s1 = css("--series-1");
    const S = 260, scaleF = 34 / (mu * m * g / 2); // px per N, disk radii in the same units
    const panels = [
      { title: "point friction: 4 corners, μ·mg/4 each", pts: [[-a, -b], [a, -b], [-a, b], [a, b]], cx: 160 },
      { title: "patch friction: 2 anchors, pooled μ·mg/2 each", pts: [[-a, 0], [a, 0]], cx: 480 },
    ];
    const res = panels.map((pn) => solveStep(pn.pts, F));
    panels.forEach((pn, k) => {
      const cy = 100, r = res[k];
      text(svg, pn.cx, 16, pn.title, { "text-anchor": "middle", fill: fg });
      svg.appendChild(el("rect", { x: pn.cx - a * S, y: cy - b * S, width: 2 * a * S, height: 2 * b * S, fill: css("--panel"), stroke: s1, "stroke-width": 1.5 }));
      arrow(pn.cx, cy, pn.cx + F[0] * scaleF, cy - F[1] * scaleF, css("--series-1"), 2.5);
      pn.pts.forEach((q, i) => {
        const x = pn.cx + q[0] * S, y = cy - q[1] * S, R = mu * r.load * scaleF;
        svg.appendChild(el("circle", { cx: x, cy: y, r: R, fill: "none", stroke: mu_, "stroke-dasharray": "3 3" }));
        svg.appendChild(el("circle", { cx: x, cy: y, r: 4, fill: k ? hot : s1 }));
        const fr = r.force[i];
        arrow(x, y, x + fr[0] * scaleF, y - fr[1] * scaleF, hot, 2);
      });
      const slides = r.speed > 1e-6;
      text(svg, pn.cx, 190, slides ? "slides: every location is at its disk's edge" : "holds this step", { "text-anchor": "middle", fill: slides ? hot : css("--series-3") });
    });
    // side view of the patch box
    const gy = 236, bx = 320;
    svg.appendChild(el("line", { x1: 20, x2: 620, y1: gy, y2: gy, stroke: css("--static"), "stroke-width": 3 }));
    svg.appendChild(el("rect", { x: bx - 60, y: gy - 30, width: 120, height: 30, fill: css("--panel"), stroke: s1 }));
    arrow(bx - 120, gy - 15, bx - 60, gy - 15, s1, 2.5);
    text(svg, bx - 124, gy - 20, "push", { "text-anchor": "end" });
    [-60, 60].forEach((dx) => arrow(bx + dx, gy + 0, bx + dx, gy - 18, mu_, 1.5));
    text(svg, bx + 70, gy - 8, "side view: normal rows", {});
    const ratio = p.mag;
    out.innerHTML = ratio <= 1
      ? `The push is ${(ratio * 100).toFixed(0)}% of μmg. Both models can hold it within one step: the total friction budget is the same. The difference is where it lives. Point friction splits the push among four disks whose loads can shift between corners from step to step; patch friction pools the load at two anchors and also corrects the displacement they carried, which stops slow creep (chart below).`
      : `The push is ${(ratio * 100).toFixed(0)}% of μmg, above the total friction budget: both models slide, with every friction location at the edge of its disk.`;
  }
  function arrow(x1, y1, x2, y2, c, w) {
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy);
    if (L < 2) return;
    const ux = dx / L, uy = dy / L, hl = Math.min(8, L * 0.4);
    svg.appendChild(el("line", { x1, y1, x2: x2 - ux * hl * 0.8, y2: y2 - uy * hl * 0.8, stroke: c, "stroke-width": w }));
    svg.appendChild(el("path", { d: `M${x2},${y2} L${x2 - ux * hl - uy * hl * 0.5},${y2 - uy * hl + ux * hl * 0.5} L${x2 - ux * hl + uy * hl * 0.5},${y2 - uy * hl - ux * hl * 0.5} Z`, fill: c }));
  }
  function drawChart() {
    if (!lib.data["patch-friction"]) return;
    lib.lineChart(chart, {
      h: 190, ylabel: "slip along the slope [mm] · measured with Steel, box at 98% of its friction limit",
      series: lib.series(lib.data, "patch-friction", "slip_mm", [css("--series-1"), css("--series-2")]),
      legend: f.querySelector("[data-role=legend]"), ymin: 0,
    });
  }
  // Minimizer of 0.5 x'Ax - b'x on |x| <= rho for a symmetric positive 2x2 A.
  function diskMin(A, bb, rho) {
    const det = A[0] * A[3] - A[1] * A[2];
    const xs = [(A[3] * bb[0] - A[1] * bb[1]) / det, (A[0] * bb[1] - A[2] * bb[0]) / det];
    if (Math.hypot(xs[0], xs[1]) <= rho) return xs;
    const tr = A[0] + A[3], d = Math.sqrt(Math.max(0, (A[0] - A[3]) ** 2 / 4 + A[1] * A[1])), l1 = tr / 2 + d, l2 = tr / 2 - d;
    let e1 = Math.abs(A[1]) > 1e-15 ? [l1 - A[3], A[1]] : A[0] >= A[3] ? [1, 0] : [0, 1];
    const ne = Math.hypot(e1[0], e1[1]); e1 = [e1[0] / ne, e1[1] / ne];
    const e2 = [-e1[1], e1[0]], b1 = e1[0] * bb[0] + e1[1] * bb[1], b2 = e2[0] * bb[0] + e2[1] * bb[1];
    let lo = 0, hi = Math.hypot(b1, b2) / rho;
    for (let i = 0; i < 80; i++) { const mid = 0.5 * (lo + hi); if (Math.hypot(b1 / (l1 + mid), b2 / (l2 + mid)) > rho) lo = mid; else hi = mid; }
    const c1 = b1 / (l1 + hi), c2 = b2 / (l2 + hi);
    return [c1 * e1[0] + c2 * e2[0], c1 * e1[1] + c2 * e2[1]];
  }
  inputs.forEach((i) => i.addEventListener("input", draw));
  lib.onTheme(() => { draw(); drawChart(); });
  draw(); drawChart();
});
