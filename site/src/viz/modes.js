/* Solve modes: serial and colored sweeps, immediate and propagation responses, on one small world. */
SteelViz.register("modes", (root, lib) => {
  const { css, el, text } = lib;
  const f = lib.frame(root, `
    <div class="vizbar">
      <span class="seg" data-role="mode"><button type="button" class="on" data-v="serial">serial</button><button type="button" data-v="colored">colored</button><button type="button" data-v="immediate">immediate</button><button type="button" data-v="propagation">propagation</button></span>
      <span class="stepbtns"><button type="button" class="btn" data-act="prev" aria-label="Previous step">◀</button><button type="button" class="btn" data-act="play">Play</button><button type="button" class="btn" data-act="next" aria-label="Next step">▶</button></span>
    </div>
    <svg class="vizsvg" viewBox="0 0 640 306" role="img" aria-label="An arm with two fingers holding a box, and a stack of two boxes, with their contact rows."></svg>
    <p class="readout" aria-live="polite"></p>`);
  const svg = f.querySelector("svg"), out = f.querySelector(".readout");
  const B = {
    base: { x: 120, y: 70, kind: "fixed", label: "base" }, L1: { x: 200, y: 110, kind: "link", label: "link 1" },
    L2a: { x: 238, y: 180, kind: "link", label: "finger a" }, L2b: { x: 302, y: 180, kind: "link", label: "finger b" },
    O1: { x: 270, y: 226, kind: "box", label: "box 1" }, O2: { x: 440, y: 226, kind: "box", label: "box 2" }, O3: { x: 440, y: 176, kind: "box", label: "box 3" },
  };
  const C = [
    { id: "c1", a: "L2a", b: "O1", x: 246, y: 206, lx: -14, ly: 4 }, { id: "c2", a: "L2b", b: "O1", x: 294, y: 206, lx: 14, ly: 4 },
    { id: "c3", a: "O1", b: null, x: 270, y: 251, lx: 0, ly: 18 }, { id: "c4", a: "O2", b: null, x: 440, y: 251, lx: 0, ly: 18 }, { id: "c5", a: "O3", b: "O2", x: 440, y: 201, lx: 40, ly: 4 },
  ];
  // Greedy coloring: no two contacts of a color share a responding body.
  const color = [];
  C.forEach((c, i) => {
    let k = 0;
    while (C.slice(0, i).some((d, j) => color[j] === k && [d.a, d.b].some((x) => x && (x === c.a || x === c.b)))) k++;
    color.push(k);
  });
  const ncolors = Math.max(...color) + 1;
  const arm = ["L1", "L2a", "L2b"];
  const DOF = [["arm", 3], ["O1", 6], ["O2", 6], ["O3", 6]];
  const STEPS = {
    serial: C.map((c, i) => ({ rows: [i], dof: dofOf([c.a, c.b]), text: `Serial sweep, row ${c.id}: one warp solves the world's rows in order; this row reads the velocity the rows before it just wrote.` })),
    colored: Array.from({ length: ncolors }, (_, k) => {
      const rows = C.map((c, i) => (color[i] === k ? i : -1)).filter((i) => i >= 0);
      return { rows, dof: dofOf([].concat(...rows.map((i) => [C[i].a, C[i].b]))), text: `Colored sweep, color ${k + 1} of ${ncolors}: ${rows.map((i) => C[i].id).join(", ")} share no responding body, so they are solved at the same time.` };
    }),
    immediate: [
      { rows: [0], dof: [], text: "Immediate response: contact c1 on finger a is a row over the arm's joint coordinates." },
      { rows: [0], dof: ["arm", "O1"], hi: arm, text: "Its response Y = H⁻¹Jᵀ is dense over the whole arm: an impulse at the fingertip moves every joint of the tree, plus box 1." },
      { rows: [1], dof: ["arm", "O1"], hi: arm, text: "The next row, c2 on finger b, already sees that change: the two fingers are coupled through the arm inside every row update." },
    ],
    propagation: [
      { rows: [0, 1, 2, 3, 4], dof: ["O1", "O2", "O3"], body: ["L2a", "L2b"], text: "Propagation response: every contact is a fixed-size row at the bodies it touches, solved with each body's 6×6 response from the tree factorization." },
      { rows: [], dof: [], up: true, body: ["L2a", "L2b", "L1"], text: "After the rows of an iteration, the impulses accumulated on the fingers are propagated up the tree to the joints..." },
      { rows: [], dof: ["arm"], down: true, hi: arm, text: "...and back down, which updates every joint velocity. The next iteration's rows see the fingers' new motion." },
    ],
  };
  let mode = "serial";
  function dofOf(bodies) {
    const s = new Set();
    bodies.forEach((b) => { if (!b) return; s.add(arm.includes(b) ? "arm" : b); });
    return [...s];
  }
  function render(i) {
    const step = STEPS[mode][i];
    svg.innerHTML = "";
    const fg = css("--fg"), mu = css("--muted"), hot = css("--hot"), s1 = css("--series-1");
    svg.appendChild(el("line", { x1: 30, x2: 610, y1: 251, y2: 251, stroke: css("--static"), "stroke-width": 3 }));
    [["base", "L1"], ["L1", "L2a"], ["L1", "L2b"]].forEach(([a, b]) => {
      const lit = (step.hi || []).includes(b) || (step.body || []).includes(b) && (step.up || step.down);
      svg.appendChild(el("line", { x1: B[a].x, y1: B[a].y, x2: B[b].x, y2: B[b].y, stroke: lit ? hot : s1, "stroke-width": lit ? 5 : 4, "stroke-linecap": "round" }));
    });
    Object.entries(B).forEach(([k, b]) => {
      const lit = (step.hi || []).includes(k) || (step.body || []).includes(k);
      if (b.kind === "box") svg.appendChild(el("rect", { x: b.x - 25, y: b.y - 25, width: 50, height: 50, rx: 3, fill: css("--panel"), stroke: s1, "stroke-width": 1.5 }));
      else if (b.kind === "fixed") svg.appendChild(el("rect", { x: b.x - 14, y: b.y - 10, width: 28, height: 20, fill: css("--static") }));
      else svg.appendChild(el("circle", { cx: b.x, cy: b.y, r: 8, fill: lit ? hot : s1 }));
      const left = k === "L2a"; text(svg, b.kind === "box" ? b.x : b.x + (left ? -12 : 12), b.kind === "box" ? b.y - 4 : b.y - 10, b.label, { "text-anchor": b.kind === "box" ? "middle" : left ? "end" : "start" });
    });
    if (step.up || step.down) {
      const p = step.up ? [[B.L2a, B.L1], [B.L2b, B.L1], [B.L1, B.base]] : [[B.base, B.L1], [B.L1, B.L2a], [B.L1, B.L2b]];
      p.forEach(([a, b]) => drawArrow(a.x, a.y, b.x, b.y, hot));
    }
    C.forEach((c, i) => {
      const on = step.rows.includes(i);
      const col = mode === "colored" ? css(["--series-2", "--series-1", "--series-3", "--series-4"][color[i]]) : on ? hot : mu;
      svg.appendChild(el("circle", { cx: c.x, cy: c.y, r: on ? 7 : 5, fill: on || mode === "colored" ? col : css("--bg"), stroke: col, "stroke-width": 2, opacity: mode === "colored" && !on ? 0.35 : 1 }));
      text(svg, c.x + c.lx, c.y + c.ly, c.id, { "text-anchor": "middle", fill: on ? fg : mu });
    });
    // velocity vector of the world, by owner
    let x = 30;
    text(svg, 30, 279, "velocity entries this step touches:", {});
    const w = 560 / DOF.reduce((s, d) => s + d[1], 0);
    DOF.forEach(([name, n]) => {
      const on = step.dof.includes(name);
      svg.appendChild(el("rect", { x, y: 286, width: n * w - 2, height: 12, fill: on ? hot : css("--panel"), stroke: css("--rule") }));
      text(svg, x + 2, 296, name === "arm" ? "arm joints" : name, { fill: on ? css("--bg") : mu, "font-size": 9.5 });
      x += n * w;
    });
    out.textContent = `Step ${i + 1} of ${STEPS[mode].length}. ${step.text}`;
  }
  function drawArrow(x1, y1, x2, y2, c) {
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
    const sx = x1 + ux * 12, sy = y1 + uy * 12, ex = x2 - ux * 12, ey = y2 - uy * 12;
    svg.appendChild(el("path", { d: `M${sx},${sy} L${ex},${ey}`, stroke: c, "stroke-width": 2, fill: "none" }));
    svg.appendChild(el("path", { d: `M${ex},${ey} l${-ux * 8 - uy * 5},${-uy * 8 + ux * 5} l${uy * 10},${-ux * 10} z`, fill: c }));
  }
  let ctl = lib.stepper(f, STEPS[mode].length, render);
  lib.tabs(f.querySelector("[data-role=mode]"), (v) => {
    mode = v; ctl.stop();
    ctl = rebuild();
  });
  function rebuild() {
    const fresh = f.querySelector(".stepbtns").cloneNode(true);
    f.querySelector(".stepbtns").replaceWith(fresh);
    return lib.stepper(f, STEPS[mode].length, render);
  }
  lib.onTheme(() => ctl.redraw());
});
