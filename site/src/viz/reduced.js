/* Reduced vs maximal coordinates: the same hanging chain with a heavy tip, simulated twice. */
SteelViz.register("reduced", (root, lib) => {
  const { css, el, text } = lib;
  const f = lib.frame(root, `
    <div class="vizbar">
      <span class="seg" data-role="view"><button type="button" class="on" data-v="2d">2D</button><button type="button" data-v="3d">3D</button></span>
      <button type="button" class="btn" data-role="play">Pause</button>
      <label>links <output></output><input type="range" min="3" max="8" step="1" value="5" data-k="n"></label>
      <label>tip mass ratio <output></output><input type="range" min="1" max="50" step="1" value="20" data-k="ratio"></label>
      <label>iterations <output></output><input type="range" min="1" max="64" step="1" value="8" data-k="iters"></label>
    </div>
    <svg class="vizsvg" viewBox="0 0 640 300" role="img" aria-label="Two copies of a chain: reduced coordinates on the left, maximal coordinates on the right."></svg>
    <div class="stage3d" hidden tabindex="0" aria-label="3D view of both chains. Drag to orbit."><button type="button" class="reset">Reset view</button></div>
    <p class="readout" aria-live="polite"></p>`);
  const svg = f.querySelector("svg"), st = f.querySelector(".stage3d"), out = f.querySelector(".readout");
  const inputs = [...f.querySelectorAll("input")];
  let run = null, frame = 0, playing = !lib.reduceMotion(), view = "2d", stage = null, raf = 0;

  const params = () => Object.fromEntries(inputs.map((i) => [i.dataset.k, +i.value]));
  function recompute() {
    const p = params();
    inputs.forEach((i) => (i.previousElementSibling.textContent = i.value));
    run = chainRun(p.n, p.ratio, p.iters, 3, 1 / 240, 8);
    frame = 0; t0 = 0;
    const worst = Math.max(...run.stretch) * 1000;
    out.innerHTML = `Maximal coordinates, ${p.iters} iterations: links stretch by up to <b>${worst.toFixed(1)} mm</b> during the swing. Reduced coordinates: <b>0 mm</b>; the state is the joint angles, so the links cannot stretch.`;
    draw();
  }
  function draw() {
    if (!run) return;
    const k = Math.min(frame, run.red.length - 1);
    if (view === "2d") draw2d(k); else draw3d(k);
  }
  function draw2d(k) {
    svg.innerHTML = "";
    const n = run.red[0].length, scale = 140 / (run.l * n);
    [[run.red[k], 165, "reduced: joint angles"], [run.max[k], 475, "maximal: free bodies + joint rows"]].forEach(([pts, ox, label], side) => {
      const oy = 60;
      text(svg, ox, 22, label, { "text-anchor": "middle", fill: css("--fg") });
      svg.appendChild(el("rect", { x: ox - 20, y: oy - 8, width: 40, height: 6, fill: css("--static") }));
      let prev = [ox, oy];
      pts.forEach((p, i) => {
        const q = [ox + p[0] * scale, oy - p[1] * scale];
        const len = Math.hypot(p[0] - (i ? pts[i - 1][0] : 0), p[1] - (i ? pts[i - 1][1] : 0));
        const strain = Math.abs(len - run.l) / run.l;
        svg.appendChild(el("line", { x1: prev[0], y1: prev[1], x2: q[0], y2: q[1], stroke: side && strain > 0.004 ? css("--hot") : css("--series-1"), "stroke-width": 4, "stroke-linecap": "round" }));
        const last = i === pts.length - 1;
        svg.appendChild(el("circle", { cx: q[0], cy: q[1], r: last ? 9 : 4, fill: last ? css("--series-2") : css("--series-1") }));
        if (!side) svg.appendChild(el("circle", { cx: prev[0], cy: prev[1], r: 3, fill: css("--bg"), stroke: css("--series-1"), "stroke-width": 1.5 }));
        prev = q;
      });
    });
    svg.appendChild(el("line", { x1: 320, x2: 320, y1: 36, y2: 290, stroke: css("--rule") }));
    text(svg, 475, 292, `max stretch now ${(run.stretch[k] * 1000).toFixed(1)} mm`, { "text-anchor": "middle" });
  }
  function draw3d(k) {
    if (!window.THREE) return;
    if (!stage) stage = lib.stage3d(st, { target: [0, 0, -0.6], dist: 4.2, theta: -1.57, phi: 0.12 });
    if (!stage) return;
    stage.clear();
    const G = stage.group, T = THREE;
    const mat = (c) => new T.MeshStandardMaterial({ color: new T.Color(c), roughness: 0.45 });
    [[run.red[k], -0.85], [run.max[k], 0.85]].forEach(([pts, ox], side) => {
      const base = new T.Mesh(new T.BoxGeometry(0.3, 0.3, 0.05), mat(css("--static"))); base.position.set(ox, 0, 0.03); G.add(base);
      let prev = new T.Vector3(ox, 0, 0);
      pts.forEach((p, i) => {
        const q = new T.Vector3(ox + p[0], 0, p[1]);
        const d = q.clone().sub(prev), len = d.length();
        const strain = Math.abs(len - run.l) / run.l;
        const cyl = new T.Mesh(new T.CylinderGeometry(0.025, 0.025, len, 16), mat(side && strain > 0.004 ? css("--hot") : css("--series-1")));
        cyl.position.copy(prev.clone().add(q).multiplyScalar(0.5));
        cyl.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), d.normalize());
        G.add(cyl);
        const s = new T.Mesh(new T.SphereGeometry(i === pts.length - 1 ? 0.07 : 0.035, 20, 14), mat(i === pts.length - 1 ? css("--series-2") : css("--series-1")));
        s.position.copy(q); G.add(s);
        prev = q;
      });
    });
    stage.req();
  }
  let t0 = 0;
  function tick(now) {
    raf = 0;
    if (!playing || !run) return;
    if (!t0) t0 = now - (frame / 30) * 1000;
    const next = Math.floor(((now - t0) / 1000) * 30) % run.red.length;   // recorded at 30 frames per second
    if (next !== frame) { frame = next; draw(); }
    raf = requestAnimationFrame(tick);
  }
  const playBtn = f.querySelector("[data-role=play]");
  playBtn.textContent = playing ? "Pause" : "Play";
  playBtn.addEventListener("click", () => {
    playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; t0 = 0;
    if (playing && !raf) raf = requestAnimationFrame(tick);
  });
  lib.tabs(f.querySelector("[data-role=view]"), (v) => {
    view = v; svg.toggleAttribute("hidden", v === "3d"); st.hidden = v !== "3d";
    draw(); if (stage) stage.resize();
  });
  inputs.forEach((i) => i.addEventListener("input", recompute));
  lib.onTheme(draw);
  recompute();
  if (playing) raf = requestAnimationFrame(tick);

  // Planar chain from a fixed pivot, released horizontal; unit point masses at link ends, the tip ratio times heavier.
  // Reduced: absolute link angles with the exact Lagrangian dynamics. Maximal: point positions with one distance row
  // per joint, solved by velocity-level Gauss-Seidel (iters sweeps, Baumgarte 0.2), root to tip.
  function chainRun(n, ratio, iters, seconds, h, every) {
    const l = 0.25, g = 9.81, m = Array.from({ length: n }, (_, i) => (i === n - 1 ? ratio : 1));
    const S = [];
    for (let j = 0; j < n; j++) { S.push([]); for (let k = 0; k < n; k++) { let s = 0; for (let i = Math.max(j, k); i < n; i++) s += m[i]; S[j].push(s); } }
    const th = new Float64Array(n), w = new Float64Array(n);
    const pos = () => { const p = []; let x = 0, y = 0; for (let j = 0; j < n; j++) { x += l * Math.cos(th[j]); y += l * Math.sin(th[j]); p.push([x, y]); } return p; };
    const P = pos(), V = P.map(() => [0, 0]), red = [], max = [], stretch = [];
    for (let s = 0; s <= Math.round(seconds / h); s++) {
      if (s % every === 0) {
        red.push(pos()); max.push(P.map((q) => q.slice()));
        let e = 0, a = [0, 0];
        for (let i = 0; i < n; i++) { e = Math.max(e, Math.abs(Math.hypot(P[i][0] - a[0], P[i][1] - a[1]) - l)); a = P[i]; }
        stretch.push(e);
      }
      const M = [], rhs = [];
      for (let j = 0; j < n; j++) {
        M.push([]); let c = 0;
        for (let k = 0; k < n; k++) { M[j].push(l * l * Math.cos(th[j] - th[k]) * S[j][k]); c += l * l * Math.sin(th[j] - th[k]) * S[j][k] * w[k] * w[k]; }
        rhs.push(-c - g * l * Math.cos(th[j]) * S[j][j]);
      }
      const acc = solve(M, rhs);
      for (let j = 0; j < n; j++) { w[j] += h * acc[j]; th[j] += h * w[j]; }
      for (let i = 0; i < n; i++) V[i][1] -= g * h;
      for (let it = 0; it < iters; it++) {
        for (let i = 0; i < n; i++) {
          const a = i ? P[i - 1] : [0, 0], va = i ? V[i - 1] : [0, 0], ima = i ? 1 / m[i - 1] : 0, imb = 1 / m[i];
          const dx = P[i][0] - a[0], dy = P[i][1] - a[1], len = Math.hypot(dx, dy), nx = dx / len, ny = dy / len;
          const lam = -(nx * (V[i][0] - va[0]) + ny * (V[i][1] - va[1]) + (0.2 * (len - l)) / h) / (ima + imb);
          V[i][0] += nx * lam * imb; V[i][1] += ny * lam * imb;
          if (i) { va[0] -= nx * lam * ima; va[1] -= ny * lam * ima; }
        }
      }
      for (let i = 0; i < n; i++) { P[i][0] += h * V[i][0]; P[i][1] += h * V[i][1]; }
    }
    return { red, max, stretch, l };
  }
  function solve(A, b) {
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) { const fct = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= fct * M[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  }
});
