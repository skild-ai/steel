// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

/* Row estimate for one world: inputs grouped by source, rows per family as stacked bars against the default
   capacities, and a schematic of the per-world row buffer. An upper-bound estimate, not a simulation. */
SteelViz.register("capacity", (root, lib) => {
  const { css, el, text } = lib;
  const num = (id, label, v) => `<label>${label}<input type="number" min="0" value="${v}" data-k="${id}"></label>`;
  const f = lib.frame(root, `
    <div class="capgrid">
      <fieldset><legend>Contacts</legend>
        <label>Response<select data-k="resp"><option value="immediate">immediate</option><option value="propagation">propagation</option></select></label>
        ${num("ac", "On articulated bodies", 8)}${num("sc", "Within one articulation", 0)}${num("fc", "Free-body contacts", 4)}
        <label>Free-body condim<select data-k="cd"><option value="0">3 (sliding)</option><option value="1">4 (+ spin)</option><option value="3">6 (+ spin, rolling)</option></select></label>
        ${num("tc", "With a torsion radius", 0)}
        <p class="capwarn" data-w="contacts" hidden></p>
      </fieldset>
      <fieldset><legend>Joints</legend>
        ${num("lim", "Limit rows (DOFs near a limit)", 2)}${num("vl", "Velocity-limited DOFs", 0)}${num("jf", "DOFs with joint friction", 0)}
        ${num("dr", "Driven DOFs, drive_mode=\"rows\"", 0)}${num("mi", "Mimic followers", 1)}
        <label>Loop closures: BALL / REVOLUTE / FIXED<span class="trio">${["lb", "lr", "lf"].map((k) => `<input type="number" min="0" value="0" data-k="${k}" aria-label="${k}">`).join("")}</span></label>
        <p class="capwarn" data-w="joints" hidden></p>
      </fieldset>
      <fieldset><legend>Free bodies</legend>
        ${num("fl", "With a max linear speed", 0)}${num("fa", "With a max angular speed", 0)}
        <p class="capwarn" data-w="free" hidden></p>
      </fieldset>
    </div>
    <svg class="vizsvg" data-role="bars" role="img" aria-label="Rows per family as stacked bars against the default capacities."></svg>
    <div class="legend" data-role="legend"></div>
    <svg class="vizsvg capbuf" data-role="buffer" role="img" aria-label="Schematic of the articulation row buffer of one world: each cell is one row; rows past the capacity are dropped."></svg>
    <p class="readout" aria-live="polite"></p>`);
  const inputs = [...f.querySelectorAll("[data-k]")];
  const bars = f.querySelector("[data-role=bars]"), buf = f.querySelector("[data-role=buffer]"), out = f.querySelector(".readout");
  const SRC = [["contacts", "--series-1"], ["torsion", "--series-4"], ["joint rows", "--series-3"], ["free-body limits", "--series-2"]];

  function compute() {
    const g = (k) => { const e = f.querySelector(`[data-k=${k}]`); return e.tagName === "SELECT" ? e.value : Math.max(0, Math.floor(+e.value || 0)); };
    const resp = g("resp"), extra = +g("cd"), ac = g("ac"), sc = g("sc"), fc = g("fc"), tc = g("tc");
    const joint = g("lim") + 2 * g("vl") + g("jf") + g("dr") + g("mi") + 3 * g("lb") + 5 * g("lr") + 6 * g("lf");
    const freeLim = 6 * g("fl") + 6 * g("fa");
    const fam = resp === "immediate"
      ? [
          { name: "articulation", cap: 32, opt: "articulation_constraint_max", parts: [3 * (ac + sc), tc, joint, 0] },
          { name: "free body", cap: 512, opt: "free_body_constraint_max", parts: [fc * (3 + extra), 0, 0, freeLim] },
        ]
      : [
          { name: "articulation", cap: 32, opt: "articulation_constraint_max", parts: [3 * sc, 0, joint, 0] },
          { name: "free body", cap: 512, opt: "free_body_constraint_max", parts: [0, 0, 0, freeLim] },
          { name: "propagation", cap: 544, opt: "propagation_constraint_max", parts: [3 * (ac + fc) + fc * extra, tc, 0, 0] },
        ];
    fam.forEach((x) => (x.total = x.parts.reduce((s, v) => s + v, 0)));
    return { fam, resp, ac, fc, joint, freeLim };
  }
  function draw() {
    const c = compute();
    const W = 900, rowH = 46, H = 18 + rowH * c.fam.length;
    bars.setAttribute("viewBox", `0 0 ${W} ${H}`);
    bars.innerHTML = "";
    c.fam.forEach((fm, i) => {
      const y = 10 + i * rowH, scale = (W - 260) / Math.max(fm.cap * 1.15, fm.total * 1.05, 1), x0 = 120;
      text(bars, 0, y + 18, fm.name, { fill: css("--fg"), "font-size": 12 });
      bars.appendChild(el("rect", { x: x0, y: y + 6, width: (W - 260), height: 18, fill: css("--panel") }));
      let x = x0;
      fm.parts.forEach((v, k) => {
        if (!v) return;
        bars.appendChild(el("rect", { x, y: y + 6, width: v * scale, height: 18, fill: css(SRC[k][1]) }));
        x += v * scale;
      });
      const cx = x0 + fm.cap * scale, over = fm.total > fm.cap;
      bars.appendChild(el("line", { x1: cx, x2: cx, y1: y, y2: y + 30, stroke: css("--fg"), "stroke-width": 2 }));
      text(bars, cx, y + 40, `default ${fm.cap}`, { "text-anchor": "middle", "font-size": 10 });
      text(bars, W - 92, y + 20, `${fm.total} rows`, { fill: over ? css("--hot") : css("--fg"), "font-weight": over ? 600 : 400 });
      if (fm.total) text(bars, W - 92, y + 34, `size ≥ ${Math.ceil(fm.total * 1.3)}`, { "font-size": 10 });
    });
    f.querySelector("[data-role=legend]").innerHTML = SRC.map(([n, t]) => `<span><i style="background:${css(t)}"></i>${n}</span>`).join("");
    // buffer schematic for the articulation family
    const art = c.fam[0], cells = Math.max(art.cap, art.total), per = 32, rows = Math.ceil(cells / per);
    buf.setAttribute("viewBox", `0 0 600 ${30 + rows * 16}`);
    buf.innerHTML = "";
    text(buf, 0, 12, "articulation row buffer of one world · one cell per row · dropped past the capacity", { "font-size": 10.5 });
    let k = 0;
    art.parts.forEach((v, s) => {
      for (let j = 0; j < v; j++, k++) {
        const r = Math.floor(k / per), col = k % per, dropped = k >= art.cap;
        buf.appendChild(el("rect", { x: col * 18, y: 20 + r * 16, width: 15, height: 12, fill: dropped ? css("--bg") : css(SRC[s][1]), stroke: dropped ? css("--hot") : "none", "stroke-dasharray": dropped ? "2 2" : "" }));
      }
    });
    for (; k < art.cap; k++) buf.appendChild(el("rect", { x: (k % per) * 18, y: 20 + Math.floor(k / per) * 16, width: 15, height: 12, fill: css("--panel") }));
    // warnings next to the inputs that feed an overflowing family
    const warn = (w, msg) => { const p = f.querySelector(`[data-w=${w}]`); p.hidden = !msg; p.textContent = msg || ""; };
    const artOver = art.total > art.cap, freeOver = c.fam[1].total > c.fam[1].cap, prop = c.fam[2];
    warn("contacts", c.resp === "immediate" && artOver && art.parts[0] >= art.parts[2] ? `Contacts on articulated bodies alone take ${art.parts[0]} articulation rows; raise articulation_constraint_max.` : prop && prop.total > prop.cap ? `Contacts take ${prop.total} propagation rows; raise propagation_constraint_max.` : c.resp === "immediate" && freeOver ? "Free-body contacts exceed free_body_constraint_max." : "");
    warn("joints", artOver && art.parts[2] > art.parts[0] ? `Joint rows (${art.parts[2]}) dominate the articulation family; raise articulation_constraint_max or gate limit rows near their bounds.` : "");
    warn("free", c.freeLim && freeOver ? `Speed limits take ${c.freeLim} free-body rows (two per limited axis, three axes each).` : "");
    out.textContent = c.fam.some((x) => x.total > x.cap)
      ? "At least one family exceeds its default capacity: rows past it would be dropped and the world flagged. Measure with row_watermark=True and size from the requested high-water mark."
      : "Every family fits its default capacity. Confirm with row_watermark=True over a full episode; patch friction often uses fewer rows than this estimate.";
  }
  inputs.forEach((i) => i.addEventListener("input", draw));
  lib.onTheme(draw);
  draw();
});
