// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

/* Steel docs visuals: shared helpers and the module registry.
   Each visual registers mount(root, lib) and reads only its own root element; colors come from CSS tokens. */
window.SteelViz = (function () {
  "use strict";
  const registry = {};
  const SVGNS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const el = (tag, attrs, text) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const text = (svg, x, y, txt, attrs) =>
    svg.appendChild(el("text", Object.assign({ x, y, "font-size": 11, fill: css("--muted"), "font-family": css("--mono") }, attrs || {}), txt));
  const reduceMotion = () => !!(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);

  function niceTicks(lo, hi, n) {
    const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0 || 1)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || mag * 10, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const fmtTick = (v) => {
    const a = Math.abs(v);
    return a >= 10 ? v.toFixed(0) : a >= 1 ? String(+v.toFixed(1)) : a === 0 ? "0" : String(+v.toPrecision(2));
  };

  // Line chart. series: [{label, values, dt, color, dash, width}]; values may hold nulls.
  function lineChart(svg, opt) {
    svg.innerHTML = "";
    const W = opt.w || 900, H = opt.h || 220, L = 54, R = 30, T = 18, B = 30;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const pts = opt.series.map((s) => s.values.map((v, i) => [i * s.dt, v]));
    const xmax = opt.xmax || Math.max(...pts.map((p) => (p.length ? p[p.length - 1][0] : 1)));
    const ys = [].concat(...pts).map((p) => p[1]).filter((v) => v !== null && isFinite(v) && (!opt.log || v > 0));
    let lo, hi;
    if (opt.log) {
      lo = Math.log10(opt.ymin || Math.max(1e-6, Math.min(...ys)));
      hi = Math.log10(Math.max(...ys) * 1.5);
    } else {
      lo = opt.ymin !== undefined ? opt.ymin : Math.min(...ys);
      hi = opt.ymax !== undefined ? opt.ymax : Math.max(...ys);
      const pad = (hi - lo || 1) * 0.06;
      if (opt.ymin === undefined) lo -= pad;
      if (opt.ymax === undefined) hi += pad;
    }
    if (hi - lo < 1e-9) hi = lo + 1;
    const X = (x) => L + (W - L - R) * (x / xmax);
    const Y = (y) => {
      const v = opt.log ? Math.log10(Math.max(y, Math.pow(10, lo))) : Math.min(Math.max(y, lo), hi);
      return T + (H - T - B) * (1 - (v - lo) / (hi - lo));
    };
    const ticks = opt.log
      ? Array.from({ length: Math.floor(hi) - Math.ceil(lo) + 1 }, (_, i) => Math.pow(10, Math.ceil(lo) + i))
      : niceTicks(lo, hi, 4);
    ticks.forEach((t) => {
      svg.appendChild(el("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), stroke: css("--rule") }));
      text(svg, L - 6, Y(t) + 4, opt.log ? (t >= 1 ? String(t) : t.toExponential(0)) : fmtTick(t), { "text-anchor": "end" });
    });
    niceTicks(0, xmax, 5).forEach((t) => { if (t <= xmax + 1e-9) text(svg, X(t), H - 10, fmtTick(t) + " s", { "text-anchor": "middle" }); });
    if (opt.ylabel) text(svg, L, 10, opt.ylabel, { "font-size": 10.5 });
    (opt.hlines || []).forEach((hl) => {
      svg.appendChild(el("line", { x1: L, x2: W - R, y1: Y(hl.y), y2: Y(hl.y), stroke: hl.color || css("--muted"), "stroke-dasharray": "4 4" }));
      if (hl.label) text(svg, W - R, Y(hl.y) - 4, hl.label, { "text-anchor": "end", fill: hl.color || css("--muted") });
    });
    opt.series.forEach((s, k) => {
      let d = "", pen = false;
      pts[k].forEach(([x, y]) => {
        if (y === null || !isFinite(y) || x > xmax + 1e-9) { pen = false; return; }
        d += (pen ? "L" : "M") + X(x).toFixed(1) + "," + Y(y).toFixed(1);
        pen = true;
      });
      const a = { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round" };
      if (s.dash) a["stroke-dasharray"] = s.dash;
      svg.appendChild(el("path", a));
    });
    if (opt.cursor !== undefined) svg.appendChild(el("line", { x1: X(opt.cursor), x2: X(opt.cursor), y1: T, y2: H - B, stroke: css("--hot"), "stroke-dasharray": "3 3" }));
    if (opt.legend) opt.legend.innerHTML = opt.series.filter((s) => s.label).map((s) => `<span><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join("");
  }

  // Previous / play / next control over count steps; render(i) draws step i.
  function stepper(root, count, render) {
    const prev = root.querySelector("[data-act=prev]"), next = root.querySelector("[data-act=next]"), play = root.querySelector("[data-act=play]");
    let i = 0, timer = null;
    const go = (k) => { i = ((k % count) + count) % count; render(i); };
    const stop = () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "Play"; } };
    prev.addEventListener("click", () => { stop(); go(i - 1); });
    next.addEventListener("click", () => { stop(); go(i + 1); });
    play.addEventListener("click", () => {
      if (timer) return stop();
      play.textContent = "Pause";
      timer = setInterval(() => go(i + 1), reduceMotion() ? 2600 : 1300);
    });
    go(0);
    return { go: (k) => { stop(); go(k); }, current: () => i, stop, redraw: () => render(i) };
  }
  // Segmented buttons inside root; calls onChange(value) with the clicked button's data-v.
  function tabs(root, onChange) {
    const btns = [...root.querySelectorAll("button")];
    const set = (b) => btns.forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-pressed", String(x === b)); });
    btns.forEach((b) => b.addEventListener("click", () => { set(b); onChange(b.dataset.v); }));
    set(btns.find((b) => b.classList.contains("on")) || btns[0]);
  }
  // A measured series from the tutorials' recordings, as chart input.
  function series(data, id, key, colors, opts) {
    opts = opts || {};
    return data[id].variants.map((v, i) => ({
      label: opts.labels ? opts.labels[i] : v.label, values: v.values[key], dt: 1 / v.fps,
      color: colors[i % colors.length], dash: opts.dashes ? opts.dashes[i] : undefined,
    })).filter((s, i) => !opts.only || opts.only.includes(i));
  }


  // A three.js stage with drag-to-orbit, wheel zoom, arrow keys and a Reset view button inside el.
  function stage3d(el, view) {
    if (!window.THREE) { el.insertAdjacentHTML("beforeend", '<p class="nogl">The 3D view needs three.js, which did not load.</p>'); return null; }
    let renderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true }); }
    catch (e) { el.insertAdjacentHTML("beforeend", '<p class="nogl">The 3D view needs WebGL, which this browser did not provide.</p>'); return null; }
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    el.insertBefore(renderer.domElement, el.firstChild);
    const scene = new THREE.Scene();
    const cam = new THREE.PerspectiveCamera(35, 1, 0.01, 200);
    cam.up.set(0, 0, 1);
    // three.js r155+ lights are physical: these are the legacy intensities times pi.
    scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa3ad, 0.9 * Math.PI));
    const sun = new THREE.DirectionalLight(0xffffff, 0.7 * Math.PI); sun.position.set(3, -4, 6); scene.add(sun);
    const fill = new THREE.DirectionalLight(0xffffff, 0.3 * Math.PI); fill.position.set(-4, 3, 2); scene.add(fill);
    const group = new THREE.Group(); scene.add(group);
    const st = Object.assign({}, view), target = new THREE.Vector3(...view.target);
    let queued = false;
    const draw = () => { queued = false; renderer.render(scene, cam); };
    const req = () => { if (!queued) { queued = true; requestAnimationFrame(draw); } };
    const place = () => {
      const c = Math.cos(st.phi);
      cam.position.set(target.x + st.dist * c * Math.cos(st.theta), target.y + st.dist * c * Math.sin(st.theta), target.z + st.dist * Math.sin(st.phi));
      cam.lookAt(target); req();
    };
    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight; if (!w || !h) return;
      renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); req();
    };
    if ("ResizeObserver" in window) new ResizeObserver(resize).observe(el);
    let drag = null;
    el.addEventListener("pointerdown", (e) => { if (e.target.closest(".reset")) return; drag = [e.clientX, e.clientY]; el.setPointerCapture(e.pointerId); });
    el.addEventListener("pointermove", (e) => {
      if (!drag) return;
      st.theta -= (e.clientX - drag[0]) * 0.01; st.phi = Math.max(-1.2, Math.min(1.45, st.phi + (e.clientY - drag[1]) * 0.01));
      drag = [e.clientX, e.clientY]; place();
    });
    const end = () => (drag = null);
    el.addEventListener("pointerup", end); el.addEventListener("pointercancel", end);
    el.addEventListener("wheel", (e) => { e.preventDefault(); st.dist = Math.max(view.dist * 0.3, Math.min(view.dist * 3, st.dist * Math.exp(e.deltaY * 0.001))); place(); }, { passive: false });
    el.addEventListener("keydown", (e) => {
      const d = { ArrowLeft: [0.08, 0], ArrowRight: [-0.08, 0], ArrowUp: [0, 0.08], ArrowDown: [0, -0.08] }[e.key];
      if (!d) return; e.preventDefault(); st.theta += d[0]; st.phi = Math.max(-1.2, Math.min(1.45, st.phi + d[1])); place();
    });
    const rb = el.querySelector(".reset");
    if (rb) rb.addEventListener("click", () => { Object.assign(st, view); place(); });
    const clear = () => {
      group.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
      group.clear();
      scene.background = new THREE.Color(css("--stage"));
    };
    resize(); place();
    return { group, req, clear, resize };
  }

  function frame(root, html) {
    const f = document.createElement("div");
    f.className = "frame";
    f.innerHTML = html;
    root.insertBefore(f, root.firstChild);
    return f;
  }

  const themeHooks = [];
  const onTheme = (cb) => themeHooks.push(cb);
  const fire = () => themeHooks.forEach((cb) => cb());
  if (window.matchMedia) { const mq = matchMedia("(prefers-color-scheme: dark)"); if (mq.addEventListener) mq.addEventListener("change", fire); }
  new MutationObserver(fire).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  const lib = { css, el, esc, text, niceTicks, fmtTick, lineChart, stepper, tabs, series, onTheme, reduceMotion, stage3d, frame, data: {} };
  return {
    lib,
    register: (name, mount) => { registry[name] = mount; },
    mountAll: (scope, data) => {
      lib.data = data || {};
      (scope || document).querySelectorAll("[data-viz]").forEach((root) => {
        const m = registry[root.dataset.viz];
        // data-viz-state lets scripts/check-figures.mjs tell a mounted figure from a silent failure.
        if (!m) { root.dataset.vizState = "unregistered"; console.error(`No visual named "${root.dataset.viz}".`); return; }
        try { m(root, lib); root.dataset.vizState = "mounted"; } catch (e) {
          root.dataset.vizState = "failed";
          console.error(e);
          root.insertAdjacentHTML("beforeend", `<p class="nogl">This figure failed to load: ${esc(e.message)}</p>`);
        }
      });
    },
  };
})();
