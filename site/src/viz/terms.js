// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
// SPDX-License-Identifier: Apache-2.0

/* Concept bubbles: <button class="term" data-viz="term" data-term="warp">warp</button>.
   One shared popover; tap or click opens it, Escape, the close button or a click outside closes it. */
SteelViz.register("term", (root, lib) => {
  const TERMS = {
    "warp": ["Warp", "A group of 32 GPU threads that execute together. Steel's serial sweep gives each world one warp: the world's rows run one after another, and the 32 lanes share the arithmetic of each row, such as the dot product J·v.",
      '<rect x="8" y="22" width="224" height="22" rx="3" class="tp-box"/>' + Array.from({ length: 32 }, (_, i) => `<rect x="${10 + i * 7}" y="25" width="5" height="16" class="${i < 32 ? "tp-hot" : ""}"/>`).join("") + '<text x="8" y="62" class="tp-t">32 lanes, one world, rows in order</text>'],
    "block": ["Thread block and SM", "A thread block is a group of warps scheduled together on one streaming multiprocessor (SM). An SM can hold several blocks at once when registers and shared memory allow, and a block's resources are freed for new blocks once all of its warps finish.",
      [0, 1, 2].map((k) => `<rect x="${8 + k * 78}" y="12" width="70" height="44" rx="4" class="tp-box"/><text x="${14 + k * 78}" y="26" class="tp-t">SM ${k}</text>` + [0, 1, 2].map((j) => `<rect x="${14 + k * 78}" y="${31 + j * 7}" width="${[52, 30, 44][(j + k) % 3]}" height="5" class="${j === 0 ? "tp-hot" : "tp-acc"}"/>`).join("")).join("") + '<text x="8" y="72" class="tp-t">blocks of warps on each multiprocessor</text>'],
    "smem": ["Shared memory", "Fast on-chip memory shared by the threads of one block, much faster than global GPU memory. Steel keeps a world's velocity vector in it during the sweep, and for small worlds the whole row block.",
      '<rect x="8" y="10" width="110" height="46" rx="4" class="tp-box"/><text x="14" y="24" class="tp-t">SM</text><rect x="14" y="30" width="98" height="18" class="tp-hot"/><text x="18" y="43" class="tp-t2">shared: v</text><rect x="140" y="10" width="92" height="46" rx="4" class="tp-box"/><text x="146" y="24" class="tp-t">global memory</text><text x="146" y="44" class="tp-t">J, Y, contacts</text><path d="M118 39 H138" class="tp-line"/>'],
    "launch": ["Kernel launch", "One call that starts a GPU function over a grid of threads. Each launch costs CPU time; a step made of many small launches can spend more time launching than computing.",
      '<path d="M8 30 H232" class="tp-line"/>' + [0, 1, 2, 3, 4].map((k) => `<rect x="${12 + k * 44}" y="22" width="${10 + (k % 2) * 8}" height="16" class="tp-hot"/>`).join("") + '<text x="8" y="58" class="tp-t">kernels with launch gaps between them</text>'],
    "graph": ["CUDA graph", "A recorded sequence of GPU work that is replayed with one call. Replay skips Python and per-launch CPU cost. The recorded kernels, arrays and launch arguments stay fixed until you capture again; values inside the arrays can change.",
      '<path d="M8 30 H232" class="tp-line"/><rect x="12" y="22" width="150" height="16" class="tp-hot"/>' + [1, 2, 3, 4].map((k) => `<path d="M${12 + k * 30} 22 V38" class="tp-sep"/>`).join("") + '<text x="8" y="58" class="tp-t">one launch replays the whole step</text>'],
    "gs": ["Gauss-Seidel and Jacobi", "Both update constraint rows iteratively. Jacobi updates every row from the previous iterate, all in parallel. Gauss-Seidel updates rows one at a time, and each row sees the impulses the rows before it just computed, so it typically converges in fewer sweeps but is sequential. Steel keeps Gauss-Seidel inside a world and runs worlds in parallel.",
      '<text x="8" y="16" class="tp-t">Jacobi</text>' + [0, 1, 2, 3].map((k) => `<rect x="${70 + k * 40}" y="6" width="30" height="14" class="tp-acc"/>`).join("") + '<text x="8" y="50" class="tp-t">Gauss-Seidel</text>' + [0, 1, 2, 3].map((k) => `<rect x="${70 + k * 40}" y="40" width="30" height="14" class="${k === 1 ? "tp-hot" : "tp-acc"}"/>` + (k < 3 ? `<path d="M${100 + k * 40} 47 H${110 + k * 40}" class="tp-line"/>` : "")).join("")],
    "delassus": ["Delassus matrix", "A = J H⁻¹ Jᵀ: how an impulse on one constraint row changes the velocity of every other row, through the bodies' inertia. Its diagonal is each row's inverse effective mass. The general matrix-free sweep applies row responses through the shared velocity instead of assembling a world-wide A. Small-world kernels can build a local dense response in shared memory, and coupled friction updates use small row blocks.",
      Array.from({ length: 5 }, (_, i) => Array.from({ length: 5 }, (_, j) => `<rect x="${60 + j * 14}" y="${4 + i * 14}" width="12" height="12" class="${i === j ? "tp-hot" : Math.abs(i - j) === 1 ? "tp-acc" : "tp-box"}"/>`).join("")).join("") + '<text x="140" y="40" class="tp-t">rows × rows</text>'],
    "lcp": ["Complementarity", "For a hard normal row, the normal impulse and the biased velocity residual uₙ + bₙ are both non-negative, and their product is zero. The bias sets the allowed approach or required separation speed for speculative contacts, penetration correction and restitution. Projected Gauss-Seidel solves such complementarity conditions one row at a time by clamping each impulse.",
      '<path d="M30 60 V10 M30 60 H200" class="tp-line"/><path d="M30 60 H200" class="tp-hot-line"/><path d="M30 60 V10" class="tp-hot-line"/><text x="36" y="16" class="tp-t">λ (push)</text><text x="96" y="54" class="tp-t">biased velocity residual</text>'],
    "cone": ["Coulomb cone", "Friction can be at most μ times the normal force, ‖λₜ‖ ≤ μ λₙ. All allowed contact impulses form a cone around the normal; at a fixed normal impulse its cross-section is the friction disk Steel projects onto.",
      '<path d="M120 70 L70 14 M120 70 L170 14" class="tp-line"/><ellipse cx="120" cy="14" rx="50" ry="8" class="tp-hot-fill"/><path d="M120 70 V4" class="tp-line"/><text x="176" y="40" class="tp-t">μλₙ</text>'],
    "crba": ["Featherstone and CRBA", "Algorithms for articulated rigid bodies in joint coordinates (Featherstone, Rigid Body Dynamics Algorithms, 2014). The composite rigid body algorithm (CRBA) builds the joint-space mass matrix H(q) in one pass from the leaves to the root; inverse dynamics gives the bias forces.",
      '<circle cx="40" cy="60" r="6" class="tp-hot-fill"/><path d="M40 60 L80 36 L120 46 M80 36 L110 14" class="tp-line"/><circle cx="80" cy="36" r="5" class="tp-acc-fill"/><circle cx="120" cy="46" r="5" class="tp-acc-fill"/><circle cx="110" cy="14" r="5" class="tp-acc-fill"/><text x="140" y="40" class="tp-t">tree → H(q)</text>'],
    "chol": ["Cholesky factorization", "Writes a symmetric positive definite matrix as H = L Lᵀ with L lower triangular. Solving H x = b then takes one forward and one backward substitution; Steel uses it for the free velocity and each row's response H⁻¹Jᵀ.",
      Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => `<rect x="${30 + j * 14}" y="${8 + i * 14}" width="12" height="12" class="${j <= i ? "tp-hot" : "tp-box"}"/>`).join("")).join("") + '<text x="96" y="40" class="tp-t">L</text><text x="120" y="40" class="tp-t">× Lᵀ = H</text>'],
  };
  const t = TERMS[root.dataset.term];
  if (!t) return;
  root.setAttribute("aria-haspopup", "dialog");
  root.setAttribute("aria-expanded", "false");
  let pop = document.getElementById("termpop");
  if (!pop) {
    pop = document.createElement("div");
    pop.id = "termpop"; pop.setAttribute("role", "dialog"); pop.setAttribute("aria-labelledby", "termpop-t"); pop.hidden = true;
    pop.innerHTML = '<div class="tp-h"><b id="termpop-t"></b><button type="button" class="tp-x" aria-label="Close">×</button></div><div class="tp-svg"></div><p class="tp-text"></p>';
    document.body.appendChild(pop);
    const close = () => {
      if (pop.hidden) return;
      pop.hidden = true;
      const from = pop.__from; pop.__from = null;
      if (from) { from.setAttribute("aria-expanded", "false"); from.focus(); }
    };
    pop.__close = close;
    pop.querySelector(".tp-x").addEventListener("click", close);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
    document.addEventListener("click", (e) => { if (!pop.hidden && !e.target.closest("#termpop") && !e.target.closest(".term")) close(); });
  }
  root.setAttribute("aria-controls", "termpop");
  root.addEventListener("click", (e) => {
    e.preventDefault();
    if (!pop.hidden && pop.__from === root) return pop.__close();
    if (pop.__from) pop.__from.setAttribute("aria-expanded", "false");
    pop.querySelector("#termpop-t").textContent = t[0];
    pop.querySelector(".tp-text").textContent = t[1];
    pop.querySelector(".tp-svg").innerHTML = `<svg viewBox="0 0 240 80" aria-hidden="true">${t[2]}</svg>`;
    pop.hidden = false; pop.__from = root; root.setAttribute("aria-expanded", "true");
    const sheet = innerWidth < 600;
    pop.classList.toggle("sheet", sheet);
    if (sheet) { pop.style.left = ""; pop.style.top = ""; pop.style.width = ""; }
    else {
      const r = root.getBoundingClientRect(), w = Math.min(340, innerWidth - 32);
      pop.style.width = w + "px";
      const left = Math.max(16, Math.min(innerWidth - w - 16, r.left));
      let top = r.bottom + 8;
      const hgt = pop.offsetHeight;
      if (top + hgt > innerHeight - 8) top = Math.max(8, r.top - hgt - 8);
      pop.style.left = left + scrollX + "px"; pop.style.top = top + scrollY + "px";
    }
    pop.querySelector(".tp-x").focus();
  });
});
