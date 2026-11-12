// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

/* Sleeping islands: a schematic of four towers going to sleep and one waking, with the awake count Steel measured. */
SteelViz.register("sleep", (root, lib) => {
  const { css, el, text } = lib;
  const f = lib.frame(root, `
    <div class="vizbar"><span class="stepbtns"><button type="button" class="btn" data-act="prev" aria-label="Previous">◀</button><button type="button" class="btn" data-act="play">Play</button><button type="button" class="btn" data-act="next" aria-label="Next">▶</button></span></div>
    <svg class="vizsvg" data-role="scene" viewBox="0 0 640 170" role="img" aria-label="Four towers of five boxes and a ball; awake bodies in one color, sleeping bodies in another."></svg>
    <p class="readout" aria-live="polite"></p>
    <svg class="vizsvg" data-role="chart" role="img" aria-label="Measured number of awake bodies over 10 seconds."></svg>`);
  const scene = f.querySelector("[data-role=scene]"), chart = f.querySelector("[data-role=chart]"), out = f.querySelector(".readout");
  const STEPS = [
    { t: 0.4, asleep: [], wake: false, text: "Islands are rebuilt every step from contacts. Each tower is one island; the ground supports it without joining it. The towers are still settling." },
    { t: 0.8, asleep: [0], wake: false, text: "An island falls asleep once every body stays below 0.05 m/s and 0.15 rad/s for 0.5 s; islands settle one at a time. A sleeping island's state is frozen and its rows and dynamics are skipped." },
    { t: 3.0, asleep: [0, 1, 2, 3], ball: true, wake: false, text: "By about 2.7 s every body is asleep, the ball included. Collision detection keeps running, so wake events are not missed." },
    { t: 7.2, asleep: [0, 2, 3], wake: true, text: "At 7 s a ball is pushed into the second tower. The contact with a moving body wakes that island only; the other three stay asleep." },
  ];
  function render(i) {
    const s = STEPS[i], awakeC = css("--series-2"), sleepC = css("--sleep");
    scene.innerHTML = "";
    scene.appendChild(el("line", { x1: 10, x2: 630, y1: 160, y2: 160, stroke: css("--static"), "stroke-width": 3 }));
    [110, 250, 390, 530].forEach((x, k) => {
      const asleep = s.asleep.includes(k), fallen = s.wake && k === 1;
      for (let j = 0; j < 5; j++) {
        const dx = fallen ? j * 14 + 20 : 0, dy = fallen ? -j * 4 : 0, rot = fallen ? 12 * j : 0;
        scene.appendChild(el("rect", { x: x - 18 + dx, y: 134 - j * 26 - dy + (fallen ? j * 22 : 0), width: 36, height: 24, rx: 2, fill: asleep ? sleepC : awakeC, opacity: asleep ? 0.85 : 1, transform: `rotate(${rot} ${x + dx} ${146 - j * 26})` }));
      }
      text(scene, x, 14, asleep ? "asleep" : "awake", { "text-anchor": "middle", fill: asleep ? sleepC : awakeC });
    });
    const bx = s.wake ? 200 : 40;
    scene.appendChild(el("circle", { cx: bx, cy: 146, r: 13, fill: s.ball ? sleepC : awakeC }));
    out.textContent = `t = ${s.t} s. ${s.text}`;
    if (lib.data.sleeping) {
      lib.lineChart(chart, { h: 170, ylabel: "awake bodies of 21 · measured with Steel", series: lib.series(lib.data, "sleeping", "awake_bodies", [css("--series-2")]), ymin: 0, cursor: s.t });
    }
  }
  const ctl = lib.stepper(f, STEPS.length, render);
  lib.onTheme(() => ctl.redraw());
});
