// SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
// SPDX-License-Identifier: Apache-2.0

/* Determinism: the largest pose difference between two runs from identical inputs, as Steel measured it. */
SteelViz.register("det", (root, lib) => {
  const { css } = lib;
  const f = lib.frame(root, `
    <svg class="vizsvg" data-role="chart" role="img" aria-label="Largest position difference between two runs of a 40-object pour, log scale: default ordering grows to about 20 cm, RUN_TO_RUN stays at zero."></svg>
    <div class="legend" data-role="legend"></div>`);
  const chart = f.querySelector("[data-role=chart]");
  function draw() {
    if (!lib.data.determinism) return;
    lib.lineChart(chart, {
      h: 210, log: true, ymin: 1e-3, ylabel: "largest position difference between run 1 and run 2 [mm] · measured with Steel",
      series: lib.series(lib.data, "determinism", "divergence_mm", [css("--series-1"), css("--series-2")], { labels: ["default: atomic row order", "RUN_TO_RUN: 0 at every frame, bitwise identical"] }),
      legend: f.querySelector("[data-role=legend]"),
    });
  }
  lib.onTheme(draw);
  draw();
});
