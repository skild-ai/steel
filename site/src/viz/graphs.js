/* CUDA graphs: launch-by-launch stepping against one captured replay, and what stays outside capture. Schematic. */
SteelViz.register("graphs", (root, lib) => {
  const { css, el, text } = lib;
  const f = lib.frame(root, `
    <div class="vizbar"><span class="seg" data-role="mode"><button type="button" class="on" data-v="eager">eager steps</button><button type="button" data-v="graph">captured graph</button></span></div>
    <svg class="vizsvg" viewBox="0 0 640 150" role="img" aria-label="Timeline of CPU launches and GPU kernels for one frame, eager or replayed from a captured graph."></svg>
    <p class="readout" aria-live="polite"></p>`);
  const svg = f.querySelector("svg"), out = f.querySelector(".readout");
  const K = [14, 6, 22, 9, 30, 8, 12, 18, 7, 26, 10, 16]; // schematic kernel lengths
  let mode = "eager";
  function draw() {
    svg.innerHTML = "";
    const hot = css("--hot"), s1 = css("--series-1"), fg = css("--fg");
    text(svg, 0, 46, "CPU", { fill: fg }); text(svg, 0, 106, "GPU", { fill: fg });
    [40, 100].forEach((y) => svg.appendChild(el("line", { x1: 40, x2: 630, y1: y, y2: y, stroke: css("--rule") })));
    let cpu = 44, gpu = 44;
    if (mode === "eager") {
      K.forEach((k) => {
        svg.appendChild(el("rect", { x: cpu, y: 32, width: 16, height: 16, fill: s1 }));
        const start = Math.max(gpu, cpu + 16);
        svg.appendChild(el("rect", { x: start, y: 92, width: k, height: 16, fill: hot }));
        cpu += 22; gpu = start + k + 2;
      });
      text(svg, 44, 140, "each kernel waits for its own launch; the GPU idles between short kernels", { "font-size": 10.5 });
    } else {
      svg.appendChild(el("rect", { x: cpu, y: 32, width: 20, height: 16, fill: s1 }));
      gpu = 70;
      K.forEach((k) => { svg.appendChild(el("rect", { x: gpu, y: 92, width: k, height: 16, fill: hot })); gpu += k + 2; });
      text(svg, 44, 140, "one replay call; the recorded kernels run back to back", { "font-size": 10.5 });
    }
    out.innerHTML = mode === "eager"
      ? "Eager stepping launches every kernel from Python. Small scenes are often limited by launch cost rather than by the GPU."
      : "A captured graph replays the whole frame with one call. Kept outside capture: <code>check_constraint_capacity()</code>, <code>constraint_row_watermarks()</code>, compliance counts, body-flag notifications and sleeping host checks, host torsion preparation (use <code>contact_torsion_device=True</code>), and anything that changes array objects or construction-time options.";
  }
  lib.tabs(f.querySelector("[data-role=mode]"), (v) => { mode = v; draw(); });
  lib.onTheme(draw);
  draw();
});
