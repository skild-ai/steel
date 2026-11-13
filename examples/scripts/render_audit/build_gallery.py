# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Build index.html, a static gallery page, from results.json in a run dir.

usage: python3 build_gallery.py <run dir>

Every card, the summary, the changes and the skipped lists are static HTML; the script only adds filters.
Excluded (out-of-scope) examples appear only as a count.
"""

import html
import json
import sys
from pathlib import Path

ORDER = {"unstable": 0, "failed": 1, "suspect": 2, "ok": 3}
SKIP_ORDER = {"no Steel preset yet": 0, "needs Steel feature": 1}

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steel Render Audit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@500;600;700&family=Barlow+Semi+Condensed:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: summary strip, filter bar, then a responsive grid of clip cards, flagged scenes first. */
:root {
  --bg: #eef1f4; --panel: #ffffff; --fg: #1b232c; --muted: #5b6875; --line: #d3dae1; --accent: #2f5f8a;
  --ok: #2f7d4f; --ok-bg: #e2f1e8; --sus: #9a6700; --sus-bg: #fbf0d6; --bad: #b4232c; --bad-bg: #fbe3e4;
  --fail: #5b6875; --fail-bg: #e6e9ed;
  --display: "Barlow", "Helvetica Neue", Arial, sans-serif;
  --body: "Barlow Semi Condensed", "Helvetica Neue", Arial, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12171c; --panel: #1a2128; --fg: #e3e8ed; --muted: #94a2b0; --line: #2c3640; --accent: #7fb0dc;
  --ok: #6fcf97; --ok-bg: #173325; --sus: #f2c14e; --sus-bg: #3a2f12; --bad: #ff7a80; --bad-bg: #3d1a1d;
  --fail: #b4c0cc; --fail-bg: #262e36; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #12171c; --panel: #1a2128; --fg: #e3e8ed; --muted: #94a2b0; --line: #2c3640; --accent: #7fb0dc;
  --ok: #6fcf97; --ok-bg: #173325; --sus: #f2c14e; --sus-bg: #3a2f12; --bad: #ff7a80; --bad-bg: #3d1a1d;
  --fail: #b4c0cc; --fail-bg: #262e36; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.45; }
.wrap { max-width: 1320px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 48px; display: grid; gap: 22px; }
h1 { font-family: var(--display); font-weight: 700; font-size: 1.9rem; margin: 0; letter-spacing: .01em; text-wrap: balance; }
h2 { font-family: var(--display); font-weight: 600; font-size: 1.15rem; margin: 0; }
.sub { color: var(--muted); margin: 4px 0 0; }
.mono { font-family: var(--mono); font-size: .82em; }
.summary { display: flex; flex-wrap: wrap; gap: 10px 28px; align-items: baseline; border-block: 1px solid var(--line); padding-block: 14px; }
.stat { display: grid; }
.stat b { font-family: var(--display); font-size: 1.6rem; font-variant-numeric: tabular-nums; line-height: 1.1; }
.stat span { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }
.stat.ok b { color: var(--ok); } .stat.suspect b { color: var(--sus); } .stat.unstable b { color: var(--bad); } .stat.failed b { color: var(--fail); }
.meta { display: grid; gap: 2px; color: var(--muted); font-size: .9rem; min-width: 0; }
.meta code { font-family: var(--mono); font-size: .8rem; color: var(--fg); overflow-wrap: anywhere; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.filters button, .filters select { font: inherit; font-size: .9rem; border: 1px solid var(--line); background: var(--panel); color: var(--fg); border-radius: 999px; padding: 4px 12px; cursor: pointer; }
.filters select { border-radius: 6px; }
.filters button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--panel); }
.filters button:focus-visible, .filters select:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.count { color: var(--muted); font-size: .9rem; margin-left: auto; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 380px), 1fr)); gap: 16px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; display: grid; grid-template-rows: auto 1fr; min-width: 0; }
.card video, .card .novideo { width: 100%; aspect-ratio: 16 / 9; max-width: 100%; display: block; background: var(--fail-bg); object-fit: cover; }
.novideo { display: grid !important; place-items: center; color: var(--muted); font-size: .9rem; }
.body { padding: 12px 14px 14px; display: grid; gap: 8px; align-content: start; min-width: 0; }
.head { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.head h3 { font-family: var(--display); font-weight: 600; font-size: 1.05rem; margin: 0; overflow-wrap: anywhere; }
.chip { font-family: var(--display); font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; padding: 2px 8px; border-radius: 4px; }
.chip.ok { color: var(--ok); background: var(--ok-bg); } .chip.suspect { color: var(--sus); background: var(--sus-bg); }
.chip.unstable { color: var(--bad); background: var(--bad-bg); } .chip.failed { color: var(--fail); background: var(--fail-bg); }
.tag { color: var(--muted); font-size: .8rem; }
.reason { margin: 0; font-size: .92rem; overflow-wrap: anywhere; }
.reasons { margin: 0; padding-left: 18px; color: var(--muted); font-size: .86rem; display: grid; gap: 2px; }
.nums { display: flex; flex-wrap: wrap; gap: 4px 14px; font-family: var(--mono); font-size: .74rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.nums b { color: var(--fg); font-weight: 500; }
details { font-size: .85rem; color: var(--muted); }
details pre { font-family: var(--mono); font-size: .72rem; white-space: pre-wrap; overflow-wrap: anywhere; margin: 6px 0 0; color: var(--fg); }
summary { cursor: pointer; }
.skipped { display: grid; gap: 8px; }
.skiplist { columns: 3 260px; column-gap: 24px; margin: 0; padding: 0; list-style: none; font-size: .88rem; }
.skiplist li { break-inside: avoid; padding-block: 2px; color: var(--muted); }
.skiplist li b { color: var(--fg); font-weight: 500; }
.changes { display: grid; gap: 8px; }
.changes ul { margin: 0; padding-left: 18px; display: grid; gap: 3px; font-size: .9rem; }
.changes li { overflow-wrap: anywhere; }
.changes .none { color: var(--muted); margin: 0; font-size: .9rem; }
.arrow { color: var(--muted); }
[hidden] { display: none !important; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
</style>
<div class="wrap">
  <header>
    <h1>Steel Render Audit</h1>
    <p class="sub">Every example run with its Steel preset (coupled presets where Steel drives the rigid part), rendered headless and graded from stability metrics plus a visual pass over each clip. Presets are as committed; nothing was tuned.</p>
  </header>
  <section class="summary">__SUMMARY__</section>
  __CHANGES__
  <section class="filters" aria-label="Filters" id="filters" hidden></section>
  <section class="grid" id="grid">__CARDS__</section>
  <section class="skipped">
    <h2>Skipped (__NSKIP__): no Steel preset to render</h2>
    __SKIPPED__
    <p class="sub">__EXCLUDED__</p>
  </section>
</div>
<script>
// Progressive enhancement: grade/category filters and play-on-scroll over the pre-rendered cards.
(() => {
  const cards = [...document.querySelectorAll("#grid .card")];
  const flt = document.getElementById("filters");
  const cats = [...new Set(cards.map(c => c.dataset.category))].sort();
  let grade = "all", cat = "all";
  try { const s = JSON.parse(localStorage.getItem("ssra-filter") || "{}"); grade = s.grade || grade; cat = s.cat || cat; } catch (e) {}
  const gs = ["all", "flagged", "unstable", "failed", "suspect", "ok"];
  flt.innerHTML = gs.map(g => `<button type="button" aria-pressed="${g === grade}" data-g="${g}">${g}</button>`).join("")
    + `<select id="f-cat" aria-label="Category"><option value="all">all categories</option>${cats.map(c => `<option ${c === cat ? "selected" : ""}>${c}</option>`).join("")}</select><span class="count" id="count"></span>`;
  const save = () => { try { localStorage.setItem("ssra-filter", JSON.stringify({grade, cat})); } catch (e) {} };
  const apply = () => {
    let n = 0;
    cards.forEach(c => {
      const g = c.dataset.grade;
      const show = (grade === "all" || (grade === "flagged" ? g !== "ok" : g === grade)) && (cat === "all" || c.dataset.category === cat);
      c.hidden = !show; n += show;
    });
    flt.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.g === grade));
    document.getElementById("count").textContent = `${n} of ${cards.length} scenes`;
  };
  flt.querySelectorAll("button").forEach(b => b.onclick = () => { grade = b.dataset.g; save(); apply(); });
  document.getElementById("f-cat").onchange = e => { cat = e.target.value; save(); apply(); };
  flt.hidden = false;
  apply();
  if ("IntersectionObserver" in window && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) e.target.play().catch(() => {}); else e.target.pause(); }), {threshold: 0.5});
    document.querySelectorAll("#grid video").forEach(v => io.observe(v));
  }
})();
</script>
"""


def main() -> None:
    d = Path(sys.argv[1])
    r = json.loads((d / "results.json").read_text())
    gpus = ", ".join(sorted({s["gpu"] for s in r["scenes"] if s.get("gpu")})) or "unknown GPU"
    summary = "".join(f'<div class="stat {g}"><b>{r["counts"].get(g, 0)}</b><span>{g}</span></div>' for g in ORDER)
    summary += f'<div class="stat"><b>{len(r["skipped"])}</b><span>skipped</span></div>'
    newton = r.get("newton_commit") or r["newton_pin"][:9]
    dirty = " (with uncommitted changes)" if r.get("dirty") else ""
    summary += (f'<div class="meta"><div>examples <code>{esc(r["commit"][:9])}</code>{dirty} · newton <code>{esc(newton)}</code></div>'
                f'<div>{esc(r["date"])} · {r["rendered"]} rendered on {esc(gpus)}</div></div>')  # fmt: skip
    rows = sorted(r["scenes"], key=lambda s: (ORDER[s["grade"]], s["category"], s["scene"]))
    excluded = r.get("excluded", [])
    note = (f"{len(excluded)} examples outside Steel's scope (no physics solver, another solver's own feature, or cloth,"
            " soft-body, MPM and particle scenes with no rigid part for Steel) are left out; results.json lists them under excluded.") if excluded else ""  # fmt: skip
    page = (PAGE.replace("__SUMMARY__", summary).replace("__CHANGES__", changes(r.get("changes")))
            .replace("__CARDS__", "\n".join(card(s) for s in rows)).replace("__NSKIP__", str(len(r["skipped"])))
            .replace("__SKIPPED__", skipped_lists(r["skipped"])).replace("__EXCLUDED__", esc(note)))  # fmt: skip
    (d / "index.html").write_text(page)
    print(f"wrote {d / 'index.html'} ({(d / 'index.html').stat().st_size / 1e3:.0f} KB)")


def skipped_lists(skipped: list[dict]) -> str:
    """One list per skip label: "needs Steel feature" and "no Steel preset yet" stay apart."""
    groups: dict[str, list[dict]] = {}
    for s in skipped:
        groups.setdefault(s.get("label", "no Steel preset"), []).append(s)
    out = []
    for label in sorted(groups, key=lambda k: SKIP_ORDER.get(k, 9)):
        lis = "".join(
            f"<li><b>{esc(s['scene'])}</b> · {esc((s.get('detail') or s['reason'])[:90])}</li>" for s in groups[label]
        )
        out.append(f'<h3 class="tag">{esc(label)} ({len(groups[label])})</h3><ul class="skiplist">{lis}</ul>')
    return "".join(out)


def changes(c: dict | None) -> str:
    """The "Changes since <sha>" section from results.json's grade diff."""
    if not c:
        return ""
    parts = []
    for key, label in (("regressed", "Regressed"), ("improved", "Improved"), ("new", "New scenes"), ("removed", "Removed")):
        items = c.get(key, [])
        if not items:
            continue
        lis = "".join(
            f'<li><b>{esc(x["scene"])}</b> <span class="chip {esc(x.get("before") or "")}">{esc(x.get("before") or "–")}</span>'
            f'<span class="arrow"> → </span><span class="chip {esc(x.get("after", ""))}">{esc(x.get("after", "gone"))}</span>'
            f'{" · " + esc(x["reason"]) if x.get("reason") else ""}</li>'
            for x in items
        )
        parts.append(f"<h3 class=\"tag\">{label} ({len(items)})</h3><ul>{lis}</ul>")
    body = "".join(parts) or '<p class="none">No scene changed grade.</p>'
    return f'<section class="changes"><h2>Changes since <span class="mono">{esc(c["previous_commit"][:7])}</span></h2>{body}</section>'


def card(s: dict) -> str:
    f = s["files"]
    if f.get("video"):
        poster = f' poster="{esc(f["poster"])}"' if f.get("poster") else ""
        media = f'<video controls muted loop playsinline preload="metadata"{poster} src="{esc(f["video"])}" aria-label="{esc(s["scene"])} clip"></video>'
    else:
        media = f'<div class="novideo">no clip: {esc((s.get("render_error") or "render failed")[:90])}</div>'
    first, *rest = s["reasons"] or ["no flags"]
    speed = f" · {s['video_speed']:g}× speed" if s["video_speed"] > 1 else ""
    reasons = f'<ul class="reasons">{"".join(f"<li>{esc(x)}</li>" for x in rest)}</ul>' if rest else ""
    chk = s["check"]
    return f"""<article class="card" data-grade="{esc(s["grade"])}" data-category="{esc(s["category"])}"><div>{media}</div><div class="body">
      <div class="head"><span class="chip {esc(s["grade"])}">{esc(s["grade"])}</span><h3>{esc(s["scene"])}</h3><span class="tag">{esc(s["category"])} · [{esc(s["preset"])}]{speed}</span></div>
      <p class="reason">{esc(first)}</p>
      {reasons}
      <div class="nums">{nums(s["metrics"])}</div>
      <details><summary>preset and check</summary><pre>{esc(json.dumps(s["settings"], indent=1))}
check: {esc(chk.get("result") or "none")} {esc(chk.get("note") or "")}
auto grade: {esc(s["auto_grade"])} · {esc(s["short"])}</pre></details>
    </div></article>"""


def nums(m: dict) -> str:
    if not m or m.get("fps") is None:
        return ""
    it = [("v", f"{fmt(m.get('max_lin_speed'))} m/s"), ("ω", f"{fmt(m.get('max_ang_speed'))} rad/s"),
          ("pen p95", f"{fmt((m.get('penetration_p95_m') or 0) * 1000)} mm"), ("pen peak", f"{fmt((m.get('penetration_peak_m') or 0) * 1000)} mm"),
          ("step", f"{fmt(m.get('step_ms_mean'), 2)} ms"), ("bodies", m.get("bodies")), ("contacts", m.get("contacts_max") or 0)]  # fmt: skip
    if m.get("particles"):
        it.append(("particles", m["particles"]))
    if m.get("overflow_frames"):
        it.append(("overflow frames", m["overflow_frames"]))
    if m.get("bodies_inside_static"):
        it.append(("inside static", m["bodies_inside_static"]))
    return "".join(f"<span>{k} <b>{esc(v)}</b></span>" for k, v in it)


def fmt(v, d: int = 1) -> str:
    if v is None:
        return "–"
    if isinstance(v, (int, float)):
        return f"{v:.0f}" if abs(v) >= 100 else f"{v:.{d}f}"
    return str(v)


def esc(v) -> str:
    return html.escape("" if v is None else str(v), quote=True)


if __name__ == "__main__":
    main()
