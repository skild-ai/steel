#!/usr/bin/env python3
"""Regenerate CHANGELOG.md (the repository root) for skild-ai/steel from upstream Newton and the Steel integration branch.

Sources:
  * newton-physics/newton: merged PRs whose squash commit on main touches Steel paths, plus merged
    PRs with "Steel" in the title; open PRs that touch Steel paths ("In review upstream").
  * skild-ai/newton main (the Steel integration branch): first-parent commits that still own, by
    blame, at least FORK_MIN_LINES of the branch's Steel diff against upstream main.

Needs a Newton clone with the upstream remote fetched and the integration head available, and the
GitHub CLI (``gh api``) for PR metadata. Responses are cached in ``.changelog_cache.json`` next to
this script; pass ``--refresh`` to refetch.

    python gen_changelog.py --repo path/to/newton --upstream upstream/main --release v1.8.0 \
        --branch skild/main --out CHANGELOG.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = "newton-physics/newton"
FORK_SLUG = "skild-ai/newton"  # author lookup for integration-branch commits
FORK_LINK = "https://github.com/skild-ai/newton/commit/{sha}"
STEEL_PATHS = [
    "newton/_src/solvers/steel",
    "newton/tests/steel",
    "newton/examples/steel",
    "newton/examples/robot/example_steel_bolt_sweep.py",
    "newton/examples/robot/example_steel_cup_unpacking.py",
    "newton/examples/robot/example_steel_pelican_bicycle.py",
    "newton/examples/robot/example_steel_plate_rack.py",
    "newton/examples/robot/example_steel_toy_well.py",
    "newton/examples/robot/example_steel_wrecking_ball.py",
    "newton/examples/robot/_plate_rack.py",
    "newton/examples/robot/_wrecking_crane.py",
]
SOLVER_DIR = "newton/_src/solvers/steel"
SINCE = "2026-10-01"
FORK_MIN_LINES = 10

AREAS = [
    "Solver core",
    "Contacts and friction",
    "Joints and drives",
    "Performance",
    "API and observables",
    "Coupling",
    "Fixes",
]

# Fork titles are checked against the regexes of --scrub-patterns (one per line) and against absolute paths.
# A matching title must be rewritten in REWRITE or listed in SKIP.
PATH_LIKE = re.compile(r"(^|\s)/[\w.-]+/")

# Fork commits that are housekeeping or already upstream under another title (sha prefix -> reason).
SKIP = {
    "e579b7aab": "solver rename before the upstream proposal",
    "c42985a6c": "test bookkeeping after merging upstream",
    "d28fe04f4": "re-land of beb9c0881; upstream as #5229",
    "beb9c0881": "upstream as #5229",
    "397c508f7": "test coverage only",
    "d2943dae3": "test fixture only",
    "c1615774d": "dead-code removal",
    "6c672c9e7": "upstream in Newton 1.8 (serial and colored solve modes)",
    "59e0c6d36": "upstream as #5204",
    "b69ca7283": "upstream as #4999",
    "13afbf739": "review changes of #4522, upstream",
    "a193ad082": "upstream as #4824",
    "28b5dab32": "upstream as #5006",
    "37b514731": "upstream as #5070",
    "4accdd372": "articulated torsion is upstream (#4522); its rolling rows continue in 14cf39b77",
}

# Plain rewrites for fork titles (sha prefix -> text), applied before the scrub check.
REWRITE = {
    "2e4d770af": "Cut single-world latency for large articulations",
    "6516b18e5": "Add multi-DOF warp tree kernels, cooperative overlap refinement and a one-launch latency mode",
    "f6fc82a8b": "Add contact adhesion through the per-shape steel:adhesion attribute",
    "baffc2abb": "Keep carried contacts matched and add a rotating color order (propagation_color_order=\"rotating\")",
    "8776ff940": "Compile the row phase into the fused matrix-free Gauss-Seidel kernel",
    "33f6fe76c": "Prebuild the matrix-free Gauss-Seidel phase kernels the propagation schedule launches",
    "4ba4f149e": "Add interactive tabletop examples",
    "14cf39b77": "Add experimental rolling and spin friction rows for free-body contacts (steel:condim 4 and 6)",
    "b642e9313": "Solve free-body rolling and spin rows on the propagation response",
    "c9820fbc3": "Support free-body rolling and spin rows on colored propagation",
    "3ba3ce9d4": "Keep shapes with different condim in separate body-pair reduction groups",
    "7b4994403": "Apply rigid-body linear damping (steel:rigid_body_linear_damping)",
}

AREA_OVERRIDE: dict[str, str] = {
    "f6fc82a8b": "Contacts and friction",
    "#5201": "Performance",
    "#5200": "Performance",
    "#5175": "Performance",
    "#5176": "Performance",
    "#5153": "Performance",
    "#5204": "Fixes",
    "#5118": "API and observables",
    "#5072": "API and observables",
    "#4881": "API and observables",
    "#5117": "Contacts and friction",  # "coupled" here is the coupled normal-friction solve
    "#4907": "Contacts and friction",
    "#4993": "Joints and drives",
    "#4760": "Joints and drives",
    "#4885": "Solver core",
    "#5113": "Solver core",
}

# Upstream PRs that only touch tests or tooling.
UPSTREAM_SKIP = {5230, 4962}

# Plain rewrites for upstream titles whose user-visible change the title does not name.
UPSTREAM_REWRITE = {
    5113: "Accept same-articulation propagation rows with the colored sweep",
}

AREA_RULES = [
    ("Coupling", r"coupl|\bvbd\b|\bxpbd\b|proxy|admm|cloth|soft"),
    ("API and observables", r"observable|body_parent_f|contact_f|warning|deprecat|rename|attribute|watermark|"
     r"notify|\bapi\b|namespace|option|docs?\b|document|example"),
    ("Contacts and friction", r"contact|friction|torsion|restitution|anchor|patch|adhesion|compliance|condim|"
     r"rolling|spin|speculative|surface velocity|combine"),
    ("Joints and drives", r"joint|drive|limit|mimic|loop|weld|closure|damping|armature|spring|\bd6\b|gravity"),
    ("Performance", r"latency|launch|kernel|cache|parallel|shared.memory|faster|speed|tiled|compile|prebuild|"
     r"sparse|fuse|fused|throughput|one body per thread|stream|cost"),
    ("Fixes", r"^fix|\bfix\b|keep |guard|correct|restore|avoid|prevent|handle|stale|crash|nan|overflow|bound"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="Newton clone with the upstream remote and the branch head")
    ap.add_argument("--scrub-patterns", help="file of regexes, one per line, that must not appear in branch titles")
    ap.add_argument("--upstream", default="upstream/main")
    ap.add_argument("--release", default="v1.8.0")
    ap.add_argument("--branch", default="baffc2abb2")
    ap.add_argument("--out", default=str(HERE.parent.parent / "CHANGELOG.md"))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cache_path = HERE / ".changelog_cache.json"
    cache = {} if args.refresh or not cache_path.exists() else json.loads(cache_path.read_text())

    upstream = collect_upstream(args, cache)
    open_prs = collect_open(cache)
    scrub = []
    if args.scrub_patterns:
        lines = Path(args.scrub_patterns).read_text().splitlines()
        scrub = [re.compile(line, re.I) for line in lines if line.strip() and not line.startswith("#")]
    fork = collect_fork(args, cache, upstream, scrub)
    cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))

    text = render(upstream, open_prs, fork, args)
    Path(args.out).write_text(text)
    n_up = len(upstream)
    n_rel = sum(1 for e in upstream if e["released"])
    print(f"upstream merged: {n_up} ({n_rel} in {args.release}), open upstream: {len(open_prs)}, "
          f"integration branch only: {len(fork)} -> {args.out}")


def collect_upstream(args, cache) -> list[dict]:
    log = git(args.repo, "log", "--first-parent", f"--since={SINCE}", "--format=%H%x09%s", args.upstream, "--", *STEEL_PATHS)
    numbers: dict[int, str] = {}
    for line in log.splitlines():
        sha, subject = line.split("\t", 1)
        m = re.search(r"\(#(\d+)\)\s*$", subject)
        if m:
            numbers[int(m.group(1))] = sha
    for item in search(cache, f"repo:{UPSTREAM} is:pr is:merged Steel in:title merged:>={SINCE}"):
        numbers.setdefault(item["number"], "")
    # The release branch cherry-picks from main, so membership is by PR number in the release history.
    released = {int(n) for n in re.findall(r"\(#(\d+)\)", git(args.repo, "log", "--format=%s", args.release))}
    entries = []
    for number in sorted(numbers):
        pr = gh_cached(cache, f"repos/{UPSTREAM}/pulls/{number}")
        if number in UPSTREAM_SKIP or not pr.get("merged_at") or pr["base"]["ref"] != "main":
            continue
        sha = numbers[number] or pr.get("merge_commit_sha") or ""
        if "steel" not in pr["title"].lower() and not touches_solver(args.repo, sha):
            continue
        entries.append({
            "title": UPSTREAM_REWRITE.get(number) or clean_title(pr["title"]),
            "number": number,
            "url": pr["html_url"],
            "author": pr["user"]["login"],
            "date": pr["merged_at"][:10],
            "released": number in released,
            "source": "upstream",
            "key": f"#{number}",
        })
    return entries


def collect_open(cache) -> list[dict]:
    out = []
    for item in search(cache, f"repo:{UPSTREAM} is:pr is:open steel"):
        files = gh_cached(cache, f"repos/{UPSTREAM}/pulls/{item['number']}/files?per_page=100")
        touches = any(f["filename"].startswith(SOLVER_DIR) for f in files)
        if not touches and "steel" not in item["title"].lower():
            continue
        out.append({
            "title": clean_title(item["title"]),
            "number": item["number"],
            "url": item["html_url"],
            "author": item["user"]["login"],
            "date": item["created_at"][:10],
        })
    return sorted(out, key=lambda e: -e["number"])


def collect_fork(args, cache, upstream, scrub) -> list[dict]:
    """Branch commits that still own at least FORK_MIN_LINES of the upstream..branch diff."""
    owned = fork_only_lines(args.repo, args.upstream, args.branch)
    log = git(args.repo, "log", "--first-parent", f"--since={SINCE}", "--format=%H%x09%ad%x09%s",
              "--date=short", args.branch, f"^{args.upstream}")
    out = []
    for line in log.splitlines():
        sha, date, subject = line.split("\t", 2)
        if sha[:9] in SKIP or re.match(r"merge ", subject, re.I) or owned.get(sha, 0) < FORK_MIN_LINES:
            continue
        title = REWRITE.get(sha[:9]) or clean_title(subject)
        if PATH_LIKE.search(title) or any(p.search(title) for p in scrub):
            raise SystemExit(f"{sha[:9]} needs a REWRITE or SKIP entry: {subject}")
        commit = gh_cached(cache, f"repos/{FORK_SLUG}/commits/{sha}")
        login = (commit.get("author") or {}).get("login") or commit["commit"]["author"]["name"]
        out.append({
            "title": title,
            "sha": sha,
            "url": FORK_LINK.format(sha=sha),
            "author": login,
            "date": date,
            "source": "branch",
            "key": sha[:9],
        })
    return out


def fork_only_lines(repo: str, upstream: str, branch: str) -> dict[str, int]:
    """Count, per first-parent commit, the lines of the branch that differ from upstream (blame)."""
    counts: dict[str, int] = defaultdict(int)
    files = git(repo, "diff", "--name-only", upstream, branch, "--", *STEEL_PATHS, "newton/examples/robot").split()
    for path in files:
        if not git(repo, "ls-tree", branch, "--", path).strip():
            continue
        hunks = git(repo, "diff", "-U0", upstream, branch, "--", path)
        lines = set()
        for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", hunks, re.M):
            start, n = int(m.group(1)), int(m.group(2) or 1)
            lines.update(range(start, start + n))
        if not lines:
            continue
        blame = git(repo, "blame", "--first-parent", "-l", "-s", branch, "--", path)
        for i, row in enumerate(blame.splitlines(), 1):
            if i in lines:
                counts[row.lstrip("^")[:40]] += 1
    return counts


def render(upstream, open_prs, fork, args) -> str:
    lines = [
        "# Changelog",
        "",
        "Steel changes, newest first, grouped by month and area. Steel is `newton.solvers.SolverSteel` in",
        "[Newton](https://github.com/newton-physics/newton).",
        "",
        "- **Upstream** entries are merged pull requests on newton-physics/newton. Unmarked entries shipped in",
        f"  Newton {args.release.lstrip('v')}; entries marked *(Newton main)* are on main after that release.",
        "- **Integration branch** entries are commits on [skild-ai/newton](https://github.com/skild-ai/newton)",
        "  `main` that are not upstream yet. They reach Newton through upstream pull requests.",
        "",
        f"Generated by `gen_changelog.py` on {dt.date.today().isoformat()}.",
        "",
        "## In review upstream",
        "",
    ]
    for e in open_prs:
        lines.append(f"- {e['title']}. [#{e['number']}]({e['url']}) by @{e['author']}")
    lines += [
        "- MuJoCo-style tendons for SolverSteel, in review on the integration branch.",
        "",
    ]
    by_month: dict[str, list[dict]] = defaultdict(list)
    for e in upstream + fork:
        by_month[e["date"][:7]].append(e)
    for month in sorted(by_month, reverse=True):
        name = dt.date.fromisoformat(month + "-01").strftime("%B %Y")
        lines += [f"## {name}", ""]
        by_area: dict[str, list[dict]] = defaultdict(list)
        for e in by_month[month]:
            by_area[area_of(e)].append(e)
        for area in AREAS:
            if not by_area.get(area):
                continue
            lines += [f"### {area}", ""]
            for e in sorted(by_area[area], key=lambda e: (e["date"], e["key"]), reverse=True):
                lines.append(entry_line(e, args))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def entry_line(e, args) -> str:
    if e["source"] == "upstream":
        tag = "" if e["released"] else " *(Newton main)*"
        return f"- {e['title']}. [#{e['number']}]({e['url']}) by @{e['author']}{tag}"
    return f"- {e['title']}. [`{e['sha'][:9]}`]({e['url']}) by @{e['author']} *(integration branch)*"


def area_of(e) -> str:
    if e["key"] in AREA_OVERRIDE:
        return AREA_OVERRIDE[e["key"]]
    title = e["title"].lower()
    for area, pattern in AREA_RULES:
        if re.search(pattern, title):
            return area
    return "Solver core"


def clean_title(title: str) -> str:
    title = re.sub(r"\s*\(#\d+\)\s*$", "", title.strip())
    title = re.sub(r"^\[?steel\]?:\s*", "", title, flags=re.I)
    title = title.replace("MF-GS", "matrix-free Gauss-Seidel")
    return title[:1].upper() + title[1:].rstrip(".")


def norm(title: str) -> str:
    title = re.sub(r"\b(steel|solversteel|steel's|in|to|the|a|an)\b", " ", title.lower())
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def search(cache, q: str) -> list[dict]:
    items, page = [], 1
    while True:
        res = gh_cached(cache, f"search/issues?q={q}&per_page=100&page={page}", raw_query=True)
        items += res.get("items", [])
        if len(res.get("items", [])) < 100:
            return items
        page += 1


def gh_cached(cache, path: str, raw_query: bool = False):
    if path in cache:
        return cache[path]
    if raw_query:
        base, query = path.split("?", 1)
        cmd = ["gh", "api", "-X", "GET", base]
        for part in query.split("&"):
            k, v = part.split("=", 1)
            cmd += ["-f", f"{k}={v}"]
    else:
        cmd = ["gh", "api", path]
    env = dict(os.environ, GH_HOST="github.com")
    cache[path] = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout)
    return cache[path]


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


def touches_solver(repo: str, sha: str) -> bool:
    if not sha:
        return False
    names = git(repo, "show", "--format=", "--name-only", sha)
    return any(n.startswith(SOLVER_DIR) for n in names.splitlines())



if __name__ == "__main__":
    main()
