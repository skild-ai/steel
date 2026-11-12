"""Charts for the Steel tuning PR, from the Solver Tuning Book's evidence log (evidence.json).

usage: python charts.py OUT_DIR  -> frame_time_{light,dark}.png, summary_{light,dark}.png, data.csv
"""

import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
rows = json.load(open(Path(__file__).with_name("evidence.json")))[1:]

GROUPS = {
    "Basics": "basic_pendulum basic_dzhanibekov basic_shapes basic_joints basic_urdf selection_cartpole sleeping "
    "basic_mimic_joint",
    "Contacts": "balance_bird basic_conveyor basic_conveyor_forces newton_cradle domino_spiral pyramid "
    "vbd_rigid_rigid_contact nut_bolt_hydro nut_bolt_sdf robot_panda_hydro",
    "Articulations": "controller_operational_space_hybrid_force_motion robot_allegro_hand robot_anymal_c_walk "
    "robot_anymal_d robot_asroballet robot_cartpole robot_g1 robot_omniwheel robot_policy robot_ur10 "
    "sensor_contact",
    "Linkages and soft": "kamino_basic_dr_testmech kamino_basic_fourbar kamino_robot_dr_legs kamino_robot_anymal_d "
    "rigid_soft_contact softbody_franka franka_cable_ik_pick_place mpm_anymal",
    "Coupling": "franka_cloth_block franka_shirt_fold_stack kamino_mujoco_admm_solver mpm_twoway_coupling "
    "mujoco_franka_vbd_cable_admm_solver mujoco_mpm_coupled_solver mujoco_vbd_admm_solver "
    "mujoco_vbd_coupled_solver proxy_joint_gripper admm_contact_solver",
}
group_of = {ex: g for g, names in GROUPS.items() for ex in names.split()}
BASELINE_FIX = {"basic_dzhanibekov": "VBD"}  # the evidence log lists XPBD; its default before PR #1 was VBD


def num(s):
    s = s.strip()
    m = re.match(r"([\d.]+)\s*[–-]\s*([\d.]+)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    return float(re.match(r"[\d.]+", s).group(0))


data = []
for r in rows:
    ex = r[0].split(" (")[0]
    base, rest = r[2].split("→")
    tuned, template = rest.split("(")
    data.append(
        dict(example=ex, label=r[0], group=group_of[ex], baseline_solver=BASELINE_FIX.get(ex, r[1]),
             baseline=num(base), tuned=num(tuned), template=num(template))
    )
assert len(data) == len(rows) and all(d["group"] for d in data)
with open(OUT / "data.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["group", "example", "baseline_solver", "baseline", "template", "tuned"])
    w.writeheader()
    for d in data:
        w.writerow({k: d[k] for k in w.fieldnames})


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e4e3df",
                  steel="#2a78d6", base="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#383835",
                 steel="#3987e5", base="#d95926"),
}
faster = sum(d["tuned"] < d["baseline"] for d in data)


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["muted"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def ms_ticks(ax, lo, hi):
    ticks = [v for v in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200) if lo <= v <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])


for mode, t in THEMES.items():
    # 1. per-example dumbbell: baseline -> tuned Steel, template Steel as a ring
    order = []
    for g in GROUPS:
        members = sorted((d for d in data if d["group"] == g), key=lambda d: d["tuned"] / d["baseline"])
        order.append(("head", g))
        order += [("row", d) for d in members]
    n = len(order)
    fig, ax = plt.subplots(figsize=(10, 0.24 * n + 1.6), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 200)
    ms_ticks(ax, 0.1, 200)
    yt, yl = [], []
    for i, (kind, item) in enumerate(order):
        y = n - i
        if kind == "head":
            ax.annotate(item, xy=(0.015, y), xycoords=("figure fraction", "data"), ha="left", va="center",
                        fontsize=10.5, fontweight="bold", color=t["ink"], annotation_clip=False)
            ax.axhline(y - 0.5, color=t["grid"], linewidth=0.8, zorder=0)
            continue
        d = item
        ax.plot([d["baseline"], d["tuned"]], [y, y], color=t["muted"], linewidth=1.5, zorder=1, solid_capstyle="round")
        ax.scatter(d["template"], y, s=34, facecolors="none", edgecolors=t["muted"], linewidths=1.2, zorder=2)
        ax.scatter(d["baseline"], y, s=44, color=t["base"], edgecolors=t["surface"], linewidths=1.5, zorder=3)
        ax.scatter(d["tuned"], y, s=44, color=t["steel"], edgecolors=t["surface"], linewidths=1.5, zorder=4)
        yt.append(y)
        yl.append(f"{d['label']}  ({d['baseline_solver']})")
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=8.5, color=t["ink2"])
    ax.set_ylim(0.3, n + 0.7)
    ax.set_xlabel("mean frame time, ms (log scale; RTX PRO 6000 Blackwell)", color=t["ink2"], fontsize=9.5)
    fig.suptitle(f"Tuned Steel is faster than the previous default in {faster} of {len(data)} examples",
                 x=0.02, ha="left", fontsize=14, fontweight="bold", color=t["ink"])
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=7, color=t["base"], label="previous default (solver in brackets)"),
        Line2D([], [], marker="o", linestyle="", markersize=7, color=t["steel"], label="Steel, tuned"),
        Line2D([], [], marker="o", linestyle="", markersize=6.5, markerfacecolor="none", markeredgecolor=t["muted"],
               label="Steel, generic template before tuning"),
    ]
    H = 0.24 * n + 1.6
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1 - 0.42 / H), ncol=3,
                     frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / H))
    fig.savefig(OUT / f"frame_time_{mode}.png", facecolor=t["surface"])
    plt.close(fig)

    # 2. per-group geometric mean of frame time relative to the previous default
    groups = list(GROUPS) + [f"All {len(data)} examples"]
    stats = []
    for g in groups:
        ds = data if g.startswith("All") else [d for d in data if d["group"] == g]
        stats.append((g, gmean([d["template"] / d["baseline"] for d in ds]),
                      gmean([d["tuned"] / d["baseline"] for d in ds]), len(ds)))
    all_tmpl, all_tuned = stats[-1][1], stats[-1][2]
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10)
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["10× faster", "5×", "2×", "same", "2×", "5×", "10× slower"])
    ax.axvline(1, color=t["ink2"], linewidth=1.2, zorder=1)
    for i, (g, tmpl, tuned, k) in enumerate(stats):
        y = len(stats) - 1 - i
        ax.annotate("", xy=(tuned, y), xytext=(tmpl, y),
                    arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.5, shrinkA=6, shrinkB=7))
        ax.scatter(tmpl, y, s=60, facecolors="none", edgecolors=t["muted"], linewidths=1.4, zorder=3)
        ax.scatter(tuned, y, s=70, color=t["steel"], edgecolors=t["surface"], linewidths=1.5, zorder=4)
        ax.text(tuned / 1.12, y + 0.28, f"{1 / tuned:.1f}× faster" if tuned < 1 else f"{tuned:.1f}× slower",
                ha="right", va="center", fontsize=9, color=t["ink"])
        ax.text(tmpl * 1.12, y + 0.28, f"{tmpl:.1f}× slower" if tmpl > 1 else f"{1 / tmpl:.1f}× faster",
                ha="left", va="center", fontsize=9, color=t["ink2"])
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([f"{g} ({k})" if not g.startswith("All") else g for g, _, _, k in reversed(stats)],
                       fontsize=9.5, color=t["ink"])
    ax.set_ylim(-0.6, len(stats) - 0.3)
    ax.set_xlabel("frame time relative to the previous default solver (geometric mean per group)", color=t["ink2"],
                  fontsize=9.5)
    fig.suptitle(f"Tuning took Steel from {all_tmpl:.1f}× slower than the previous defaults to "
                 f"{1 / all_tuned:.1f}× faster", x=0.02, ha="left", fontsize=14, fontweight="bold", color=t["ink"])
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="none", markeredgecolor=t["muted"],
               label="Steel, generic template"),
        Line2D([], [], marker="o", linestyle="", markersize=7, color=t["steel"], label="Steel, tuned"),
    ]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=2, frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / f"summary_{mode}.png", facecolor=t["surface"])
    plt.close(fig)

print(f"faster in {faster}/{len(data)}; geometric mean template {all_tmpl:.2f}x, tuned {all_tuned:.2f}x of baseline")
for g, tmpl, tuned, k in stats:
    print(f"  {g:22s} n={k:2d} template {tmpl:.2f}x tuned {tuned:.2f}x")
