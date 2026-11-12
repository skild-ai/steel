#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Generate the Steel configuration reference from SolverSteel's source.

Reads ``newton/_src/solvers/steel/solver_steel.py`` with ``ast`` (nothing is imported, so no GPU or Warp
is needed) and extracts every constructor option (type, default, docstring, units, whether it is read
every step), every ``steel:`` custom attribute, the Model/State/Control/Contacts inputs the docstrings
name, and the public methods. Option availability is checked against other refs of the same clone.

Writes ``site/src/data/reference.json``, which the reference pages render at build time. From ``site/``:

    git -C path/to/newton fetch upstream --tags
    python scripts/gen_reference.py --source path/to/newton \
        --refs "Newton 1.8.0=v1.8.0" "Newton main=upstream/main"

``--source`` is a clone of the Newton build the docs describe (skild-ai/newton ``main``) with the upstream
remote fetched, so the availability refs resolve.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLVER = "newton/_src/solvers/steel/solver_steel.py"
OUT = HERE.parent / "src" / "data" / "reference.json"

CATEGORIES = {
    "Solve": ["solve_mode", "articulated_contact_response", "iterations", "relaxation", "cfm", "row_schedule",
              "nesterov_momentum", "velocity_iterations", "velocity_drive_mode", "deterministic",
              "enable_bilateral_preelimination", "bilateral_preelimination_include_mimics"],
    "Stabilization": ["baumgarte", "contact_regularization", "contact_speculative_scale", "contact_gap_gate"],
    "Contacts and friction": ["friction_anchor_beta", "friction_mode", "friction_combine", "contact_friction_scale",
                              "contact_warmstart", "contact_warmstart_decay", "enable_restitution",
                              "restitution_velocity_threshold", "restitution_combine",
                              "contact_friction_position_iterations", "contact_friction_gap_max",
                              "contact_shared_anchor", "contact_friction_shared_anchor", "contact_torsion_device",
                              "contact_compliance", "free_rigid_torsional_friction", "free_rigid_rolling_friction"],
    "Joints and drives": ["enable_joint_limits", "joint_limit_activation_gap", "enable_joint_velocity_limits",
                          "velocity_limit_activation_fraction", "fuse_joint_velocity_limits", "enable_joint_friction",
                          "drive_mode"],
    "Capacity and diagnostics": ["articulation_constraint_max", "free_body_constraint_max",
                                 "propagation_constraint_max", "row_watermark", "warn_constraint_overflow"],
    "Propagation response": ["propagation_same_articulation_rows", "propagation_cached_response",
                             "propagation_cached_response_body_max", "propagation_color_order"],
    "Sleeping": ["enable_sleeping", "sleep_linear_threshold", "sleep_angular_threshold", "sleep_quiet_time",
                 "sleep_skip_constraints"],
    "Performance": ["update_mass_matrix_interval", "parallel_tree", "reuse_kinematics", "use_parallel_streams",
                    "double_buffer"],
}

# When to change an option. Hand-written guidance; every option must have an entry.
WHEN = {
    "solve_mode": "Use \"colored\" (with the propagation response) for worlds with many free bodies, such as piles "
                  "and pours. Keep \"serial\" for articulated robots, small scenes and loop closures.",
    "articulated_contact_response": "\"propagation\" is required by the colored sweep. Keep \"immediate\" for "
                                    "loop-closing joints, Nesterov momentum and coupled solvers.",
    "iterations": "Raise for tall stacks, large mass ratios, loop closures and slipping grasps. Lower only after a "
                  "full-episode check passes at the lower value.",
    "relaxation": "Leave at 1. Under-relaxation (about 0.6) helped one momentum-accelerated rope; values above 1 "
                  "made a heavy impactor sink further.",
    "cfm": "Rarely changed. On ordinary scalar rows it shifts the diagonal, which damps the sweep like a relaxation below 1 and "
           "leaves the converged impulses of unregularized rows unchanged. It deepens the resting penetration of "
           "contacts under contact_regularization and floors the pre-elimination regularization; soften contacts "
           "with contact_regularization or contact_compliance instead.",
    "row_schedule": "Try \"phased\" for grasps tuned against PhysX. Avoid \"contact_then_internal\" when contacts "
                    "load a joint limit or drive: the load sinks through at any iteration count. Contact torsion, "
                    "compliance and the sparse mass factors need \"interleaved\".",
    "nesterov_momentum": "Resting stacks with mass ratios of roughly 100:1 to 1000:1 and long chains, when more "
                         "iterations alone are too slow. Serial sweep, immediate response, CUDA only.",
    "velocity_iterations": "Set 2 to 4 when position correction launches bodies (deep start overlaps). Must be 0 "
                           "with sleeping, compliance, and torsion under propagation.",
    "velocity_drive_mode": "Only matters with drive_mode=\"rows\" and velocity_iterations > 0.",
    "deterministic": "Set wp.DeterministicMode.RUN_TO_RUN for tests, debugging and data generation that must "
                     "reproduce bit for bit. Pair it with CollisionPipeline(deterministic=True).",
    "enable_bilateral_preelimination": "Loop closures or mimic couplings that drift at your iteration budget.",
    "bilateral_preelimination_include_mimics": "Set False if a mimic row is nearly dependent on a closure and the "
                                               "eliminated block becomes singular.",
    "baumgarte": "Up to 0.5 for light bodies pressed into the ground by heavy ones. Higher values launched bodies "
                 "in testing; prefer velocity_iterations for deep start overlaps.",
    "contact_regularization": "About 0.01 removes warm-start bursts on deeply wedged contacts and makes "
                              "indeterminate load splits unique, at a small resting penetration.",
    "contact_speculative_scale": "Rarely changed. 1 lets a contact close exactly its gap in one step; 0 removes the "
                                 "speculative allowance.",
    "contact_gap_gate": "Set 0 where speculative rows never engage (for example many small rollers) to save rows.",
    "friction_anchor_beta": "Keep the default (anchors) for anything held, carried or pinched. Set 0 (point "
                            "friction) for scenes where nothing is held; it is cheaper.",
    "friction_mode": "Experimental. Leave at \"fixed_normal\"; \"coulomb_newton\" for free bodies near the friction "
                     "limit. At this commit \"bisection\" and \"bisection_desaxce\" let a box near the friction "
                     "limit slide; a fix is in review in skild-ai/newton#26.",
    "friction_combine": "Match the combine mode of the engine you are porting from (PhysX average, min, multiply "
                        "or max).",
    "contact_friction_scale": "Sweeps or domain randomization over friction without editing shape materials.",
    "contact_warmstart": "Off by default. Turn on only where a measured scene needs it (for example a house of "
                         "cards with momentum). Needs contact matching; never for a grasp without anchors.",
    "contact_warmstart_decay": "Values below 1 weaken the carried impulses of warm start.",
    "enable_restitution": "Set False to ignore authored restitution everywhere, or to use contact_compliance with "
                          "shapes that author restitution.",
    "restitution_velocity_threshold": "Lower to let slower impacts bounce; raise to make slow impacts inelastic.",
    "restitution_combine": "Match the combine mode of the engine you are porting from.",
    "contact_friction_position_iterations": "Rarely changed. Solves friction only in the last k position iterations; "
                                            "rejects warm start and compliance.",
    "contact_friction_gap_max": "-inf for a frictionless scene, which saves the friction rows.",
    "contact_shared_anchor": "Match PhysX's single-point contact application.",
    "contact_friction_shared_anchor": "Remove the friction couple of witness points separated along the normal.",
    "contact_torsion_device": "Set True whenever shapes author steel:min_torsional_patch_radius and you capture "
                              "CUDA graphs or use the propagation response.",
    "contact_compliance": "Experimental. Soft fingertips or pads modeled with hydroelastic contacts.",
    "free_rigid_torsional_friction": "Sweeps over spin friction of free bodies without editing shapes.",
    "free_rigid_rolling_friction": "Sweeps over rolling friction of free bodies without editing shapes.",
    "enable_joint_limits": "Set True whenever the model authors joint limits. Off, limits are not simulated. "
                           "DISTANCE joint limits are ignored either way at this commit.",
    "joint_limit_activation_gap": "A fixed gap such as 0.1 builds limit rows only near a limit and is often faster "
                                  "than the speed-aware default. Keep it above 2 * max|qd| * dt.",
    "enable_joint_velocity_limits": "Set True when the task depends on joint velocity limits.",
    "velocity_limit_activation_fraction": "Raise toward 1 to skip rows of DOFs far below their limit.",
    "fuse_joint_velocity_limits": "Leave on; set False to keep explicit velocity-limit rows on driven DOFs.",
    "enable_joint_friction": "Set True for models that author joint friction (MJCF frictionloss). It disables "
                             "several fast paths, so leave it off otherwise.",
    "drive_mode": "\"rows\" when drive torques must respect the effort limit exactly (torque-limited robots, "
                  "sim-to-real), or to solve drives together with contacts.",
    "articulation_constraint_max": "Size from row_watermark: about 2x the requested mark while overflowing, then "
                                   "1.3x a non-overflowing mark. Robots with limits or self-contacts often need "
                                   "more than 32. An overflowing run can gain energy and diverge.",
    "free_body_constraint_max": "Size from row_watermark under the immediate response. Very large values stop the "
                                "immediate kernels from building; move such scenes to colored propagation.",
    "propagation_constraint_max": "Size from row_watermark under the propagation response.",
    "row_watermark": "On while tuning. Read constraint_row_watermarks() after a full episode.",
    "warn_constraint_overflow": "Turn off only when you check constraint_overflow yourself.",
    "propagation_same_articulation_rows": "Robots with many self-contacts under the propagation response, to move those "
                                          "contacts off the dense rows.",
    "propagation_cached_response": "Leave on; compare both on your model if the propagation sweep dominates.",
    "propagation_cached_response_body_max": "Raise when many contact-active bodies per world fall back to the "
                                            "tree walk.",
    "propagation_color_order": "\"rotating\" spreads Gauss-Seidel bias over a symmetric grasp on a sorted contact "
                               "stream; \"longest_first\" can shorten sweeps with uneven row counts.",
    "enable_sleeping": "Settling scenes with many passive objects. Excludes warm start and velocity iterations. In "
                       "many worlds, pair it with CollisionPipeline(deterministic=True), or most islands can stay "
                       "awake; check body_awake.",
    "sleep_linear_threshold": "Lower if bodies fall asleep while still creeping.",
    "sleep_angular_threshold": "Lower if slowly spinning bodies fall asleep. Raise (for example to 0.3) if resting "
                               "stacks stay awake with the default contact order.",
    "sleep_quiet_time": "Raise if islands fall asleep before they settle.",
    "sleep_skip_constraints": "Set False when consumers read contact forces of sleeping bodies.",
    "update_mass_matrix_interval": "Above 1 only for slowly moving articulations; it must divide the steps per "
                                   "captured graph. It does not cure energy gain on fast, light chains: on the "
                                   "measured chains the stale matrix lost energy instead.",
    "parallel_tree": "Broad trees such as hands; time the full step before keeping it.",
    "reuse_kinematics": "Time it on your scene; call notify_state_changed after writing state outside step().",
    "use_parallel_streams": "Models with several articulation sizes, or many independent single-DOF joints on one "
                            "base.",
    "double_buffer": "Time it on your scene; results are unchanged.",
}

MODEL_ROLE = r":(?:attr|class|meth|func):`~?newton\.((?:Model|State|Control|Contacts|ModelBuilder\.ShapeConfig)\.\w+)`"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="Newton checkout to document")
    ap.add_argument("--refs", nargs="*", default=["Newton 1.8.0=v1.8.0", "Newton main=upstream/main"],
                    help="label=git-ref pairs checked for option availability in the same clone")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    source = Path(args.source).expanduser()
    text = (source / SOLVER).read_text()
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SolverSteel")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    head = git(source, "rev-parse", "HEAD").strip()

    step_options = step_option_names(cls)
    docs = parse_args_section(ast.get_docstring(init))
    availability = {label: signature_names(source, ref) for label, ref in (r.split("=", 1) for r in args.refs)}

    options = []
    positional = init.args.kwonlyargs
    for arg, default in zip(positional, init.args.kw_defaults):
        name = arg.arg
        desc = docs.get(name, "")
        options.append({
            "name": name,
            "type": ast.unparse(arg.annotation) if arg.annotation else "",
            "default": ast.unparse(default) if default is not None else "",
            "units": "" if arg.annotation is not None and ast.unparse(arg.annotation) == "bool" else units_of(desc),
            "step": name in step_options,
            "category": category_of(name),
            "doc": rst_to_html(desc),
            "when": WHEN[name],
            "available": {label: name in sig for label, sig in availability.items()},
            "differences": value_differences(name, ast.unparse(arg.annotation) if arg.annotation else "",
                                             ast.unparse(default) if default is not None else "", availability),
            "line": arg.lineno,
        })
    missing = [n for n in WHEN if n not in {o["name"] for o in options}]
    if missing:
        raise SystemExit(f"WHEN entries for options that do not exist: {missing}")

    constants = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
                 for t in n.targets if isinstance(t, ast.Name)}
    attributes = custom_attributes(cls, constants)
    class_doc = ast.get_docstring(cls) + "\n" + ast.get_docstring(init)
    inputs = model_inputs(class_doc)
    methods = public_methods(cls)

    data = {
        "source": {"commit": head, "file": SOLVER, "init_line": init.lineno},
        "options": options,
        "attributes": attributes,
        "inputs": inputs,
        "methods": methods,
        "counts": {
            "options": len(options),
            "documented": sum(1 for o in options if o["doc"] and o["when"]),
            "step_options": len(step_options),
            "attributes": len(attributes),
            "inputs": len(inputs),
            "methods": len(methods),
        },
    }
    Path(args.out).write_text(json.dumps(data, indent=1) + "\n")
    c = data["counts"]
    print(f"{c['documented']}/{c['options']} options documented ({c['step_options']} read every step), "
          f"{c['attributes']} steel: attributes, {c['inputs']} model inputs, {c['methods']} methods; "
          f"source {head[:10]}")


def step_option_names(cls: ast.ClassDef) -> set[str]:
    names = set()
    for node in cls.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = ast.unparse(node.value.func)
            if func in ("_StepOption", "_FrictionGapOption"):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def parse_args_section(doc: str) -> dict[str, str]:
    """Split a Google-style ``Args:`` section into ``{name: text}``, keeping nested paragraphs."""
    lines = doc.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "Args:")
    out, name, buf, base = {}, None, [], None
    for line in lines[start + 1:]:
        if line.strip() and not line.startswith(" "):
            break
        m = re.match(r"^(\s+)(\w+):\s?(.*)$", line)
        if m and (base is None or len(m.group(1)) == base):
            if name:
                out[name] = "\n".join(buf).strip()
            base = len(m.group(1))
            name, buf = m.group(2), [m.group(3)]
        elif name:
            buf.append(line[base + 4:] if line.startswith(" " * (base + 4)) else line.strip())
    if name:
        out[name] = "\n".join(buf).strip()
    return out


def units_of(desc: str) -> str:
    for m in re.finditer(r"\[([^\]]{1,24})\]", desc):
        unit = m.group(1)
        if re.search(r"\b(m|s|rad|N|kg)\b|m/s|rad/s|N·m|N s/m|N/m|1/s", unit):
            return unit
    return ""


def category_of(name: str) -> str:
    for cat, names in CATEGORIES.items():
        if name in names:
            return cat
    raise SystemExit(f"option {name} has no category in CATEGORIES")


def leading_units(text: str) -> str:
    m = re.match(r"\s*\[([^\]]+)\]", text)
    return m.group(1) if m else ""


def usd_name(kw: dict, constants: dict[str, str]) -> str:
    if "usd_attribute_name" not in kw:  # ModelBuilder.CustomAttribute's default: newton:<namespace>:<name>
        return f"newton:steel:{ast.literal_eval(kw['name'])}"
    name = ast.literal_eval(kw["usd_attribute_name"])
    if name == "*":  # resolved by a transformer that reads a module constant
        return constants.get("_JOINT_FRICTION_LOAD_COEFFICIENT_USD", "")
    return name


def custom_attributes(cls: ast.ClassDef, constants: dict[str, str]) -> list[dict]:
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "register_custom_attributes")
    doc = ast.get_docstring(method)
    bullets = {}
    for m in re.finditer(r"^- ``steel:(\w+)``(.*?)(?=^- ``steel:|^\S|\Z)", doc, re.M | re.S):
        bullets[m.group(1)] = (m.group(1), m.group(2))
    pair = r"^- ``steel:(\w+)`` (\[[^\]]*\]) and ``steel:(\w+)``(.*?)(?=^- ``steel:|^\S|\Z)"
    for m in re.finditer(pair, doc, re.M | re.S):
        bullets[m.group(1)] = (m.group(1), " " + m.group(2) + re.sub(r"^\s*\[[^\]]*\]", "", m.group(4)))
        bullets[m.group(3)] = (m.group(3), m.group(4))
    out = []
    for call in ast.walk(method):
        if not (isinstance(call, ast.Call) and ast.unparse(call.func).endswith("CustomAttribute")):
            continue
        kw = {k.arg: k.value for k in call.keywords}
        name = ast.literal_eval(kw["name"])
        text = bullets.get(name, ("", ""))[1]
        mjcf = ast.literal_eval(kw["mjcf_attribute_name"]) if "mjcf_attribute_name" in kw else ""
        # A prefixed name is not well-formed MJCF without a namespace declaration, so no file can author it.
        mjcf = "" if ":" in mjcf else mjcf
        out.append({
            "name": f"steel:{name}",
            "frequency": ast.unparse(kw["frequency"]).split(".")[-1].lower(),
            "dtype": ast.unparse(kw["dtype"]).replace("wp.", ""),
            "default": ast.unparse(kw["default"]).replace("_DEFAULT_ANGULAR_DAMPING", "0.05"),
            "usd": usd_name(kw, constants),
            "mjcf": mjcf,
            "units": leading_units(text),
            "doc": rst_to_html(re.sub(r"^\s*\[[^\]]*\]\s*", "", re.sub(r"\s+", " ", text)).strip(" .") + "."),
        })
    return out


def model_inputs(doc: str) -> list[dict]:
    """Every newton.Model/State/Control/Contacts attribute the docstrings name, with the first sentence naming it."""
    # Split into paragraphs, bullets and Args entries first, so a sentence never spans two of them.
    blocks = re.split(r"\n\s*\n|\n(?=\s*- )|\n(?=\s{4,}\w+: )", doc)
    found: dict[str, str] = {}
    for block in blocks:
        for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z:`(])", squash(block).strip()):
            for m in re.finditer(MODEL_ROLE, sentence):
                name = m.group(1)
                text = re.sub(r"^\s*(- |\w+: )", "", sentence)
                if len(text) > 420:
                    start = max(0, m.start() - 180)
                    chunk = text[start:start + 400].rsplit(" ", 1)[0]
                    text = ("…" + chunk.split(" ", 1)[1] if start else chunk) + "…"
                if name not in found:
                    found[name] = text
    return [{"name": k, "context": rst_to_html(v)} for k, v in sorted(found.items())]


def public_methods(cls: ast.ClassDef) -> list[dict]:
    out = []
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_") or node.name.startswith("coupling_"):
            continue
        decos = [ast.unparse(d) for d in node.decorator_list]
        kind = "property" if "property" in decos else "classmethod" if "classmethod" in decos else "method"
        first = (ast.get_docstring(node) or "").split("\n\n")[0]
        out.append({"name": node.name, "kind": kind, "doc": rst_to_html(re.sub(r"\s+", " ", first))})
    return out


def signature_names(source: Path, ref: str) -> dict[str, tuple[str, str]]:
    """``{option: (annotation, default)}`` of ``SolverSteel.__init__`` at ``ref``; empty if the ref is missing."""
    try:
        text = git(source, "show", f"{ref}:{SOLVER}")
    except subprocess.CalledProcessError:
        return {}
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SolverSteel")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    return {a.arg: (ast.unparse(a.annotation) if a.annotation else "", ast.unparse(d) if d is not None else "")
            for a, d in zip(init.args.kwonlyargs, init.args.kw_defaults)}


def literal_values(annotation: str) -> list[str]:
    m = re.match(r"Literal\[(.*)\]$", annotation)
    return [v.strip() for v in m.group(1).split(",")] if m else []


def value_differences(name: str, annotation: str, default: str, availability: dict) -> list[str]:
    """Values or defaults of an option that differ in other refs, as short notes."""
    notes = []
    for label, sig in availability.items():
        if name not in sig:
            continue
        other_annotation, other_default = sig[name]
        missing = [v for v in literal_values(annotation) if v not in literal_values(other_annotation)]
        if missing and literal_values(other_annotation):
            notes.append(f"{', '.join(missing)} not in {label}")
        if other_default != default:
            notes.append(f"default in {label}: {other_default}")
    return notes


def rst_to_html(text: str) -> str:
    """Convert the docstrings' RST inline markup to small HTML."""
    s = html.escape(text, quote=False)
    s = re.sub(r"\.\. (deprecated|experimental)::\s*([\d.]*)", lambda m: f"<b>{m.group(1).capitalize()}"
               f"{' ' + m.group(2) if m.group(2) else ''}.</b>", s)
    s = re.sub(r":\w+:`([^`<]+?) &lt;[^`]+&gt;`", r"\1", s)
    s = re.sub(r":\w+:`~?([\w.]+)`", lambda m: f"<code>{m.group(1).split('.')[-1]}</code>", s)
    s = re.sub(r"``(.+?)``", lambda m: "<code>" + squash(m.group(1)) + "</code>", s, flags=re.S)
    s = re.sub(r"(?<![\w*])\*(\S[^*]*?)\*(?![\w*])", r"<em>\1</em>", s)
    paras = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    out = []
    for p in paras:
        if re.match(r"^- ", p):
            items = [i.strip() for i in re.split(r"(?m)^- ", p) if i.strip()]
            out.append("<ul>" + "".join("<li>" + squash(i) + "</li>" for i in items) + "</ul>")
        else:
            out.append("<p>" + squash(p) + "</p>")
    return "".join(out)


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


if __name__ == "__main__":
    main()
