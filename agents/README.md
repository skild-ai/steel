# Agent workflows

Procedures for working on Steel scenes, for any coding agent or person. Three layers:

- **Tools** are code in this repository: `python -m shelf list | play | bench | tune | check | trial | autotune |
  advise` (run from `examples/`) and the scripts in `examples/scripts/` (`render_audit/`, `live_check.py`). All logic
  lives there.
- **Steps** (`steps/<verb-noun>/README.md`) are self-contained procedures: inputs, exact commands, pass/fail
  criteria, outputs. A step may call tools; it never calls a workflow.
- **Workflows** (`workflows/<verb-noun>/README.md`) are goals: when to use one, its inputs, an ordered list of
  steps, decision points, exit criteria and deliverables.

Every README opens with a three-line summary (`Goal / Use when / Done when` for workflows, `Does / Use in / Passes
when` for steps). Run commands from the repository root unless a step says `cd examples`.

## What are you trying to do?

| You want to | Workflow |
|---|---|
| make one Steel scene stable or faster, or pick its preset | [tune-a-scene](workflows/tune-a-scene/README.md) |
| find out why a Steel scene explodes, jitters, sinks, slips or overflows | [debug-instability](workflows/debug-instability/README.md) |
| know whether a Newton or Steel change breaks or slows any example | [validate-a-solver-change](workflows/validate-a-solver-change/README.md) |
| add a scene to the collection, or give an example a Steel preset | [add-an-example](workflows/add-an-example/README.md) |
| move a PhysX project to Steel | [port-from-physx](workflows/port-from-physx/README.md) |

## Steps

| Step | Does |
|---|---|
| [run-trial](steps/run-trial/README.md) | one full-episode trial at given settings, with ordered fixes |
| [autotune-settings](steps/autotune-settings/README.md) | fixes until none is left, then cheaper settings that stay healthy |
| [size-capacities](steps/size-capacities/README.md) | row capacities from measured demand, with headroom |
| [tune-one-knob](steps/tune-one-knob/README.md) | one setting changed, every gate re-checked |
| [check-example](steps/check-example/README.md) | the example's own test, recorded in `presets.toml` |
| [render-and-grade](steps/render-and-grade/README.md) | clips, stability metrics and grades for many examples |
| [visual-review](steps/visual-review/README.md) | frames looked at, grades corrected with notes |
| [record-baseline-trace](steps/record-baseline-trace/README.md) | a frozen PhysX project and its deterministic trace |
| [check-parity](steps/check-parity/README.md) | two traces compared by name, each difference traced to a cause |
| [randomize-one-factor](steps/randomize-one-factor/README.md) | domain randomization one factor at a time, both backends |
| [benchmark-throughput](steps/benchmark-throughput/README.md) | matched throughput and memory, and the tuning order |

## Reference

- [Steel tuning guide](../examples/docs/steel_tuning/guide.md): symptoms, recipes, costs, every option.
- [examples/AGENTS.md](../examples/AGENTS.md): the presets format and the rules for changing examples.

## Adapters

`python agents/make_adapters.py` writes optional pointers for tools that discover skill folders
(`.claude/skills/steel-<workflow>/SKILL.md`): front matter from the workflow's summary and one link to its README.
Steps get none. `--check` fails when a pointer is stale; never edit one by hand.
