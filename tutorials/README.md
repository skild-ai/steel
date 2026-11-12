# Steel tutorials

One short, runnable script per Steel feature. Each script builds a small scene, runs it with one or more
settings side by side, and can record the run so the [tutorials site](../site/) replays it in 3D next to
the script's exact source.

```bash
cd tutorials
python 03_patch_friction.py                    # prints each variant's final metric
python 03_patch_friction.py --record out.json  # also writes the replay
```

The scripts need the Newton build the docs describe (see the site's Installation page) and a CUDA GPU for
most scenes.

## Layout

```
index.toml               group order and the tutorials in each group, by id; the planned list
NN_name.py               a tutorial script; it imports Recorder from replay.py
NN_name.toml             its page text (fields below)
dr_name.py, dr_name.toml domain randomization tutorials, same format
replay.py                the recorder every script imports
recordings/<id>.json     the recorded replay of each tutorial, committed
DR_GAPS.md               domain randomization cases without a tutorial, shown on the tutorials index
tools/record_all.py      record every tutorial (or some) on a local GPU
tools/check.py           check metadata, order and that every recording matches its script
tools/timing*.py         the cost tables some tutorials show (timing.py holds the shared method)
tools/test_tools.py      CPU-only tests of the tools
```

A tutorial's page is `/tutorials/<id>/` on the site. Ids are stable: other pages link to them, so do not
rename one.

## Adding a tutorial

1. Write `NN_name.py`. Build the scene, create a `Recorder()`, add one `rec.variant(label, model)` per
   setting, call `run.frame(state, metric=value, ...)` once per replay frame and `run.check(solver)` after
   the run, then `rec.save()`. Keep it short and self-contained; copy the shape of an existing tutorial.
2. Write `NN_name.toml` with the page text, and add its `id` to a group in `index.toml`.
3. Record it: `python tools/record_all.py --only <id>` writes `recordings/<id>.json`. Add `--timing` if the
   tutorial has a `[timing]` table.
4. Check: `python tools/check.py --poses`, and `ruff check .` from this folder.
5. Preview it on the site: `cd ../site && npm run dev`, then open `/tutorials/<id>/`.

A recording keeps its cost table only while its script, the Newton commit and the timed workload (the
runner, `timing.py` and the scripts the runner builds scenes from) stay the same; otherwise re-recording
drops the table until it is measured again with `--timing`, and `tools/check.py` fails until then.
Measured tables carry the Newton commit, the runner and a hash of that workload.

Changing a script after it was recorded makes `tools/check.py` and the site flag the recording as out of
date, because the page would show code that did not produce the replay. Record it again in the same change.

## Page text fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | The page slug, lowercase words joined by `-`. |
| `title` | yes | Page title. |
| `about` | yes | One paragraph: the scene and the idea it shows. |
| `look` | yes | What to watch in the replay, one sentence each. |
| `try` | yes | Changes the reader can make to the script. |
| `[param]` | yes | `name`: the key option or attribute; `text`: what it does and its default. |
| `[metric]` | no | The chart: `key` (a value the script records), `label`, `unit`; `multi` plots several keys per variant, `log = true` a log axis, `[metric.alt]` a second chart. Use `[[metric.charts]]` for several charts, each with an optional `scene` that selects the variants recorded with that `scene=`. |
| `spacing` | no | Offset between variants in metres, `[x, y, z]`; default `[0, 0.6, 0]`. |
| `speed` | no | Initial playback speed; default 1. |
| `fit` | no | `"end"` frames the camera on the final pose instead of the motion. |
| `[concept]` | no | `name`, `what`, `eq` (optional, plain text), `here`, and `[[concept.consider]]` entries with `term` and `text`. |
| `[timing]` | no | A cost table: `group` (a group defined in one of `tools/timing*.py`: `GROUPS` in `timing.py`, `GROUP_NAMES` in the others), `scene` (one sentence), `per = "frame"` to report per frame, `rounds`, and `[timing.notes]` mapping each option label to the result it gives in this tutorial. |
| `notify_flag`, `update_route` | no | Domain randomization tutorials: the notification flag and when Steel sees the change. |
| `allow_dropped_rows` | no | `true` when the tutorial drops rows on purpose. |
| `[[in_review]]` | no | A solver change in review that bears on the page: `pr`, the [skild-ai/newton](https://github.com/skild-ai/newton) pull request number, and `text`, one sentence on what it changes. The page shows it with a link. |

Text is plain: no Markdown. Quote measured numbers with the scene they were measured on.
