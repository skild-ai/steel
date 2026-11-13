#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Render one example with its Steel preset and record metrics, then run its check without writing results.
# usage: render_one.sh RUN_DIR SCENE     (EXAMPLES and PYTHON override the defaults)
# Exits nonzero when the render or the check failed; collect.py reads the same outcome from the logs.
set -u
RUN=$(cd "$1" && pwd)
SCENE=$2
HERE=$(cd "$(dirname "$0")" && pwd)
EXAMPLES=${EXAMPLES:-$(cd "$HERE/../.." && pwd)}
PY=${PYTHON:-$EXAMPLES/.venv/bin/python}
OUT=$RUN/out/$SCENE
mkdir -p "$OUT"
# Nothing from an earlier attempt may be read as this one's.
rm -f "$OUT"/{clip_raw.mp4,poster.jpg,sheet.jpg,metrics.json,render.log,render_headless_arg.log,check.log,attempt}
ATTEMPT=$(date +%s%N)-$$-$RANDOM
echo "$ATTEMPT" > "$OUT/attempt"
cd "$EXAMPLES"
unset VIRTUAL_ENV SHELF_SET SHELF_PRESETS SHELF_SOLVER
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
PRESET=$(python3 -c "import json,sys; print(next(e['preset'] for e in json.load(open(sys.argv[1]))['render'] if e['scene'] == sys.argv[2]))" "$RUN/plan.json" "$SCENE")
render() {
  timeout 3000 "$PY" "$HERE/render_scene.py" --plan "$RUN/plan.json" --scene "$SCENE" --out "$RUN/out" --wall-budget 2400 --attempt "$ATTEMPT" "$@" > "$OUT/render.log" 2>&1
  RENDER_RC=$?
  echo "RENDER_RC $RENDER_RC" >> "$OUT/render.log"
}
render
if grep -q "ViewerNull does not support frame capture" "$OUT/render.log"; then
  # the example swaps to the null viewer under --headless; pyglet stays headless either way
  mv "$OUT/render.log" "$OUT/render_headless_arg.log"
  render --no-headless-arg
fi
timeout 1500 "$PY" -m shelf check "$SCENE" --solver "$PRESET" --dry-run --timeout 1200 > "$OUT/check.log" 2>&1
CHECK_RC=$?
echo "CHECK_RC $CHECK_RC" >> "$OUT/check.log"
echo "$SCENE: render rc $RENDER_RC, check rc $CHECK_RC"
[ "$RENDER_RC" = 0 ] && [ "$CHECK_RC" = 0 ]
