#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Render every scene of RUN_DIR/plan.json: one worker per listed GPU, each rendering its share of scenes in turn.
# usage: run_all.sh RUN_DIR [GPU ...]      (default: GPU 0). Exits nonzero if any scene's render or check failed.
set -u
RUN=$(cd "$1" && pwd)
shift
GPUS=("${@:-0}")
HERE=$(cd "$(dirname "$0")" && pwd)
mapfile -t SCENES < <(python3 -c "import json,sys; [print(e['scene']) for e in json.load(open(sys.argv[1]))['render']]" "$RUN/plan.json")
FAILED=$RUN/failed.txt
: > "$FAILED"
echo "${#SCENES[@]} scenes on GPUs ${GPUS[*]}; logs in $RUN/out/<scene>/"
for k in "${!GPUS[@]}"; do
  (
    for i in "${!SCENES[@]}"; do
      if (( i % ${#GPUS[@]} == k )); then
        CUDA_VISIBLE_DEVICES=${GPUS[$k]} WARP_CACHE_PATH=${WARP_CACHE_PATH:-$RUN/warpcache-$k} \
          "$HERE/render_one.sh" "$RUN" "${SCENES[$i]}" || echo "${SCENES[$i]}" >> "$FAILED"
      fi
    done
  ) &
done
wait
N=$(wc -l < "$FAILED")
echo "done: ${#SCENES[@]} scenes, $N with a failed render or check (listed in $FAILED)"
[ "$N" = 0 ]
