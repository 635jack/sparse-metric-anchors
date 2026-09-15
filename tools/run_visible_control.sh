#!/usr/bin/env bash
# The control the paper names first among its limitations: anchors taken on the face
# the camera sees, as a depth camera would give them, instead of tactile contacts on
# the hidden face.
#
# visible_128 from data/contacts_render (drawn from the camera that rendered the
# images: about 1 % of these points are hidden from it), 42 objects, 3 seeds, oracle
# frame. It reuses the unanchored passes of the Table I rerun. The oracle frame depends
# only on that unanchored pass, so these anchors and the palpation anchors of
# full42_render_oracle.json are placed by exactly the same transform: the two arms
# differ by the anchor set and nothing else.
#
# Queued behind tools/run_table1_render.sh: it waits for that script to exit, and
# starts only if Table I finished. Mains power only (tools/mains_guard.sh).
#
#   nohup caffeinate -i bash tools/run_visible_control.sh > logs/visible_control.log 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1
source tools/mains_guard.sh
PY=${PY:-/Users/kojack/miniforge3/bin/python3}
OBJECTS=$($PY -c "import json; print(' '.join(sorted(json.load(open('full42_combined.json')))))")

if pgrep -f "tools/run_table1_render.sh" > /dev/null; then
  echo "[$(stamp)] waiting for the Table I rerun to finish"
  while pgrep -f "tools/run_table1_render.sh" > /dev/null; do sleep 60; done
fi
if ! grep -q "Table I with the rendering camera: finished" logs/table1_render.log; then
  echo "[$(stamp)] Table I did not finish: the control is not started"
  exit 1
fi

echo "[$(stamp)] visible-face control, $(echo $OBJECTS | wc -w | tr -d ' ') objects"
run_on_mains $PY tools/run_guidance_campaign.py --oracle_frame \
    --sets visible:128 --contacts_dir data/contacts_render \
    --baseline_from out/full42_render_combined \
    --out_dir out/full42_render_oracle_visible --out full42_render_oracle_visible.json \
    --objects $OBJECTS || exit 1

echo "[$(stamp)] visible-face control: finished"
