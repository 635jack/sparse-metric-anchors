#!/usr/bin/env bash
# Table I again, with the anchors drawn from the camera that rendered the images.
#
#   1. palpation8_128 regenerated for the 42 objects with `make_contacts.py --camera render`,
#      into data/contacts_render (the published contacts stay untouched);
#   2. frame estimated (--combined_frame), unanchored passes regenerated;
#   3. frame given (--oracle_frame), reusing the unanchored passes of step 2.
#
# Runs on mains power only. When the machine is unplugged, the running campaign is
# stopped and resumes when power comes back: results are written after every
# prediction and the campaign skips objects that are already complete.
#
#   nohup caffeinate -i bash tools/run_table1_render.sh > logs/table1_render.log 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-/Users/kojack/miniforge3/bin/python3}
OBJECTS=$($PY -c "import json; print(' '.join(sorted(json.load(open('full42_combined.json')))))")

stamp() { date "+%Y-%m-%d %H:%M"; }
on_ac() { pmset -g batt | head -1 | grep -q "AC Power"; }

wait_for_ac() {
  local said=0
  while ! on_ac; do
    [ $said -eq 0 ] && echo "[$(stamp)] on battery: waiting for mains power" && said=1
    sleep 60
  done
}

# Runs a command on mains power; stops it if unplugged and starts it again later.
run_on_mains() {
  local pid rc
  while true; do
    wait_for_ac
    echo "[$(stamp)] start: $*"
    "$@" &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      if ! on_ac; then
        echo "[$(stamp)] unplugged: stopping, will resume on mains power"
        kill "$pid"
        break
      fi
      sleep 20
    done
    wait "$pid"
    rc=$?
    if on_ac; then
      [ $rc -eq 0 ] && echo "[$(stamp)] done: $*" && return 0
      echo "[$(stamp)] FAILED (exit $rc): $*"
      return $rc
    fi
  done
}

echo "[$(stamp)] Table I with the rendering camera, $(echo $OBJECTS | wc -w | tr -d ' ') objects"

if [ "$(ls data/contacts_render/*/palpation8_128.pt 2>/dev/null | wc -l | tr -d ' ')" -ne 42 ]; then
  $PY tools/make_contacts.py --camera render --sites 8 --densities 128 \
      --out_dir data/contacts_render --objects $OBJECTS || exit 1
else
  echo "[$(stamp)] render-camera contacts already present for the 42 objects"
fi

run_on_mains $PY tools/run_guidance_campaign.py --combined_frame \
    --sets palpation8:128 --contacts_dir data/contacts_render \
    --out_dir out/full42_render_combined --out full42_render_combined.json \
    --objects $OBJECTS || exit 1

run_on_mains $PY tools/run_guidance_campaign.py --oracle_frame \
    --sets palpation8:128 --contacts_dir data/contacts_render \
    --baseline_from out/full42_render_combined \
    --out_dir out/full42_render_oracle --out full42_render_oracle.json \
    --objects $OBJECTS || exit 1

echo "[$(stamp)] Table I with the rendering camera: finished"
