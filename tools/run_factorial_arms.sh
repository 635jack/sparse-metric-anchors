#!/usr/bin/env bash
# The two missing cells of a 2 x 2 design on the anchor set, oracle frame, 42 objects:
#
#                       hidden face                    visible face
#   8 palpation sites   palpation8_128 (done)          palpation8visible_128  <- here
#   uniform coverage    occluded_128   <- here         visible_128 (done)
#
# The visible-face control gained twice what the hidden-face palpation sites gained,
# but those two sets differ both in where the anchors are and in how they spread.
# These two arms separate the two. Both reuse the unanchored passes of the Table I
# rerun, so all four cells share the same unanchored pass and the same oracle frame.
#
# Mains power only (tools/mains_guard.sh).
#
#   nohup caffeinate -i bash tools/run_factorial_arms.sh > logs/factorial_arms.log 2>&1 &

set -u
cd "$(dirname "$0")/.." || exit 1
source tools/mains_guard.sh
PY=${PY:-/Users/kojack/miniforge3/bin/python3}
OBJECTS=$($PY -c "import json; print(' '.join(sorted(json.load(open('full42_combined.json')))))")

echo "[$(stamp)] 2 x 2 anchor-set arms, $(echo $OBJECTS | wc -w | tr -d ' ') objects"

if [ "$(ls data/contacts_render_vispatch/*/palpation8visible_128.pt 2>/dev/null | wc -l | tr -d ' ')" -ne 42 ]; then
  $PY tools/make_contacts.py --camera render --face visible --sites 8 --densities 128 \
      --out_dir data/contacts_render_vispatch --objects $OBJECTS || exit 1
fi

run_on_mains $PY tools/run_guidance_campaign.py --oracle_frame \
    --sets occluded:128 --contacts_dir data/contacts_render \
    --baseline_from out/full42_render_combined \
    --out_dir out/full42_render_oracle_occluded --out full42_render_oracle_occluded.json \
    --objects $OBJECTS || exit 1

run_on_mains $PY tools/run_guidance_campaign.py --oracle_frame \
    --sets palpation8visible:128 --contacts_dir data/contacts_render_vispatch \
    --baseline_from out/full42_render_combined \
    --out_dir out/full42_render_oracle_vispatch --out full42_render_oracle_vispatch.json \
    --objects $OBJECTS || exit 1

echo "[$(stamp)] 2 x 2 anchor-set arms: finished"
