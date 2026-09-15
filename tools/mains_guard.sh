# Shared by the long campaign scripts: run a command on mains power only.
#
# `run_on_mains cmd args...` starts the command when the machine is plugged in, stops
# it if the machine is unplugged, and starts it again when power comes back. The
# campaigns write their results after every prediction and skip completed objects,
# so a restart loses at most the object in progress. Returns the command's exit code
# once it has run to the end on mains power.
#
# Same functions as tools/run_table1_render.sh, whose stop-and-resume logic was tested
# with simulated power before the Table I rerun.

stamp() { date "+%Y-%m-%d %H:%M"; }
on_ac() { pmset -g batt | head -1 | grep -q "AC Power"; }

wait_for_ac() {
  local said=0
  while ! on_ac; do
    [ $said -eq 0 ] && echo "[$(stamp)] on battery: waiting for mains power" && said=1
    sleep 60
  done
}

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
