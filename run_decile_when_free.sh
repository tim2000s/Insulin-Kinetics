#!/usr/bin/env bash
# Wait for capacity, then fit one kernel per dose decile across the cohort, with the
# no-effect negative control that the reported tilt has to be read against.
set -uo pipefail
cd "$(dirname "$0")"
THRESH=${LOAD_THRESHOLD:-6}
JOBS=${DECILE_JOBS:-4}

while true; do
  load=$(uptime | sed 's/.*load averages*: *//' | awk '{print int($1)}')
  [ "$load" -lt "$THRESH" ] && break
  sleep 60
done
echo "$(date -u +%H:%M:%SZ) load below $THRESH; fitting decile kernels on $JOBS workers"
python3 dose_decile_response.py --config cohort.json --pool --jobs "$JOBS" \
        --out build/DOSE_DECILE.md > build/decile_run.log 2>&1
echo "$(date -u +%H:%M:%SZ) decile exit $?"
