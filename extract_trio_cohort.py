#!/usr/bin/env python3
"""Extract the Trio site cohort into the analysis database with real treatment streams.

These sites were reachable all along. The archived table for this cohort holds a decision stream
but no treatments, which sent an earlier attempt down the route of reconstructing doses from
cumulative delivered insulin — recovering only 62% of deliveries. The live sites carry the
treatments outright, so the reconstruction is unnecessary.

User ids are the site INDEX only (S01..S21); no handle, URL or token is stored or printed.
"""
import datetime as dt
import json
import os
import subprocess
import sys

EXTRACT = os.path.expanduser("~/StudioProjects/Boost-AAPS-core/backtesting/scripts/extractor")
SITES = json.load(open("/Users/timstreet/SID-evaluation/multi_user/sites.json"))
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
since = (dt.datetime.now(dt.UTC) - dt.timedelta(days=DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

for i, s in enumerate(SITES, 1):
    uid = f"S{i:02d}"
    if not s.get("url") or s.get("token") is None:
        # some registry entries carry a null token (open sites or lapsed access)
        print(f"[skip] {uid} no token in registry", flush=True)
        continue
    for script in ("boost_extractor.py", "boost_treatments.py"):
        r = subprocess.run(
            [sys.executable, os.path.join(EXTRACT, script),
             "--url", s["url"], "--token", s.get("token") or "", "--user-id", uid, "--since", since],
            capture_output=True, text=True)
        out = (r.stdout + r.stderr).replace(s["url"], "<site>").replace(s["token"], "<token>")
        tail = [l for l in out.splitlines() if l.strip()][-1:] or ["(no output)"]
        tag = "ok " if r.returncode == 0 else "FAIL"
        print(f"[{tag}] {uid} {script.split('_')[1][:9]:<10} {tail[0][:90]}", flush=True)
