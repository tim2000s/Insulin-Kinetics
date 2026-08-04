#!/usr/bin/env python3
"""RUN COHORT — Gate 1 and Gate 4 across a set of users, into one comparison table.

Gate 1 recovers what the loop BELIEVES (its configured curve, by algebraic identity from logged
IOB). Gate 4 recovers what the insulin appears to DO (non-parametric deconvolution of glucose).
The interesting quantity is the gap between them, and that only means anything when both are
produced the same way for every user, which is what this driver is for.

Also runs Gate 1 on two disjoint halves of the period. A user whose configured curve changed
mid-period will show it here and nowhere else — on one cohort this caught a switch from a ~75-80
min curve to a ~55 min curve that no treatment record mentioned.

USER LIST AND TIMEZONES are supplied at runtime and never committed. Either:
    --users "A:Etc/GMT-1,B:Etc/GMT-2"
    --config cohort.json          {"A": "Etc/GMT-1", "B": "Etc/GMT-2"}
For a fixed window with no DST transition inside it, a fixed offset is the safe choice — note
Etc/GMT-1 is UTC+1, the sign is inverted. A wrong timezone silently selects the wrong hours.

Usage:
  python3 run_cohort.py --config cohort.json [--days 28] [--workers 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, args, out_path):
    r = subprocess.run([sys.executable, os.path.join(HERE, script)] + args + ["--out", out_path],
                       capture_output=True, text=True, cwd=HERE)
    return r.stdout, r.returncode


def f(m, i=1, cast=float):
    return cast(m.group(i)) if m else None


def parse_gate1(txt):
    return dict(
        peak=f(re.search(r"peak \(min\) \|[^|]*\| \*\*([\d.]+)\*\*", txt)),
        dia=f(re.search(r"DIA \(min\)\s+\|[^|]*\| \*\*([\d.]+)\*\*", txt)),
        rel=f(re.search(r"relative ([\d.]+)\)", txt)),
        dia_unidentified="NOT identified" in txt,
        residual_flag="Residual is large" in txt,
    )


def parse_gate4(txt):
    return dict(
        peak=f(re.search(r"Peak of the estimated activity curve: (\d+) min", txt)),
        lo=f(re.search(r"95% CI\s*\[(\d+), (\d+)\]", txt), 1),
        hi=f(re.search(r"95% CI\s*\[(\d+), (\d+)\]", txt), 2),
        t50=f(re.search(r"Half of the total effect has landed by (\d+) min", txt)),
        nsample=f(re.search(r"([\d,]+) usable 5-minute samples",
                            txt.replace(",", "")), 1, int),
    )


def job(item, days, boot):
    user, tz = item
    d = {"user": user, "tz": tz}
    g1, _ = run("gate1_recover_known_curve.py",
                ["--user", user, "--days", str(days), "--boot", "60"],
                os.path.join(HERE, f".cohort_g1_{user}.md"))
    d["gate1"] = parse_gate1(g1)
    h1, _ = run("gate1_recover_known_curve.py",
                ["--user", user, "--days", str(days // 2), "--offset-days", str(days // 2),
                 "--boot", "40"], os.path.join(HERE, f".cohort_g1h1_{user}.md"))
    h2, _ = run("gate1_recover_known_curve.py",
                ["--user", user, "--days", str(days // 2), "--boot", "40"],
                os.path.join(HERE, f".cohort_g1h2_{user}.md"))
    d["half1"], d["half2"] = parse_gate1(h1)["peak"], parse_gate1(h2)["peak"]
    g4, _ = run("gate4_deconvolution.py",
                ["--user", user, "--tz", tz, "--boot", str(boot)],
                os.path.join(HERE, f".cohort_g4_{user}.md"))
    d["gate4"] = parse_gate4(g4)
    print(f"  [{user}] done", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON mapping user id -> IANA timezone")
    ap.add_argument("--users", help='inline, e.g. "A:Etc/GMT-1,B:Etc/GMT-2"')
    ap.add_argument("--days", type=int, default=28, help="Gate 1 window; halves are days/2 each")
    ap.add_argument("--boot", type=int, default=40, help="Gate 4 bootstrap draws (slow)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "COHORT_REPORT.md"))
    a = ap.parse_args()

    if a.config:
        users = json.load(open(a.config))
    elif a.users:
        users = dict(kv.split(":", 1) for kv in a.users.split(","))
    else:
        ap.error("supply --config or --users")

    print(f"running {len(users)} users, {a.days}-day Gate 1 window")
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(lambda it: job(it, a.days, a.boot), users.items()))
    res.sort(key=lambda r: r["user"])

    L, P = [], None
    P = L.append
    P("# Cohort — configured curve vs observed response\n")
    P(f"{len(res)} users. Gate 1 window {a.days} days, split into two disjoint halves of "
      f"{a.days // 2} days to detect a mid-period change. Gate 4 uses full history.\n")
    P("\n| user | configured (Gate 1) | DIA | fit | half 1 | half 2 | observed (Gate 4) | gap |")
    P("|---|---|---|---|---|---|---|---|")
    for r in res:
        g1, g4 = r["gate1"], r["gate4"]
        if g1["peak"] is None:
            P(f"| {r['user']} | no fit | | | | | | |"); continue
        dia = "n/a" if g1["dia_unidentified"] else f"{g1['dia']:.0f}"
        fit = f"{g1['rel']:.3f}" + ("!" if g1["residual_flag"] else "")
        obs = f"{g4['peak']:.0f}" if g4["peak"] is not None else "-"
        gap = (f"{g4['peak'] - g1['peak']:+.0f}" if g4["peak"] is not None else "-")
        h1 = f"{r['half1']:.0f}" if r["half1"] else "-"
        h2 = f"{r['half2']:.0f}" if r["half2"] else "-"
        P(f"| {r['user']} | {g1['peak']:.1f} | {dia} | {fit} | {h1} | {h2} | {obs} | {gap} |")

    P("\n**DIA `n/a`** means the leverage test found no alternative duration the data can "
      "distinguish — the recovered number is arbitrary, do not quote it.\n")
    P("\n**fit with `!`** means the relative residual exceeds 15% on what is an exact identity, so "
      "that user's dose records are not what the app saw. Treat their curve as provisional and run "
      "`gate1_controls.py` before using it.\n")
    P("\n**half 1 vs half 2** differing by more than a few minutes indicates the configured curve "
      "CHANGED mid-period. Localise it with rolling `--offset-days` windows.\n")
    P("\n**gap** is observed minus configured. Sensor lag biases the observed value late by a few "
      "minutes and is not corrected, so small negative gaps are the conservative direction.\n")

    open(a.out, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
