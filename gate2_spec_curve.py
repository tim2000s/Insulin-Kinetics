#!/usr/bin/env python3
"""SPECIFICATION CURVE for Gate 2 — how much does the answer depend on the analyst? (2026-08-04)

A single Gate 2 number with a bootstrap interval understates the real uncertainty, because the
bootstrap holds every analysis choice fixed. This refits across a grid of defensible choices —
window length, night hours, minimum insulin, drift term on or off, DIA prior — and reports the
spread. If that spread is larger than the effect being claimed, the claim is an artefact of the
specification and not a finding.

It also reports the collinearity that drives the spread. Over a fasting window the insulin activity
profile is slowly varying and largely monotone, so it is nearly a straight line in time — which is
exactly what the dawn-ramp control also is. Median |r(insulin regressor, time)| runs 0.73-0.81 at
4-hour windows and up to 0.99 at 2-hour windows. The two terms compete for the same slow component,
and whichever one is allowed to absorb it decides the answer.

Usage:
  python3 gate2_spec_curve.py --user <id> --tz <tz> --dia <min> [--configured 55]
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

from gate2_peak_from_glucose import load, build_windows, fit_peak, window_regressor

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tz", required=True)
    ap.add_argument("--dia", type=float, required=True)
    ap.add_argument("--configured", type=float, help="configured peak, for a reference line")
    ap.add_argument("--out")
    a = ap.parse_args()

    dec, tre = load(a.user, a.tz)
    est = []
    for hours, night, mi, drift, dscale in itertools.product(
            (3.0, 4.0, 5.0), ((0, 7), (0, 4), (1, 6)), (0.3, 1.0), (True, False), (0.7, 1.0, 1.4)):
        w = build_windows(dec, tre, hours=hours, night=night, min_insulin=mi)
        if len(w) < 8:
            continue
        p, _ = fit_peak(w, a.dia * dscale, linear_drift=drift)
        est.append((p, drift, hours, mi, dscale, len(w)))
    if len(est) < 5:
        print("too few viable specifications"); return

    v = np.array([e[0] for e in est])
    on = np.array([e[0] for e in est if e[1]])
    off = np.array([e[0] for e in est if not e[1]])

    base = build_windows(dec, tre)
    pk0, _ = fit_peak(base, a.dia, True)
    rs = []
    for ts, bg, d_s, d_u in base:
        x = window_regressor(ts, d_s, d_u, pk0, a.dia)[:-1]
        if np.std(x) > 1e-12:
            rs.append(abs(np.corrcoef(x, ts[:-1])[0, 1]))
    rs = np.array(rs)

    L, P = [], None
    P = L.append
    P("# Gate 2 specification curve\n")
    P(f"User **{a.user}**, {len(est)} defensible specifications.\n")
    P(f"\n| | peak (min) |\n|---|---|")
    P(f"| median across specifications | **{np.median(v):.1f}** |")
    P(f"| range | [{v.min():.1f}, {v.max():.1f}] |")
    P(f"| interquartile range | [{np.percentile(v, 25):.1f}, {np.percentile(v, 75):.1f}] |")
    P(f"| drift ramp ON (median) | {np.median(on):.1f} |")
    P(f"| drift ramp OFF (median) | {np.median(off):.1f} |")
    P(f"| swing from that one choice | **{abs(np.median(on) - np.median(off)):.1f} min** |")
    if a.configured:
        P(f"| within +-10 min of configured ({a.configured:.0f}) | "
          f"{100 * np.mean(abs(v - a.configured) <= 10):.0f}% of specifications |")
    P(f"\nCollinearity of the insulin regressor with time, within windows: median "
      f"|r| = {np.median(rs):.2f}, {100 * np.mean(rs > 0.9):.0f}% of windows above 0.9.\n")
    P("\nA swing of more than about 10 minutes from the drift choice alone means the design cannot "
      "separate insulin action from the dawn ramp for this user, and no bootstrap interval on a "
      "single specification will reveal that.\n")

    open(a.out or os.path.join(HERE, "GATE2_SPEC_CURVE.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
