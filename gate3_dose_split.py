#!/usr/bin/env python3
"""GATE 3 — does the action peak differ between LARGE and SMALL doses? (2026-08-04)

Pharmacology says it should: rapid analogues peak later and last longer as the dose grows, because
absorption from a bigger subcutaneous depot is slower relative to its volume. If that holds in loop
data, a single fitted curve is a compromise that fits neither the micro-boluses nor the meal
boluses, and the loop's IOB is wrong in opposite directions at the two ends.

MODEL. Same as Gate 2, but the delivered insulin is split at a dose threshold and each stratum gets
its OWN action peak, with its own amplitude per window:

    dBG_i = -k_s * conv(small, act(.; peak_s, DIA))_i
            -k_l * conv(large, act(.; peak_l, DIA))_i  +  c  +  drift_i  +  e_i

k_s, k_l, c and drift are profiled out per window by OLS, so only (peak_s, peak_l) are searched.
Separate amplitudes matter: they stop a difference in EFFECT SIZE between strata from masquerading
as a difference in TIMING.

WHY A NEGATIVE CONTROL COMES FIRST. Run the same two-kernel fit through Gate 1, on logged IOB. The
app applies ONE configured curve to every dose regardless of size, so the truth there is
peak_s == peak_l. Any split Gate 1 reports is an artefact of the estimator or of the dose records,
not pharmacology. On this cohort that control passed for four users (differences -0.9 to +3.1 min)
and failed for one at +48.7 min — which turned out to be a data problem with that user's large
doses, not a discovery. Do not interpret a Gate 3 result for a user who fails the Gate 1 control.

WHAT THIS CANNOT DO. Overnight fasting windows are chosen precisely because no meal is present, so
they are dominated by small automatic boluses. Users differ enormously in how much large-dose
content survives that filter — from 0.9% of doses above 1 U to 17.6%. Where large doses are rare
the split is not estimable at all, and the script says so rather than returning a number.

Usage:
  python3 gate3_dose_split.py --user <id> --tz <tz> --dia <min> [--threshold 1.0]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from gate1_recover_known_curve import activity
from gate2_peak_from_glucose import load, build_windows

HERE = os.path.dirname(os.path.abspath(__file__))


def split_regressors(ts, d_s, d_u, peak_s, peak_l, dia, thr):
    small = d_u < thr
    xs = np.zeros(len(ts))
    xl = np.zeros(len(ts))
    for s, u in zip(d_s[small], d_u[small]):
        xs += u * activity((ts - s) / 60.0, peak_s, dia)
    for s, u in zip(d_s[~small], d_u[~small]):
        xl += u * activity((ts - s) / 60.0, peak_l, dia)
    return xs, xl


def sse(peaks, windows, dia, thr):
    ps, pl = peaks
    tot = 0.0
    for ts, bg, d_s, d_u in windows:
        xs, xl = split_regressors(ts, d_s, d_u, ps, pl, dia, thr)
        y = np.diff(bg)
        A = np.column_stack([xs[:-1], xl[:-1], np.ones(len(y)), (ts[:-1] - ts[0]) / 3600.0])
        if np.std(A[:, 0]) < 1e-12 or np.std(A[:, 1]) < 1e-12:
            continue
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        tot += float(np.sum((y - A @ beta) ** 2))
    return tot


def fit(windows, dia, thr, grid=np.arange(15.0, 121.0, 7.5)):
    best, arg = np.inf, (np.nan, np.nan)
    for ps in grid:
        for pl in grid:
            v = sse((ps, pl), windows, dia, thr)
            if v < best:
                best, arg = v, (ps, pl)
    # local refinement on the coarse winner
    for step in (3.0, 1.0):
        for _ in range(6):
            improved = False
            for dps, dpl in ((step, 0), (-step, 0), (0, step), (0, -step)):
                cand = (min(max(arg[0] + dps, 10.0), 150.0), min(max(arg[1] + dpl, 10.0), 150.0))
                v = sse(cand, windows, dia, thr)
                if v < best:
                    best, arg, improved = v, cand, True
            if not improved:
                break
    return arg, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tz", required=True)
    ap.add_argument("--dia", type=float, required=True)
    ap.add_argument("--threshold", type=float, default=1.0, help="U; doses at or above are 'large'")
    ap.add_argument("--since")
    ap.add_argument("--boot", type=int, default=120)
    ap.add_argument("--out")
    a = ap.parse_args()

    dec, tre = load(a.user, a.tz)
    if a.since:
        cut = pd.Timestamp(a.since, tz="UTC")
        dec = dec[dec.ts >= cut]; tre = tre[tre.ts >= cut - pd.Timedelta(hours=6)]
    wins = build_windows(dec, tre)

    # A window is only informative about BOTH peaks if it contains both kinds of dose.
    usable = [w for w in wins if (w[3] < a.threshold).any() and (w[3] >= a.threshold).any()]
    frac_large = float(np.mean(np.concatenate([w[3] for w in wins]) >= a.threshold)) if wins else 0.0

    L, P = [], None
    P = L.append
    P("# Gate 3 — does the peak differ by dose size?\n")
    P(f"User **{a.user}**, threshold {a.threshold:g} U, DIA held at {a.dia:.0f} min.\n")
    P(f"\n{len(wins)} isolated fasting windows, of which **{len(usable)} contain both a sub-"
      f"{a.threshold:g} U and a {a.threshold:g} U+ dose**. {100 * frac_large:.1f}% of in-window "
      f"doses are large.\n")

    if len(usable) < 8:
        P("\n**NOT ESTIMABLE.** Fewer than 8 windows carry both dose classes, so the two peaks are "
          "not separately identified. This is the normal outcome for a user whose overnight "
          "windows contain only automatic micro-boluses — which is most of them, because the "
          "window filter deliberately excludes meals.\n")
        open(a.out or os.path.join(HERE, "GATE3_REPORT.md"), "w").write("\n".join(L))
        print("\n".join(L)); return

    (ps, pl), _ = fit(usable, a.dia, a.threshold)
    rng = np.random.default_rng(20260804)
    diffs, pss, pls = [], [], []
    for _ in range(a.boot):
        idx = rng.integers(0, len(usable), len(usable))
        (bs, bl), _ = fit([usable[j] for j in idx], a.dia, a.threshold)
        pss.append(bs); pls.append(bl); diffs.append(bl - bs)
    dlo, dhi = np.percentile(diffs, [2.5, 97.5])
    slo, shi = np.percentile(pss, [2.5, 97.5])
    llo, lhi = np.percentile(pls, [2.5, 97.5])

    P("\n| stratum | peak (min) | 95% CI |")
    P("|---|---|---|")
    P(f"| small (< {a.threshold:g} U) | **{ps:.1f}** | [{slo:.1f}, {shi:.1f}] |")
    P(f"| large (>= {a.threshold:g} U) | **{pl:.1f}** | [{llo:.1f}, {lhi:.1f}] |")
    P(f"\n**Difference (large - small): {pl - ps:+.1f} min, 95% CI [{dlo:+.1f}, {dhi:+.1f}]**\n")
    P("\n" + ("The interval excludes zero, so the two strata are distinguishable in this data.\n"
              if (dlo > 0 or dhi < 0) else
              "The interval spans zero: **unproven**. Splitting the dose path on size is not "
              "supported by this data, which is not the same as the strata being equal.\n"))
    P("\nRead against the Gate 1 two-kernel negative control for the same user. That control has a "
      "known answer of zero difference, so it calibrates how much apparent split this estimator "
      "produces from artefact alone.\n")

    open(a.out or os.path.join(HERE, "GATE3_REPORT.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
