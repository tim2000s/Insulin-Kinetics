#!/usr/bin/env python3
"""Why do observed peaks fall so far below published pharmacodynamics?

Across this cohort the observed action peak lands at 25-50 minutes. Published euglycaemic-clamp
data for the same analogues put the maximum glucose-lowering effect at one to three hours, and the
delivery systems assume 45-75. Three descriptions of ostensibly the same quantity differ by factors
of two to four. Omitted basal insulin has been ruled out (gate4_with_basal.py: shifts of 0-5 min).

This tests a physiological explanation. A euglycaemic clamp holds glucose CONSTANT by infusing
glucose to match insulin action; that is the point of the design, and it deliberately prevents the
counter-regulatory response. Free-living glucose is not held constant. It falls, and as it falls
counter-regulation opposes the fall — glucagon, reduced insulin-independent uptake, eventually
hepatic glucose release. That opposition grows with the depth and duration of the fall, so it
truncates the later part of the observed response while leaving the early part intact. The
resulting dBG impulse response would peak EARLIER and decay FASTER than the clamp GIR curve, even
with identical underlying insulin kinetics.

The prediction is directional and testable: where glucose stays high, counter-regulation is weak
and the observed peak should sit later; where glucose runs low, it should sit earlier. The estimator
is therefore run on subsets of each user's record split by glucose level, with everything else held
identical.

A confound to keep in view: the loop doses differently at different glucose levels, so the dose
pattern is not identical between strata. The test is directional evidence, not proof.

Usage:
  python3 counterregulation_test.py --config cohort.json [--jobs 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os

import numpy as np

from gate4_deconvolution import design, fit_fir, gcv, load_grid, peak_of

HERE = os.path.dirname(os.path.abspath(__file__))


def peak_for(user, tz, bg_lo, bg_hi, max_lag=360.0):
    grid, bg, dose, ok, clock, day, _ = load_grid(user, tz)
    sel = ok & np.isfinite(bg) & (bg >= bg_lo) & (bg < bg_hi)
    if sel.sum() < 1500:
        return None, int(sel.sum())
    X_d, X_c, X_n, rows, K = design(dose, clock, day, sel, max_lag)
    if len(rows) < 1200:
        return None, len(rows)
    y = bg[rows + 1] - bg[rows]
    lam = gcv(y, X_d, X_c, X_n, np.logspace(1, 5, 5))
    return peak_of(fit_fir(y, X_d, X_c, X_n, lam)[0]), len(rows)


def job(item):
    user, tz = item
    out = {"user": user}
    for lbl, lo, hi in (("low", 40, 110), ("mid", 110, 160), ("high", 160, 350)):
        pk, n = peak_for(user, tz, lo, hi)
        out[lbl] = pk
        out[lbl + "_n"] = n
    print(f"  [{user}] low={out['low']} mid={out['mid']} high={out['high']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "cohort.json"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    users = json.load(open(a.config))

    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(job, users.items()))

    L, P = [], None
    P = L.append
    P("# Does counter-regulation explain the short observed peaks?\n")
    P("\nThe estimator is run on each participant's record split by glucose level, everything else "
      "held identical. If counter-regulation truncates the response, the peak should sit later "
      "where glucose is high and earlier where it is low.\n")
    P("\n| user | glucose < 110 | 110-160 | > 160 | high − low |")
    P("|---|---|---|---|---|")
    deltas = []
    for r in sorted(res, key=lambda x: x["user"]):
        lo, mid, hi = r["low"], r["mid"], r["high"]
        d = (hi - lo) if (lo is not None and hi is not None) else None
        if d is not None:
            deltas.append(d)
        f = lambda v: f"{v:.0f}" if v is not None else "—"
        P(f"| {r['user']} | {f(lo)} | {f(mid)} | {f(hi)} | {f(d) if d is not None else '—'} |")
    if deltas:
        d = np.array(deltas, dtype=float)
        pos = int((d > 0).sum())
        P(f"\nAcross {len(d)} participants with both strata estimable, the peak in the high-glucose "
          f"stratum is later in **{pos} of {len(d)}**, median difference "
          f"**{np.median(d):+.0f} min** (mean {d.mean():+.1f}).\n")
        P("\n" + ("Direction is consistent with counter-regulation truncating the response when "
                  "glucose falls: the clamp prevents that by design, which would make published "
                  "peaks later than anything observable in free-living data.\n" if np.median(d) > 5
                  else "No consistent difference by glucose level, so counter-regulation does not "
                       "appear to explain the discrepancy on this evidence.\n"))
        P("\nThe loop doses differently at different glucose levels, so the strata differ in dose "
          "pattern as well as in glucose. This is directional evidence, not proof.\n")

    open(a.out or os.path.join(HERE, "COUNTERREGULATION.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
