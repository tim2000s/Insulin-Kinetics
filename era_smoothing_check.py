#!/usr/bin/env python3
"""Is the era-shift peak stable under the smoothing choice, or does it move with it?

The peak of a kernel estimated over a short post-boundary window can be an artefact of the
smoothing penalty rather than a property of the response. This refits the with-basal era model
across a range of penalties and reports the peak either side of the boundary and the ratio of
kernel areas, so that a shift can be judged against the spread the penalty alone produces.

It also reports the penalty generalised cross-validation selects on each design separately. The
four-kernel design carries twice the parameters of the bolus-only one and lands an order of
magnitude lower; applying the bolus-only value to it over-smooths and produces a degenerate,
bimodal null.

Run for the dilution boundary this reports on:
  python3 era_smoothing_check.py --user <id> --tz <tz> --boundary <YYYY-MM-DD>
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from era_shift_basal import build, describe, era_fit
from gate4_deconvolution import ALIGN_DEFAULT, gcv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tz", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--history-days", type=float, default=180.0)
    a = ap.parse_args()

    grid, rows, y, Xd, Xb, X_c, X_n, K, cover, bas = build(a.user, a.tz)
    tb = pd.Timestamp(a.boundary, tz="UTC").timestamp()
    sel = grid[rows] >= tb - a.history_days * 86400
    rows_s, y_s, Xd_s, Xb_s = rows[sel], y[sel], Xd[sel], Xb[sel]
    X_c_s, X_n_s = X_c[sel], X_n[sel]
    keep = X_n_s[:, X_n_s.sum(axis=0) > 0]

    lam_o = gcv(y_s, Xd_s, X_c_s, keep, np.logspace(1, 5, 5))
    lam_b = gcv(y_s, np.hstack([Xd_s, Xb_s]), X_c_s, keep, np.logspace(1, 5, 5))
    print(f"GCV selects {lam_o:g} on the bolus-only design and {lam_b:g} with basal admitted\n")

    print(f"{'lambda':>10s} {'pre peak':>10s} {'post peak':>10s} {'area ratio':>11s}")
    for lam in sorted({lam_o * 0.1, lam_o, lam_o * 10, lam_b}):
        (bp, bq, _cp, _cq), _n = era_fit(grid, rows_s, y_s, Xd_s, Xb_s, X_c_s, X_n_s, tb, lam)
        p, q = describe(bp, ALIGN_DEFAULT), describe(bq, ALIGN_DEFAULT)
        ratio = q["area"] / p["area"] if p["area"] > 0 else float("nan")
        print(f"{lam:10g} {p['peak']:10.0f} {q['peak']:10.0f} {ratio:11.2f}")
    print("\nA peak that moves across this range by as much as the shift being claimed is not a "
          "finding. An area ratio that holds across it is.")


if __name__ == "__main__":
    main()
