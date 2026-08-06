#!/usr/bin/env python3
"""Does omitting basal insulin explain why observed peaks fall short of published ones?

Observed action peaks across this cohort land at 25-50 minutes. Published euglycaemic-clamp
pharmacodynamics put the maximum glucose-lowering effect of the same analogues at one to three
hours, and the delivery systems themselves assume 45-75 minutes. Three quantities that should
describe the same thing differ by factors of two to four, so something in the chain is not what it
is taken to be.

One candidate is a plain modelling gap. Gate 4 regresses glucose change on BOLUS insulin only. A
closed loop also modulates basal continuously, and does so in reaction to the same glucose it is
trying to control — cutting basal when it expects a fall, raising it when it does not. That input
is currently absorbed into the intercept and the time-of-day profile, which can only represent the
part of it that is constant or clock-locked. The reactive part is neither.

This adds it. `netbasalinsulin` is the net basal insulin delivered within the lookback window
relative to the scheduled profile, so its increments are the net basal input per interval — signed,
because a zero-temp delivers less than schedule and appears as a negative contribution:

    dg_t = -SUM_k b_k dose_{t-k} - SUM_k c_k basal_{t-k} + tod + day + e

with two independent kernels, each smoothness-penalised, the bolus one constrained non-negative.
The basal kernel is left unconstrained: a negative net basal input acting through a positive kernel
must be allowed to raise glucose.

If the bolus peak moves materially when basal is admitted, the short observed peaks are partly an
artefact of the omission. If it does not, the discrepancy lies elsewhere and this rules one
explanation out.

Usage:
  python3 gate4_with_basal.py --user <id> --tz <tz>
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import lsq_linear

from gate1_recover_known_curve import DSN
from gate4_deconvolution import STEP_S, gcv, load_grid, peak_of

HERE = os.path.dirname(os.path.abspath(__file__))


def basal_series(user, grid, t0):
    """Net basal insulin delivered per 5-minute bin, from increments of netbasalinsulin."""
    conn = psycopg2.connect(DSN)
    d = pd.read_sql("""
        SELECT DISTINCT ON (floor(ts_epoch/300.0)) ts_epoch, iob_netbasalinsulin AS nb
        FROM boost_decisions
        WHERE user_id = %s AND iob_netbasalinsulin IS NOT NULL
        ORDER BY floor(ts_epoch/300.0), ts_epoch DESC""", conn, params=(user,))
    conn.close()
    if len(d) < 100:
        return None
    d = d.sort_values("ts_epoch")
    t = d.ts_epoch.values.astype(float)
    nb = d.nb.values.astype(float)
    out = np.zeros(len(grid))
    gap = np.diff(t)
    ok = (gap > 0) & (gap <= 900)
    inc = np.diff(nb)[ok]
    idx = np.clip(((t[1:][ok] - t0) / STEP_S).round().astype(int), 0, len(grid) - 1)
    np.add.at(out, idx, inc)
    return out


def fit_two(y, X_d, X_b, X_c, X_n, lam):
    """Two smoothness-penalised kernels; bolus non-negative, basal free."""
    K = X_d.shape[1]
    X = np.hstack([X_d, X_b, X_c, X_n])
    p = X.shape[1]
    D = np.zeros((2 * (K - 2), p))
    for i in range(K - 2):                       # bolus block
        D[i, i] = 1.0; D[i, i + 1] = -2.0; D[i, i + 2] = 1.0
    for i in range(K - 2):                       # basal block
        r = K - 2 + i
        D[r, K + i] = 1.0; D[r, K + i + 1] = -2.0; D[r, K + i + 2] = 1.0
    A = np.vstack([X, np.sqrt(lam) * D])
    b = np.concatenate([y, np.zeros(2 * (K - 2))])
    lo = np.full(p, -np.inf); hi = np.full(p, np.inf)
    lo[:K] = 0.0
    r = lsq_linear(A, b, bounds=(lo, hi), max_iter=200, tol=1e-8)
    return r.x[:K], r.x[K:2 * K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tz", required=True)
    ap.add_argument("--max-lag", type=float, default=360.0)
    ap.add_argument("--out")
    a = ap.parse_args()

    grid, bg, dose, ok, clock, day, _ = load_grid(a.user, a.tz)
    t0 = grid[0]
    bas = basal_series(a.user, grid, t0)
    if bas is None:
        print(f"{a.user}: no basal series"); return

    K = int(a.max_lag / 5)
    rows = np.flatnonzero(ok[:-1]); rows = rows[rows >= K]
    y = bg[rows + 1] - bg[rows]
    X_d = -np.column_stack([dose[rows - k] for k in range(K + 1)])
    X_b = -np.column_stack([bas[rows - k] for k in range(K + 1)])
    cl, dy = clock[rows], day[rows]
    X_c = (cl[:, None] == np.unique(cl)[None, :]).astype(float)[:, 1:]
    X_n = (dy[:, None] == np.unique(dy)[None, :]).astype(float)

    lam = gcv(y, X_d, X_c, X_n, np.logspace(1, 5, 5))
    b_only, _ = fit_two(y, X_d, np.zeros_like(X_b), X_c, X_n, lam)
    b_both, c_both = fit_two(y, X_d, X_b, X_c, X_n, lam)

    L, P = [], None
    P = L.append
    P(f"# Does admitting basal insulin move the peak? — user {a.user}\n")
    P(f"\n{len(y):,} samples, {K + 1} lag coefficients per kernel, smoothing {lam:g}. "
      f"Net basal input over the period: {bas.sum():+.0f} U relative to schedule.\n")
    P("\n| model | bolus peak (min) |")
    P("|---|---|")
    P(f"| bolus only (as reported elsewhere) | {peak_of(b_only):.0f} |")
    P(f"| bolus + basal | {peak_of(b_both):.0f} |")
    shift = peak_of(b_both) - peak_of(b_only)
    P(f"\nShift from admitting basal: **{shift:+.0f} min**.\n")
    if np.any(c_both != 0):
        # min_lag_min=0 deliberately: a basal kernel peaking at lag zero IS the finding here
        # (it identifies the controller's reaction rather than insulin action), and the usual
        # guard would mask it by reporting 15.
        P(f"\nThe basal kernel peaks at {peak_of(np.abs(c_both), min_lag_min=0):.0f} min and carries "
          f"{100 * np.abs(c_both).sum() / (np.abs(c_both).sum() + b_both.sum() + 1e-9):.0f}% of "
          f"the total absolute kernel mass.\n")
    P("\n" + ("Admitting basal moves the bolus peak by less than five minutes, so omitting it does "
              "not explain the gap against published values.\n" if abs(shift) < 5 else
              "**Admitting basal moves the bolus peak materially.** The bolus-only estimate is "
              "confounded by the loop's basal modulation and should not be compared with published "
              "pharmacodynamics as though it were a clean bolus response.\n"))

    open(a.out or os.path.join(HERE, f"BASAL_{a.user}.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
