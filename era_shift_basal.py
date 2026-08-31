#!/usr/bin/env python3
"""Did the impulse response change at a known date, once basal insulin is admitted?

`era_shift.py` regresses glucose change on BOLUS insulin alone. For a closed-loop participant that
omits roughly two fifths of the insulin, and at a concentration change it omits it asymmetrically:
diluting the preparation doubles the units recorded for basal exactly as it does for bolus, so the
omitted term has one scale before the boundary and another after it. That is the structure that
biases a before/after comparison, so the era test has to carry basal explicitly.

Net basal relative to schedule (`iob_netbasalinsulin`) is NULL for some participants, which is why
`gate4_with_basal.py` cannot run for them. The delivered rate the loop set each cycle (`sug_rate`)
is populated instead, and absolute delivery works here because the per-day intercepts absorb each
day's mean level; what identifies the basal kernel is the departure from that day's mean, which is
the reactive modulation the loop applies.

    dg_t = - SUM_k b_k^pre  d_{t-k} 1[t<T]  - SUM_k b_k^post d_{t-k} 1[t>=T]
           - SUM_k c_k^pre  a_{t-k} 1[t<T]  - SUM_k c_k^post a_{t-k} 1[t>=T]
           + tod_clock(t) + day_n(t) + e_t

Both dose kernels are held non-negative; both basal kernels are left free, since basal is cut in
anticipation of a fall and a negative departure acting through a positive kernel must be able to
raise glucose. All four get their own second-difference penalty.

The placebo calibration is unchanged and is the whole point: a 26-day kernel returns a number
whatever the truth is, so the estimate is judged against boundaries placed where nothing changed.

Usage:
  python3 era_shift_basal.py --user <id> --tz <tz> --boundary <YYYY-MM-DD> [--placebo 24]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import lsq_linear

from gate4_deconvolution import (ALIGN_DEFAULT, DSN, STEP_S, gcv, load_grid, peak_of, peak_shape)

HERE = os.path.dirname(os.path.abspath(__file__))
FFILL_BINS = 6          # carry a set rate forward at most 30 min before treating it as unknown


def basal_series(user, grid, t0, ffill=FFILL_BINS):
    """Basal insulin delivered per 5-minute bin, from the rate the loop set each cycle."""
    conn = psycopg2.connect(DSN)
    d = pd.read_sql("""
        SELECT DISTINCT ON (floor(ts_epoch/300.0)) ts_epoch, sug_rate
        FROM boost_decisions
        WHERE user_id = %s AND sug_rate IS NOT NULL
        ORDER BY floor(ts_epoch/300.0), ts_epoch DESC""", conn, params=(user,))
    conn.close()
    if len(d) < 500:
        return None, 0.0
    d = d.sort_values("ts_epoch")
    n = len(grid)
    idx = ((d.ts_epoch.values.astype(float) - t0) / STEP_S).round().astype(int)
    keep = (idx >= 0) & (idx < n)
    rate = np.full(n, np.nan)
    rate[idx[keep]] = d.sug_rate.values.astype(float)[keep]
    rate = pd.Series(rate).ffill(limit=ffill).values
    cover = float(np.isfinite(rate).mean())
    return np.nan_to_num(rate, nan=0.0) * STEP_S / 3600.0, cover


def fit_four(y, blocks, X_c, X_n, lam, nonneg):
    """Four smoothness-penalised lag kernels plus shared drift terms."""
    K = blocks[0].shape[1]
    nb = len(blocks)
    X = np.hstack(list(blocks) + [X_c, X_n])
    p = X.shape[1]
    D = np.zeros((nb * (K - 2), p))
    for j in range(nb):
        off = j * K
        for i in range(K - 2):
            r = j * (K - 2) + i
            D[r, off + i] = 1.0
            D[r, off + i + 1] = -2.0
            D[r, off + i + 2] = 1.0
    A = np.vstack([X, np.sqrt(lam) * D])
    b = np.concatenate([y, np.zeros(nb * (K - 2))])
    lo = np.full(p, -np.inf)
    hi = np.full(p, np.inf)
    for j in nonneg:
        lo[j * K:(j + 1) * K] = 0.0
    r = lsq_linear(A, b, bounds=(lo, hi), max_iter=200, tol=1e-8)
    return [r.x[j * K:(j + 1) * K] for j in range(nb)]


def build(user, tz, max_lag=360.0, thin=3):
    grid, bg, dose, ok, clock, day, _hs, _ne, _np = load_grid(user, tz)
    bas, cover = basal_series(user, grid, grid[0])
    if bas is None:
        raise SystemExit(f"{user}: no basal rate series")
    K = int(max_lag / 5)
    rows = np.flatnonzero(ok[:-1])
    rows = rows[rows >= K]
    if thin > 1:
        rows = rows[::thin]
    y = bg[rows + 1] - bg[rows]
    Xd = -np.column_stack([dose[rows - k] for k in range(K + 1)])
    Xb = -np.column_stack([bas[rows - k] for k in range(K + 1)])
    cl, dy = clock[rows], day[rows]
    X_c = (cl[:, None] == np.unique(cl)[None, :]).astype(float)[:, 1:]
    X_n = (dy[:, None] == np.unique(dy)[None, :]).astype(float)
    return grid, rows, y, Xd, Xb, X_c, X_n, K, cover, bas


def era_fit(grid, rows, y, Xd, Xb, X_c, X_n, t_bound, lam, with_basal=True):
    post = grid[rows] >= t_bound
    m_pre, m_post = (~post)[:, None], post[:, None]
    blocks = [Xd * m_pre, Xd * m_post]
    nonneg = [0, 1]
    if with_basal:
        blocks += [Xb * m_pre, Xb * m_post]
    keep_n = X_n[:, X_n.sum(axis=0) > 0]
    out = fit_four(y, blocks, X_c, keep_n, lam, nonneg)
    return out, int(post.sum())


def describe(b, align):
    if b.max() <= 0:
        return dict(peak=float("nan"), conc=float("nan"), prom=float("nan"), area=0.0)
    c, p = peak_shape(b)
    return dict(peak=peak_of(b) + align, conc=c, prom=p, area=float(b.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tz", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--placebo", type=int, default=24)
    ap.add_argument("--history-days", type=float, default=180.0)
    ap.add_argument("--out")
    a = ap.parse_args()

    align = ALIGN_DEFAULT
    grid, rows, y, Xd, Xb, X_c, X_n, K, cover, bas = build(a.user, a.tz)
    t_bound = pd.Timestamp(a.boundary, tz="UTC").timestamp()
    t_start = t_bound - a.history_days * 86400
    sel = grid[rows] >= t_start
    rows_s, y_s, Xd_s, Xb_s = rows[sel], y[sel], Xd[sel], Xb[sel]
    X_c_s, X_n_s = X_c[sel], X_n[sel]
    post_days = (grid[rows_s].max() - t_bound) / 86400

    # Smoothing must be selected on the design actually fitted. The with-basal design carries
    # twice the kernel parameters, and GCV lands an order of magnitude lower on it; reusing the
    # bolus-only value over-smooths the four-kernel fit.
    keep_n = X_n_s[:, X_n_s.sum(axis=0) > 0]
    lam_o = gcv(y_s, Xd_s, X_c_s, keep_n, np.logspace(1, 5, 5))
    lam = gcv(y_s, np.hstack([Xd_s, Xb_s]), X_c_s, keep_n, np.logspace(1, 5, 5))

    (bo_pre, bo_post), _ = era_fit(grid, rows_s, y_s, Xd_s, Xb_s, X_c_s, X_n_s, t_bound, lam_o,
                                   with_basal=False)
    (b_pre, b_post, c_pre, c_post), n_post = era_fit(grid, rows_s, y_s, Xd_s, Xb_s, X_c_s, X_n_s,
                                                     t_bound, lam, with_basal=True)
    pre, post = describe(b_pre, align), describe(b_post, align)
    o_pre, o_post = describe(bo_pre, align), describe(bo_post, align)

    rng = np.random.default_rng(20260807)
    lo = grid[rows_s].min() + 30 * 86400
    hi = t_bound - post_days * 86400 - 7 * 86400
    null = []
    if a.placebo > 0 and hi > lo:
        for tb in rng.uniform(lo, hi, a.placebo):
            keep = grid[rows_s] < tb + post_days * 86400
            if keep.sum() < 500:
                continue
            try:
                (bp, bq, _cp, _cq), npq = era_fit(grid, rows_s[keep], y_s[keep], Xd_s[keep],
                                                  Xb_s[keep], X_c_s[keep], X_n_s[keep], tb, lam)
            except Exception as e:                                     # noqa: BLE001
                print(f"  placebo fit failed: {type(e).__name__}", flush=True)
                continue
            d, dp = describe(bq, align), describe(bp, align)
            d["ratio"] = (d["area"] / dp["area"]) if dp["area"] > 0 else float("nan")
            null.append(d)
            print(f"  placebo {len(null)}/{a.placebo}: peak {d['peak']:.0f}, "
                  f"area ratio {d['ratio']:.2f}, n={npq}", flush=True)

    npk = np.array([d["peak"] for d in null if np.isfinite(d["peak"])])

    L = []
    P = L.append
    P(f"# Did the response change on {a.boundary}, with basal admitted? — user {a.user}\n")
    P(f"\n{len(y_s):,} thinned samples over the {a.history_days:.0f} days before the boundary plus "
      f"{post_days:.1f} days after it; {n_post:,} fall after. Basal delivery is reconstructed from "
      f"the rate set each cycle, covering {100 * cover:.0f}% of bins, {bas.sum():.0f} U over the "
      f"whole grid. Both eras share one time-of-day profile and per-day intercepts. "
      f"Smoothing is chosen by GCV on each design separately: {lam_o:g} bolus-only, "
      f"{lam:g} with basal.\n")

    P("\n## Bolus kernel either side of the boundary\n")
    P("\n| model | era | peak (min) | concentration | prominence | area (per unit) |")
    P("|---|---|---|---|---|---|")
    P(f"| bolus only | before | {o_pre['peak']:.0f} | {o_pre['conc']:.2f} | {o_pre['prom']:.2f} | "
      f"{o_pre['area']:.3f} |")
    P(f"| bolus only | after | {o_post['peak']:.0f} | {o_post['conc']:.2f} | {o_post['prom']:.2f} "
      f"| {o_post['area']:.3f} |")
    P(f"| bolus + basal | before | {pre['peak']:.0f} | {pre['conc']:.2f} | {pre['prom']:.2f} | "
      f"{pre['area']:.3f} |")
    P(f"| bolus + basal | after | {post['peak']:.0f} | {post['conc']:.2f} | {post['prom']:.2f} | "
      f"{post['area']:.3f} |")
    P(f"\nAdmitting basal moves the before-era peak by {pre['peak'] - o_pre['peak']:+.0f} min and "
      f"the after-era peak by {post['peak'] - o_post['peak']:+.0f} min.\n")

    ratio = (post["area"] / pre["area"]) if pre["area"] > 0 else float("nan")
    if np.isfinite(ratio):
        P(f"\nArea ratio after/before: **{ratio:.2f}** per unit delivered. Halving the strength "
          f"doubles the units recorded for the same mass, so the change itself predicts a ratio "
          f"near 0.5 independent of any shift in timing.\n")

    for nm, kk in (("before", c_pre), ("after", c_post)):
        if np.any(kk != 0):
            share = 100 * np.abs(kk).sum() / (np.abs(kk).sum() + max(
                b_pre.sum() if nm == "before" else b_post.sum(), 1e-9))
            P(f"\nThe {nm}-era basal kernel peaks at "
              f"{peak_of(np.abs(kk), min_lag_min=0) + align:.0f} min and carries {share:.0f}% of "
              f"that era's total absolute kernel mass.\n")

    P("\n## What a window this short can resolve\n")
    if len(npk) >= 5:
        P(f"\nThe same fit was repeated with the boundary moved to {len(npk)} earlier dates, each "
          f"with a post-window of the same {post_days:.1f} days, where nothing changed.\n")
        P(f"\nNull distribution of the post-window peak: median **{np.median(npk):.0f} min**, "
          f"range {npk.min():.0f} to {npk.max():.0f}, 10th-90th percentile "
          f"{np.percentile(npk, 10):.0f} to {np.percentile(npk, 90):.0f}.\n")
        nra = np.array([d["ratio"] for d in null if np.isfinite(d.get("ratio", np.nan))])
        if len(nra) >= 5 and np.isfinite(ratio):
            rp = 100.0 * (nra < ratio).mean()
            alt = 0.5 * nra
            P(f"\nNull distribution of the area ratio: median **{np.median(nra):.2f}**, "
              f"10th-90th percentile {np.percentile(nra, 10):.2f} to "
              f"{np.percentile(nra, 90):.2f}. The observed {ratio:.2f} sits at the {rp:.0f}th "
              f"percentile. Under a true halving the estimate would be distributed like that null "
              f"scaled by one half: median {np.median(alt):.2f}, 10th-90th "
              f"{np.percentile(alt, 10):.2f} to {np.percentile(alt, 90):.2f}.\n")
        if np.isfinite(post["peak"]):
            pctl = 100.0 * (npk < post["peak"]).mean()
            extreme = min(pctl, 100 - pctl)
            P(f"\nThe observed post-boundary peak of {post['peak']:.0f} min sits at the "
              f"{pctl:.0f}th percentile of that null, so a value at least as extreme arises by "
              f"chance about {2 * extreme:.0f}% of the time when nothing has changed.\n")
            verdict = ("**No indication of a change.** " if 2 * extreme > 20 else
                       "**Suggestive, not conclusive.** " if 2 * extreme > 5 else
                       "**The shift exceeds what a window this short produces by chance.** ")
            P(f"\n{verdict}The null spans {npk.max() - npk.min():.0f} min, which is the resolution "
              f"a window of this length has.\n")
    else:
        P("\nToo few placebo windows completed to calibrate the null.\n")

    open(a.out or os.path.join(HERE, f"ERAB_{a.user}.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
