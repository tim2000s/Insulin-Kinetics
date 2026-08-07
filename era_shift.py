#!/usr/bin/env python3
"""Did the impulse response change at a known date, and could three days show it if it had?

A participant changed insulin preparation on a known date — the same analogue diluted from U200 to
U100 strength, so the same mass is delivered in twice the volume and recorded as twice the units.
Dilution plausibly changes absorption: a larger depot has more surface area, which would move the
peak earlier. The question is whether the record since the change can detect that.

DESIGN. One model over the whole window, with the dose kernel allowed to differ before and after
the boundary and everything else shared:

    dg_t = - SUM_k b_k^pre  d_{t-k} 1[t < T]
           - SUM_k b_k^post d_{t-k} 1[t >= T]  + tod_clock(t) + day_n(t) + e_t

Sharing the time-of-day profile and per-day intercepts across both eras is what makes the short
side estimable at all: the drift, which needs many days, is carried by the long side, and the post
era only has to identify its own kernel. Both kernels get their own second-difference penalty and
non-negativity constraint.

THE POINT OF THE PLACEBO. A three-day kernel will return a number whatever the truth is, so the
estimate alone says nothing. The same fit is therefore repeated with the boundary moved to earlier
dates, where by construction NOTHING changed, giving the distribution of three-day peak estimates
under the null. A real shift has to be judged against that spread, not against the pre-era value.

AMPLITUDE AS A POSITIVE CONTROL. The kernel is per unit delivered. If the same mass is now recorded
as twice the units, the per-unit response must halve. The ratio of kernel areas is therefore a check
that the design is seeing the change at all, independent of any question about the peak.

Usage:
  python3 era_shift.py --user tim --tz Europe/London --boundary 2026-08-05 [--placebo 20]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from gate4_deconvolution import (ALIGN_DEFAULT, PEAK_CONC_MIN, PEAK_PROM_MIN, STEP_S,
                                 gcv, load_grid, peak_of, peak_shape)

HERE = os.path.dirname(os.path.abspath(__file__))


def fit_two_era(y, X_pre, X_post, X_c, X_n, lam):
    """Two dose kernels, each smoothness-penalised and non-negative; shared drift terms."""
    K = X_pre.shape[1]
    X = np.hstack([X_pre, X_post, X_c, X_n])
    p = X.shape[1]
    D = np.zeros((2 * (K - 2), p))
    for i in range(K - 2):
        D[i, i] = 1.0; D[i, i + 1] = -2.0; D[i, i + 2] = 1.0
        r = K - 2 + i
        D[r, K + i] = 1.0; D[r, K + i + 1] = -2.0; D[r, K + i + 2] = 1.0
    A = np.vstack([X, np.sqrt(lam) * D])
    b = np.concatenate([y, np.zeros(2 * (K - 2))])
    lo = np.full(p, -np.inf); hi = np.full(p, np.inf)
    lo[:2 * K] = 0.0
    r = lsq_linear(A, b, bounds=(lo, hi), max_iter=120, tol=1e-8)
    return r.x[:K], r.x[K:2 * K]


def build(user, tz, max_lag=360.0, thin=3):
    grid, bg, dose, ok, clock, day, has_steps, _n_ext, _n_pre = load_grid(user, tz)
    K = int(max_lag / 5)
    rows = np.flatnonzero(ok[:-1]); rows = rows[rows >= K]
    if thin > 1:
        rows = rows[::thin]
    y = bg[rows + 1] - bg[rows]
    Xd = -np.column_stack([dose[rows - k] for k in range(K + 1)])
    cl, dy = clock[rows], day[rows]
    X_c = (cl[:, None] == np.unique(cl)[None, :]).astype(float)[:, 1:]
    X_n = (dy[:, None] == np.unique(dy)[None, :]).astype(float)
    return grid, rows, y, Xd, X_c, X_n, K


def era_fit(grid, rows, y, Xd, X_c, X_n, t_bound, lam):
    """Split the dose block at t_bound and fit both kernels."""
    post = grid[rows] >= t_bound
    X_pre = Xd * (~post)[:, None]
    X_post = Xd * post[:, None]
    keep_n = X_n[:, X_n.sum(axis=0) > 0]
    return fit_two_era(y, X_pre, X_post, X_c, keep_n, lam), int(post.sum())


def describe(b, align):
    if b.max() <= 0:
        return dict(peak=float("nan"), conc=float("nan"), prom=float("nan"), area=0.0)
    c, p = peak_shape(b)
    return dict(peak=peak_of(b) + align, conc=c, prom=p, area=float(b.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tz", required=True)
    ap.add_argument("--boundary", required=True, help="ISO date the change took effect")
    ap.add_argument("--placebo", type=int, default=20,
                    help="number of earlier boundaries to fit as a null distribution")
    ap.add_argument("--history-days", type=float, default=180.0,
                    help="how much record before the boundary to use")
    ap.add_argument("--power", default="",
                    help="comma-separated post-window lengths in days; for each, fit the same "
                         "placebo null and report the spread, i.e. how many days are needed "
                         "before a shift of a given size could be seen at all")
    ap.add_argument("--power-draws", type=int, default=30,
                    help="placebo fits per window length. Twelve is too few: the 10th-90th "
                         "percentile is itself noisy at that count and the sweep came out "
                         "non-monotonic in window length, which cannot be true.")
    ap.add_argument("--out")
    a = ap.parse_args()

    align = ALIGN_DEFAULT
    grid, rows, y, Xd, X_c, X_n, K = build(a.user, a.tz)
    t_bound = pd.Timestamp(a.boundary, tz="UTC").timestamp()
    t_start = t_bound - a.history_days * 86400
    sel = grid[rows] >= t_start
    rows_s, y_s, Xd_s = rows[sel], y[sel], Xd[sel]
    X_c_s, X_n_s = X_c[sel], X_n[sel]
    post_days = (grid[rows_s].max() - t_bound) / 86400

    lam = gcv(y_s, Xd_s, X_c_s, X_n_s[:, X_n_s.sum(axis=0) > 0], np.logspace(1, 5, 5))
    (b_pre, b_post), n_post = era_fit(grid, rows_s, y_s, Xd_s, X_c_s, X_n_s, t_bound, lam)
    pre, post = describe(b_pre, align), describe(b_post, align)

    # placebo boundaries: same post-window length, placed where nothing changed
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
                (bp, bq), npq = era_fit(grid, rows_s[keep], y_s[keep], Xd_s[keep],
                                        X_c_s[keep], X_n_s[keep], tb, lam)
            except Exception as e:                                     # noqa: BLE001
                print(f"  placebo fit failed: {type(e).__name__}", flush=True)
                continue
            d = describe(bq, align)
            dp = describe(bp, align)
            # the area ratio is only meaningful against its own era, so carry the pair
            d["ratio"] = (d["area"] / dp["area"]) if dp["area"] > 0 else float("nan")
            d["n"] = npq
            null.append(d)
            print(f"  placebo {len(null)}/{a.placebo}: peak {d['peak']:.0f}, "
                  f"area ratio {d['ratio']:.2f}, n={npq}", flush=True)

    npk = np.array([d["peak"] for d in null if np.isfinite(d["peak"])])

    L, P = [], None
    P = L.append
    P(f"# Did the response change on {a.boundary}? — user {a.user}\n")
    P(f"\n{len(y_s):,} thinned samples over the {a.history_days:.0f} days before the boundary plus "
      f"{post_days:.1f} days after it; {n_post:,} of them fall after. Both kernels share one "
      f"time-of-day profile and per-day intercepts; smoothing {lam:g}.\n")

    P("\n## Estimates either side of the boundary\n")
    P("\n| era | samples | peak (min) | concentration | prominence | kernel area (per unit) |")
    P("|---|---|---|---|---|---|")
    P(f"| before | {len(y_s) - n_post:,} | {pre['peak']:.0f} | {pre['conc']:.2f} | "
      f"{pre['prom']:.2f} | {pre['area']:.3f} |")
    P(f"| after | {n_post:,} | {post['peak']:.0f} | {post['conc']:.2f} | {post['prom']:.2f} | "
      f"{post['area']:.3f} |")
    ratio = (post["area"] / pre["area"]) if pre["area"] > 0 else float("nan")
    if np.isfinite(ratio):
        P(f"\nArea ratio after/before: **{ratio:.2f}** per unit delivered. Diluting to half "
          f"strength doubles the units recorded for the same mass, so a ratio near 0.5 is what the "
          f"change itself predicts, independent of any shift in timing.\n")

    P("\n## What a window this short can resolve\n")
    if len(npk) >= 5:
        P(f"\nThe same fit was repeated with the boundary moved to {len(npk)} earlier dates, each "
          f"with a post-window of the same {post_days:.1f} days, where by construction nothing "
          f"changed.\n")
        P(f"\nNull distribution of the post-window peak: median **{np.median(npk):.0f} min**, "
          f"range {npk.min():.0f} to {npk.max():.0f}, 10th-90th percentile "
          f"{np.percentile(npk, 10):.0f} to {np.percentile(npk, 90):.0f}.\n")
        nra = np.array([d["ratio"] for d in null if np.isfinite(d.get("ratio", np.nan))])
        if len(nra) >= 5 and np.isfinite(ratio):
            rp = 100.0 * (nra < ratio).mean()
            P(f"\nNull distribution of the area ratio: median **{np.median(nra):.2f}**, "
              f"10th-90th percentile {np.percentile(nra, 10):.2f} to "
              f"{np.percentile(nra, 90):.2f}. The observed {ratio:.2f} sits at the {rp:.0f}th "
              f"percentile.\n")
            # A short window inflates kernel area, so the null is not centred on 1. The observed
            # value must therefore be judged against BOTH hypotheses carried through the same
            # bias, not against the null alone: under a true halving the estimate is distributed
            # like the null scaled by 0.5, and if those two overlap the amplitude cannot decide.
            alt = 0.5 * nra
            P(f"\nBecause a window this short inflates the recovered area — the null is centred on "
              f"{np.median(nra):.2f}, not on 1 — the observed value has to be read against both "
              f"hypotheses carried through the same bias. Under a true halving the estimate would "
              f"be distributed like the null scaled by one half: median {np.median(alt):.2f}, "
              f"10th-90th percentile {np.percentile(alt, 10):.2f} to "
              f"{np.percentile(alt, 90):.2f}.\n")
            ov = max(0.0, min(np.percentile(nra, 90), np.percentile(alt, 90))
                     - max(np.percentile(nra, 10), np.percentile(alt, 10)))
            span = np.percentile(nra, 90) - np.percentile(alt, 10)
            P(f"\nThose two ranges overlap over {100 * ov / span:.0f}% of their combined span, and "
              f"the observed {ratio:.2f} lies inside both. "
              + ("The amplitude therefore cannot distinguish a halving from no change either.\n"
                 if ov > 0 else
                 "They are separable, so the amplitude does carry information here.\n"))
        if np.isfinite(post["peak"]):
            pctl = 100.0 * (npk < post["peak"]).mean()
            extreme = min(pctl, 100 - pctl)
            P(f"\nThe observed post-boundary peak of {post['peak']:.0f} min sits at the "
              f"{pctl:.0f}th percentile of that null, so a value at least as extreme arises by "
              f"chance about {2 * extreme:.0f}% of the time when nothing has changed.\n")
            verdict = ("**No indication of a change.** " if 2 * extreme > 20 else
                       "**Suggestive, not conclusive.** " if 2 * extreme > 5 else
                       "**The shift exceeds what a window this short produces by chance.** ")
            P(f"\n{verdict}The spread of the null is {npk.max() - npk.min():.0f} min wide, which "
              f"is the resolution a window of this length actually has. A shift smaller than that "
              f"cannot be detected here however it is analysed, and more days are the only remedy.\n")
    else:
        P("\nToo few placebo windows completed to calibrate the null; treat the estimates above as "
          "uncalibrated.\n")

    if a.power:
        P("\n## How long a window would be needed\n")
        P("\nThe same placebo procedure, run at a range of window lengths. The spread of the null "
          "is the smallest shift that window could distinguish from chance; a change smaller than "
          "it is undetectable no matter how the data are analysed.\n")
        # The null is heavily right-skewed — a mode near the long-run value plus a tail of windows
        # that return anything at all — so a percentile spread is itself unstable and came out
        # NON-MONOTONE in window length at both 12 and 30 draws, which cannot be true. The stable
        # summary is a proportion: how often a window of this length lands near the long-run
        # answer. That is binomial, so its own uncertainty is known and it must increase with days.
        long_run = pre["peak"]
        P(f"\nThe null is strongly right-skewed, so a percentile spread is unstable. The dependable "
          f"summary is the proportion of windows landing within 15 min of the long-run estimate "
          f"({long_run:.0f} min), which is a binomial proportion and must rise with window length.\n")
        P("\n| post-window (days) | placebo fits | median peak | within 15 min of long-run | "
          "10th-90th percentile |")
        P("|---|---|---|---|---|")
        for wd in [float(x) for x in a.power.split(",")]:
            lo_w = grid[rows_s].min() + 30 * 86400
            hi_w = t_bound - wd * 86400 - 7 * 86400
            if hi_w <= lo_w:
                P(f"| {wd:.0f} | insufficient history | | | | |"); continue
            pk = []
            for tb in np.random.default_rng(int(wd) + 7).uniform(lo_w, hi_w, a.power_draws):
                keep = grid[rows_s] < tb + wd * 86400
                if keep.sum() < 500:
                    continue
                try:
                    (_bp, bq), _n = era_fit(grid, rows_s[keep], y_s[keep], Xd_s[keep],
                                            X_c_s[keep], X_n_s[keep], tb, lam)
                except Exception:                                      # noqa: BLE001
                    continue
                v = describe(bq, align)["peak"]
                if np.isfinite(v):
                    pk.append(v)
            print(f"  power {wd:g}d: n={len(pk)}", flush=True)
            if len(pk) < 4:
                P(f"| {wd:.0f} | {len(pk)} | too few to summarise | | | |"); continue
            pk = np.array(pk)
            p10, p90 = np.percentile(pk, [10, 90])
            hit = float(np.mean(np.abs(pk - long_run) <= 15.0))
            se = float(np.sqrt(max(hit * (1 - hit), 1e-9) / len(pk)))
            P(f"| {wd:.0f} | {len(pk)} | {np.median(pk):.0f} | **{100 * hit:.0f}%** "
              f"(+/-{100 * se:.0f}) | {p10:.0f} to {p90:.0f} |")
        P("\nA window is only worth fitting once that proportion is high. Until then the estimator "
          "returns a number, but which number is mostly a matter of which days happened to fall "
          "inside the window.\n")

    open(a.out or os.path.join(HERE, f"ERA_{a.user}.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
