#!/usr/bin/env python3
"""How long a post-boundary window is needed to resolve a peak shift, with basal admitted?

The companion to `era_shift_basal.py`. For each candidate window length it places the boundary at
many earlier dates, where nothing changed, and fits the same four-kernel model. The summary is the
proportion of windows landing within 15 min of the long-run estimate: the null is right-skewed with
a tail of degenerate fits at the end of the lag range, so a percentile spread is unstable, whereas
a proportion is binomial and must rise with window length.

Fits are independent, so they run in a pool. Workers re-import this module under spawn, so each
builds its own copy of the design in the initialiser rather than receiving it through a job tuple.

Usage:
  python3 era_power_basal.py --user <id> --tz <tz> --boundary <YYYY-MM-DD> \
      --windows 26,40,60,90 --draws 30 --workers 7
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from era_shift_basal import build, era_fit, describe
from gate4_deconvolution import ALIGN_DEFAULT, gcv

HERE = os.path.dirname(os.path.abspath(__file__))
_G: dict = {}


def _init(user, tz, boundary, history_days):
    grid, rows, y, Xd, Xb, X_c, X_n, K, cover, bas = build(user, tz)
    t_bound = pd.Timestamp(boundary, tz="UTC").timestamp()
    sel = grid[rows] >= t_bound - history_days * 86400
    keep_n_all = X_n[sel][:, X_n[sel].sum(axis=0) > 0]
    lam = gcv(y[sel], np.hstack([Xd[sel], Xb[sel]]), X_c[sel], keep_n_all, np.logspace(1, 5, 5))
    _G.update(grid=grid, rows=rows[sel], y=y[sel], Xd=Xd[sel], Xb=Xb[sel],
              X_c=X_c[sel], X_n=X_n[sel], lam=lam, t_bound=t_bound)


def _one(job):
    wd, tb = job
    g = _G
    keep = g["grid"][g["rows"]] < tb + wd * 86400
    if keep.sum() < 500:
        return wd, float("nan")
    try:
        (_bp, bq, _cp, _cq), _n = era_fit(g["grid"], g["rows"][keep], g["y"][keep],
                                          g["Xd"][keep], g["Xb"][keep], g["X_c"][keep],
                                          g["X_n"][keep], tb, g["lam"])
    except Exception:                                                  # noqa: BLE001
        return wd, float("nan")
    return wd, describe(bq, ALIGN_DEFAULT)["peak"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tz", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--windows", default="26,40,60,90")
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--history-days", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--long-run", type=float, default=None,
                    help="reference peak in minutes; defaults to the pre-era estimate")
    ap.add_argument("--out")
    a = ap.parse_args()

    _init(a.user, a.tz, a.boundary, a.history_days)
    g = _G
    long_run = a.long_run
    if long_run is None:
        (bp, _bq, _cp, _cq), _n = era_fit(g["grid"], g["rows"], g["y"], g["Xd"], g["Xb"],
                                          g["X_c"], g["X_n"], g["t_bound"], g["lam"])
        long_run = describe(bp, ALIGN_DEFAULT)["peak"]
    print(f"long-run (pre-era) peak {long_run:.0f} min, smoothing {g['lam']:g}", flush=True)

    windows = [float(x) for x in a.windows.split(",")]
    jobs, spans = [], {}
    t0g = g["grid"][g["rows"]].min()
    for wd in windows:
        lo = t0g + 30 * 86400
        hi = g["t_bound"] - wd * 86400 - 7 * 86400
        spans[wd] = (hi - lo) / 86400
        if hi <= lo:
            continue
        for tb in np.random.default_rng(int(wd) + 7).uniform(lo, hi, a.draws):
            jobs.append((wd, tb))

    with mp.Pool(a.workers, initializer=_init,
                 initargs=(a.user, a.tz, a.boundary, a.history_days)) as pool:
        res = pool.map(_one, jobs, chunksize=1)

    by = {wd: np.array([v for w, v in res if w == wd and np.isfinite(v)]) for wd in windows}

    L = []
    P = L.append
    P(f"# How long a window is needed to resolve a peak shift? — user {a.user}\n")
    P(f"\nFour-kernel model with basal admitted, smoothing {g['lam']:g} chosen by GCV on that "
      f"design. Each row places the boundary at {a.draws} earlier dates where nothing changed and "
      f"fits the same model to a window of that length. The reference is the pre-era estimate of "
      f"{long_run:.0f} min.\n")
    P("\n| post-window (days) | fits | median peak | within 15 min of long-run | 10th-90th | "
      "boundary range drawn from (days) |")
    P("|---|---|---|---|---|---|")
    for wd in windows:
        pk = by[wd]
        if len(pk) < 4:
            P(f"| {wd:.0f} | {len(pk)} | too few to summarise | | | {spans[wd]:.0f} |")
            continue
        hit = float(np.mean(np.abs(pk - long_run) <= 15.0))
        se = float(np.sqrt(max(hit * (1 - hit), 1e-9) / len(pk)))
        p10, p90 = np.percentile(pk, [10, 90])
        P(f"| {wd:.0f} | {len(pk)} | {np.median(pk):.0f} | **{100 * hit:.0f}%** "
          f"(+/-{100 * se:.0f}) | {p10:.0f} to {p90:.0f} | {spans[wd]:.0f} |")
    P("\nThe last column is how much boundary range remains for a window of that length. Where it "
      "is small the placebo windows overlap heavily and share most of their days, so the spread is "
      "understated and the proportion is optimistic.\n")
    open(a.out or os.path.join(HERE, f"POWERB_{a.user}.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
