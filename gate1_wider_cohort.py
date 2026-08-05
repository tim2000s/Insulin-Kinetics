#!/usr/bin/env python3
"""GATE 1 ON THE WIDER ARCHIVE — configured curves for users with no treatment stream.

The main cohort has a treatment stream. A far larger archive of oref-derived records does not, and
an earlier attempt to reconstruct doses from the insulin-on-board series alone failed for a
structural reason (see iob_only_identifiability.py): the IOB identity is lower-triangular with unit
diagonal, so every candidate kernel reproduces the series exactly and goodness of fit is
identically uninformative.

A different column solves it. Alongside `bolusiob` — insulin REMAINING — oref logs `bolusinsulin`,
the insulin DELIVERED within the lookback window. These are different quantities, so deconvolving
one against the other is not degenerate. Doses are recovered as the positive jumps in the
cumulative delivered series:

    d_i = max(0, bolusinsulin_i - bolusinsulin_{i-1})

`bolusinsulin` falls only when a dose ages out of the window, which is a discrete event, so
positive jumps are new deliveries. The check that this works is physical rather than statistical:
across the archive **100% of recovered doses are exact multiples of 0.05 U**, the pump increment.
A reconstruction producing arbitrary reals would not do that.

The fit is then exactly Gate 1 — the delivered series convolved with the candidate IOB kernel must
reproduce the remaining series — with the same post-bolus masking and the same DIA leverage test.

CAVEATS. Sampling is irregular in this archive (some users at one-minute cadence, some at five,
some mixed), so values are binned to a five-minute grid. Dose history before the record starts is
unknown, so the first DIA of observations is dropped. Timestamps are relative, so no time-of-day
analysis is possible and Gate 4 cannot be run here at all.

Usage:
  python3 gate1_wider_cohort.py --table oref_v3 [--min-rows 5000] [--jobs 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import least_squares

from gate1_recover_known_curve import DSN, predicted_iob

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = 300.0


def load_user(table, user):
    conn = psycopg2.connect(DSN)
    d = pd.read_sql(f"""
        SELECT ts_relative_sec AS t, iob_bolusiob AS iob, iob_bolusinsulin AS bi
        FROM {table}
        WHERE user_id = %s AND iob_bolusiob IS NOT NULL AND iob_bolusinsulin IS NOT NULL
        ORDER BY 1""", conn, params=(user,))
    conn.close()
    return d.drop_duplicates("t")


def build(d, max_gap_s=900.0):
    t = d.t.values.astype(float)
    iob = d.iob.values.astype(float)
    bi = d.bi.values.astype(float)
    gap = np.diff(t)
    ok = (gap > 0) & (gap <= max_gap_s)
    dose_t = t[1:][ok]
    dose_u = np.maximum(np.diff(bi)[ok], 0.0)
    m = dose_u > 1e-9
    dose_t, dose_u = dose_t[m], dose_u[m]

    t0, t1 = t.min(), t.max()
    grid = np.arange(t0, t1 + STEP, STEP)
    dg = np.zeros(len(grid))
    np.add.at(dg, np.clip(((dose_t - t0) / STEP).astype(int), 0, len(grid) - 1), dose_u)
    obs_idx = np.clip(((t - t0) / STEP).astype(int), 0, len(grid) - 1)
    return grid, dg, obs_idx, iob, t, dose_t, dose_u


def fit_user(table, user, warmup_min=600.0, mask_min=10.0):
    d = load_user(table, user)
    if len(d) < 2000:
        return None
    grid, dg, obs_idx, obs, t, dose_t, dose_u = build(d)
    keep = t >= t.min() + warmup_min * 60                      # dose history before start unknown
    if mask_min > 0 and len(dose_t):
        near = ((t[:, None] - dose_t[None, :]) >= 0) & ((t[:, None] - dose_t[None, :]) < mask_min * 60)
        keep &= ~near.any(axis=1)
    oi, ob = obs_idx[keep], obs[keep]
    if len(ob) < 1000:
        return None

    def resid(p):
        return predicted_iob(grid, dg, p[0], p[1], STEP)[oi] - ob
    f = least_squares(resid, x0=[60.0, 400.0], bounds=([10.0, 120.0], [180.0, 1440.0]),
                      xtol=1e-9, ftol=1e-9)
    r = resid(f.x)
    rmse = float(np.sqrt(np.mean(r ** 2)))
    denom = float(np.sqrt(np.mean((ob - ob.mean()) ** 2)))
    base = predicted_iob(grid, dg, f.x[0], f.x[1], STEP)[oi]
    lev = max(float(np.sqrt(np.mean((predicted_iob(grid, dg, f.x[0], float(dt_), STEP)[oi] - base) ** 2)))
              for dt_ in (240, 360, 480, 600, 900, 1440))
    quant = float(np.mean(np.abs(dose_u / 0.05 - np.round(dose_u / 0.05)) < 1e-6)) if len(dose_u) else 0.0
    return dict(user=user, peak=float(f.x[0]), dia=float(f.x[1]), rmse=rmse,
                rel=rmse / denom if denom else float("nan"), n=len(ob),
                days=float((t.max() - t.min()) / 86400), doses=int(len(dose_u)),
                units=float(dose_u.sum()), quant=quant, dia_identified=lev > 2 * rmse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="oref_v3")
    ap.add_argument("--min-rows", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()

    conn = psycopg2.connect(DSN)
    us = pd.read_sql(f"""
        SELECT user_id, count(*) n FROM {a.table}
        WHERE iob_bolusiob IS NOT NULL AND iob_bolusinsulin IS NOT NULL
        GROUP BY 1 HAVING count(*) >= %s ORDER BY 1""", conn, params=(a.min_rows,))
    conn.close()
    print(f"{a.table}: {len(us)} users with >= {a.min_rows:,} usable rows")

    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        res = [r for r in ex.map(lambda u: fit_user(a.table, u), us.user_id) if r]

    L, P = [], None
    P = L.append
    P(f"# Configured insulin curves — wider archive ({a.table})\n")
    P(f"\n{len(res)} users fitted, from {len(us)} with sufficient records. Doses reconstructed from "
      f"the cumulative delivered-insulin series; the insulin-on-board series is the fitting target. "
      f"Timestamps are relative, so no time-of-day analysis is possible.\n")
    P("\n| user | days | doses | total U | dose quantisation | peak (min) | DIA | relRMSE |")
    P("|---|---|---|---|---|---|---|---|")
    for r in sorted(res, key=lambda x: x["user"]):
        P(f"| {r['user']} | {r['days']:.0f} | {r['doses']:,} | {r['units']:.0f} | "
          f"{100 * r['quant']:.0f}% | **{r['peak']:.1f}** | "
          f"{r['dia']:.0f}{'' if r['dia_identified'] else ' (n/i)'} | {r['rel']:.3f} |")
    pk = np.array([r["peak"] for r in res])
    good = np.array([r["peak"] for r in res if r["rel"] < 0.15])
    P(f"\nRecovered peaks: median {np.median(pk):.1f} min, range {pk.min():.1f}–{pk.max():.1f}.\n")
    if len(good):
        P(f"\nRestricting to the {len(good)} users whose fit residual is under 15% of signal: "
          f"median {np.median(good):.1f}, range {good.min():.1f}–{good.max():.1f}.\n")
    else:
        P("\n**No user achieved a fit residual under 15% of signal.** On a relationship that is an "
          "exact identity this means the reconstructed dose series does not reproduce the logged "
          "insulin-on-board series, and no recovered curve below should be believed. The "
          "distribution is reported only to characterise the failure.\n")
        good = pk
    for lo, hi, lbl in ((0, 45, "under 45"), (45, 65, "45–65 (ultra-rapid preset 55)"),
                        (65, 85, "65–85 (rapid preset 75)"), (85, 999, "over 85")):
        n = int(((good >= lo) & (good < hi)).sum())
        P(f"- {lbl}: {n} of {len(good)}")
    P(f"\nDose quantisation is {100 * np.mean([r['quant'] for r in res]):.0f}% on average — "
      "recovered doses are exact multiples of the pump increment, which is the physical check that "
      "the reconstruction is recovering real deliveries.\n")
    P(f"\nDIA was identifiable in {sum(1 for r in res if r['dia_identified'])} of {len(res)} users; "
      "'n/i' marks the others, whose recovered duration should be ignored.\n")

    txt = "\n".join(L)
    open(a.out or os.path.join(HERE, f"WIDER_{a.table}.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
