#!/usr/bin/env python3
"""Does the response depend on dose size? A decile-stratified impulse response.

Pharmacology says the peak should move later as the dose grows, because a larger subcutaneous depot
absorbs more slowly relative to its volume, and the product labels document dose-dependence
directly. Earlier attempts here could not test it: a per-user split into two strata left regressors
with median correlation up to 0.94, and the resulting intervals were 30-100 minutes wide.

This splits each participant's doses into DECILES of their own dose distribution and fits one
kernel per decile:

    dg_t = - SUM_q SUM_k beta_{q,k} dose^{(q)}_{t-k} + tod + day + e

Ten kernels on collinear regressors is exactly the setting where a spurious trend appears, so two
things guard against it.

REGULARISATION IN TWO DIRECTIONS. Each kernel is smoothness-penalised along lag as before, and
adjacent deciles are additionally penalised toward each other. A dose-size effect must therefore be
strong enough to overcome a prior that says neighbouring deciles behave alike, rather than emerging
from noise in a weakly-determined fit.

A NEGATIVE CONTROL THAT MUST COME FIRST. Glucose is replaced by a simulation in which EVERY dose,
of whatever size, acts through ONE known kernel. The true decile profile is then flat by
construction. Whatever tilt the estimator reports on that data is the tilt it manufactures from
collinearity alone, and any real result has to be read against it — not against zero.

Usage:
  python3 dose_decile_response.py --user <id> --tz <tz> [--deciles 10]
  python3 dose_decile_response.py --config cohort.json --pool
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os

import numpy as np
import psycopg2
from scipy.optimize import lsq_linear

from gate1_recover_known_curve import DSN, activity
from gate4_deconvolution import STEP_S, gcv, load_grid

HERE = os.path.dirname(os.path.abspath(__file__))


def dose_series(user, grid, t0, nq):
    """Split a user's boluses into nq quantile bins, returning one dose series per bin."""
    import pandas as pd
    conn = psycopg2.connect(DSN)
    d = pd.read_sql("""SELECT ts_utc, insulin FROM boost_treatments
                       WHERE user_id=%s AND insulin>0 ORDER BY ts_utc""", conn, params=(user,))
    conn.close()
    if len(d) < 500:
        return None, None
    ts = (pd.to_datetime(d.ts_utc, utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds().values
    u = d.insulin.values.astype(float)
    edges = np.unique(np.quantile(u, np.linspace(0, 1, nq + 1)))
    if len(edges) < 3:
        return None, None
    q = np.clip(np.digitize(u, edges[1:-1]), 0, len(edges) - 2)
    nq_eff = len(edges) - 1
    out = np.zeros((nq_eff, len(grid)))
    idx = np.clip(((ts - t0) / STEP_S).round().astype(int), 0, len(grid) - 1)
    for j in range(nq_eff):
        m = q == j
        np.add.at(out[j], idx[m], u[m])
    med = [float(np.median(u[q == j])) if (q == j).any() else np.nan for j in range(nq_eff)]
    return out, med


def fit_deciles(y, Xq, X_c, X_n, lam, lam_q):
    """One kernel per decile; smooth in lag, and adjacent deciles tied to each other."""
    nq, n, K1 = Xq.shape[0], Xq.shape[1], Xq.shape[2]
    X = np.hstack([Xq[j] for j in range(nq)] + [X_c, X_n])
    p = X.shape[1]
    rows = []
    for j in range(nq):                                   # smoothness along lag
        base = j * K1
        for i in range(K1 - 2):
            r = np.zeros(p); r[base + i] = 1.0; r[base + i + 1] = -2.0; r[base + i + 2] = 1.0
            rows.append(np.sqrt(lam) * r)
    for j in range(nq - 1):                               # adjacent deciles tied
        a, b = j * K1, (j + 1) * K1
        for i in range(K1):
            r = np.zeros(p); r[a + i] = 1.0; r[b + i] = -1.0
            rows.append(np.sqrt(lam_q) * r)
    D = np.array(rows)
    A = np.vstack([X, D])
    bvec = np.concatenate([y, np.zeros(len(D))])
    lo = np.full(p, -np.inf); hi = np.full(p, np.inf)
    lo[:nq * K1] = 0.0
    r = lsq_linear(A, bvec, bounds=(lo, hi), max_iter=120, tol=1e-7)
    return [r.x[j * K1:(j + 1) * K1] for j in range(nq)]


def peaks_of(ks):
    return [float(np.argmax(k) * 5.0) if k.max() > 0 else np.nan for k in ks]


def run_user(user, tz, nq=10, max_lag=240.0, control=True):
    grid, bg, dose, ok, clock, day, _ = load_grid(user, tz)
    t0 = grid[0]
    Dq, med = dose_series(user, grid, t0, nq)
    if Dq is None:
        return None
    K = int(max_lag / 5)
    rows = np.flatnonzero(ok[:-1]); rows = rows[rows >= K]
    if len(rows) < 3000:
        return None
    y = bg[rows + 1] - bg[rows]
    Xq = np.stack([-np.column_stack([Dq[j][rows - k] for k in range(K + 1)])
                   for j in range(Dq.shape[0])])
    cl, dy = clock[rows], day[rows]
    X_c = (cl[:, None] == np.unique(cl)[None, :]).astype(float)[:, 1:]
    X_n = (dy[:, None] == np.unique(dy)[None, :]).astype(float)
    lam = gcv(y, -np.column_stack([dose[rows - k] for k in range(K + 1)]), X_c, X_n,
              np.logspace(1, 5, 5))
    obs = peaks_of(fit_deciles(y, Xq, X_c, X_n, lam, lam * 10))

    ctl = None
    if control:
        # every dose acts through ONE kernel: the true decile profile is flat
        kern = activity(np.arange(K + 1) * 5.0, 55.0, 480.0); kern /= kern.max()
        sig = sum(np.column_stack([Dq[j][rows - k] for k in range(K + 1)]) @ kern
                  for j in range(Dq.shape[0]))
        rng = np.random.default_rng(7)
        sd = float(np.std(y))
        ysyn = -0.35 * sd * sig / max(np.std(sig), 1e-9) + rng.normal(0, 0.9 * sd, len(rows))
        ctl = peaks_of(fit_deciles(ysyn, Xq, X_c, X_n, lam, lam * 10))
    return dict(user=user, med=med, observed=obs, control=ctl, n=len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user"); ap.add_argument("--tz")
    ap.add_argument("--config"); ap.add_argument("--pool", action="store_true")
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.config:
        users = json.load(open(a.config))
        with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
            res = [r for r in ex.map(lambda kv: run_user(kv[0], kv[1], a.deciles), users.items()) if r]
    else:
        res = [r for r in [run_user(a.user, a.tz, a.deciles)] if r]
    if not res:
        print("no users estimable"); return

    nq = max(len(r["observed"]) for r in res)
    obs = np.full((len(res), nq), np.nan); ctl = np.full((len(res), nq), np.nan)
    dz = np.full((len(res), nq), np.nan)
    for i, r in enumerate(res):
        obs[i, :len(r["observed"])] = r["observed"]
        dz[i, :len(r["med"])] = r["med"]
        if r["control"]:
            ctl[i, :len(r["control"])] = r["control"]

    L, P = [], None
    P = L.append
    P("# Response by dose decile\n")
    P(f"\n{len(res)} participants, doses split into {nq} bins of each participant's own "
      f"distribution. One kernel per bin, smoothness-penalised along lag and tied between adjacent "
      f"bins. Peaks in minutes.\n")
    P("\n| decile | median dose (U) | observed peak | negative control | observed − control |")
    P("|---|---|---|---|---|")
    for j in range(nq):
        o = np.nanmedian(obs[:, j]); c = np.nanmedian(ctl[:, j]); d = np.nanmedian(dz[:, j])
        P(f"| {j + 1} | {d:.2f} | {o:.0f} | {c:.0f} | {o - c:+.0f} |")
    lo_o, hi_o = np.nanmedian(obs[:, 0]), np.nanmedian(obs[:, nq - 1])
    lo_c, hi_c = np.nanmedian(ctl[:, 0]), np.nanmedian(ctl[:, nq - 1])
    ctl_med = np.nanmedian(ctl, axis=0)
    ctl_spread = float(np.nanmax(ctl_med) - np.nanmin(ctl_med))
    obs_med = np.nanmedian(obs, axis=0)
    obs_spread = float(np.nanmax(obs_med) - np.nanmin(obs_med))

    P(f"\nThe negative control should be FLAT: every dose in it acts through one 55-minute kernel "
      f"by construction. It is not. Across deciles it spans **{ctl_spread:.0f} min** "
      f"({np.nanmin(ctl_med):.0f} to {np.nanmax(ctl_med):.0f}), which is the tilt the estimator "
      f"manufactures from collinearity between dose strata with no dose effect present.\n")
    P(f"\nThe observed profile spans {obs_spread:.0f} min, and its end-to-end change is "
      f"{hi_o - lo_o:+.0f} min against {hi_c - lo_c:+.0f} for the control.\n")

    if ctl_spread >= 15:
        P("\n**NOT ESTIMABLE.** The control artefact is as large as any effect worth detecting, so "
          "no dose-size conclusion can be drawn from this design. The decile split does not "
          "overcome the collinearity between dose strata; regularising toward neighbouring deciles "
          "constrains the fit but does not identify it. A dose-size effect of the magnitude "
          "pharmacology predicts would be indistinguishable from this artefact.\n")
    elif abs((hi_o - lo_o) - (hi_c - lo_c)) < 10:
        P("\nThe observed tilt is within the artefact the estimator produces on a no-effect "
          "simulation, so it is not evidence of a dose-size effect.\n")
    else:
        P(f"\nThe observed tilt exceeds the control artefact by "
          f"{(hi_o - lo_o) - (hi_c - lo_c):+.0f} min. That residual is the only part attributable "
          "to dose size, and it should be read against a control that is itself imperfect.\n")
    open(a.out or os.path.join(HERE, "DOSE_DECILE.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
