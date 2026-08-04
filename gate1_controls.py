#!/usr/bin/env python3
"""GATE 1 CONTROLS — three diagnostics that decide whether a recovered curve means anything.

Each of these was originally run ad hoc while investigating a cohort. They are here because a
number quoted in a report has to be reproducible by whoever reads it.

A. POST-BOLUS MASK SWEEP
   IOB steps by the whole dose at delivery, so a small error in placing that step leaves a residual
   the size of the dose. Unmasked, the residual on what is supposed to be an exact identity ran
   11-32% of signal across a cohort. Masking observations near a bolus cuts it to 7-10% and, more
   importantly, removes a DOWNWARD bias on DIA — one user whose configured DIA is known to be 600
   moved from 314 to 487. The peak barely moves, so the mask is close to free. This sweep shows
   both effects for a given user.

B. TWO-KERNEL DOSE-SIZE NEGATIVE CONTROL
   Fit separate kernels to small and large doses. The app applies ONE configured curve to every
   dose regardless of size, so the true difference is exactly zero and any split this reports is
   estimator or data artefact. On one cohort four users returned -0.9 to +3.1 min (clean) and one
   returned +48.7 min, which identified a data problem with that user's large doses rather than a
   discovery about insulin. Run this before believing any dose-stratified result.

C. TAIL CONTRIBUTION BEYOND N HOURS
   How much of the predicted IOB series comes from kernel lags beyond N hours, in RMS terms,
   against the fit residual. This is what decides whether DIA is worth quoting: across a cohort the
   post-5h tail contributed 0.000-0.051 U against residuals of 0.15-0.55 U, i.e. below the noise
   floor for every user at p90 doses as well as median ones. A dose-size split does not rescue that.

Usage:
  python3 gate1_controls.py --user <id> [--days 45] [--threshold 1.0] [--tail-hours 5]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.optimize import least_squares

from gate1_recover_known_curve import DSN, iob_fraction, load, predicted_iob  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = 300.0


def build(dec, bol):
    t0 = float(bol.ts.min().timestamp())
    t1 = float(dec.ts_epoch.max())
    grid = np.arange(t0, t1 + STEP, STEP)
    d_s = bol.ts.values.astype("datetime64[s]").astype(float)
    d_u = bol.insulin.values.astype(float)
    obs_t = dec.ts_epoch.values.astype(float)
    obs = dec.bolus_iob.values.astype(float)
    obs_idx = np.clip(((obs_t - t0) / STEP).astype(int), 0, len(grid) - 1)
    return t0, grid, d_s, d_u, obs_t, obs, obs_idx


def mask_keep(obs_t, d_s, minutes):
    if minutes <= 0:
        return np.ones(len(obs_t), bool)
    gap = obs_t[:, None] - d_s[None, :]
    return ~((gap >= 0) & (gap < minutes * 60)).any(axis=1)


def fit_single(grid, dose_grid, obs_idx, obs):
    f = least_squares(lambda p: predicted_iob(grid, dose_grid, p[0], p[1], STEP)[obs_idx] - obs,
                      x0=[60.0, 400.0], bounds=([10.0, 120.0], [180.0, 1440.0]),
                      xtol=1e-10, ftol=1e-10)
    r = predicted_iob(grid, dose_grid, f.x[0], f.x[1], STEP)[obs_idx] - obs
    rmse = float(np.sqrt(np.mean(r ** 2)))
    rel = rmse / float(np.sqrt(np.mean((obs - obs.mean()) ** 2)))
    return f.x[0], f.x[1], rmse, rel


def control_a(grid, d_s, d_u, t0, obs_t, obs, obs_idx, P):
    P("\n## A. Post-bolus mask sweep\n")
    P("| mask (min) | observations kept | peak | DIA | relRMSE |")
    P("|---|---|---|---|---|")
    dose_grid = np.zeros(len(grid))
    np.add.at(dose_grid, np.clip(((d_s - t0) / STEP).astype(int), 0, len(grid) - 1), d_u)
    for m in (0, 5, 10, 15):
        keep = mask_keep(obs_t, d_s, m)
        pk, dia, _, rel = fit_single(grid, dose_grid, obs_idx[keep], obs[keep])
        P(f"| {m} | {100 * keep.mean():.0f}% | {pk:.1f} | {dia:.0f} | {rel:.3f} |")
    P("\nA peak that moves with the mask means the fit is being driven by the delivery step rather "
      "than the curve. A DIA that moves a lot was biased short by that step.\n")


def control_b(grid, d_s, d_u, t0, obs_t, obs, obs_idx, thr, mask_min, P):
    P(f"\n## B. Two-kernel dose-size negative control (threshold {thr:g} U)\n")
    small, large = d_u < thr, d_u >= thr
    if large.sum() < 20 or small.sum() < 20:
        P(f"Not run: {int(small.sum())} small / {int(large.sum())} large doses — too few to "
          "separate.\n")
        return
    gs, gl = np.zeros(len(grid)), np.zeros(len(grid))
    idx = np.clip(((d_s - t0) / STEP).astype(int), 0, len(grid) - 1)
    np.add.at(gs, idx[small], d_u[small])
    np.add.at(gl, idx[large], d_u[large])
    keep = mask_keep(obs_t, d_s, mask_min)
    oi, ob = obs_idx[keep], obs[keep]

    def pred(ps, pl, dia):
        n = int(np.ceil(dia * 60 / STEP)) + 1
        lag = np.arange(n) * STEP / 60.0
        return (np.convolve(gs, iob_fraction(lag, ps, dia))[:len(grid)]
                + np.convolve(gl, iob_fraction(lag, pl, dia))[:len(grid)])

    f = least_squares(lambda p: pred(p[0], p[1], p[2])[oi] - ob, x0=[60.0, 60.0, 400.0],
                      bounds=([10.0, 10.0, 120.0], [180.0, 180.0, 1440.0]),
                      xtol=1e-9, ftol=1e-9)
    r = pred(*f.x)[oi] - ob
    rel = float(np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean((ob - ob.mean()) ** 2)))
    diff = f.x[1] - f.x[0]
    P(f"| stratum | n doses | recovered peak |")
    P("|---|---|---|")
    P(f"| small (< {thr:g} U) | {int(small.sum()):,} | {f.x[0]:.1f} |")
    P(f"| large (>= {thr:g} U) | {int(large.sum()):,} | {f.x[1]:.1f} |")
    P(f"\nShared DIA {f.x[2]:.0f} min, relRMSE {rel:.3f}. "
      f"**Difference {diff:+.1f} min, true answer 0.**\n")
    P("\n" + (f"Within tolerance — no dose-size artefact.\n" if abs(diff) <= 5 else
              f"**FLAG: {abs(diff):.0f} min split where the app applies one curve to all doses.** "
              "This is a data or estimator problem for this user, not pharmacology. Do not run a "
              "dose-stratified analysis on them until it is understood.\n"))


def control_c(grid, d_s, d_u, t0, obs_t, obs, obs_idx, tail_hours, mask_min, P):
    P(f"\n## C. Kernel contribution beyond {tail_hours:g} h\n")
    dose_grid = np.zeros(len(grid))
    np.add.at(dose_grid, np.clip(((d_s - t0) / STEP).astype(int), 0, len(grid) - 1), d_u)
    keep = mask_keep(obs_t, d_s, mask_min)
    oi, ob = obs_idx[keep], obs[keep]
    pk, dia, rmse, _ = fit_single(grid, dose_grid, oi, ob)
    full = predicted_iob(grid, dose_grid, pk, dia, STEP)[oi]
    n = int(np.ceil(dia * 60 / STEP)) + 1
    lag = np.arange(n) * STEP / 60.0
    kern = iob_fraction(lag, pk, dia).copy()
    kern[lag > tail_hours * 60.0] = 0.0
    trunc = np.convolve(dose_grid, kern)[:len(grid)][oi]
    contrib = float(np.sqrt(np.mean((full - trunc) ** 2)))
    P(f"Fitted curve peak {pk:.1f} min, DIA {dia:.0f} min, fit RMSE {rmse:.3f} U.\n")
    P(f"\nRemoving all kernel mass beyond {tail_hours:g} h changes the predicted IOB series by "
      f"**{contrib:.4f} U RMS**, against a fit residual of {rmse:.3f} U "
      f"(ratio {contrib / rmse if rmse else float('nan'):.2f}x).\n")
    P("\n" + ("The tail carries real signal; a DIA quoted for this user is meaningful.\n"
              if contrib > rmse else
              "**The tail is below the noise floor**, so DIA is not measurable for this user at "
              "these dose sizes. Do not quote it, and do not expect a dose-size split to help — "
              "the tail is small relative to the residual in both strata.\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--tail-hours", type=float, default=5.0)
    ap.add_argument("--mask-post-bolus", type=float, default=10.0)
    ap.add_argument("--out")
    a = ap.parse_args()

    dec, bol = load(a.user, a.days, 0)
    if dec.empty or bol.empty:
        print("no data"); return
    t0, grid, d_s, d_u, obs_t, obs, obs_idx = build(dec, bol)

    L, P = [], None
    P = L.append
    P("# Gate 1 controls\n")
    P(f"User **{a.user}**, last {a.days} days: {len(obs):,} IOB samples, {len(d_u):,} boluses "
      f"({d_u.sum():.0f} U).\n")
    control_a(grid, d_s, d_u, t0, obs_t, obs, obs_idx, P)
    control_b(grid, d_s, d_u, t0, obs_t, obs, obs_idx, a.threshold, a.mask_post_bolus, P)
    control_c(grid, d_s, d_u, t0, obs_t, obs, obs_idx, a.tail_hours, a.mask_post_bolus, P)

    open(a.out or os.path.join(HERE, "GATE1_CONTROLS.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
