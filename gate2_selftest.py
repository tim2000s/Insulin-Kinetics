#!/usr/bin/env python3
"""GATE 2 SELF-TEST — can the glucose estimator recover a peak it was given? (2026-08-04)

Gate 1 has a positive control: logged IOB is an exact function of the configured curve, so the
answer is known. Gate 2 had none. Everything it reported rested on the assumption that fitting
dBG against convolved insulin activity returns the peak that generated the data.

This tests that assumption directly. It takes a user's REAL windows — real dose times, real dose
sizes, real sampling — and replaces the observed glucose with glucose SIMULATED from a known peak:

    dBG_i = -k * conv(dose, activity(.; peak_true, DIA))_i  +  c  +  drift_i  +  noise_i

then runs the unmodified Gate 2 estimator over it. The recovered peak must come back as peak_true.
Amplitude, drift and noise are taken from the user's own fitted values so the signal-to-noise ratio
is the real one, not a flattering one.

Three variants, because the interesting question is not whether it works in the clean case:

  clean     signal + white noise. If this fails, the estimator is broken.
  basal     adds an unmodelled basal-insulin action term. Gate 2 uses only the BOLUS series; a
            loop moves temp basal in step with its boluses, so the omitted term is correlated with
            the regressor and can bias the shape rather than just adding noise.
  meal      adds an unannounced carb absorption bump in a share of windows. For users who never
            enter carbs the COB and carb filters do nothing, so this is not a hypothetical.

Usage:
  python3 gate2_selftest.py --user <id> --tz <tz> --dia <min> [--true-peaks 35,45,55,75]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from gate1_recover_known_curve import activity
from gate2_peak_from_glucose import load, build_windows, fit_peak, window_regressor

HERE = os.path.dirname(os.path.abspath(__file__))


def synth_window(ts, d_s, d_u, peak, dia, k, c, drift, noise_sd, rng,
                 basal_u_per_h=0.0, basal_peak=None, meal=None):
    """Return a synthetic BG series over the same timestamps, generated from a known peak."""
    x = window_regressor(ts, d_s, d_u, peak, dia)
    dbg = -k * x[:-1] + c + drift * (ts[:-1] - ts[0]) / 3600.0

    if basal_u_per_h > 0:
        # Basal delivered as a micro-dose every 5 min, acting through the SAME curve. Gate 2 never
        # sees these doses, so this is the omitted-variable case.
        b_s = np.arange(ts[0] - 6 * 3600, ts[-1], 300.0)
        b_u = np.full(len(b_s), basal_u_per_h / 12.0)
        xb = window_regressor(ts, b_s, b_u, basal_peak or peak, dia)
        dbg = dbg - k * xb[:-1]

    if meal is not None:
        grams, t_off, tau_c = meal
        tt = (ts[:-1] - ts[0]) / 60.0 - t_off
        # simple gamma-ish absorption bump
        bump = np.where(tt > 0, (tt / tau_c) * np.exp(1 - tt / tau_c), 0.0)
        dbg = dbg + grams * bump

    dbg = dbg + rng.normal(0, noise_sd, len(dbg))
    return np.concatenate([[120.0], 120.0 + np.cumsum(dbg)])


def calibrate(windows, dia, peak_hat):
    """Recover realistic k, c, drift and noise from the user's own data at their fitted peak."""
    ks, cs, ds, ss = [], [], [], []
    for ts, bg, d_s, d_u in windows:
        x = window_regressor(ts, d_s, d_u, peak_hat, dia)[:-1]
        y = np.diff(bg)
        if np.std(x) < 1e-12:
            continue
        A = np.column_stack([x, np.ones(len(y)), (ts[:-1] - ts[0]) / 3600.0])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        ks.append(-beta[0]); cs.append(beta[1]); ds.append(beta[2])
        ss.append(np.std(y - A @ beta))
    return (float(np.median(ks)), float(np.median(cs)), float(np.median(ds)),
            float(np.median(ss)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tz", required=True)
    ap.add_argument("--dia", type=float, required=True)
    ap.add_argument("--true-peaks", default="35,45,55,75")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--basal-u-per-h", type=float, default=0.6)
    ap.add_argument("--meal-share", type=float, default=0.25)
    ap.add_argument("--out")
    a = ap.parse_args()

    dec, tre = load(a.user, a.tz)
    wins = build_windows(dec, tre)
    if len(wins) < 8:
        print(f"only {len(wins)} windows"); return
    seed_peak, _ = fit_peak(wins, a.dia, linear_drift=True)
    k, c, drift, noise = calibrate(wins, a.dia, seed_peak)

    L, P = [], None
    P = L.append
    P("# Gate 2 self-test — recovering a known peak from simulated glucose\n")
    P(f"User **{a.user}**: {len(wins)} real windows, real dose series. Calibrated from this user's "
      f"own fit — amplitude k={k:.1f}, intercept {c:.2f} mg/dL per 5 min, drift {drift:.2f} "
      f"mg/dL/h, residual noise SD {noise:.2f} mg/dL per 5 min. DIA {a.dia:.0f} min.\n")
    P("\nGlucose is REPLACED by a simulation from a known peak; the unmodified estimator is then "
      "run over it. Recovered should equal true.\n")

    rng = np.random.default_rng(20260804)
    P("\n| true peak | clean | + unmodelled basal | + unannounced meals |")
    P("|---|---|---|---|")
    rows = []
    for tp in [float(v) for v in a.true_peaks.split(",")]:
        got = {"clean": [], "basal": [], "meal": []}
        for _ in range(a.reps):
            for mode in got:
                sw = []
                for ts, bg, d_s, d_u in wins:
                    meal = None
                    if mode == "meal" and rng.random() < a.meal_share:
                        meal = (rng.uniform(0.5, 2.5), rng.uniform(0, 120), rng.uniform(30, 60))
                    sbg = synth_window(
                        ts, d_s, d_u, tp, a.dia, k, c, drift, noise, rng,
                        basal_u_per_h=(a.basal_u_per_h if mode == "basal" else 0.0),
                        basal_peak=tp, meal=meal)
                    sw.append((ts, sbg, d_s, d_u))
                pk, _ = fit_peak(sw, a.dia, linear_drift=True)
                got[mode].append(pk)
        r = {m: (float(np.mean(v)), float(np.std(v))) for m, v in got.items()}
        rows.append((tp, r))
        P(f"| {tp:.0f} | {r['clean'][0]:.1f} ± {r['clean'][1]:.1f} | "
          f"{r['basal'][0]:.1f} ± {r['basal'][1]:.1f} | {r['meal'][0]:.1f} ± {r['meal'][1]:.1f} |")

    bias = [r["clean"][0] - tp for tp, r in rows]
    P(f"\nClean-case bias across the range: {min(bias):+.1f} to {max(bias):+.1f} min.\n")
    P("\n" + ("**The estimator is unbiased on its own generating model.** Any discrepancy against a "
              "configured curve is therefore about the data or an unmodelled term, not the fit.\n"
              if max(abs(b) for b in bias) < 5 else
              "**The estimator does NOT return the peak it was given.** Reported peaks are not "
              "trustworthy until this is explained; treat every Gate 2 number as provisional.\n"))
    bb = [r["basal"][0] - tp for tp, r in rows]
    mb = [r["meal"][0] - tp for tp, r in rows]
    P(f"\nOmitted basal action shifts it by {min(bb):+.1f} to {max(bb):+.1f} min; unannounced meals "
      f"by {min(mb):+.1f} to {max(mb):+.1f} min. These are the terms Gate 2 does not model, and "
      "they are the honest bound on how far an observed peak can sit from the truth.\n")

    open(a.out or os.path.join(HERE, "GATE2_SELFTEST.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
