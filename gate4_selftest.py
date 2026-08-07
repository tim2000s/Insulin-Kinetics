#!/usr/bin/env python3
"""GATE 4 SELF-TEST — positive control for the non-parametric deconvolution (2026-08-04).

Gate 4 is the method that is actually used, so this is the test that matters. It takes a user's
REAL dose series, real sampling and real eligibility mask, replaces the observed glucose with
glucose SIMULATED from a known activity curve, and checks the estimator returns that curve's peak.

Three generating families, because the whole point of dropping the AAPS parametric form was to stop
assuming the shape:

  exponential    the AAPS/oref family the parametric gates assumed. Matched case.
  gamma          different shape parameter, so a different rise and tail.
  bi-exponential different tail ratio again.

A method that only recovers its own family is not non-parametric. Gate 2's parametric estimator
biased -10.4 to +5.3 min on the wrong families; this one should not.

It also runs a ZERO-NOISE case. That is not a formality: it caught two real defects that no amount
of staring at output would have found.

  * a carb record timestamped before the grid start produced a negative slice bound, and
    ok[0:negative] silently blanked 112 of 123 days
  * the design regressed on +dose while constraining beta >= 0, which pins every true (negative)
    coefficient at zero. At zero noise the estimator returned a peak of 355 min for a true 45

At zero noise the answer must be exact. If it is not, stop and fix the pipeline.

Usage:
  python3 gate4_selftest.py --user <id> --tz <tz>
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.optimize import brentq

from gate1_recover_known_curve import activity as exp_activity
from gate4_deconvolution import design, fit_fir, gcv, load_grid, peak_of

HERE = os.path.dirname(os.path.abspath(__file__))


def gamma_activity(t, peak, shape):
    """Gamma-shaped activity with its mode placed at `peak`. shape>2 narrows, ->2 lengthens tail."""
    b = peak / (shape - 1.0)
    return np.where(t > 0, (t / b) ** (shape - 1) * np.exp(-t / b), 0.0)


def biexp_activity(t, peak, ratio):
    """exp(-t/tau1) - exp(-t/tau2) with tau1 = ratio*tau2, solved so the mode sits at `peak`."""
    def mode(t2):
        t1 = ratio * t2
        return np.log(t1 / t2) / (1.0 / t2 - 1.0 / t1) - peak
    t2 = brentq(mode, 0.5, 400.0)
    t1 = ratio * t2
    return np.where(t > 0, np.exp(-t / t1) - np.exp(-t / t2), 0.0)


def simulate(dose, rows, K, kernel, noise_sd, clock, rng, tod_amp=2.5):
    """Glucose deltas generated from a KNOWN kernel, plus a clock-locked drift and white noise."""
    X = np.column_stack([dose[rows - k] for k in range(K + 1)])
    signal = X @ (kernel / max(kernel.max(), 1e-12))
    tod = tod_amp * np.sin(2 * np.pi * (clock[rows] - 14) / 48.0)
    return -signal + tod + rng.normal(0.0, noise_sd, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tz", required=True)
    ap.add_argument("--max-lag", type=float, default=360.0)
    ap.add_argument("--noise", type=float, default=0.9,
                    help="realistic-noise level as a multiple of the user's own dBG SD")
    ap.add_argument("--out")
    a = ap.parse_args()

    grid, bg, dose, ok, clock, day, has_steps, *_ = load_grid(a.user, a.tz)
    X_d, X_c, X_n, rows, K = design(dose, clock, day, ok, a.max_lag)
    if len(rows) < 800:
        print(f"only {len(rows)} usable samples"); return
    sd = float(np.std(bg[rows + 1] - bg[rows]))
    rng = np.random.default_rng(20260804)
    lag = np.arange(K + 1) * 5.0

    cases = [
        ("AAPS exponential", lambda t: exp_activity(t, 35.0, 480.0), 35.0),
        ("AAPS exponential", lambda t: exp_activity(t, 55.0, 480.0), 55.0),
        ("AAPS exponential", lambda t: exp_activity(t, 75.0, 480.0), 75.0),
        ("gamma, shape 3 (wrong family)", lambda t: gamma_activity(t, 45.0, 3.0), 45.0),
        ("gamma, shape 6 (wrong family)", lambda t: gamma_activity(t, 35.0, 6.0), 35.0),
        ("bi-exponential, ratio 8 (wrong family)", lambda t: biexp_activity(t, 45.0, 8.0), 45.0),
    ]

    L, P = [], None
    P = L.append
    P("# Gate 4 self-test — recovering a known peak from simulated glucose\n")
    P(f"User **{a.user}**: {len(rows):,} usable samples over {len(np.unique(day[rows]))} days, "
      f"{K + 1} free lag coefficients, observed dBG SD {sd:.2f} mg/dL per 5 min.\n")
    P("\nThe real dose series and sampling are kept; only glucose is replaced, by a simulation from "
      "a known curve. Recovered must equal true — exactly, in the zero-noise column.\n")
    P("\n| generating curve | true peak | no noise | realistic noise |")
    P("|---|---|---|---|")

    worst_clean = 0.0
    for name, fn, true_peak in cases:
        kern = fn(lag)
        got = []
        for nm in (0.0, a.noise):
            y = simulate(dose, rows, K, kern, nm * sd, clock, rng)
            lam = gcv(y, X_d, X_c, X_n, np.logspace(1, 5, 5))
            got.append(peak_of(fit_fir(y, X_d, X_c, X_n, lam)[0]))
        worst_clean = max(worst_clean, abs(got[0] - true_peak))
        P(f"| {name} | {true_peak:.0f} | {got[0]:.0f} | {got[1]:.0f} |")

    P(f"\nWorst zero-noise error across all families: **{worst_clean:.0f} min**.\n")
    P("\n" + ("**PASS** — the estimator recovers the generating peak, including when the "
              "generating family is not the one the parametric gates assumed.\n"
              if worst_clean <= 5 else
              "**FAIL** — the estimator does not return the peak it was given at zero noise. This "
              "is a pipeline defect, not a finding about insulin. Do not interpret any Gate 4 "
              "output until it is fixed.\n"))

    open(a.out or os.path.join(HERE, "GATE4_SELFTEST.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
