#!/usr/bin/env python3
"""FEEDBACK SIMULATION — does the controller's own reaction bias the recovered peak?

Every method here regresses glucose change on lagged doses. That assumes the dose is an EXOGENOUS
input. In a closed loop it is not: the controller doses BECAUSE glucose is rising. Measured on real
data across five users:

    corr(dose, glucose change in the preceding 5 min)   +0.25 to +0.55
    corr(dose, glucose change in the following 15-30 min) -0.05 to +0.04

The reaction function dominates the raw dose-glucose relationship by an order of magnitude. This is
the classic closed-loop system identification problem, and it is invisible to a positive control
built by simulating glucose FROM the doses — because that construction makes the input exogenous by
definition. Both self-tests in this repo have that blind spot. This file is the test that does not.

It simulates a closed loop end to end: glucose responds to insulin through a KNOWN kernel, and a
proportional controller doses in response to glucose. Then it runs the estimator over the result and
asks whether the recovered peak is still the one that generated the data.

`--gain 0` gives an open loop (doses uncorrelated with glucose) as the reference case; higher gains
give progressively more reactive control. If bias grows with gain, endogeneity is biting.

Usage:
  python3 feedback_simulation.py [--true-peaks 35,55,75] [--gains 0,0.5,1.0]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from gate1_recover_known_curve import activity as exp_activity
from gate4_deconvolution import design, fit_fir, gcv, peak_of

HERE = os.path.dirname(os.path.abspath(__file__))
STEP_S = 300.0


def simulate(true_peak, dia, days, gain, rng, isf=40.0, target=110.0):
    """5-minute closed loop. Returns (bg, dose, clock, day) on a regular grid."""
    n = days * 288
    K = int(dia / 5)
    kern = exp_activity(np.arange(K + 1) * 5.0, true_peak, dia)
    kern = kern / kern.max()

    bg = np.zeros(n); bg[0] = 120.0
    dose = np.zeros(n)
    # clock-locked drift (dawn) plus slow wander, so the estimator has a real confounder to fight
    tod = 2.0 * np.sin(2 * np.pi * (np.arange(n) % 288 - 160) / 288.0)
    wander = np.cumsum(rng.normal(0, 0.05, n)); wander -= wander.mean()

    for t in range(1, n):
        lo = max(0, t - K - 1)
        infl = float(np.dot(dose[lo:t][::-1], kern[:t - lo]))
        d = -isf * infl * 0.02 + tod[t] + wander[t] + rng.normal(0, 2.2)
        bg[t] = float(np.clip(bg[t - 1] + d, 50, 350))
        if gain > 0 and t >= 3:
            rise = bg[t] - bg[t - 3]                       # controller sees the last 15 min
            u = gain * max(0.0, (bg[t] - target) / 60.0 + rise / 12.0)
            dose[t] = round(min(u, 2.0), 2) if (bg[t] > 100 and u > 0.04) else 0.0
        elif gain == 0:
            # open loop: same total insulin, but delivered at times unrelated to glucose
            if rng.random() < 0.12:
                dose[t] = round(float(rng.uniform(0.05, 1.2)), 2)

    clock = (np.arange(n) % 288) // 6                       # 48 half-hour bins
    day = np.arange(n) // 288
    return bg, dose, clock, day


def endogeneity(bg, dose):
    """Correlation of dose with glucose change before vs after it — the diagnostic itself."""
    db = np.diff(bg); m = len(db)

    def cc(shift):
        a = max(0, -shift); b = min(m, m - shift)
        return float(np.corrcoef(dose[a:b], db[a + shift:b + shift])[0, 1])
    return cc(-1), cc(3)


def recover(bg, dose, clock, day, max_lag=360.0):
    ok = np.ones(len(bg), bool); ok[-1] = False
    X_d, X_c, X_n, rows, K = design(dose, clock, day, ok, max_lag)
    y = bg[rows + 1] - bg[rows]
    lam = gcv(y, X_d, X_c, X_n, np.logspace(1, 5, 5))
    return peak_of(fit_fir(y, X_d, X_c, X_n, lam)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true-peaks", default="35,55,75")
    ap.add_argument("--gains", default="0,0.5,1.0")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--reps", type=int, default=3, help="repeats per cell, averaged")
    ap.add_argument("--dia", type=float, default=480.0)
    ap.add_argument("--out")
    a = ap.parse_args()

    peaks = [float(v) for v in a.true_peaks.split(",")]
    gains = [float(v) for v in a.gains.split(",")]

    L, P = [], None
    P = L.append
    P("# Feedback simulation — is the recovered peak robust to controller reaction?\n")
    P(f"{a.days} simulated days per cell, DIA {a.dia:.0f} min. Gain 0 is an open loop (doses "
      "unrelated to glucose); higher gains dose more aggressively off the recent rise.\n")
    P("\n| true peak | " + " | ".join(f"gain {g:g}" for g in gains) + " |")
    P("|---" * (len(gains) + 1) + "|")

    endo, errs = {}, {g: [] for g in gains}
    for tp in peaks:
        cells = []
        for g in gains:
            reps = []
            for rep in range(a.reps):
                rng = np.random.default_rng(int(tp) * 1000 + int(g * 10) * 7 + rep)
                bg, dose, clock, day = simulate(tp, a.dia, a.days, g, rng)
                if rep == 0:
                    endo[(tp, g)] = endogeneity(bg, dose)
                reps.append(recover(bg, dose, clock, day))
            got = float(np.mean(reps))
            errs[g].append(got - tp)
            cells.append(f"{got:.0f} ({got - tp:+.0f})")
        P(f"| {tp:.0f} | " + " | ".join(cells) + " |")

    # The verdict must compare against the OPEN-LOOP reference. A method that is simply noisy will
    # miss at gain 0 too, and calling that "feedback bias" would be wrong.
    mae = {g: float(np.mean(np.abs(errs[g]))) for g in gains}
    P("\n| gain | mean |error| |")
    P("|---|---|")
    for g in gains:
        P(f"| {g:g}{' (open loop reference)' if g == 0 else ''} | {mae[g]:.1f} min |")
    base = mae.get(0.0)
    excess = (max(mae[g] for g in gains if g > 0) - base) if base is not None else float('nan')
    P(f"\nExcess error attributable to feedback, over the open-loop reference: "
      f"**{excess:+.1f} min**.\n")
    P("\n## Endogeneity actually induced\n")
    P("| true peak | gain | corr(dose, dBG before) | corr(dose, dBG after) |")
    P("|---|---|---|---|")
    for (tp, g), (b, af) in endo.items():
        P(f"| {tp:.0f} | {g:g} | {b:+.3f} | {af:+.3f} |")
    P("\nFor comparison, real cohort data shows +0.25 to +0.55 before and roughly zero after. If "
      "the simulated 'before' correlations do not reach that range, this test has not actually "
      "exercised the failure mode and its pass is worth little.\n")
    if base is not None and base > 10:
        P("\n**INCONCLUSIVE** — the OPEN-loop reference is already off by "
          f"{base:.0f} min, so this simulation is too noisy to attribute anything to feedback. "
          "Raise --days or --reps before reading the gain columns.\n")
    elif excess <= 5:
        P("\n**PASS** — error does not grow with controller gain beyond the open-loop reference, "
          "so endogenous dosing is not measurably biasing the peak at these gains.\n")
    else:
        P("\n**FAIL** — error grows with controller gain over and above the open-loop reference. "
          "Peaks estimated from closed-loop data are confounded by the controller's reaction "
          "function, and need an instrument or deliberate excitation.\n")

    open(a.out or os.path.join(HERE, "FEEDBACK_SIM.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
