#!/usr/bin/env python3
"""GATE 1 — can we recover an insulin curve we already know? (2026-08-04)

Before asking what a user's REAL insulin peak is, establish that the estimator can recover a
curve whose answer is known by construction. If it cannot, nothing downstream is worth building.

THE TRICK. `boost_decisions.iob_bolusiob` is AAPS's bolus-only IOB. It is a deterministic
function of the bolus history and the configured insulin curve:

    iob_bolusiob(t) = SUM_j  dose_j * iobFraction(t - t_j)

So deconvolving iob_bolusiob against the delivered-bolus series must return the CONFIGURED
curve. Pass that configured curve in via --expect-peak/--expect-dia; any answer far from it is a
failure of the method, not a discovery about physiology.

WHY THIS IS THE RIGHT GATE, rather than a simulator test. It runs on the real dose series, so it
exercises the actual spacing and burst structure of a closed loop dosing every five minutes —
which is where the real risk lies. The danger here was never sample size (~19,000 boluses per
user); it is COLLINEARITY between adjacent lags. If "insulin 30 minutes ago" and "insulin 35
minutes ago" are nearly the same regressor, the kernel is not identifiable no matter how much
data there is. A simulator with tidy dosing would hide exactly that failure.

Bolus-only IOB is used deliberately: total IOB folds in basal and temp-basal delivery, which is
NOT in the database (Temp Basal records carry a rate, not an insulin amount), so it cannot be
reconstructed. Restricting to the bolus channel keeps the identity exact.

`iob_bolusiob` is logged as a column but is entirely NULL — Nightscout's devicestatus payload does
not carry it — so it is DERIVED here as `iob_iob - iob_basaliob`, which is AAPS's own
decomposition (basaliob is net of scheduled basal and is routinely negative under zero-temps).

MODEL. AAPS's exponential ("free-peak") insulin model, parameterised by peak time tp and
duration td, both in minutes:

    tau  = tp * (1 - tp/td) / (1 - 2*tp/td)
    a    = 2*tau/td
    S    = 1 / (1 - a + (1 + a) * exp(-td/tau))
    IOBfrac(t) = 1 - S * (1 - a) * ((t^2/(tau*td*(1-a)) - t/tau - 1) * exp(-t/tau) + 1)
    activity(t) = (S/tau^2) * t * (1 - t/td) * exp(-t/tau)

Fit (tp, td) by least squares against the logged IOB series. Both are shape parameters and the
amplitude is fixed by the doses, so this is well posed IF the lags are identifiable.

Usage:  python3 gate1_recover_known_curve.py --user <id> [--expect-peak 38] [--expect-dia 600]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import least_squares

DSN = "dbname=oref host=127.0.0.1 port=5432"
HERE = os.path.dirname(os.path.abspath(__file__))


# ── AAPS exponential insulin model ──────────────────────────────────────────────
def iob_fraction(t_min: np.ndarray, peak: float, dia_min: float) -> np.ndarray:
    """Fraction of a dose still active t minutes after delivery (0 outside [0, dia])."""
    t = np.asarray(t_min, dtype=float)
    tau = peak * (1 - peak / dia_min) / (1 - 2 * peak / dia_min)
    a = 2 * tau / dia_min
    s = 1 / (1 - a + (1 + a) * np.exp(-dia_min / tau))
    with np.errstate(over="ignore", invalid="ignore"):
        frac = 1 - s * (1 - a) * (
            (t * t / (tau * dia_min * (1 - a)) - t / tau - 1) * np.exp(-t / tau) + 1)
    frac = np.where((t < 0) | (t > dia_min), 0.0, frac)
    return np.clip(frac, 0.0, 1.0)


def activity(t_min: np.ndarray, peak: float, dia_min: float) -> np.ndarray:
    """Insulin ACTION at t minutes — the curve whose peak the loop actually cares about."""
    t = np.asarray(t_min, dtype=float)
    tau = peak * (1 - peak / dia_min) / (1 - 2 * peak / dia_min)
    a = 2 * tau / dia_min
    s = 1 / (1 - a + (1 + a) * np.exp(-dia_min / tau))
    act = (s / (tau * tau)) * t * (1 - t / dia_min) * np.exp(-t / tau)
    return np.where((t < 0) | (t > dia_min), 0.0, act)


# ── data ────────────────────────────────────────────────────────────────────────
def load(user: str, days: int, offset_days: int = 0, drop_external: bool = False):
    conn = psycopg2.connect(DSN)
    dec = pd.read_sql(f"""
        SELECT DISTINCT ON (floor(ts_epoch/300.0))
          ts_utc, ts_epoch, (iob_iob - iob_basaliob) AS bolus_iob
        FROM boost_decisions
        WHERE user_id = %s AND iob_iob IS NOT NULL AND iob_basaliob IS NOT NULL
          AND ts_utc > now() - interval '{days + offset_days} days'
          AND ts_utc <= now() - interval '{offset_days} days'
        ORDER BY floor(ts_epoch/300.0), ts_epoch DESC""", conn, params=(user,))
    bol = pd.read_sql(f"""
        SELECT ts_utc, insulin, event_type FROM boost_treatments
        WHERE user_id = %s AND insulin > 0
          AND ts_utc > now() - interval '{days + offset_days + 2} days'
          AND ts_utc <= now() - interval '{offset_days} days'
        ORDER BY ts_utc""", conn, params=(user,))
    conn.close()
    # Unlike Gate 4, Gate 1 deconvolves the app's OWN logged insulin-on-board, and the app counts
    # externally-logged insulin into that series essentially in full (measured ratio 0.99 U of IOB
    # per unit logged). Dropping these records would therefore break the identity Gate 1 rests on,
    # so they are retained by default. The switch exists to test whether the app applies the same
    # decay curve to them as to pump boluses.
    if drop_external and "event_type" in bol.columns:
        bol = bol[~bol.event_type.eq("External Insulin")]
    for d in (dec, bol):
        d["ts"] = pd.to_datetime(d.ts_utc, utc=True)
    return dec.sort_values("ts").reset_index(drop=True), bol


def predicted_iob(grid_s: np.ndarray, dose_grid: np.ndarray, peak: float, dia: float,
                  step_s: float) -> np.ndarray:
    """IOB on a regular grid = dose series convolved with the IOB-remaining kernel.

    Done as a convolution rather than a per-dose superposition: the naive form is
    O(n_doses x n_times) (~2e8 per residual evaluation here) and the optimiser calls it hundreds
    of times. On a regular grid the same quantity is one convolution with a DIA-length kernel.
    """
    ntap = int(np.ceil(dia * 60.0 / step_s)) + 1
    lag_min = np.arange(ntap) * step_s / 60.0
    kern = iob_fraction(lag_min, peak, dia)
    return np.convolve(dose_grid, kern)[:len(grid_s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="user id as stored in the database")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--offset-days", type=int, default=0,
                    help="end the window N days ago instead of now. Two disjoint windows "
                         "(e.g. --days 14 --offset-days 14, then --days 14) test whether the "
                         "configured curve CHANGED mid-period — an insulin or settings switch "
                         "would otherwise be pooled into one blended estimate.")
    ap.add_argument("--expect-peak", type=float, default=38.0)
    ap.add_argument("--expect-dia", type=float, default=600.0)
    ap.add_argument("--drop-external", action="store_true",
                    help="exclude 'External Insulin' records. Off by default: the app counts them "
                         "into the logged IOB this gate deconvolves, so removing them would break "
                         "the identity. Use to test whether they decay on the same curve.")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", help="report path (default GATE1_REPORT.md beside the script)")
    ap.add_argument("--fix-dia", action="store_true",
                    help="hold DIA at the configured value and fit peak alone")
    ap.add_argument("--mask-post-bolus", type=float, default=10.0,
                    help="drop observations within N minutes of any bolus. The IOB step at a bolus "
                         "is the size of the dose, so a sub-grid timing error there produces a "
                         "residual of up to a whole dose. Masking cuts the relative residual by "
                         "roughly a third to a half and, more importantly, removes a DOWNWARD bias "
                         "on DIA (one user moved 314 -> 486 min against a known 600). The peak "
                         "estimate barely moves (<0.2 min), so this is close to free.")
    a = ap.parse_args()

    dec, bol = load(a.user, a.days, a.offset_days, drop_external=a.drop_external)
    if dec.empty or bol.empty:
        print("no data"); return
    STEP = 300.0                                    # the loop's own 5-minute cadence
    t0 = float(bol.ts.min().timestamp())
    t1 = float(dec.ts.max().timestamp())
    grid_s = np.arange(t0, t1 + STEP, STEP)
    # bin doses onto the grid (sum within a bin; sub-grid timing is below CGM resolution anyway)
    d_s = bol.ts.values.astype("datetime64[s]").astype(float)
    d_u = bol.insulin.values.astype(float)
    dose_grid = np.zeros(len(grid_s))
    np.add.at(dose_grid, np.clip(((d_s - t0) / STEP).astype(int), 0, len(grid_s) - 1), d_u)
    # observations aligned to the same grid
    obs_idx = np.clip(((dec.ts_epoch.values.astype(float) - t0) / STEP).astype(int),
                      0, len(grid_s) - 1)
    obs_all = dec.bolus_iob.values.astype(float)
    obs_t = dec.ts_epoch.values.astype(float)
    # Warm-up: dose history before t0 is unknown, so early IOB is under-supplied. The bolus query
    # reaches 2 days further back than the decision query, which already covers any DIA the fit can
    # return (bounded at 1440 min), so in practice this drops nothing — it is a guard, not a filter.
    keep = obs_t >= t0 + a.expect_dia * 60
    if a.mask_post_bolus > 0:
        gap = obs_t[:, None] - d_s[None, :]
        keep &= ~((gap >= 0) & (gap < a.mask_post_bolus * 60)).any(axis=1)
    obs_idx, obs = obs_idx[keep], obs_all[keep]
    masked_pct = 100.0 * (1.0 - keep.mean())

    def resid(p):
        return predicted_iob(grid_s, dose_grid, p[0], p[1], STEP)[obs_idx] - obs

    if a.fix_dia:
        # DIA held at the configured value. The IOB tail beyond ~5 h is a fraction of a percent of
        # a 0.1-0.3 U dose, i.e. below rounding, so DIA carries almost no signal — fitting it only
        # lets it absorb residual that belongs elsewhere and drags the peak with it.
        f1 = least_squares(lambda q: resid([q[0], a.expect_dia]), x0=[60.0],
                           bounds=([10.0], [180.0]), xtol=1e-10, ftol=1e-10)
        fit = type("F", (), {"x": np.array([f1.x[0], a.expect_dia])})()
    else:
        fit = least_squares(resid, x0=[60.0, 360.0], bounds=([10.0, 120.0], [180.0, 1440.0]),
                            xtol=1e-10, ftol=1e-10)
    peak_hat, dia_hat = fit.x
    rmse = float(np.sqrt(np.mean(resid(fit.x) ** 2)))
    denom = float(np.sqrt(np.mean((obs - obs.mean()) ** 2)))

    # block bootstrap over days, so the CI reflects the real correlation structure
    rng = np.random.default_rng(20260804)
    day = (grid_s[obs_idx] // 86400).astype(int)
    days_u = np.unique(day)
    peaks, dias, boot_failed = [], [], []
    for _ in range(a.boot):
        pick = rng.choice(days_u, len(days_u), replace=True)
        idx = np.concatenate([np.flatnonzero(day == dd) for dd in pick])
        ib, ob = obs_idx[idx], obs[idx]
        try:
            if a.fix_dia:
                fb = least_squares(
                    lambda q: predicted_iob(grid_s, dose_grid, q[0], a.expect_dia, STEP)[ib] - ob,
                    x0=[fit.x[0]], bounds=([10.0], [180.0]), xtol=1e-8, ftol=1e-8)
                peaks.append(fb.x[0]); dias.append(a.expect_dia)
            else:
                fb = least_squares(lambda p: predicted_iob(grid_s, dose_grid, p[0], p[1], STEP)[ib] - ob,
                                   x0=fit.x, bounds=([10.0, 120.0], [180.0, 1440.0]),
                                   xtol=1e-8, ftol=1e-8)
                peaks.append(fb.x[0]); dias.append(fb.x[1])
        except Exception as e:                                       # noqa: BLE001
            # A dropped replicate is not free. The interval would then be computed over the
            # replicates that CONVERGED — a survivorship-selected subset, not the bootstrap
            # distribution. Count them so the report discloses the loss rather than hiding it.
            boot_failed.append(repr(e)[:80])
    pk_lo, pk_hi = np.percentile(peaks, [2.5, 97.5]) if len(peaks) > 20 else (np.nan, np.nan)
    di_lo, di_hi = np.percentile(dias, [2.5, 97.5]) if len(dias) > 20 else (np.nan, np.nan)

    L = []
    P = L.append
    P("# Gate 1 — recovering a known insulin curve\n")
    P(f"User **{a.user}**, last {a.days} days: {len(obs)} cycles, {int((dose_grid>0).sum())} dosing bins "
      f"({dose_grid.sum():.0f} U total).\n")
    if boot_failed:
        P(f"\n**{len(boot_failed)} of {a.boot} bootstrap replicates failed to converge** and are "
          f"absent from the interval below, which is therefore computed over survivors and may be "
          f"optimistically narrow. First failure: `{boot_failed[0]}`.\n")
    P("\nDeconvolves the logged bolus-only IOB series against the delivered boluses. The answer is "
      "known by construction — it must return the configured curve.\n")
    P("\n| | configured | recovered | 95% CI |")
    P("|---|---|---|---|")
    P(f"| peak (min) | {a.expect_peak:.0f} | **{peak_hat:.1f}** | [{pk_lo:.1f}, {pk_hi:.1f}] |")
    P(f"| DIA (min)  | {a.expect_dia:.0f} | **{dia_hat:.0f}** | [{di_lo:.0f}, {di_hi:.0f}] |")
    rel = rmse / denom if denom else float("nan")
    P(f"\nFit RMSE {rmse:.4f} U against an IOB series of RMS {denom:.4f} U (relative {rel:.4f}); "
      f"{masked_pct:.0f}% of observations masked as within {a.mask_post_bolus:g} min of a bolus.\n")
    if rel > 0.15:
        P(f"\n**Residual is large for an exact identity ({rel:.0%}).** Something in this user's "
          "dose record is not reaching the model the way the app saw it — candidates are extended "
          "or multiwave boluses, partial delivery, or bolus timestamps that differ from the "
          "delivery the loop counted. Treat the recovered curve as provisional and run the "
          "two-kernel dose-size control before using it.\n")

    # DIA LEVERAGE. The bootstrap CI on DIA is not trustworthy — it is narrowest where the
    # likelihood is flattest. What actually decides whether a DIA is identified is how much the
    # predicted IOB series moves when DIA moves, measured against the fit residual. Anything below
    # ~1x the fit RMSE is a DIA the data cannot distinguish from the fitted one.
    P("\n## Is DIA identified at all?\n")
    P("| DIA (min) | RMS change in predicted IOB | vs fit RMSE |")
    P("|---|---|---|")
    lev = []
    for dtest in (240, 360, 480, 600, 900, 1440):
        alt = predicted_iob(grid_s, dose_grid, peak_hat, float(dtest), STEP)[obs_idx]
        base = predicted_iob(grid_s, dose_grid, peak_hat, dia_hat, STEP)[obs_idx]
        v = float(np.sqrt(np.mean((alt - base) ** 2)))
        lev.append(v)
        P(f"| {dtest} | {v:.3f} U | {v / rmse if rmse else float('nan'):.2f}x |")
    dia_identified = max(lev) > 2.0 * rmse
    P("\n" + ("Some alternatives move the fit well clear of the residual, so DIA carries real "
              "information here — though neighbouring values may still be indistinguishable.\n"
              if dia_identified else
              "**Every alternative DIA sits within about twice the fit residual, so DIA is NOT "
              "identified for this user.** Ignore the recovered value and its interval; the curve "
              "is pinned by its rising limb, not its tail.\n"))

    ok_pk = abs(peak_hat - a.expect_peak) <= max(3.0, 0.1 * a.expect_peak)
    # Only hold the fit to its DIA when DIA is identified. Failing a user on a parameter the data
    # cannot constrain reports a method defect where there is none — the peak is what Gate 2 needs.
    ok_di = (abs(dia_hat - a.expect_dia) <= max(30.0, 0.1 * a.expect_dia)) or not dia_identified
    if ok_pk and not dia_identified:
        P("\n*(The verdict below rests on the peak alone: DIA is unidentified for this user, so a "
          "mismatch there is not evidence of anything.)*\n")
    P(f"\n**GATE: {'PASS' if (ok_pk and ok_di) else 'FAIL'}** — "
      + ("both parameters recovered within tolerance, so the kernel is identifiable under this "
         "user's real dose spacing and the method may proceed to observed glucose.\n"
         if ok_pk and ok_di else
         "the estimator did NOT return the configured curve. Since the relationship is an exact "
         "identity, this is a defect in the method or an unmodelled term (check: does the logged "
         "IOB include basal? is the configured curve actually what we assumed?), NOT a finding "
         "about insulin.\n"))
    P("\n## Peak of ACTION vs peak of IOB decay\n")
    tt = np.arange(0, a.expect_dia, 1.0)
    # argmax-ok: this is the ANALYTIC activity curve evaluated on a 1-minute grid from recovered
    # parameters, not an estimated kernel. It is unimodal by construction and carries no noise, so
    # the reverse-causality guard that peak_of applies would be meaningless here.
    P(f"For the recovered curve, action peaks at **{tt[np.argmax(activity(tt, peak_hat, dia_hat))]:.0f} min** "
      "— this is the number the loop uses, and the one a glucose-based estimate is comparable to.\n")

    open(a.out or os.path.join(HERE, "GATE1_REPORT.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
