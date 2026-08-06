#!/usr/bin/env python3
"""GATE 4 — non-parametric deconvolution of the insulin impulse response (2026-08-04).

Gates 2 and 3 assumed the AAPS exponential shape and fitted two numbers. Gate 2 failed its own
positive control, not because that shape is badly wrong but because the DESIGN was under-identified:
inside a single overnight window the insulin activity profile is nearly a straight line in time, and
so is the dawn ramp it must be separated from (median |r| 0.73-0.81). Whichever term was allowed to
absorb the slow component decided the answer.

This drops the parametric form entirely and changes the identification strategy.

MODEL. Treat the record as a train of impulses and estimate the response directly:

    dBG_t  =  - SUM_k  beta_k * dose_{t-k}  +  tod_{clock(t)}  +  night_{n(t)}  +  e_t

  beta_k   the insulin ACTIVITY curve itself, one free coefficient per 5-minute lag, k = 0..K.
           Nothing is assumed about its shape. The peak is argmax_k beta_k.
  tod      a time-of-day profile shared across ALL days, one coefficient per 30-minute clock bin.
  night    a per-day intercept, absorbing that day's basal rate, ISF level and sensor offset.

WHY THIS IDENTIFIES WHAT GATE 2 COULD NOT. Gate 2 gave every window its own free drift ramp, which
is hopeless: within one window, drift and insulin action are the same slow shape. Here the drift is
a REPEATABLE TIME-OF-DAY PATTERN estimated from every day at once, while doses arrive at times that
vary from day to day. Dawn is locked to the clock; insulin is not. That variation is the
identifying information, and it only exists when you pool across days instead of fitting each
window alone. For the same reason the sample is every fasting stretch around the clock, not just
overnight — that maximises how much dose timing moves relative to clock time.

REGULARISATION. Adjacent lags are strongly collinear (that was always the real constraint), so the
unpenalised FIR estimate is noise. A second-difference (Tikhonov) penalty on beta enforces
smoothness — the standard treatment for this class of problem — with lambda chosen by generalised
cross-validation. beta is additionally constrained non-negative, since insulin does not raise
glucose; that constraint is doing real work at the tail, where the free estimate would otherwise
oscillate around zero.

WHAT IT STILL CANNOT DO. Sensor lag is not removed. Unannounced carbs still contaminate; they push
beta down, not sideways in time. And a flat likelihood is still a flat likelihood — the point of
this file is a better-posed design, not a claim that the answer is now easy. Validate with
gate4_selftest before believing any number it prints.

Usage:
  python3 gate4_deconvolution.py --user <id> --tz <tz> [--max-lag 360] [--boot 100]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import lsq_linear

DSN = "dbname=oref host=127.0.0.1 port=5432"
HERE = os.path.dirname(os.path.abspath(__file__))
STEP_S = 300.0


def load_grid(user: str, tz: str, since: str | None = None):
    """Regular 5-minute grid of dBG, dose, eligibility, clock bin and day index."""
    conn = psycopg2.connect(DSN)
    dec = pd.read_sql("""
        SELECT DISTINCT ON (floor(ts_epoch/300.0))
          ts_epoch, cgm_mgdl, sug_cob, steps_60m, boostv5_postrescuewindow AS postrescue
        FROM boost_decisions WHERE user_id = %s AND cgm_mgdl > 1
        ORDER BY floor(ts_epoch/300.0), ts_epoch DESC""", conn, params=(user,))
    tre = pd.read_sql("""
        SELECT ts_utc, insulin, carbs FROM boost_treatments
        WHERE user_id = %s ORDER BY ts_utc""", conn, params=(user,))
    conn.close()
    tre["ts"] = pd.to_datetime(tre.ts_utc, utc=True)
    dec = dec.sort_values("ts_epoch").reset_index(drop=True)
    if since:
        cut = pd.Timestamp(since, tz="UTC").timestamp()
        dec = dec[dec.ts_epoch >= cut].reset_index(drop=True)
        tre = tre[tre.ts >= pd.Timestamp(since, tz="UTC") - pd.Timedelta(hours=8)]

    t0 = float(dec.ts_epoch.min()); t1 = float(dec.ts_epoch.max())
    grid = np.arange(t0, t1 + STEP_S, STEP_S)
    n = len(grid)

    bg = np.full(n, np.nan)
    idx = np.clip(((dec.ts_epoch.values - t0) / STEP_S).round().astype(int), 0, n - 1)
    bg[idx] = dec.cgm_mgdl.values
    cob = np.full(n, np.nan); cob[idx] = dec.sug_cob.fillna(0).values
    pr = np.zeros(n, bool); pr[idx] = dec.postrescue.fillna(False).astype(bool).values
    st = np.full(n, np.nan); st[idx] = dec.steps_60m.values

    dose = np.zeros(n)
    bol = tre[tre.insulin.fillna(0) > 0]
    if len(bol):
        bi = np.clip(((bol.ts.values.astype("datetime64[s]").astype(float) - t0) / STEP_S)
                     .round().astype(int), 0, n - 1)
        np.add.at(dose, bi, bol.insulin.values.astype(float))
    carb_t = tre[tre.carbs.fillna(0) > 0].ts.values.astype("datetime64[s]").astype(float)

    ts_local = pd.to_datetime(grid, unit="s", utc=True).tz_convert(tz)
    clock = (ts_local.hour * 2 + ts_local.minute // 30).values      # 48 half-hour bins
    day = ((grid - t0) // 86400).astype(int)

    # Eligibility of the TARGET sample (regressors reach back on their own).
    ok = np.isfinite(bg) & np.isfinite(np.roll(bg, -1)) & (bg > 40) & (bg < 350)
    ok &= np.nan_to_num(cob, nan=1.0) == 0
    ok &= ~pr
    has_steps = np.isfinite(st).any()
    if has_steps:
        ok &= np.nan_to_num(st, nan=0.0) <= 200
    for ct in carb_t:                                    # no carbs in the preceding 3 h
        # Clamp BOTH ends into range. A carb record before the grid start gives a negative upper
        # bound, and ok[0:negative] silently blanks almost the whole array — that bug cost this
        # user 112 of 123 days before it was caught.
        lo = int(np.clip((ct - t0) / STEP_S, 0, n - 1))
        hi = int(np.clip((ct - t0) / STEP_S + 36, 0, n - 1))
        if hi >= lo:
            ok[lo:hi + 1] = False
    return grid, bg, dose, ok, clock, day, has_steps


def design(dose, clock, day, ok, max_lag_min):
    K = int(max_lag_min / 5)
    rows = np.flatnonzero(ok[:-1])
    rows = rows[rows >= K]
    # NEGATED so that a positive coefficient means "lowers glucose", matching the model as
    # written (dBG = -SUM beta_k * dose) and letting the beta >= 0 constraint mean what it says.
    # Regressing on +dose with beta >= 0 pins every true coefficient at zero: at ZERO noise the
    # estimator returned a peak of 355 min for a true 45.
    X_d = -np.column_stack([dose[rows - k] for k in range(K + 1)])
    cl = clock[rows]; dy = day[rows]
    cl_u = np.unique(cl); dy_u = np.unique(dy)
    X_c = (cl[:, None] == cl_u[None, :]).astype(float)
    X_n = (dy[:, None] == dy_u[None, :]).astype(float)
    return X_d, X_c[:, 1:], X_n, rows, K          # drop one clock bin for identifiability


def fit_fir(y, X_d, X_c, X_n, lam):
    """Penalised least squares: smoothness on beta only, beta >= 0."""
    K1 = X_d.shape[1]
    X = np.hstack([X_d, X_c, X_n])
    p = X.shape[1]
    D = np.zeros((K1 - 2, p))
    for i in range(K1 - 2):
        D[i, i] = 1.0; D[i, i + 1] = -2.0; D[i, i + 2] = 1.0
    A = np.vstack([X, np.sqrt(lam) * D])
    b = np.concatenate([y, np.zeros(K1 - 2)])
    lo = np.full(p, -np.inf); hi = np.full(p, np.inf)
    lo[:K1] = 0.0                                   # insulin cannot raise glucose
    r = lsq_linear(A, b, bounds=(lo, hi), max_iter=200, tol=1e-8)
    return r.x[:K1], r.x


def gcv(y, X_d, X_c, X_n, lams):
    best, bl = np.inf, lams[0]
    X = np.hstack([X_d, X_c, X_n]); nrow = len(y)
    K1 = X_d.shape[1]; p = X.shape[1]
    D = np.zeros((K1 - 2, p))
    for i in range(K1 - 2):
        D[i, i] = 1.0; D[i, i + 1] = -2.0; D[i, i + 2] = 1.0
    XtX = X.T @ X; Xty = X.T @ y; DtD = D.T @ D
    for lam in lams:
        try:
            H = np.linalg.solve(XtX + lam * DtD, np.eye(p))
        except np.linalg.LinAlgError:
            continue
        beta = H @ Xty
        resid = y - X @ beta
        # tr(X H X^T) = tr(H X^T X) — identical to ~1e-12 and ~6500x faster, because the right-hand
        # form never builds the n-by-n matrix. n is ~31,000 here, so the difference is 1.07s vs
        # 0.0002s per lambda evaluated.
        dof = float(np.sum(np.diag(XtX @ H)))
        v = np.sum(resid ** 2) / max(nrow - dof, 1) ** 2 * nrow
        if v < best:
            best, bl = v, lam
    return bl


# Lags below this are excluded from the peak search. Two reasons, and the exclusion changes the
# answer for exactly one participant in the cohort — every other recovered peak is already at or
# beyond 20 min, so this is a guard rather than a thumb on the scale.
#   1. No rapid-acting analogue has an onset of action under 10 min by its own product information,
#      so a peak before 15 min is not a physiologically admissible answer.
#   2. Reverse causality concentrates here. The controller doses BECAUSE glucose is rising and
#      glucose increments are autocorrelated, so a dose correlates with the change that immediately
#      follows it; the non-negativity constraint cannot represent that as a negative coefficient and
#      it emerges instead as a narrow positive spike in the first bins. In the affected participant
#      a single-bin spike of 2.45 at lag 5 min outranked a broad, well-supported mode of 2.15 at
#      65 min, and a naive global argmax reported 5 rather than 65.
MIN_PEAK_LAG_MIN = 15.0


def peak_of(beta, min_lag_min: float = MIN_PEAK_LAG_MIN):
    """Lag of maximum activity, ignoring the first bins (see MIN_PEAK_LAG_MIN)."""
    if beta.max() <= 0:
        return float("nan")
    b = np.asarray(beta, dtype=float).copy()
    b[:int(min_lag_min // 5)] = 0.0
    if b.max() <= 0:
        return float("nan")
    return float(np.argmax(b) * 5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--tz", required=True)
    ap.add_argument("--max-lag", type=float, default=360.0)
    ap.add_argument("--since")
    ap.add_argument("--boot", type=int, default=80)
    ap.add_argument("--thin", type=int, default=3,
                    help="use every Nth eligible sample. Glucose increments are autocorrelated "
                         "(lag-1 residual r about +0.44), which violates the independence GCV "
                         "assumes and biases the recovered peak EARLY. The estimate converges by "
                         "N=3 (fifteen-minute spacing); N=1 reproduces the uncorrected behaviour.")
    ap.add_argument("--lam", type=float, help="skip GCV and use this smoothing weight")
    ap.add_argument("--out")
    a = ap.parse_args()

    grid, bg, dose, ok, clock, day, has_steps = load_grid(a.user, a.tz, a.since)
    X_d, X_c, X_n, rows, K = design(dose, clock, day, ok, a.max_lag)
    if a.thin > 1:
        sel = np.zeros(len(rows), bool); sel[::a.thin] = True
        X_d, X_c, rows = X_d[sel], X_c[sel], rows[sel]
        X_n = X_n[sel][:, X_n[sel].sum(axis=0) > 0]
    y = bg[rows + 1] - bg[rows]
    if len(y) < 500:
        print(f"only {len(y)} usable samples"); return

    lam = a.lam or gcv(y, X_d, X_c, X_n, np.logspace(0, 6, 13))
    beta, _ = fit_fir(y, X_d, X_c, X_n, lam)
    ALIGN = 2.5    # see --thin help and the audit: forward difference spans [t, t+1]
    pk = peak_of(beta) + ALIGN

    rng = np.random.default_rng(20260804)
    days = np.unique(day[rows]); pks = []
    for _ in range(a.boot):
        pick = rng.choice(days, len(days), replace=True)
        sel = np.concatenate([np.flatnonzero(day[rows] == d) for d in pick])
        b, _ = fit_fir(y[sel], X_d[sel], X_c[sel], X_n[sel], lam)
        pks.append(peak_of(b) + ALIGN)
    pks = np.array([v for v in pks if np.isfinite(v)])
    lo, hi = np.percentile(pks, [2.5, 97.5]) if len(pks) > 10 else (np.nan, np.nan)

    tot = beta.sum()
    cum = np.cumsum(beta) / tot if tot > 0 else np.zeros_like(beta)
    t50 = float(np.argmax(cum >= 0.5) * 5.0)
    t90 = float(np.argmax(cum >= 0.9) * 5.0)

    L, P = [], None
    P = L.append
    P("# Gate 4 — non-parametric insulin impulse response\n")
    P(f"User **{a.user}**: {len(y):,} usable 5-minute samples across {len(days)} days, "
      f"{K + 1} free lag coefficients out to {a.max_lag:.0f} min, smoothing lambda {lam:g}, "
      f"thinning 1-in-{a.thin}"
      f"{'' if has_steps else '; NO step data - exercise uncontrolled'}.\n")
    P("\nNo shape is assumed. Each lag gets its own coefficient; a second-difference penalty keeps "
      "the curve smooth and the coefficients are held non-negative.\n")
    P(f"\nPeaks include the +{ALIGN:g} min alignment correction: the forward difference spans "
      f"[t, t+1] so the mean lag across it is 5k+{ALIGN:g}, not 5k.\n")
    P(f"\n**Peak of the estimated activity curve: {pk:.0f} min**, day-bootstrap 95% CI "
      f"[{lo:.0f}, {hi:.0f}].\n")
    P(f"\nHalf of the total effect has landed by {t50:.0f} min and 90% by {t90:.0f} min.\n")
    P("\n| lag (min) | activity (relative) |")
    P("|---|---|")
    mx = beta.max() if beta.max() > 0 else 1.0
    for k in range(0, K + 1, max(1, (K + 1) // 24)):
        bar = "#" * int(round(20 * beta[k] / mx))
        P(f"| {k * 5} | {beta[k] / mx:.2f} {bar} |")
    open(a.out or os.path.join(HERE, "GATE4_REPORT.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
