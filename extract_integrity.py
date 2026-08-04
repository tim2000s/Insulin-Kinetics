#!/usr/bin/env python3
"""EXTRACT INTEGRITY — could the data pipeline itself be faking the recovered peak? (2026-08-04)

Gate 1 recovers a peak by deconvolving a loop's logged bolus IOB against its delivered doses. That
answer is only as good as the extract. Four things in the pipeline could produce a wrong peak with
no other symptom, and each has a check that does not involve the kernel at all:

  1 DERIVATION   bolus IOB is normally derived as (iob - basaliob) because most uploaders do not
                 send bolusiob. Where an uploader DOES send it, the two must agree.
  2 DUPLICATION  a bolus counted twice inflates the modelled input; the fit compensates by
                 flattening the kernel.
  3 COMPLETENESS a bolus missing from the extract leaves IOB the model cannot explain; the fit
                 compensates by stretching the kernel later.
  4 ALIGNMENT    a systematic offset between dose timestamps and decision timestamps shifts the
                 recovered peak by roughly that offset. This is the dangerous one, because it is
                 invisible in every summary statistic and looks exactly like a different insulin.

Checks 2-4 are kernel-free: at each recorded bolus the logged IOB must step up by that dose, in
that 5-minute bin. If it does, the doses are complete, unduplicated and correctly timestamped,
whatever curve the loop is using.

Check 4 also reports a SENSITIVITY: refit with the dose series deliberately shifted, to show how
large a timing error would have to be to explain the gap you are investigating. Zero shift should
give the best residual; if it does not, trust the shift, not the peak.

Usage:
  python3 extract_integrity.py --user <id> [--days 120]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import least_squares

from gate1_recover_known_curve import DSN, load, predicted_iob

STEP = 300.0


def build_grid(dec, bol):
    t0 = float(bol.ts.min().timestamp())
    t1 = float(dec.ts_epoch.max())
    grid = np.arange(t0, t1 + STEP, STEP)
    iob = np.full(len(grid), np.nan)
    # round, not truncate, so a sample lands in its nearest bin
    iob[np.clip(((dec.ts_epoch.values.astype(float) - t0) / STEP).round().astype(int),
                0, len(grid) - 1)] = dec.bolus_iob.values
    d_s = bol.ts.values.astype("datetime64[s]").astype(float)
    d_u = bol.insulin.values.astype(float)
    dose = np.zeros(len(grid))
    np.add.at(dose, np.clip(((d_s - t0) / STEP).round().astype(int), 0, len(grid) - 1), d_u)
    return t0, grid, iob, dose, d_s, d_u


def check_derivation(user):
    """1. Where the uploader sends bolusiob directly, does our derived value match it?"""
    conn = psycopg2.connect(DSN)
    df = pd.read_sql("""
        SELECT iob_bolusiob, (iob_iob - iob_basaliob) AS derived
        FROM boost_decisions
        WHERE user_id = %s AND iob_bolusiob IS NOT NULL
          AND iob_iob IS NOT NULL AND iob_basaliob IS NOT NULL""", conn, params=(user,))
    dup = pd.read_sql("""
        SELECT count(*) n, count(DISTINCT ns_id) uniq FROM boost_treatments
        WHERE user_id = %s""", conn, params=(user,))
    conn.close()
    if len(df):
        print(f"  derived (iob - basaliob) vs uploaded bolusiob: n={len(df):,}, "
              f"mean |diff| = {(df.iob_bolusiob - df.derived).abs().mean():.6f} U")
    else:
        print("  uploader does not send bolusiob — derivation cannot be cross-checked here")
    r = dup.iloc[0]
    print(f"  treatment rows {r.n:,}, distinct Nightscout ids {r.uniq:,}"
          f"{'  <-- DUPLICATES' if r.n != r.uniq else '  (no duplicates)'}")


def check_steps(grid, iob, dose, min_dose=1.0, show=6):
    """2+3. The IOB step at an isolated bolus must equal the dose."""
    rows, ratios = [], []
    for i in np.flatnonzero(dose >= min_dose):
        if i < 4 or i > len(grid) - 5:
            continue
        if dose[i - 3:i].sum() > 0 or dose[i + 1:i + 4].sum() > 0:
            continue                                   # isolated: nothing either side
        w = iob[i - 3:i + 4]
        if not np.isfinite(w).all():
            continue
        step = w[3] - w[2]                             # into the bin the dose is stamped in
        rows.append((dose[i], w, step)); ratios.append(step / dose[i])
    print(f"  isolated doses >= {min_dose:g} U with unbroken IOB either side: n={len(rows)}")
    for d_u, w, step in rows[:show]:
        print(f"    dose {d_u:5.2f} U   IOB " + " ".join(f"{v:6.3f}" for v in w)
              + f"   step {step:+.3f}")
    if ratios:
        r = np.array(ratios)
        print(f"    step / dose: median {np.median(r):.3f}  IQR "
              f"[{np.percentile(r, 25):.3f}, {np.percentile(r, 75):.3f}]")
        print("    expect slightly BELOW 1.0 (pre-existing IOB decays within the bin). "
              "Above 1.0 => doses missing from the extract; far below => double counting.")


def check_alignment(t0, grid, iob, dose, d_s, d_u, shifts=(-30, -20, -10, -5, 0, 5, 10, 20, 30)):
    """4. Where does IOB actually step, and how much would a timing error move the peak?"""
    # kernel-free: correlate the dose series against the IOB increment ARRIVING at each bin
    step_in = np.full(len(grid), np.nan)
    step_in[1:] = iob[1:] - iob[:-1]
    m = np.isfinite(step_in) & np.isfinite(dose)
    print("  correlation of dose with the IOB increment arriving in that bin, by lag:")
    for lag in range(-3, 4):
        b = np.roll(step_in, lag)
        mm = m & np.isfinite(b)
        r = np.corrcoef(dose[mm], b[mm])[0, 1]
        print(f"    dose vs increment {lag * 5:+3d} min: r = {r:+.3f}"
              + ("   <-- expected best (same bin)" if lag == 0 else ""))

    print("\n  sensitivity: refit with the dose series deliberately shifted")
    ts = np.arange(len(grid)) * STEP + t0
    obs_ok = np.isfinite(iob)
    for sh in shifts:
        dg = np.zeros(len(grid))
        np.add.at(dg, np.clip(((d_s + sh * 60 - t0) / STEP).astype(int), 0, len(grid) - 1), d_u)
        gap = ts[:, None] - (d_s + sh * 60)[None, :]
        near = ((gap >= 0) & (gap < 600)).any(axis=1)   # 10-min post-bolus mask
        keep = obs_ok & ~near
        oi = np.flatnonzero(keep)
        ob = iob[keep]
        f = least_squares(lambda p: predicted_iob(grid, dg, p[0], p[1], STEP)[oi] - ob,
                          x0=[60.0, 500.0], bounds=([10.0, 120.0], [180.0, 1440.0]),
                          xtol=1e-9, ftol=1e-9)
        res = predicted_iob(grid, dg, f.x[0], f.x[1], STEP)[oi] - ob
        rel = np.sqrt(np.mean(res ** 2)) / np.sqrt(np.mean((ob - ob.mean()) ** 2))
        print(f"    doses {sh:+3d} min -> peak {f.x[0]:5.1f}, DIA {f.x[1]:4.0f}, relRMSE {rel:.3f}"
              + ("   <-- no shift" if sh == 0 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--min-dose", type=float, default=1.0)
    a = ap.parse_args()

    dec, bol = load(a.user, a.days, 0)
    t0, grid, iob, dose, d_s, d_u = build_grid(dec, bol)
    print(f"\nUser {a.user}, last {a.days} days: {len(dec):,} decision samples, {len(bol):,} boluses\n")
    print("1/2. DERIVATION AND DUPLICATION")
    check_derivation(a.user)
    print("\n3. COMPLETENESS AND SCALE")
    check_steps(grid, iob, dose, min_dose=a.min_dose)
    print("\n4. ALIGNMENT")
    check_alignment(t0, grid, iob, dose, d_s, d_u)
    print()


if __name__ == "__main__":
    main()
