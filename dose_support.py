#!/usr/bin/env python3
"""Which doses actually identify the peak?

A structural consequence of the eligibility rule that is not obvious from reading it. Target rows
are blanked for three hours FOLLOWING any carb entry, so a dose accompanied by announced carbs can
only contribute to rows more than three hours later — that is, to lags beyond 180 min. Every lag
below 180 min, which is where the peak sits for every participant, is therefore identified
exclusively by insulin delivered WITHOUT announced carbs: corrections and automatic microboluses.

That is not automatically a bias in time. A correction bolus should follow the same kinetics as a
meal bolus. But it bounds what the analysis can speak to:

  - the dose range supporting the peak is the correction range, not the full delivered range, which
    limits how far a "no dose dependence" result can be extended;
  - corrections are given because glucose is high or rising, so the doses identifying the peak are
    exactly the ones subject to the reverse-causality problem;
  - if large and small doses genuinely differ in kinetics, this design would not see it at the peak.

This quantifies the restriction per participant: how much dose mass supports lags at or below the
peak region, how it compares with the full delivered distribution, and what dose sizes are actually
represented there.

Usage:
  python3 dose_support.py --config cohort.json [--lag-max 180] [--out DOSE_SUPPORT.md]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from gate4_deconvolution import load_grid

HERE = os.path.dirname(os.path.abspath(__file__))


def support(user, tz, lag_max_min=180.0):
    """Dose mass reaching eligible target rows at lags <= lag_max, against all delivered dose."""
    grid, bg, dose, ok, clock, day, *_ = load_grid(user, tz)
    K = int(lag_max_min // 5)
    rows = np.flatnonzero(ok[:-1])
    rows = rows[rows >= K]
    if len(rows) == 0:
        return None
    # For each dose bin, how many eligible target rows does it reach within the peak region?
    reach = np.zeros(len(dose))
    for k in range(K + 1):
        np.add.at(reach, rows - k, 1)
    delivered = dose > 0
    supported = delivered & (reach > 0)
    sizes_all = dose[delivered]
    sizes_sup = dose[supported]
    # weight each dose by how much of the peak region it actually informs
    w = reach[supported]
    return dict(
        n_all=int(delivered.sum()), n_sup=int(supported.sum()),
        u_all=float(sizes_all.sum()), u_sup=float(sizes_sup.sum()),
        med_all=float(np.median(sizes_all)) if len(sizes_all) else float("nan"),
        med_sup=float(np.median(sizes_sup)) if len(sizes_sup) else float("nan"),
        p90_all=float(np.percentile(sizes_all, 90)) if len(sizes_all) else float("nan"),
        p90_sup=float(np.percentile(sizes_sup, 90)) if len(sizes_sup) else float("nan"),
        max_all=float(sizes_all.max()) if len(sizes_all) else float("nan"),
        max_sup=float(sizes_sup.max()) if len(sizes_sup) else float("nan"),
        wmed=float(np.median(np.repeat(sizes_sup, np.minimum(w, 60).astype(int))))
        if len(sizes_sup) else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "cohort.json"))
    ap.add_argument("--lag-max", type=float, default=180.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    users = json.load(open(a.config))

    L, P = [], None
    P = L.append
    P("# Which doses identify the peak?\n")
    P(f"\nTarget rows are blanked for three hours after any carb entry, so a dose given with "
      f"announced carbs cannot inform any lag below 180 min. Lags up to {a.lag_max:.0f} min — the "
      f"region containing every recovered peak — are therefore supported only by insulin delivered "
      f"without announced carbs. This table shows what that leaves.\n")
    P("\n| user | doses (all) | doses supporting peak | % of dose count | median U (all) | "
      "median U (supporting) | 90th pct (all) | 90th pct (supporting) | max U (supporting) |")
    P("|---|---|---|---|---|---|---|---|---|")
    ratios = []
    for u, tz in sorted(users.items()):
        try:
            s = support(u, tz, a.lag_max)
        except Exception as e:                                   # noqa: BLE001
            P(f"| {u} | error: {type(e).__name__} | | | | | | | |")
            continue
        if s is None:
            P(f"| {u} | no eligible rows | | | | | | | |")
            continue
        ratios.append(s["p90_sup"] / s["p90_all"] if s["p90_all"] else np.nan)
        P(f"| {u} | {s['n_all']:,} | {s['n_sup']:,} | {100 * s['n_sup'] / s['n_all']:.0f}% | "
          f"{s['med_all']:.2f} | {s['med_sup']:.2f} | {s['p90_all']:.2f} | {s['p90_sup']:.2f} | "
          f"{s['max_sup']:.1f} |")
    r = np.array([x for x in ratios if np.isfinite(x)])
    if len(r):
        P(f"\nAcross participants the 90th-percentile dose supporting the peak region is a median "
          f"of {100 * np.median(r):.0f}% of the 90th-percentile dose overall. Where that ratio is "
          f"well below 100% the peak is identified by systematically smaller doses than the "
          f"participant typically receives, and any statement about dose dependence at the peak is "
          f"restricted to the range shown in the 'supporting' columns.\n")
    open(a.out or os.path.join(HERE, "DOSE_SUPPORT.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
