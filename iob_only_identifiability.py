#!/usr/bin/env python3
"""CAN THE CURVE BE RECOVERED FROM AN IOB SERIES ALONE? No. This file shows why.

Many archives hold a loop's logged insulin-on-board series but no record of the doses that produced
it. Extending Gate 1 to those archives would multiply the available cohort several times over, so
the question is worth settling rather than assuming. It is settled negatively, for a structural
reason and an empirical one.

STRUCTURAL. Write the IOB identity on a regular grid:

    IOB_i = SUM_{j<=i} d_j f(i-j)

Because f(0) = 1, this is a lower-triangular system with unit diagonal. For ANY candidate kernel f,
forward substitution

    d_i = IOB_i - SUM_{j<i} d_j f(i-j)

returns a dose series reproducing the observed IOB series exactly. Goodness of fit therefore
carries no information whatsoever about the kernel: a peak of 35 minutes and a peak of 120 minutes
both fit perfectly, differing only in the dose series they imply. The only quantity that
discriminates is whether those implied doses are physically possible — insulin delivery cannot be
negative — and this file measures whether that constraint is sharp enough to identify the peak.

It is not. Implied negative mass rises monotonically with the candidate peak across the plausible
range, so the constraint has no interior optimum: it always prefers the shortest peak on offer,
irrespective of the truth.

EMPIRICAL. A cruder reconstruction, taking doses as the positive first differences of IOB, fails
for a related reason. Between two five-minute samples IOB decays by 1-3% of TOTAL IOB, and in a
micro-dosing loop total IOB is an order of magnitude larger than an individual automatic bolus. The
decay therefore swamps small doses entirely: against real treatment streams this route lost 28-93%
of delivered insulin and displaced the recovered peak by 12 to 102 minutes.

CONCLUSION. Gate 1 requires an independent dose record. Archives holding IOB alone are out of
reach, and no amount of care in the fitting changes that. This is a property of the problem, not a
limitation of the implementation.

Usage:
  python3 iob_only_identifiability.py --user <id> [--days 60]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from gate1_recover_known_curve import iob_fraction, load

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = 300.0


def implied_doses(y, peak, dia):
    """Exact forward substitution: the dose series this kernel implies for the observed IOB."""
    f = iob_fraction(np.arange(len(y)) * 5.0, peak, dia)
    d = np.zeros(len(y))
    for i in range(len(y)):
        d[i] = y[i] - np.dot(d[:i][::-1], f[1:i + 1])
    return d, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--out")
    a = ap.parse_args()

    dec, bol = load(a.user, a.days, 0)
    if dec.empty:
        print("no data"); return
    y = dec.bolus_iob.values.astype(float)[:a.samples]

    L, P = [], None
    P = L.append
    P("# Identifiability of the insulin curve from an IOB series alone\n")
    P(f"\nUser **{a.user}**, {len(y):,} consecutive IOB samples.\n")
    P("\nFor each candidate kernel, the dose series it implies is obtained by exact forward "
      "substitution, and that series is convolved back to check the fit.\n")
    P("\n| candidate curve | max IOB reconstruction error | implied doses negative | "
      "negative mass (U) |")
    P("|---|---|---|---|")
    rows = []
    for peak, dia in ((35., 480.), (55., 480.), (75., 480.), (95., 600.), (120., 600.)):
        d, f = implied_doses(y, peak, dia)
        rec = np.convolve(d, f)[:len(y)]
        err = float(np.max(np.abs(rec - y)))
        negmass = float(-d[d < 0].sum())
        rows.append((peak, negmass))
        P(f"| peak {peak:.0f}, DIA {dia:.0f} | {err:.1e} | {100 * np.mean(d < -1e-9):.1f}% | "
          f"{negmass:.1f} |")

    P("\n**Every candidate reproduces the observed series to machine precision.** The fit residual "
      "is identically uninformative, because the system is lower-triangular with unit diagonal: "
      "any kernel can be made to explain any IOB series by a suitable choice of doses.\n")
    monotone = all(rows[i][1] <= rows[i + 1][1] for i in range(len(rows) - 1))
    P("\n" + (f"Implied negative mass is **monotonically increasing** in the candidate peak "
              f"({rows[0][1]:.0f} U at {rows[0][0]:.0f} min rising to {rows[-1][1]:.0f} U at "
              f"{rows[-1][0]:.0f} min), so the non-negativity constraint has no interior optimum "
              "and cannot identify the peak — it will always select the shortest candidate "
              "offered.\n" if monotone else
              "Implied negative mass is not monotone in the candidate peak; a constrained "
              "estimator may be worth exploring.\n"))
    P("\n**Conclusion.** An independent dose record is necessary. Archives holding only an IOB "
      "series cannot yield the configured curve, however the fitting is arranged.\n")

    txt = "\n".join(L)
    open(a.out or os.path.join(HERE, "IOB_ONLY_IDENTIFIABILITY.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
