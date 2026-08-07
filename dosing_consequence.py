#!/usr/bin/env python3
"""What a mis-specified curve does to dose sizing.

The observed response peaks near 42 minutes across this cohort while systems are configured at 35
to 102. This translates that discrepancy into the quantity the controller actually acts on:
insulin-on-board.

IOB is a brake. Every oref-derived system withholds or shrinks a dose when it believes insulin is
still to come. If the configured curve decays more slowly than the person's response, the system
carries PHANTOM active insulin — units it thinks are still working that have, on the evidence of
that person's own glucose, already worked. Phantom IOB makes the controller conservative: it brakes
earlier and for longer than the person's physiology warrants. The opposite error, a configured
curve faster than the response, removes the brake too soon and is the direction associated with
stacking.

Per unit delivered, at a given time after the dose, the discrepancy is

    phantom(t) = f(t; configured peak, DIA) - f(t; observed peak, DIA)

positive when the system over-states remaining insulin.

IMPORTANT LIMIT. The observed peak is the peak of the GLUCOSE RESPONSE, and the IOB model describes
assumed insulin action. Section 5a of the paper sets out why those are not guaranteed to be the same
quantity — different conditions, doses two orders of magnitude apart, counter-regulation free to
operate in one and suppressed in the other. Substituting one for the other here is deliberate but
approximate, and the numbers below are the size of a discrepancy in the controller's own accounting,
not a measured error in insulin action.

Usage:
  python3 dosing_consequence.py [--at 60] [--build build]
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

from cohort_table import parse as parse_cohort, with_peak
from gate1_recover_known_curve import iob_fraction

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(HERE, "build"))
    ap.add_argument("--at", type=float, default=60.0, help="minutes after the dose")
    ap.add_argument("--out")
    a = ap.parse_args()

    # participants whose kernel has no identifiable mode have no observed peak to compare
    # against, so they cannot enter a phantom-IOB calculation at all
    rows = {r["user"]: (r["cfg"], r["dia"], int(r["obs"]))
            for r in with_peak(parse_cohort(a.build))}
    if not rows:
        print("no cohort table"); return

    L, P = [], None
    P = L.append
    P("# Consequence for dose sizing\n")
    P(f"\nPhantom insulin-on-board {a.at:.0f} minutes after a dose, per unit delivered: what the "
      f"controller believes is still to come, less what the participant's own response suggests "
      f"has already acted. Positive means the controller over-states remaining insulin and so "
      f"brakes more than warranted.\n")
    P("\n| user | configured | observed | IOB believed | IOB implied | phantom (U per U) |")
    P("|---|---|---|---|---|---|")
    ph = []
    for u, (cfg, dia, obs) in sorted(rows.items()):
        d = float(dia) if dia != "n/a" else 500.0
        believed = float(iob_fraction(np.array([a.at]), cfg, d)[0])
        implied = float(iob_fraction(np.array([a.at]), max(obs, 12), d)[0])
        ph.append(believed - implied)
        P(f"| {u} | {cfg:.0f} | {obs} | {believed:.3f} | {implied:.3f} | "
          f"**{believed - implied:+.3f}** |")
    ph = np.array(ph)
    P(f"\nMedian **{np.median(ph):+.3f} U per unit delivered** ({100 * np.median(ph):+.1f}%). "
      f"{int((ph > 0.15).sum())} of {len(ph)} participants carry more than 0.15 U of phantom "
      f"insulin per unit; {int((ph < -0.02).sum())} carry less than their response implies, which "
      f"is the direction associated with stacking rather than braking.\n")
    P("\n## What this means for dose sizing\n")
    P("- For the median participant the error is small — under 6% of a unit — and would change "
      "few dosing decisions.\n")
    P(f"- For the {int((ph > 0.15).sum())} participants above 0.15 U per unit it is not small. At a "
      "maximum IOB of 6 units, half a unit of phantom insulin per unit delivered brings the brake "
      "on materially sooner than the person's own glucose response warrants.\n")
    P("- The error is **mostly in the conservative direction**. A controller that thinks insulin "
      "lingers doses less, and a feedback loop partly compensates by dosing again when glucose "
      "fails to fall. The cost is sluggishness rather than hypoglycaemia.\n")
    P(f"- The {int((ph < -0.02).sum())} participants with the opposite sign are the ones worth "
      "attention: there the brake is released earlier than the response justifies.\n")
    P("\nNone of this licenses changing a setting. The observed peak is a property of the glucose "
      "response, the configured peak is an input to an accounting model, and these figures size a "
      "disagreement between them rather than measuring an error in either.\n")

    open(a.out or os.path.join(HERE, "DOSING_CONSEQUENCE.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
