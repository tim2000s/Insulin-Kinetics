#!/usr/bin/env python3
"""Identify which delivery system produced a user's records, from the records themselves.

Platform is never stated in the extract, but three independent signals separate the two families
present in this cohort, and they agree unanimously:

  bolus typing      AAPS uploads treatments as "Correction Bolus"/"Meal Bolus" carrying a `type`
                    field of SMB or NORMAL. Trio uploads eventType "SMB"/"Bolus" with no type at
                    all — which is why any filter on bolus_type silently discards every dose a
                    Trio user ever delivered.
  bolus IOB         Trio uploads `bolusiob` in the devicestatus payload; AAPS does not, so for
                    AAPS users it must be derived as (iob - basaliob).
  step counts       the AAPS uploader carries step data; the Trio one does not, so exercise is
                    uncontrolled for Trio users.

The classification matters beyond bookkeeping. The bin in which logged IOB steps at a bolus differs
between the two — same bin for Trio, the following one for AAPS, because AAPS computes IOB at the
start of its cycle and counts a bolus delivered afterwards next time round.

Usage:
  python3 platform_detect.py            # classify every user in boost_decisions
"""
from __future__ import annotations

import psycopg2

from gate1_recover_known_curve import DSN

QUERY = """
SELECT d.user_id,
       count(*) FILTER (WHERE t.event_type IN ('SMB','Bolus'))                  AS trio_style,
       count(*) FILTER (WHERE t.event_type IN ('Correction Bolus','Meal Bolus')) AS aaps_style,
       max(CASE WHEN t.bolus_type IS NOT NULL THEN 1 ELSE 0 END)                AS typed
FROM boost_treatments t
JOIN (SELECT DISTINCT user_id FROM boost_decisions) d USING (user_id)
WHERE t.insulin > 0
GROUP BY 1
"""

SIGNALS = """
SELECT user_id, count(iob_bolusiob) > 0 AS uploads_bolusiob, count(steps_60m) > 0 AS has_steps
FROM boost_decisions GROUP BY 1
"""


def detect(conn=None):
    """Return {user_id: ('AAPS'|'Trio'|'unknown', agreement_note)}."""
    own = conn is None
    if own:
        conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    cur.execute(QUERY)
    style = {r[0]: (r[1], r[2], r[3]) for r in cur.fetchall()}
    cur.execute(SIGNALS)
    sig = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    if own:
        conn.close()

    out = {}
    for u in sorted(set(style) | set(sig)):
        trio_n, aaps_n, typed = style.get(u, (0, 0, 0))
        uploads, steps = sig.get(u, (False, False))
        votes = [
            "Trio" if trio_n > aaps_n else ("AAPS" if aaps_n > 0 else None),
            "Trio" if uploads else "AAPS",
            "Trio" if not steps else "AAPS",
        ]
        votes = [v for v in votes if v]
        if not votes:
            out[u] = ("unknown", "no signal")
            continue
        winner = max(set(votes), key=votes.count)
        agree = votes.count(winner) == len(votes)
        out[u] = (winner, "unanimous" if agree else f"{votes.count(winner)}/{len(votes)} signals")
    return out


if __name__ == "__main__":
    res = detect()
    n_agree = sum(1 for _, (_, a) in res.items() if a == "unanimous")
    print(f"{'user':<6}{'platform':<10}agreement")
    for u, (p, a) in res.items():
        print(f"{u:<6}{p:<10}{a}")
    print(f"\n{n_agree}/{len(res)} classified unanimously across all three signals.")
    for p in ("AAPS", "Trio", "unknown"):
        members = [u for u, (q, _) in res.items() if q == p]
        if members:
            print(f"  {p}: {len(members)} — {', '.join(members)}")
