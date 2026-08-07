#!/usr/bin/env python3
"""Single parser for the cohort table.

Four modules used to carry their own copy of this regex and they had drifted: different group
counts, different field meanings for the same column, and an `int()` on a column that can legally
hold a non-numeric marker. Downstream code now imports from here so a change to the table format
breaks in one place instead of silently mis-reading in three.

Column contract, as emitted by run_cohort.py:

    | user | configured | DIA | fit | half 1 | half 2 | observed | gap |

`DIA` is a number or `n/a` (leverage test found the duration unidentifiable). `fit` may carry a
trailing `!` (relative residual above tolerance). `observed` is a number, `n/i` (no identifiable
peak — either no mode, or not stable to the smoothing weight) or `-` (no fit); `gap` likewise. Numeric-or-marker cells parse to None, never to
a fabricated number, and callers filter explicitly.
"""
from __future__ import annotations

import os
import re

ROW = re.compile(
    r"\|\s*(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\w/]+)\s*\|\s*([\d.]+)(!?)\s*\|"
    r"\s*([\w./+-]+)\s*\|\s*([\w./+-]+)\s*\|\s*([\w./+-]+)\s*\|\s*([\w./+-]+)\s*\|")


def _num(s):
    """A cell that may hold a marker instead of a number."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse(build_or_path):
    """Rows of the cohort table. Accepts a build directory or a direct path."""
    p = build_or_path
    if os.path.isdir(p):
        p = os.path.join(p, "cohort.md")
    if not os.path.exists(p):
        return []
    rows = []
    for ln in open(p):
        m = ROW.match(ln)
        if not m:
            continue
        rows.append(dict(
            user=m.group(1),
            cfg=float(m.group(2)),
            dia=m.group(3),
            dia_val=_num(m.group(3)),
            fit=float(m.group(4)),
            flag=m.group(5) == "!",
            h1=_num(m.group(6)),
            h2=_num(m.group(7)),
            obs=_num(m.group(8)),
            flat=m.group(8).strip() in ("n/i", "flat"),
            gap=_num(m.group(9)),
        ))
    return rows


def with_peak(rows):
    """Rows carrying a usable observed peak. Everything summarising peaks goes through this."""
    return [r for r in rows if r["obs"] is not None]
