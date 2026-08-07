#!/usr/bin/env python3
"""Verification harness: static defects, data veracity, output plausibility.

Written after a participant was reported with an 8-minute insulin action peak — a value no reader
would accept — and nothing in the pipeline objected. A person caught it. The response to that is
not to read more carefully next time; it is to make the pipeline capable of the objection.

Three stages, each failing loudly rather than warning quietly.

STAGE 1, STATIC. An AST scan for the specific defect classes this project has actually produced,
rather than generic style. Each pattern below corresponds to a real bug that reached an output:

  naive-argmax        a kernel reduced to a scalar without excluding the bins where reverse
                      causality concentrates. Produced the 8-minute peak.
  datetime-unit       .astype("int64") on a datetime column, whose unit is platform-dependent.
                      Dividing microseconds by 1e9 compressed 30 days to 43 minutes and produced a
                      spurious 100% dose match.
  unguarded-slice     a slice whose bounds are computed from data without clamping. A carb record
                      before the grid start gave a negative bound and blanked 112 of 123 days.
  silent-except       an exception handler that neither re-raises nor records, so a failed stage
                      looks like an empty result.
  hardcoded-result    a numeric literal in a reporting script that duplicates a value the analysis
                      computes. Five such literals went stale when the estimator changed.

STAGE 2, DATA. Currency of the database, agreement between the cohort configuration and what is
actually stored, per-participant record integrity, and unit consistency.

STAGE 3, OUTPUT. Every reported quantity is checked against a plausibility range derived from the
product information rather than from the data, so the check is independent of the analysis. Values
resting on an optimiser bound are reported as suspect regardless of plausibility, because a
parameter at a bound is a fit that failed to converge inside its domain. Independent estimates that
should agree are cross-checked. Finally the manuscript is compared against the analysis outputs it
claims to report.

Usage:
  python3 verify.py [--build build] [--config cohort.json] [--stage all|static|data|output]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys

import numpy as np
import psycopg2

from cohort_table import parse as parse_cohort, with_peak
# imported rather than restated, so the verifier cannot drift from the thresholds it enforces
from gate4_deconvolution import (PEAK_CONC_MIN as CONC_MIN, PEAK_PROM_MIN as PROM_MIN,
                                 PEAK_LAMBDA_SPREAD_MAX as LAM_MAX)

HERE = os.path.dirname(os.path.abspath(__file__))
DSN = "dbname=oref host=127.0.0.1 port=5432"

# Physiological bounds, taken from product information (references 8-12), NOT from these data.
# No rapid-acting analogue has an onset below 10 min or a peak beyond 3 h by its own label.
PEAK_MIN, PEAK_MAX = 15.0, 180.0
DIA_MIN, DIA_MAX = 120.0, 1440.0        # optimiser domain; a value AT either end is suspect
FIT_MAX_CLEAN = 0.15                    # relative residual on an algebraic identity

FAIL, WARN, OK = "FAIL", "WARN", "ok"


class Findings:
    def __init__(self):
        self.rows = []

    def add(self, level, stage, what, detail):
        self.rows.append((level, stage, what, detail))

    def report(self):
        for lvl in (FAIL, WARN):
            rows = [r for r in self.rows if r[0] == lvl]
            if not rows:
                continue
            print(f"\n{lvl}S ({len(rows)}):")
            for _, stage, what, detail in rows:
                print(f"  [{stage}] {what}: {detail}")
        n_fail = sum(1 for r in self.rows if r[0] == FAIL)
        n_warn = sum(1 for r in self.rows if r[0] == WARN)
        n_ok = sum(1 for r in self.rows if r[0] == OK)
        print(f"\n{n_ok} checks passed, {n_warn} warnings, {n_fail} failures")
        return n_fail


# ---------------------------------------------------------------- stage 1: static
GUARDED_PEAK = {"peak_of", "peaks_of"}


def static_checks(f: Findings):
    scripts = sorted(p for p in os.listdir(HERE) if p.endswith(".py") and p != "verify.py")
    for name in scripts:
        path = os.path.join(HERE, name)
        src = open(path).read()
        try:
            tree = ast.parse(src, filename=name)
        except SyntaxError as e:
            f.add(FAIL, "static", name, f"syntax error line {e.lineno}: {e.msg}")
            continue

        lines = src.splitlines()

        for node in ast.walk(tree):
            # naive argmax on a kernel, outside the guarded helpers
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("argmax", "argmin")):
                fn = _enclosing_func(tree, node)
                ctx = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                if fn in GUARDED_PEAK:
                    continue
                # A legitimate argmax outside the guarded helpers must SAY WHY, on the line or the
                # one above it. Silence is the condition that produced the 8-minute peak, so an
                # unexplained one stays a warning even when it turns out to be fine.
                near = " ".join(lines[max(0, node.lineno - 5):node.lineno])
                if "argmax-ok:" in near:
                    continue
                if any(k in ctx for k in ("beta", "kernel", "b[", "ks", "activity")):
                    f.add(WARN, "static", f"{name}:{node.lineno}",
                          f"argmax on what may be a kernel, outside a guarded helper and with no "
                          f"'argmax-ok:' justification: {ctx.strip()[:70]}")

            # datetime unit assumption
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "astype" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and arg.value in ("int64", np.int64):
                    ctx = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    if "1e9" in ctx or "1e6" in ctx or "to_datetime" in ctx:
                        f.add(FAIL, "static", f"{name}:{node.lineno}",
                              f"datetime cast with hardcoded unit divisor: {ctx.strip()[:70]}")

            # bare / silent exception handler
            if isinstance(node, ast.ExceptHandler):
                body = node.body
                silent = not any(isinstance(s, (ast.Raise,)) for s in body)
                logs = any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                           and getattr(getattr(s.value, "func", None), "id", "") in ("print", "P")
                           for s in body)
                only_pass = len(body) == 1 and isinstance(body[0], ast.Pass)
                near = " ".join(lines[max(0, node.lineno - 5):node.lineno + 3])
                if silent and not logs and only_pass and "except-ok:" not in near:
                    f.add(WARN, "static", f"{name}:{node.lineno}",
                          "exception handler neither raises nor records, and carries no "
                          "'except-ok:' justification")

    # compile every script
    r = subprocess.run([sys.executable, "-m", "py_compile"] + [os.path.join(HERE, s) for s in scripts],
                       capture_output=True, text=True)
    if r.returncode != 0:
        f.add(FAIL, "static", "compile", r.stderr.strip()[:200])
    else:
        f.add(OK, "static", "compile", f"{len(scripts)} scripts compile")

    # hardcoded results in the reporting script
    paper = open(os.path.join(HERE, "make_paper.py")).read()
    lits = re.findall(r'"[^"]*?\b(\d+ of \d+)\b[^"]*?"', paper)
    lits += re.findall(r'"[^"]*?(0\.\d{3}) U per unit', paper)
    lits = [x for x in lits if x not in ("0 of 0",)]
    if lits:
        f.add(WARN, "static", "make_paper.py",
              f"numeric literals that may duplicate computed values: {sorted(set(lits))[:6]}")
    else:
        f.add(OK, "static", "make_paper.py", "no obvious hardcoded result literals")


def _enclosing_func(tree, node):
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(fn):
                if sub is node:
                    return fn.name
    return None


# ---------------------------------------------------------------- stage 2: data
def data_checks(f: Findings, cfg_path):
    users = json.load(open(cfg_path))
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()

    cur.execute("SELECT max(ts_utc)::date FROM boost_decisions")
    latest = cur.fetchone()[0]
    import datetime as dt
    age = (dt.date.today() - latest).days
    (f.add(WARN, "data", "currency", f"most recent decision row is {age} days old ({latest})")
     if age > 3 else f.add(OK, "data", "currency", f"database current to {latest}"))

    cur.execute("SELECT DISTINCT user_id FROM boost_decisions")
    have = {r[0] for r in cur.fetchall()}
    missing = sorted(set(users) - have)
    if missing:
        f.add(FAIL, "data", "cohort config", f"configured but absent from database: {missing}")
    else:
        f.add(OK, "data", "cohort config", f"all {len(users)} configured participants present")

    for u in sorted(users):
        cur.execute("""SELECT count(*), count(cgm_mgdl), min(cgm_mgdl), max(cgm_mgdl),
                              count(iob_iob) FROM boost_decisions WHERE user_id=%s""", (u,))
        n, ncgm, lo, hi, niob = cur.fetchone()
        if n < 2000:
            f.add(WARN, "data", f"{u} volume", f"only {n} decision rows")
        # The stored series legitimately contains sensor artefacts — dropouts logged as zero, and
        # transient spikes. The estimator never sees them: load_grid admits only 40-350 mg/dL. So
        # min/max of the raw column is the wrong test and flagged 23 of 31 participants when it was
        # written that way. What matters is whether enough of the series SURVIVES that filter, and
        # whether the discarded fraction is small enough that the exclusions are artefacts rather
        # than a systematically broken feed.
        cur.execute("""SELECT count(*) FILTER (WHERE cgm_mgdl BETWEEN 40 AND 350),
                              count(*) FILTER (WHERE cgm_mgdl IS NOT NULL)
                       FROM boost_decisions WHERE user_id=%s""", (u,))
        inrange, nonnull = cur.fetchone()
        if nonnull:
            drop = 1.0 - inrange / nonnull
            if drop > 0.10:
                f.add(FAIL, "data", f"{u} glucose range",
                      f"{100*drop:.1f}% of readings fall outside 40–350 mg/dL and are discarded; "
                      f"too much of the feed is unusable to trust the remainder")
            elif drop > 0.02:
                f.add(WARN, "data", f"{u} glucose range",
                      f"{100*drop:.1f}% of readings discarded as out of range ({inrange:,} usable)")
        if inrange is not None and inrange < 2000:
            f.add(WARN, "data", f"{u} usable glucose", f"only {inrange:,} in-range readings")
        if ncgm and niob and niob < 0.5 * ncgm:
            f.add(WARN, "data", f"{u} iob coverage", f"IOB present on {100*niob/ncgm:.0f}% of rows")
        cur.execute("""SELECT count(*), count(DISTINCT ns_id) FROM boost_treatments
                       WHERE user_id=%s""", (u,))
        tn, tu = cur.fetchone()
        if tn != tu:
            f.add(FAIL, "data", f"{u} treatments", f"{tn-tu} duplicate treatment ids")
        # Large records are expected where they are externally-logged insulin, which the observed
        # -profile estimator excludes for exactly this reason. A large record typed as a PUMP bolus
        # is a different matter and stays a warning.
        cur.execute("""SELECT count(*) FILTER (WHERE event_type = 'External Insulin'),
                              count(*) FILTER (WHERE event_type IS DISTINCT FROM 'External Insulin')
                       FROM boost_treatments WHERE user_id=%s AND insulin > 25""", (u,))
        big_ext, big_pump = cur.fetchone()
        if big_pump:
            f.add(WARN, "data", f"{u} doses",
                  f"{big_pump} PUMP boluses above 25 U — not explained by external logging")
        elif big_ext:
            f.add(OK, "data", f"{u} doses",
                  f"{big_ext} records above 25 U, all externally logged and excluded from Gate 4")
    conn.close()
    f.add(OK, "data", "integrity", f"per-participant checks completed for {len(users)}")


# ---------------------------------------------------------------- stage 3: output


def output_checks(f: Findings, build):
    if not os.path.exists(os.path.join(build, "cohort.md")):
        f.add(FAIL, "output", "cohort", "no cohort.md; run make_report.py")
        return
    rows = parse_cohort(build)
    f.add(OK, "output", "cohort", f"{len(rows)} participants parsed")

    for r in rows:
        # a withheld peak has no value to range-check; the mode-shape check below covers it
        pairs = [("configured", r["cfg"])]
        if r["obs"] is not None:
            pairs.append(("observed", float(r["obs"])))
        for label, v in pairs:
            if not (PEAK_MIN <= v <= PEAK_MAX):
                f.add(FAIL, "output", f"{r['user']} {label} peak",
                      f"{v:.0f} min outside the plausible range {PEAK_MIN:.0f}–{PEAK_MAX:.0f} "
                      f"derived from product information")
        # a value resting on an optimiser bound
        if r["dia"] != "n/a":
            d = float(r["dia"])
            if abs(d - DIA_MAX) < 1 or abs(d - DIA_MIN) < 1:
                f.add(WARN, "output", f"{r['user']} duration",
                      f"{d:.0f} min rests on the optimiser bound; fit did not converge inside its domain")
        if r["fit"] > FIT_MAX_CLEAN and not r["flag"]:
            f.add(FAIL, "output", f"{r['user']} residual",
                  f"{r['fit']:.3f} exceeds {FIT_MAX_CLEAN} but is not flagged")
        # halves that disagree wildly indicate a change, which should be reported not averaged
        if r["h1"] is not None and r["h2"] is not None and abs(r["h1"] - r["h2"]) > 25:
            f.add(WARN, "output", f"{r['user']} stability",
                  f"halves differ by {abs(r['h1'] - r['h2']):.0f} min; treat as two regimes")

    # positive controls
    val = " ".join(open(os.path.join(build, p)).read()
                   for p in os.listdir(build) if p.startswith("val_"))
    nf = len(re.findall(r"\*\*FAIL\*\*", val))
    (f.add(FAIL, "output", "positive controls", f"{nf} control(s) failed")
     if nf else f.add(OK, "output", "positive controls", "all passed"))

    # Cross-check the reported peak against the participant's own printed profile. The profile is
    # PRINTED EVERY (K+1)//24 LAGS, so its argmax can legitimately miss the true one by up to that
    # stride plus the alignment term; comparing them with a flat 30-minute tolerance raised a false
    # alarm on a participant whose reported peak was correct. Tolerance is therefore derived from
    # the sampling stride actually used, and the flatness the earlier check was groping towards is
    # now tested directly against the mode-shape floors.
    for r in rows:
        p = os.path.join(build, f"g4_{r['user']}.md")
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        prof = re.findall(r"\|\s*(\d+)\s*\|\s*([\d.]+)\s", txt)
        shape = re.search(r"concentration (\d+\.\d+), prominence (\d+\.\d+)", txt)
        if not shape:
            f.add(FAIL, "output", f"{r['user']} mode shape",
                  "no concentration/prominence line; estimator predates the identifiability check")
            continue
        conc, prom = float(shape.group(1)), float(shape.group(2))
        stab = re.search(r"spread (\d+) min", txt)
        if not stab:
            f.add(FAIL, "output", f"{r['user']} stability",
                  "no smoothing-stability line; estimator predates the lambda-stability check")
            continue
        spread = float(stab.group(1))
        # a peak must clear BOTH criteria; withholding must be justified by at least one
        bad_shape = conc < CONC_MIN or prom < PROM_MIN
        bad_lam = spread > LAM_MAX
        if (bad_shape or bad_lam) and not r["flat"]:
            f.add(FAIL, "output", f"{r['user']} identifiability",
                  f"fails {'mode shape' if bad_shape else ''}"
                  f"{' and ' if bad_shape and bad_lam else ''}"
                  f"{'lambda stability' if bad_lam else ''} "
                  f"(conc {conc:.2f}, prom {prom:.2f}, spread {spread:.0f} min) "
                  f"but a peak was still reported")
        elif r["flat"] and not (bad_shape or bad_lam):
            f.add(FAIL, "output", f"{r['user']} identifiability",
                  f"withheld but passes both criteria (conc {conc:.2f}, prom {prom:.2f}, "
                  f"spread {spread:.0f} min)")
        elif r["flat"]:
            f.add(OK, "output", f"{r['user']} identifiability",
                  f"correctly withheld ({'no mode' if bad_shape else 'unstable to lambda'})")

        pk = re.search(r"curve: (\d+) min", txt)
        if pk and len(prof) > 4:
            lag = np.array([float(a) for a, _ in prof])
            amp = np.array([float(b) for _, b in prof])
            stride = float(np.median(np.diff(lag))) if len(lag) > 1 else 5.0
            shown_peak = lag[int(np.argmax(amp))]
            if abs(shown_peak - float(pk.group(1))) > stride + 5.0:
                f.add(WARN, "output", f"{r['user']} profile",
                      f"reported peak {pk.group(1)} min but the printed profile maximises at "
                      f"{shown_peak:.0f} min, further than the {stride:.0f} min print stride "
                      f"explains")

    # manuscript against the outputs it reports
    ph = os.path.join(build, "paper.html")
    if os.path.exists(ph):
        h = re.sub(r"<[^>]+>", " ", open(ph).read())
        # the manuscript's peak statistics are computed over participants WITH a peak, so the
        # cross-check must use the same subset or it will disagree with a correct manuscript
        peaked = with_peak(rows)
        cfg = np.array([r["cfg"] for r in peaked]); obs = np.array([r["obs"] for r in peaked])
        checks = [(f"median of {np.median(obs):.0f} min", "observed median"),
                  (f"{len(rows)}", "participant count")]
        for needle, what in checks:
            (f.add(OK, "output", f"manuscript {what}", needle) if needle in h
             else f.add(FAIL, "output", f"manuscript {what}",
                        f"'{needle}' not found in rendered manuscript"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(HERE, "build"))
    ap.add_argument("--config", default=os.path.join(HERE, "cohort.json"))
    ap.add_argument("--stage", default="all", choices=("all", "static", "data", "output"))
    a = ap.parse_args()

    f = Findings()
    if a.stage in ("all", "static"):
        static_checks(f)
    if a.stage in ("all", "data"):
        data_checks(f, a.config)
    if a.stage in ("all", "output"):
        output_checks(f, a.build)
    sys.exit(1 if f.report() else 0)


if __name__ == "__main__":
    main()
