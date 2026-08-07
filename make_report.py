#!/usr/bin/env python3
"""MAKE REPORT — run the whole suite and assemble one PDF.

Runs every analysis in the repo against a cohort, collects the Markdown each one writes, and
renders a single PDF. Entirely local: markdown -> HTML -> weasyprint. No network, no service.

The ordering is the order the results should be READ in, which is not the order they are cheapest
to compute:

  1 validation   the positive controls, first, because nothing below them means anything if they
                 fail. Gate 4 self-test (recovery of a known curve, including wrong generating
                 families) and the feedback simulation (does controller reaction bias the answer).
  2 integrity    per user, can the extract itself be faking the result — derivation, duplication,
                 dose completeness, timestamp alignment.
  3 cohort       the comparison table: what each loop BELIEVES vs what the insulin appears to DO.
  4 per user     Gate 1 with its controls, then Gate 4.

Anything that fails is recorded in the PDF as a failure rather than silently omitted — a missing
section in a report reads as "not applicable" when it usually means "crashed".

Usage:
  python3 make_report.py --config cohort.json [--jobs 4] [--skip-run] [--out report.pdf]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm;
        @bottom-center { content: counter(page) " / " counter(pages);
                         font: 8pt "Helvetica"; color: #888; } }
body { font: 10pt/1.45 "Helvetica Neue", Helvetica, Arial, sans-serif; color: #1a1a1a; }
h1 { font-size: 19pt; margin: 0 0 2mm 0; color: #111; }
h2 { font-size: 13pt; margin: 7mm 0 2mm 0; padding-bottom: 1mm;
     border-bottom: 1px solid #ccc; break-after: avoid; }
h3 { font-size: 11pt; margin: 5mm 0 1.5mm 0; color: #333; break-after: avoid; }
p { margin: 0 0 2.5mm 0; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 4mm 0;
        font-size: 8.5pt; break-inside: avoid; }
th { background: #f0f0f0; text-align: left; padding: 1.4mm 2mm; border-bottom: 1.5px solid #999;
     font-weight: 600; }
td { padding: 1.2mm 2mm; border-bottom: 0.5px solid #ddd; vertical-align: top; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.5pt; background: #f5f5f5;
       padding: 0.3mm 1mm; }
pre { background: #f7f7f7; border-left: 2px solid #bbb; padding: 2mm 3mm; font-size: 8pt;
      overflow-wrap: break-word; white-space: pre-wrap; break-inside: avoid; }
strong { color: #000; }
.cover { text-align: left; margin-bottom: 8mm; }
.cover .sub { color: #555; font-size: 10pt; margin-top: 2mm; }
.meta { color: #777; font-size: 8.5pt; margin-top: 4mm; }
.fail { color: #a00; font-weight: 600; }
.section-break { break-before: page; }
"""


def run(script, args, out_md, timeout=5400):
    """Run one analysis. Returns (markdown, ok)."""
    cmd = [sys.executable, os.path.join(HERE, script)] + args + ["--out", out_md]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"**Did not finish within {timeout}s.**\n", False
    if r.returncode != 0 or not os.path.exists(out_md):
        tail = "\n".join((r.stderr or r.stdout or "").strip().splitlines()[-12:])
        return f"**Failed (exit {r.returncode}).**\n\n```\n{tail}\n```\n", False
    return open(out_md).read(), True


def strip_h1(md):
    """Drop the leading '# ...' so the assembled document keeps one heading hierarchy."""
    lines = md.splitlines()
    out, dropped = [], False
    for ln in lines:
        if not dropped and ln.startswith("# "):
            dropped = True
            continue
        out.append("#" + ln if ln.startswith("#") else ln)   # demote everything one level
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "cohort.json"))
    ap.add_argument("--days", type=int, default=28, help="Gate 1 window")
    ap.add_argument("--boot", type=int, default=20, help="Gate 4 bootstrap draws")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-run", action="store_true", help="reuse build/ from a previous run")
    ap.add_argument("--out", default=os.path.join(HERE, "insulin-kinetics-report.pdf"))
    a = ap.parse_args()

    users = json.load(open(a.config))
    os.makedirs(BUILD, exist_ok=True)
    ref_user, ref_tz = next(iter(users.items()))

    # ---- what to run -------------------------------------------------------------------
    jobs = [
        ("validation", "Gate 4 self-test (recovery of a known curve)",
         "gate4_selftest.py", ["--user", ref_user, "--tz", ref_tz], "val_g4selftest.md"),
        ("validation", "Feedback simulation (does controller reaction bias the peak?)",
         "feedback_simulation.py", ["--days", "150", "--reps", "3"], "val_feedback.md"),
        ("validation", "Which doses identify the peak region?",
         "dose_support.py", ["--config", "cohort.json"], "val_dosesupport.md"),
        ("validation", "Insulin logged as delivered outside the pump",
         "external_insulin_sensitivity.py", ["--config", "cohort.json"], "val_external.md"),
    ]
    for u, tz in users.items():
        jobs.append(("integrity", f"User {u}", "extract_integrity.py",
                     ["--user", u, "--days", "120"], f"int_{u}.md"))
    jobs.append(("cohort", "Configured curve vs observed response",
                 "run_cohort.py", ["--config", a.config, "--days", str(a.days),
                                   "--boot", str(a.boot), "--workers", str(a.jobs)],
                 "cohort.md"))
    for u, tz in users.items():
        jobs.append(("peruser", f"User {u} — Gate 1 controls", "gate1_controls.py",
                     ["--user", u, "--days", "45"], f"g1c_{u}.md"))
        jobs.append(("peruser", f"User {u} — Gate 4 impulse response", "gate4_deconvolution.py",
                     ["--user", u, "--tz", tz, "--boot", str(a.boot)], f"g4_{u}.md"))

    results = {}
    if a.skip_run:
        for grp, title, _s, _ar, fn in jobs:
            p = os.path.join(BUILD, fn)
            results[fn] = (open(p).read(), True) if os.path.exists(p) else ("**Not run.**\n", False)
    else:
        print(f"running {len(jobs)} analyses, {a.jobs} at a time")

        def go(j):
            grp, title, script, args, fn = j
            md, ok = run(script, args, os.path.join(BUILD, fn))
            print(f"  [{'ok ' if ok else 'FAIL'}] {title}", flush=True)
            return fn, (md, ok)
        with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
            for fn, val in ex.map(go, jobs):
                results[fn] = val

    # ---- assemble ----------------------------------------------------------------------
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_ok = sum(1 for v in results.values() if v[1])
    parts = [
        '<div class="cover">',
        "<h1>Insulin kinetics from closed-loop data</h1>",
        f'<div class="sub">Cohort report — {len(users)} users</div>',
        f'<div class="meta">Generated {now} &nbsp;·&nbsp; {n_ok} of {len(results)} analyses '
        f'completed &nbsp;·&nbsp; Gate 1 window {a.days} d &nbsp;·&nbsp; '
        f'Gate 4 bootstrap {a.boot} draws</div>',
        "</div>",
    ]
    md_parts = [
        "\n## How to read this\n",
        "**Gate 1** recovers what a loop *believes* — its configured insulin curve — by "
        "deconvolving the loop's own logged IOB against its delivered doses. That is an algebraic "
        "identity, so it is the solid part.\n",
        "**Gate 4** recovers what the insulin appears to *do*, by non-parametric deconvolution of "
        "glucose against the dose series, with a time-of-day drift profile shared across all days. "
        "It assumes no curve shape.\n",
        "**Gate 2 is withdrawn** and is not included: its design was under-identified against dawn "
        "drift. Any Gate 2 figure quoted previously should be disregarded.\n",
        "Read the validation section first. If the positive controls fail, nothing below them "
        "means anything.\n",
    ]
    groups = [("validation", "1. Validation — do the methods work at all?"),
              ("integrity", "2. Extract integrity — could the pipeline be faking it?"),
              ("cohort", "3. Cohort summary"),
              ("peruser", "4. Per-user detail")]
    for key, heading in groups:
        md_parts.append(f'\n<div class="section-break"></div>\n')
        md_parts.append(f"\n## {heading}\n")
        for grp, title, _s, _ar, fn in jobs:
            if grp != key:
                continue
            md, ok = results.get(fn, ("**Missing.**\n", False))
            md_parts.append(f"\n### {title}" + ("" if ok else ' <span class="fail">— FAILED</span>')
                            + "\n")
            md_parts.append(strip_h1(md))

    import markdown as md_mod
    body = md_mod.markdown("\n".join(md_parts),
                           extensions=["tables", "fenced_code", "sane_lists"])
    html = (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"
            + "".join(parts) + body + "</body></html>")
    html_path = os.path.join(BUILD, "report.html")
    open(html_path, "w").write(html)

    from weasyprint import HTML
    HTML(string=html, base_url=HERE).write_pdf(a.out)
    print(f"\nwrote {a.out}  ({os.path.getsize(a.out) / 1024:.0f} KB)")
    if n_ok < len(results):
        print(f"NOTE: {len(results) - n_ok} analyses failed and are marked as such in the PDF")


if __name__ == "__main__":
    main()
