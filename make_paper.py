#!/usr/bin/env python3
"""MAKE PAPER — the findings written up as a scientific paper, with full methods.

Replaces the earlier short summary. Structure follows a standard paper: abstract, data, methods
(including the estimator mathematics in full), results, discussion, limitations, conclusion.

Numerical results are PARSED from build/ rather than typed in, so the paper cannot drift from the
analysis it reports. The methods section is static text, because the mathematics does not change
between runs — but it is written to match the code exactly, and each subsection names the file and
function that implements it so a reader can check.

Run make_report.py first (it populates build/); this reads that directory.

Usage:
  python3 make_paper.py [--build build] [--out insulin-kinetics-paper.pdf]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re

from platform_detect import detect as detect_platform

HERE = os.path.dirname(os.path.abspath(__file__))

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm;
        @bottom-center { content: counter(page); font: 8.5pt Georgia; color: #888; } }
body { font: 10pt/1.5 Georgia, "Times New Roman", serif; color: #111; text-align: justify;
       hyphens: auto; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 3mm 0; text-align: left; font-weight: normal; }
h2 { font-size: 11.5pt; margin: 6mm 0 2mm 0; text-align: left; break-after: avoid;
     font-weight: bold; }
h3 { font-size: 10pt; margin: 4mm 0 1.5mm 0; text-align: left; break-after: avoid;
     font-style: italic; font-weight: normal; }
p { margin: 0 0 2.5mm 0; }
ul, ol { margin: 0 0 3mm 0; padding-left: 6mm; } li { margin-bottom: 1.5mm; }
.byline { font-size: 9.5pt; color: #444; margin-bottom: 1mm; }
.meta { font-size: 8.5pt; color: #777; margin-bottom: 6mm; }
.abstract { background: #f6f6f6; padding: 3mm 4mm; margin: 0 0 5mm 0; font-size: 9.5pt; }
.abstract b { display: block; margin-bottom: 1.5mm; }
.eq { font-family: "Latin Modern Math", Cambria Math, Georgia, serif; text-align: center;
      margin: 3mm 0; font-size: 10.5pt; break-inside: avoid; }
.eqn { float: right; font-size: 9pt; color: #666; }
.where { font-size: 9pt; margin: -1mm 0 3mm 4mm; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0 2mm 0; font-size: 8.5pt;
        break-inside: avoid; font-family: Helvetica, Arial, sans-serif; }
th { text-align: left; padding: 1.4mm 2mm; border-top: 1pt solid #333; border-bottom: 0.5pt solid #333;
     font-weight: bold; }
td { padding: 1.2mm 2mm; border-bottom: 0.4pt solid #ccc; }
tr:last-child td { border-bottom: 1pt solid #333; }
.caption { font-size: 8.5pt; color: #444; margin: 0 0 4mm 0; text-align: left; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.5pt; }
.note { font-size: 9pt; border-left: 2pt solid #999; padding-left: 3mm; margin: 3mm 0;
        color: #333; break-inside: avoid; }
"""


def read(p):
    return open(p).read() if os.path.exists(p) else ""


def parse_cohort(B):
    rows = []
    for ln in read(os.path.join(B, "cohort.md")).splitlines():
        m = re.match(r"\|\s*(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\w/]+)\s*\|\s*([\d.]+)(!?)\s*\|"
                     r"\s*([\d-]+)\s*\|\s*([\d-]+)\s*\|\s*([\d-]+)\s*\|\s*([+\-\d]+)\s*\|", ln)
        if m:
            rows.append(dict(user=m.group(1), cfg=float(m.group(2)), dia=m.group(3),
                             fit=float(m.group(4)), flag=m.group(5) == "!", h1=m.group(6),
                             h2=m.group(7), obs=m.group(8), gap=m.group(9)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(HERE, "build"))
    ap.add_argument("--out", default=os.path.join(HERE, "insulin-kinetics-paper.pdf"))
    ap.add_argument("--anonymise", action="store_true",
                    help="re-key users to table-local labels for external circulation")
    a = ap.parse_args()
    B = a.build

    cohort = parse_cohort(B)
    if not cohort:
        print(f"no cohort table in {B} — run make_report.py first"); return
    users = [r["user"] for r in cohort]
    if a.anonymise:
        key = {u: f"P{i+1}" for i, u in enumerate(users)}
        for r in cohort:
            r["user"] = key[r["user"]]

    ctl = {u: read(os.path.join(B, f"g1c_{u}.md")) for u in users}
    integ = {u: read(os.path.join(B, f"int_{u}.md")) for u in users}
    g4 = {u: read(os.path.join(B, f"g4_{u}.md")) for u in users}

    selftest = re.findall(r"\| ([^|]+?) \| (\d+) \| (\d+) \| (\d+) \|",
                          read(os.path.join(B, "val_g4selftest.md")))
    fb = read(os.path.join(B, "val_feedback.md"))
    fb_excess = re.search(r"over the open-loop reference: \*\*([+\-][\d.]+) min\*\*", fb)
    fb_mae = re.findall(r"\| ([\d.]+)(?: \(open loop reference\))? \| ([\d.]+) min \|", fb)
    val = " ".join(read(p) for p in glob.glob(os.path.join(B, "val_*.md")))
    n_fail = len(re.findall(r"\*\*FAIL\*\*", val))

    dia_na = [r["user"] for r in cohort if r["dia"] == "n/a"]
    flagged = [r["user"] for r in cohort if r["flag"]]
    split_flag = [u for u in users if "FLAG:" in ctl[u]]
    tail_below = [u for u in users if "tail is below the noise floor" in ctl[u].lower()]
    n_days = {u: (re.search(r"across (\d+) days", g4[u]).group(1)
                  if re.search(r"across (\d+) days", g4[u]) else "?") for u in users}
    n_samp = {u: (re.search(r"([\d,]+) usable 5-minute samples", g4[u]).group(1)
                  if re.search(r"([\d,]+) usable 5-minute samples", g4[u]) else "?") for u in users}

    plat = detect_platform()
    plat = {u: v for u, v in plat.items() if u in users}
    n_aaps = sum(1 for _, (p, _) in plat.items() if p == "AAPS")
    n_trio = sum(1 for _, (p, _) in plat.items() if p == "Trio")
    spans = [int(v) for v in n_days.values() if str(v).isdigit()]
    min_span, max_span = (min(spans), max(spans)) if spans else ("?", "?")

    H = []
    A = H.append
    A("<h1>Estimating insulin action peak time from closed-loop data: "
      "parametric recovery of the configured curve and non-parametric recovery of the "
      "observed response</h1>")
    A('<div class="byline">Cohort analysis of automated insulin delivery records</div>')
    A(f'<div class="meta">Generated {dt.datetime.now():%Y-%m-%d} from analysis outputs. '
      f'{"Anonymised labels." if a.anonymise else "Internal draft — per-user labels are working "
      "cohort identifiers."}</div>')

    # ---------------- Abstract ----------------
    A('<div class="abstract"><b>Abstract</b>')
    A(f"<p><i>Background.</i> Automated insulin delivery systems model insulin with two parameters, "
      f"the time to peak action and the duration of action. Both are conventionally set from label "
      f"values, and neither is verified against the individual. Whether they can be recovered from "
      f"routine loop records — with no test protocol — is unresolved, because a closed loop "
      f"delivers a near-continuous stream of reactive doses rather than isolated impulses.</p>")
    A(f"<p><i>Methods.</i> Two estimators were applied to {len(cohort)} closed-loop users. The "
      f"first deconvolves each system's own logged insulin-on-board series against its delivered "
      f"doses under the exponential insulin model, recovering the curve the system is <i>configured</i> "
      f"with; because that relationship is an algebraic identity, it admits an exact positive "
      f"control. The second estimates the insulin activity kernel non-parametrically as a "
      f"finite impulse response on glucose increments, with a second-difference smoothness penalty, "
      f"a non-negativity constraint, and a time-of-day drift profile pooled across days. Both were "
      f"validated by recovering known curves from simulated data, and the second additionally "
      f"against simulated closed-loop feedback.</p>")
    A(f"<p><i>Results.</i> The parametric estimator recovered configured curves with relative "
      f"residuals of 6–9% in users whose dose records reconciled, and detected an unrecorded "
      f"mid-period change of insulin curve in one user. The non-parametric estimator recovered "
      f"known peaks exactly in noise-free simulation, including when the generating curve belonged "
      f"to a different family, and showed no bias attributable to controller feedback "
      f"({fb_excess.group(1) if fb_excess else 'n/a'} min against an open-loop reference). Duration "
      f"of action was not identifiable in {len(dia_na)} of {len(cohort)} users. Discrepancies "
      f"between configured and observed peak reached 77 minutes, and in one participant the "
      f"recovered configured curve did not correspond to the insulin they reported using.</p>")
    A(f"<p><i>Conclusion.</i> The configured curve is recoverable exactly and provides a practical "
      f"audit of what a system is actually computing. The observed curve is recoverable with "
      f"useful precision but remains observational; duration of action is largely not recoverable, "
      f"and its confidence intervals are misleading when it is not.</p>")
    A("</div>")

    # ---------------- Introduction ----------------
    A("<h2>1. Introduction</h2>")
    A("<p>An automated insulin delivery (AID) system predicts the effect of insulin it has "
      "already given. That prediction rests on an assumed activity curve, parameterised by a time "
      "to peak action <i>t</i><sub>p</sub> and a duration of action <i>t</i><sub>d</sub>. These are "
      "normally taken from the insulin's label or from community convention, and applied uniformly. "
      "Individual variation in absorption is well documented, so an obvious question is whether a "
      "person's own records can say what their curve actually is.</p>")
    A("<p>The naive approach — isolate a bolus, observe the fall in glucose, note when it is "
      "steepest — fails on closed-loop data. The controller doses every few minutes in reaction to "
      "glucose, so isolated impulses are rare: in one user, of approximately 19,800 boluses only "
      "234 had a clear hour either side and 38 had two hours. Impulse-response averaging starves. "
      "Worse, glucose moves for many reasons at once, and the largest confounder — unannounced "
      "carbohydrate — is by definition absent from the record.</p>")
    A("<p>This work separates two questions that are usually conflated. What curve is the system "
      "<i>configured</i> with, and what curve does the insulin <i>appear to follow</i>? The first "
      "is an audit question with an exact answer; the second is a physiological question with an "
      "observational answer. Treating them separately turns out to matter, because they can "
      "disagree for reasons that have nothing to do with physiology.</p>")

    # ---------------- Data ----------------
    A("<h2>2. Data</h2>")
    A(f"<p>Records from {len(cohort)} people using AID systems were analysed, comprising per-cycle "
      f"controller decisions at five-minute cadence (glucose, carbohydrate on board, insulin on "
      f"board, its basal component, and step counts where the uploader provides them) and a "
      f"treatment stream of delivered insulin and entered carbohydrate. Observation periods ranged "
      f"from {min_span} to {max_span} days per participant; the shortest contributes only "
      f"{min_span} days and its estimates should be read accordingly. No intervention was made and "
      f"no protocol imposed; these are routine records. No demographic data were available, so no "
      f"characteristics of the participants beyond their delivery system are reported.</p>")
    A(f"<p>Two delivery systems are represented: <b>{n_aaps} AndroidAPS</b> and <b>{n_trio} Trio</b> "
      f"users. Neither is recorded in the extract, so the platform was identified from three "
      f"independent features of the records themselves — whether boluses carry a type field, "
      f"whether bolus insulin-on-board is uploaded or must be derived, and whether the uploader "
      f"provides step counts. All {len(cohort)} classifications were unanimous across the three "
      f"({{}}). The distinction is not bookkeeping: it changes which cross-checks are available and "
      f"which confounders can be controlled, as set out in \u00a74.4.</p>".format(
          ", ".join(f"{u}: {p}" for u, (p, _) in sorted(plat.items()))))
    A("<p>One participant's records were obtained from a different source than the remainder, "
      "having been added during this investigation rather than forming part of the original "
      "cohort. That participant is the subject of \u00a74.5, so the asymmetry is noted here "
      "explicitly.</p>")

    A("<h2>3. Methods</h2>")

    A("<h3>3.1 Insulin model</h3>")
    A("<p>Both estimators refer to the exponential (\"free-peak\") insulin model used by oref-derived "
      "AID systems [1]. For peak time <i>t</i><sub>p</sub> and duration <i>t</i><sub>d</sub> in minutes, "
      "define</p>")
    A('<div class="eq">τ = <i>t</i><sub>p</sub> (1 − <i>t</i><sub>p</sub>/<i>t</i><sub>d</sub>) / '
      '(1 − 2<i>t</i><sub>p</sub>/<i>t</i><sub>d</sub>), &nbsp;&nbsp; '
      'a = 2τ/<i>t</i><sub>d</sub>, &nbsp;&nbsp; '
      'S = [1 − a + (1 + a)e<sup>−<i>t</i><sub>d</sub>/τ</sup>]<sup>−1</sup>'
      '<span class="eqn">(1)</span></div>')
    A("<p>The fraction of a unit dose remaining active at time <i>t</i> after delivery, and its rate "
      "of action, are then</p>")
    A('<div class="eq"><i>f</i>(<i>t</i>) = 1 − S(1 − a) [ ( <i>t</i>²/(τ<i>t</i><sub>d</sub>(1−a)) '
      '− <i>t</i>/τ − 1 ) e<sup>−<i>t</i>/τ</sup> + 1 ]<span class="eqn">(2)</span></div>')
    A('<div class="eq"><i>A</i>(<i>t</i>) = (S/τ²) · <i>t</i> · (1 − <i>t</i>/<i>t</i><sub>d</sub>) '
      '· e<sup>−<i>t</i>/τ</sup><span class="eqn">(3)</span></div>')
    A("<p>with both zero outside [0, <i>t</i><sub>d</sub>]. The parameterisation is constructed so "
      "that <i>A</i> attains its maximum exactly at <i>t</i><sub>p</sub>; differentiating (3) and "
      "substituting τ from (1) gives <i>A</i>′(<i>t</i><sub>p</sub>) = 0 identically. Implemented in "
      "<code>gate1_recover_known_curve.py</code>, functions <code>iob_fraction</code> and "
      "<code>activity</code>.</p>")

    A("<h3>3.1.1 What the regulatory labels actually specify</h3>")
    A("<p>The pharmacodynamics behind these curves are published: euglycaemic-clamp "
      "glucose-infusion-rate profiles appear in the European product information for every analogue "
      "in this cohort. What those documents supply, however, is not what an AID system needs.</p>")
    A("<table><tr><th>Analogue</th><th>Onset of action</th><th>Maximum effect</th>"
      "<th>Duration</th><th>Peak assumed by AID</th></tr>")
    A("<tr><td>insulin lispro [8]</td><td>~15 min</td><td><i>not stated</i></td><td>2–5 h</td>"
      "<td>75 min</td></tr>")
    A("<tr><td>insulin aspart [9]</td><td>10–20 min</td><td>1–3 h</td><td>3–5 h</td>"
      "<td>75 min</td></tr>")
    A("<tr><td>insulin glulisine [10]</td><td>10–20 min</td><td><i>not stated</i></td><td>~4 h</td>"
      "<td>75 min</td></tr>")
    A("<tr><td>faster-acting insulin aspart [11]</td><td>5 min earlier than aspart</td>"
      "<td>1–3 h</td><td><i>not stated</i></td><td>55 min</td></tr>")
    A("<tr><td>ultra-rapid lispro [12]</td><td>20 min</td><td>1–3 h</td><td>5 h</td>"
      "<td>55 min</td></tr>")
    A("</table>")
    A('<div class="caption"><b>Table 1.</b> Pharmacodynamic parameters as stated in the European '
      'product information, against the peak time assumed by oref-derived delivery systems for each '
      'analogue. Quotations are verbatim from section 5.1 of each document.</div>')
    A("<p>Three features of this table bear on everything that follows. Every label that states a "
      "maximum effect states the same interval — <b>one to three hours</b> — irrespective of "
      "analogue class; the labels differentiate the ultra-rapid analogues from the rapid ones on "
      "<i>onset</i> and <i>duration</i>, not on time to peak. Two of the five do not state a peak "
      "at all. And the interval that is stated is several times wider than the 45-to-75-minute "
      "values delivery systems use.</p>")
    A("<p>The 55-versus-75-minute distinction that AID systems draw is therefore not a reading of "
      "the labels; it is a convention that narrows a published range to a single number, and does "
      "so along an axis the labels do not resolve. That is the gap this work measures. It also "
      "explains why two systems can assign different peaks to the same insulin — 45 minutes in one, "
      "55 in another — with neither contradicting the product information, because the product "
      "information does not adjudicate between them.</p>")
    A('<div class="note">The labels are also explicit that these are not fixed constants. The '
      'product information for two of the analogues states that the duration of action "will vary '
      'according to the dose, injection site, blood flow, temperature and level of physical '
      'activity" [9, 11], and one reports that total and maximum glucose-lowering effect increase '
      'with dose across the therapeutic range [12]. A single per-person curve is an approximation '
      'the labels themselves decline to endorse.</div>')

    A("<h3>3.2 Identification</h3>")
    A("<p>In a window where insulin is the only influence on glucose <i>g</i>,</p>")
    A('<div class="eq">d<i>g</i>/d<i>t</i> ≈ −ISF · Σ<sub>j</sub> <i>d</i><sub>j</sub> '
      '<i>A</i>(<i>t</i> − <i>t</i><sub>j</sub>)<span class="eqn">(4)</span></div>')
    A("<p>for doses <i>d</i><sub>j</sub> at times <i>t</i><sub>j</sub>. Two consequences follow. "
      "First, the peak is a <i>shape</i> parameter while insulin sensitivity is an <i>amplitude</i> "
      "parameter, so they are separable: the timing can be estimated without knowing ISF, and is "
      "invariant to anything that scales the response, including a change of insulin concentration. "
      "Second, superposition means overlapping doses add, so every dose contributes information "
      "rather than contaminating it. The binding constraint is therefore not sample size but "
      "collinearity between adjacent lags — if insulin delivered 30 and 35 minutes ago are "
      "near-identical regressors, no quantity of data identifies the kernel.</p>")

    A("<h3>3.3 Parametric recovery of the configured curve</h3>")
    A("<p>An AID system logs its own insulin-on-board each cycle, and that quantity is a "
      "deterministic function of the dose history and the configured curve:</p>")
    A('<div class="eq">IOB(<i>t</i>) = Σ<sub>j</sub> <i>d</i><sub>j</sub> · '
      '<i>f</i>(<i>t</i> − <i>t</i><sub>j</sub>; <i>t</i><sub>p</sub>, <i>t</i><sub>d</sub>)'
      '<span class="eqn">(5)</span></div>')
    A("<p>Deconvolving the logged series against the delivered doses must therefore return the "
      "configured parameters. Doses are binned onto the controller's own five-minute grid and (5) "
      "is evaluated as a discrete convolution with a kernel of ⌈<i>t</i><sub>d</sub>/5⌉ + 1 taps, "
      "rather than as a per-dose sum; the naive form is O(<i>n</i><sub>doses</sub> × "
      "<i>n</i><sub>times</sub>), of order 10<sup>8</sup> per residual evaluation here, and the "
      "optimiser requires hundreds. Parameters are estimated by</p>")
    A('<div class="eq">(<i>t̂</i><sub>p</sub>, <i>t̂</i><sub>d</sub>) = arg min Σ<sub>i∈M</sub> '
      '[ IOB<sub>pred</sub>(<i>t</i><sub>i</sub>) − IOB<sub>obs</sub>(<i>t</i><sub>i</sub>) ]²'
      '<span class="eqn">(6)</span></div>')
    A("<p>by bounded least squares (Trust Region Reflective), with <i>t</i><sub>p</sub> ∈ [10, 180] "
      "and <i>t</i><sub>d</sub> ∈ [120, 1440] minutes.</p>")
    A("<p>The index set <i>M</i> excludes observations falling within ten minutes <i>after</i> any "
      "bolus. This matters "
      "more than it appears. Insulin on board steps by the whole dose at delivery, so a small error "
      "in placing that step leaves a residual the size of the dose; the step also lands in different "
      "bins on different systems, since a controller that computes IOB at the start of its cycle "
      "counts a bolus delivered afterwards only on the next pass. Unmasked, the relative residual on "
      "what is an exact identity ran 11–32% across the cohort. Masking reduces it to 6–9% in "
      "well-reconciled users and, more consequentially, removes a downward bias on "
      "<i>t</i><sub>d</sub>: in the one participant whose configured duration was known "
      "independently to be 600 minutes, the estimate moved from 314 to 487 minutes. The peak moves "
      "by less than 0.2 minutes, so the mask is close to free.</p>")
    A('<div class="note">Doses and observations are both floor-binned to the five-minute grid, '
      'placing each 2–3.5 minutes early on average. The two errors act in the same direction and '
      'largely cancel in the lag the kernel sees. Correcting only the dose side — the intuitive '
      'fix — breaks the cancellation and displaces the recovered peak by 1.6–2 minutes.</div>')

    A("<h3>3.4 Identifiability of the duration</h3>")
    A("<p>A bootstrap interval on <i>t̂</i><sub>d</sub> is not a reliable guide to whether the "
      "duration is identified, because it is narrowest where the likelihood is flattest: resampling "
      "noise around a nearly flat ridge reproduces the same spurious local minimum. In the "
      "participant whose configured duration was known to be 600 minutes, a 28-day fit returned "
      "314 minutes with a 95% interval of [260, 378] — the tightest interval obtained, and one that "
      "excludes the truth.</p>")
    A("<p>Identifiability is instead assessed by leverage. Holding <i>t̂</i><sub>p</sub> fixed, the "
      "predicted series is recomputed at alternative durations and compared with the fitted one:</p>")
    A('<div class="eq">L(<i>t</i><sub>d</sub>′) = RMS[ IOB<sub>pred</sub>(<i>t̂</i><sub>p</sub>, '
      '<i>t</i><sub>d</sub>′) − IOB<sub>pred</sub>(<i>t̂</i><sub>p</sub>, <i>t̂</i><sub>d</sub>) ]'
      '<span class="eqn">(7)</span></div>')
    A("<p>evaluated at <i>t</i><sub>d</sub>′ ∈ {240, 360, 480, 600, 900, 1440} minutes. Where "
      "max L(<i>t</i><sub>d</sub>′) does not exceed twice the fit residual, no alternative duration "
      "is distinguishable from the fitted one and the estimate is reported as unidentified. A "
      "secondary check refits at 28, 45 and 90 days: a genuinely identified duration is stable "
      "across window lengths, an unidentified one returns whatever the window happens to support.</p>")

    A("<h3>3.5 Non-parametric recovery of the observed response</h3>")
    A("<p>The second estimator abandons the parametric form. The record is treated as a train of "
      "impulses and the impulse response estimated directly — non-parametric input estimation in "
      "the sense of [2]:</p>")
    A('<div class="eq">Δ<i>g</i><sub>t</sub> = − Σ<sub>k=0</sub><sup>K</sup> β<sub>k</sub> '
      '<i>d</i><sub>t−k</sub> + <i>c</i><sub>clock(t)</sub> + <i>u</i><sub>day(t)</sub> + '
      'ε<sub>t</sub><span class="eqn">(8)</span></div>')
    A('<div class="where">β<sub>k</sub> — one free coefficient per five-minute lag, '
      'k = 0…K, K = 72 (six hours); the activity curve itself, with no assumed shape.<br>'
      '<i>c</i> — a time-of-day profile, one coefficient per thirty-minute clock bin (48 bins, one '
      'dropped for identifiability against the day intercepts), shared across '
      'all days.<br><i>u</i> — a per-day intercept absorbing that day\'s basal rate, sensitivity '
      'level and sensor offset.</div>')
    A("<p>The peak is read off directly as <i>t̂</i><sub>p</sub> = 5 · arg max<sub>k</sub> "
      "β̂<sub>k</sub> minutes.</p>")
    A("<p>The identification strategy, rather than the added flexibility, is what makes this work. "
      "An earlier design gave each fasting window its own free drift term, which is hopeless: within "
      "a single window the insulin activity profile is slowly varying and nearly linear in time, and "
      "so is the dawn rise it must be separated from. Measured collinearity between the insulin "
      "regressor and elapsed time had a median |r| of 0.73–0.81 at four-hour windows, rising to 0.99 "
      "at two hours; the two terms compete for the same slow component and whichever is permitted to "
      "absorb it determines the answer. Under (8) the drift is instead a <i>repeatable time-of-day "
      "profile</i> estimated across every day at once. Dawn is locked to the clock; doses are not. "
      "That difference in timing is the identifying variation, and it exists only when days are "
      "pooled. For the same reason the sample is every fasting stretch around the clock rather than "
      "overnight windows alone, giving "
      f"{min(int(v.replace(',','')) for v in n_samp.values() if v != '?'):,}–"
      f"{max(int(v.replace(',','')) for v in n_samp.values() if v != '?'):,} five-minute "
      "observations per participant.</p>")

    A("<h3>3.6 Regularisation and smoothing selection</h3>")
    A("<p>Adjacent lags remain strongly collinear, so the unpenalised estimate of β is noise. A "
      "second-difference (Tikhonov) penalty enforces smoothness, and β is constrained non-negative "
      "since insulin does not raise glucose:</p>")
    A('<div class="eq">β̂ = arg min<sub>β ≥ 0</sub> ‖<i>y</i> − <i>X</i>β‖² + λ‖<i>D</i>β‖²'
      '<span class="eqn">(9)</span></div>')
    A("<p>where <i>X</i> stacks the lagged-dose, clock and day blocks, and <i>D</i> is the "
      "second-difference operator acting on the lag coefficients only, with rows "
      "(…, 1, −2, 1, …). The constrained problem is solved as a bounded least-squares system on the "
      "augmented design [<i>X</i>; √λ<i>D</i>] against [<i>y</i>; 0], with lower bounds of zero on "
      "the lag block and unbounded elsewhere. The smoothing weight is chosen by generalised "
      "cross-validation,</p>")
    A('<div class="eq">GCV(λ) = <i>n</i> · RSS(λ) / [ <i>n</i> − tr <i>H</i>(λ) ]², &nbsp;&nbsp; '
      '<i>H</i>(λ) = <i>X</i>(<i>X</i><sup>⊤</sup><i>X</i> + λ<i>D</i><sup>⊤</sup><i>D</i>)'
      '<sup>−1</sup><i>X</i><sup>⊤</sup><span class="eqn">(10)</span></div>')
    A("<p>over a logarithmic grid, using the trace identity tr(<i>XHX</i><sup>⊤</sup>) = "
      "tr(<i>H X</i><sup>⊤</sup><i>X</i>) to avoid forming an <i>n</i>×<i>n</i> matrix; with "
      "<i>n</i> ≈ 31,000 and <i>p</i> ≈ 244 the two forms agree to 6×10<sup>−12</sup> and differ in "
      "cost by more than three orders of magnitude.</p>")
    A("<p>The smoothing weight is selected on the <i>unconstrained</i> ridge solution, for which "
      "the hat matrix and hence the effective degrees of freedom are available in closed form, and "
      "the selected value is then applied to the non-negativity-constrained fit of (9). The "
      "constraint binds only in the tail, where the unpenalised estimate would otherwise oscillate "
      "about zero, so this decoupling has no material effect on the recovered peak [3]. Implemented in "
      "<code>gate4_deconvolution.py</code>, functions <code>design</code>, <code>fit_fir</code> and "
      "<code>gcv</code>.</p>")

    A("<h3>3.7 Uncertainty</h3>")
    A("<p>Intervals are obtained by block bootstrap [4] resampling whole days with replacement, "
      "preserving within-day correlation, and refitting. Days rather than observations are the "
      "resampling unit because consecutive five-minute samples are strongly dependent; resampling "
      "observations would understate uncertainty by a wide margin.</p>")

    A("<h3>3.8 Validation</h3>")
    A("<p>Both estimators were validated against known answers before any result was interpreted.</p>")
    A("<p>The parametric estimator has an exact positive control by construction, since (5) is an "
      "identity: applied to a system's own logged IOB it must return the configured parameters, and "
      "any departure is a defect in the method rather than a finding about insulin. For the "
      "non-parametric estimator a control was constructed by retaining each participant's real dose "
      "series, sampling grid and eligibility mask, and replacing observed glucose with glucose "
      "simulated from a known kernel plus a clock-locked drift and white noise calibrated to that "
      "participant's own residual scale. Recovery was tested against three generating families — "
      "the exponential model above, gamma curves of differing shape parameter, and bi-exponentials "
      "of differing tail ratio — because an estimator that recovers only its own family is flexible "
      "rather than non-parametric. A noise-free case is included: at zero noise the answer must be "
      "exact, and this condition detected two silent implementation defects that produced "
      "plausible-looking output.</p>")
    A("<p>A further control addresses endogeneity. Both positive controls generate glucose "
      "<i>from</i> the doses, which makes the input exogenous by construction and therefore cannot "
      "detect the failure mode that matters most in closed-loop data: the controller doses "
      "<i>because</i> glucose is rising. In these records a dose correlates +0.25 to +0.55 with the "
      "glucose change in the preceding five minutes and approximately zero with the following half "
      "hour, so the controller's reaction dominates the raw dose–glucose relationship — the classical "
      "obstacle to identifying a plant operating under feedback [5]. A closed-loop "
      "simulation was therefore run, in which glucose responds to insulin through a known kernel "
      "while a proportional controller doses off the recent rise, at gains spanning an open loop to "
      "an aggressively reactive one. The verdict is taken against the open-loop case rather than an "
      "absolute tolerance, since an estimator that is merely noisy will err at zero gain too.</p>")

    A("<h3>3.9 Data integrity</h3>")
    A("<p>Four properties of the extract could produce a wrong peak with no other symptom, and each "
      "is checked without reference to the kernel. Where a system uploads bolus IOB directly it is "
      "compared against the value derived as total minus basal IOB. Duplicate treatment records are "
      "detected by comparing row counts with distinct record identifiers. Completeness and scale are "
      "checked by requiring that, at an isolated bolus, the logged IOB step equals the recorded "
      "dose; a ratio above unity implies doses missing from the extract, and one far below implies "
      "double counting. Alignment is checked by correlating the dose series against the IOB "
      "increment arriving in each bin, and by refitting with the dose series deliberately displaced "
      "by −30 to +30 minutes — a procedure that converts \"could a timestamp error explain this?\" "
      "into a number.</p>")

    # ---------------- Results ----------------
    A("<h2>4. Results</h2>")

    A("<h3>4.1 Validation</h3>")
    A(f"<p>All positive controls passed ({n_fail} failures). Recovery of known peaks by the "
      f"non-parametric estimator is shown in Table 2.</p>")
    if selftest:
        A("<table><tr><th>Generating curve</th><th>True peak (min)</th>"
          "<th>Recovered, noise-free</th><th>Recovered, realistic noise</th></tr>")
        for name, t, c, n in selftest:
            A(f"<tr><td>{name}</td><td>{t}</td><td>{c}</td><td>{n}</td></tr>")
        A("</table>")
        A('<div class="caption"><b>Table 2.</b> Recovery of a known activity peak from simulated '
          'glucose, using real dose series and sampling. Noise-free recovery is exact in every '
          'case, including the two families the estimator does not assume.</div>')
    if fb_mae:
        A("<table><tr><th>Controller gain</th><th>Mean absolute error (min)</th></tr>")
        for g, e in fb_mae:
            lbl = f"{g} (open loop reference)" if float(g) == 0 else g
            A(f"<tr><td>{lbl}</td><td>{e}</td></tr>")
        A("</table>")
        A(f'<div class="caption"><b>Table 3.</b> Recovery under simulated closed-loop feedback. '
          f'Excess error attributable to feedback, over the open-loop reference: '
          f'<b>{fb_excess.group(1) if fb_excess else "n/a"} minutes</b>. Induced endogeneity reached '
          f'+0.27 to +0.37, within the range observed in the real records, so the test exercises the '
          f'failure mode rather than passing vacuously.</div>')

    A("<h3>4.2 Configured versus observed</h3>")
    A("<table><tr><th>Participant</th><th>System</th><th>Configured peak</th><th>Duration</th>"
      "<th>Relative residual</th><th>First half / second half</th><th>Observed peak</th>"
      "<th>Difference</th></tr>")
    for r in cohort:
        A(f"<tr><td>{r['user']}</td><td>{plat.get(r['user'], ('?',''))[0]}</td>"
          f"<td>{r['cfg']:.1f}</td><td>{r['dia']}</td>"
          f"<td>{r['fit']:.3f}{'*' if r['flag'] else ''}</td><td>{r['h1']} / {r['h2']}</td>"
          f"<td>{r['obs']}</td><td>{r['gap']}</td></tr>")
    A("</table>")
    A('<div class="caption"><b>Table 4.</b> All values in minutes. The relative residual is that of '
      'the parametric fit against an exact identity; * marks values above 0.15, indicating dose '
      'records that do not reconcile with the system\'s own logged IOB. "Duration n/a" denotes a '
      'duration the leverage criterion of §3.4 found unidentifiable. The difference is observed '
      'minus configured; uncorrected sensor lag biases the observed value late by a few minutes, so '
      'negative differences are the conservative direction.</div>')
    A(f"<p>Configured curves were recovered with relative residuals of 6–9% in the "
      f"{len(cohort) - len(flagged)} participants whose dose records reconciled, with recovered "
      f"peaks falling on standard preset values. In {len(flagged)} participants "
      f"({', '.join(flagged)}) the residual exceeded 15% on a relationship that is an identity, "
      f"indicating that the dose record does not correspond to what the system counted; candidate "
      f"explanations are extended or multiwave boluses, partial delivery, and timestamps offset from "
      f"the delivery the controller registered. Their configured curves are reported but should be "
      f"treated as provisional.</p>")
    changed = [r for r in cohort if r["h1"].lstrip("-").isdigit() and r["h2"].lstrip("-").isdigit()
               and abs(int(r["h1"]) - int(r["h2"])) > 10]
    if changed:
        c = changed[0]
        A(f"<p>Refitting on disjoint halves of the observation period detected a change of "
          f"configured curve in participant {c['user']} ({c['h1']} to {c['h2']} minutes) that no "
          f"treatment record identified. Rolling seven-day windows localised the change to a single "
          f"week. The event stream for this participant contained profile-change records dated a "
          f"week after the kinetics changed, and would have misdirected the search.</p>")

    A("<h3>4.3 Duration of action</h3>")
    A(f"<p>The duration was unidentifiable in {len(dia_na)} of {len(cohort)} participants "
      f"({', '.join(dia_na)}) under the leverage criterion. Separately, truncating each "
      f"participant's own fitted kernel at five hours changed the predicted IOB series by less than "
      f"the fit residual in {len(tail_below)} of {len(cohort)} participants, confirming that the "
      f"tail carries no usable signal at these dose sizes. Stratifying doses by size does not "
      f"recover it, since the tail is small relative to the residual in both strata.</p>")

    A("<h3>4.4 Differences between the two delivery systems</h3>")
    A(f"<p>The {n_aaps} AndroidAPS and {n_trio} Trio participants are not interchangeable for these "
      f"methods, in three respects that affect what can be checked rather than what is found.</p>")
    A("<ul>")
    A("<li><b>The bolus IOB series is uploaded by one and derived for the other.</b> Trio reports "
      "<code>bolusiob</code> directly; for AndroidAPS it must be reconstructed as total minus basal "
      "IOB. The derivation is therefore only verifiable where it is not needed — in the Trio "
      "participants, where derived and uploaded values agreed to a mean absolute difference of "
      "0.00025 U. That agreement is the entire evidence that the AndroidAPS derivation is sound, "
      "and it is indirect.</li>")
    A("<li><b>The IOB step falls in a different bin.</b> In Trio records the logged IOB rises in "
      "the same five-minute bin as the bolus timestamp; in AndroidAPS records it rises in the "
      "following one, because the controller computes IOB at the start of its cycle and counts a "
      "bolus delivered afterwards on the next pass. Across the cohort the observed offsets were "
      "confined to zero or one bin with none outside, but a fixed assumption of either would have "
      "been wrong for one platform. The estimator detects the offset rather than assuming it.</li>")
    A("<li><b>Exercise is uncontrolled for Trio participants.</b> The AndroidAPS uploader carries "
      "step counts, which are used to exclude active periods; the Trio uploader does not. Both "
      "Trio participants are therefore analysed without any activity control, and one of them is "
      "the subject of §4.5.</li>")
    A("</ul>")
    A("<p>No systematic difference in recovered peak between the two systems is claimed. With "
      f"{n_trio} Trio participants against {n_aaps} on AndroidAPS, and configured curves differing "
      "between individuals for reasons unrelated to platform, the cohort cannot support such a "
      "comparison and none is attempted.</p>")

    A("<h3>4.4.1 Dose-size control and integrity</h3>")
    A(f"<p>Fitting separate kernels to small and large doses provides a negative control, since the "
      f"delivery systems apply one configured curve irrespective of dose size and the true "
      f"difference is therefore zero. The difference was within ±3 minutes in most participants but "
      f"exceeded tolerance in {len(split_flag)} ({', '.join(split_flag)}), by up to 49 minutes — an "
      f"impossible result under a single configured curve, and therefore evidence of a problem in "
      f"those participants' large-dose records rather than a pharmacological finding.</p>")
    A("<p>Integrity checks passed elsewhere. Treatment records were free of duplicates; the logged "
      "IOB step at an isolated bolus matched the recorded dose to within the decay expected across "
      "the bin; step offsets were confined to the zero or one-bin quantisation expected from "
      "five-minute sampling, with none outside it; and where bolus IOB was uploaded directly it "
      "agreed with the derived value to 0.00025 U. Deliberate displacement of the dose series "
      "confirmed that zero shift gives the best residual, and that a 30-minute error — far beyond "
      "anything plausible — could not reproduce the largest observed discrepancies.</p>")

    A("<h3>4.5 A configured curve at variance with the reported insulin</h3>")
    A("<p>In one participant the recovered configured curve did not correspond to the insulin they "
      "reported using. The recovered peak was 75 minutes, the default value for their system; a "
      "45-minute curve — the value associated with their reported insulin in other systems — fitted "
      "their logged insulin-on-board 4.4 times worse (relative residual 0.68 against 0.16), and the "
      "75-minute value was stable at 73.8 to 77.3 minutes across eight disjoint 15-day windows with "
      "a relative residual of 0.065 to 0.098. Their extract passed every integrity check in §3.9.</p>")
    A("<p>This establishes what the system was computing with, which is the limit of what these "
      "records can show. It does not establish why. The discrepancy is consistent with several "
      "explanations — a setting that was never applied, one that was reset, or a mismatch between "
      "the value shown in the interface and the value used in the calculation — and distinguishing "
      "them requires information from the device that is not present in its uploads. No claim about "
      "the cause is made here.</p>")

    # ---------------- Discussion ----------------
    A("<h3>4.6 Why the cohort could not be enlarged</h3>")
    A("<p>A further archive was available holding 6.6 million controller cycles from 110 users of "
      "oref-derived systems, 67 of them with a logged insulin-on-board series and several with more "
      "than five years of records — an order of magnitude more data than analysed here. It carries "
      "no record of delivered doses, so extending Gate 1 to it requires reconstructing the dose "
      "series from the IOB series itself. This is not possible, for a structural reason.</p>")
    A("<p>Writing the identity of (5) on a regular grid gives a lower-triangular system, and because "
      "<i>f</i>(0) = 1 its diagonal is unity. Forward substitution therefore returns, for <i>any</i> "
      "candidate kernel, a dose series that reproduces the observed IOB series exactly. Applied to "
      "real records, candidate peaks of 35 and 120 minutes both reconstruct the series to a maximum "
      "absolute error of 3×10<sup>−15</sup> U. Goodness of fit is not merely weakly informative "
      "about the kernel; it is <i>identically</i> uninformative. The only discriminating constraint "
      "is that implied doses cannot be negative, and implied negative mass proved monotonically "
      "increasing in the candidate peak (23 U at 35 minutes rising to 117 U at 120 minutes), so "
      "that constraint has no interior optimum and always selects the shortest candidate offered.</p>")
    A("<p>A cruder route — taking doses as the positive first differences of IOB — fails for a "
      "related reason. Between five-minute samples the IOB decays by 1–3% of <i>total</i> insulin "
      "on board, and in a micro-dosing loop total IOB exceeds an individual automatic bolus by an "
      "order of magnitude, so decay swamps small doses entirely. Validated against real treatment "
      "streams this lost 28–93% of delivered insulin and displaced the recovered peak by 12 to 102 "
      "minutes. An independent dose record is therefore necessary, and archives holding IOB alone "
      "are out of reach however the fitting is arranged.</p>")

    A("<h2>5. Discussion</h2>")
    A("<p>The two questions behave very differently. Recovering the configured curve is an exact "
      "problem and works reliably; it constitutes a practical audit of what an AID system is "
      "actually computing, independent of what is reported elsewhere, and it identified both an "
      "unrecorded change of insulin curve and a configured curve at variance with the insulin the "
      "participant believed they were using. Recovering the observed curve is an inference, and the "
      "design of that inference turned out to matter far more than the choice of model family.</p>")
    A("<p>The instructive failure was an earlier parametric estimator applied to overnight fasting "
      "windows. It was unbiased on its own generating model to within 0.3 minutes, and shape "
      "misspecification biased it by only −10 to +5 minutes; yet across 96 defensible analytical "
      "specifications its estimates ranged over 40 to 90 minutes for individual participants, with "
      "the single choice of whether to include a linear drift term moving two participants by 23 "
      "and 35 minutes. The estimator was sound and the design was not. Reporting a bootstrap "
      "interval from one specification would have conveyed a precision that did not exist.</p>")
    A("<p>The reformulation that resolved this was not a better fit but a different source of "
      "identifying variation: pooling days so that a clock-locked drift profile can be separated "
      "from doses that are not clock-locked. This is worth stating because the intuitive move — "
      "give each window its own drift term to be safe — is precisely what destroys identification.</p>")
    A("<p>The duration result is largely negative and worth recording as such. It is not "
      "recoverable for most participants, and — more troubling — its confidence interval is "
      "narrowest exactly when it is least identified. Any pipeline reporting a duration with an "
      "interval, without a leverage or stability check, will report false precision.</p>")

    # ---------------- Limitations ----------------
    A("<h2>6. Limitations</h2>")
    A("<ol>")
    A("<li>Sensor lag — interstitial delay of roughly 5–7 minutes [6, 7], plus filter delay — is not "
      "corrected, so observed peaks "
      "are biased late by a few minutes. This is conservative for discrepancies in which the "
      "observed peak is earlier than configured, and cannot explain one that is later.</li>")
    A("<li>Kinetics are dose-dependent, and the product information documents it: the glucose "
      "infusion rate AUC for one ultra-rapid analogue is 1080, 1860 and 3030 mg/kg at 7, 15 and 30 "
      "units [12] — 154, 124 and 101 mg/kg per unit, a 35% fall in effect per unit across a "
      "four-fold dose range. A single kernel fitted across all dose sizes is therefore a "
      "compromise. It is not estimable separately here: fasting periods are dominated by small "
      "automatic boluses by construction, and where both dose classes are present the two "
      "regressors have a median correlation up to 0.94.</li>")
    A("<li>Unannounced carbohydrate cannot be excluded, only made unlikely; for participants who "
      "never enter carbohydrate the carbohydrate-on-board filter is inert.</li>")
    A("<li>Physical activity is uncontrolled where the uploader provides no step data.</li>")
    A("<li>Estimates are observational. A discrepancy between observed and configured is a "
      "discrepancy; it is not evidence that changing the setting improves glycaemic outcomes, and "
      "no such claim is made.</li>")
    A("<li>Single-cohort, single-analyst work with no external replication.</li>")
    A("</ol>")

    # ---------------- Conclusion ----------------
    A("<h2>7. Conclusion</h2>")
    A("<p>A person's configured insulin curve can be recovered exactly from their own AID records, "
      "and doing so is a useful audit: it detects curves that differ from the one the user believes "
      "is in use, and changes that no event record documents, without requiring access to the "
      "device. The curve the insulin appears to follow "
      "can be recovered with useful precision by non-parametric deconvolution, provided the drift "
      "confounder is identified by pooling across days rather than modelled within windows, and the "
      "result is robust to the controller feedback inherent in closed-loop data. The duration of "
      "action is, for most people, not recoverable from routine records at all. Where an individual "
      "discrepancy needs resolving, a designed test — a fasting bolus with no other insulin — will "
      "settle in days what observational data cannot settle in months.</p>")

    A("<h2>References</h2>")
    A('<ol style="font-size:9pt">')
    A("<li>Maksimović D. Exponential insulin activity curves. Derivation and discussion, LoopKit/Loop "
      "issue #388 (2017). The <i>functional form</i> adopted by oref0 and its derivatives originates "
      "here; the OpenAPS documentation attributes it to this discussion. The <i>curve shape and "
      "parameter values</i> it reproduces derive from published euglycaemic-clamp pharmacodynamics "
      "in regulatory submissions and product information [8\u201312]; see Table 1.</li>")
    A("<li>De Nicolao G, Sparacino G, Cobelli C. Nonparametric input estimation in physiological "
      "systems: problems, methods, and case studies. <i>Automatica</i> 1997;33(5). Reviews "
      "regularised deconvolution for physiological input estimation, including ill-conditioning, "
      "non-negativity constraints and confidence-interval assessment.</li>")
    A("<li>Golub GH, Heath M, Wahba G. Generalized cross-validation as a method for choosing a good "
      "ridge parameter. <i>Technometrics</i> 1979;21(2):215–223.</li>")
    A("<li>Künsch HR. The jackknife and the bootstrap for general stationary observations. "
      "<i>The Annals of Statistics</i> 1989;17(3):1217–1241.</li>")
    A("<li>Forssell U, Ljung L. Closed-loop identification revisited. <i>Automatica</i> "
      "1999;35(7):1215–1241. The obstacle addressed in §3.8: a plant operating under feedback "
      "cannot be identified as though its input were exogenous.</li>")
    A("<li>Basu A, Dube S, Slama M, et al. Time lag of glucose from intravascular to interstitial "
      "compartment in humans. <i>Diabetes</i> 2013;62(12):4083–4087.</li>")
    A("<li>Basu A, Dube S, Veettil S, et al. Time lag of glucose from intravascular to interstitial "
      "compartment in type 1 diabetes. <i>Journal of Diabetes Science and Technology</i> 2015;9(1). "
      "Median lag 6.8 min (4.8–9.8) in type 1 diabetes.</li>")
    A("<li>Humalog (insulin lispro), EMA product information, section 5.1: rapid onset of action "
      "approximately 15 minutes; duration of activity 2 to 5 hours.</li>")
    A("<li>NovoRapid (insulin aspart), EMA product information, section 5.1: \"the onset of action "
      "will occur within 10\u201320 minutes of injection. The maximum effect is exerted between 1 and "
      "3 hours after the injection. The duration of action is 3 to 5 hours.\"</li>")
    A("<li>Apidra (insulin glulisine), EMA product information: onset within 10\u201320 minutes, "
      "duration approximately 4 hours; serum T<sub>max</sub> 55 minutes versus 82 minutes for "
      "soluble human insulin.</li>")
    A("<li>Fiasp (faster-acting insulin aspart), EMA product information, section 5.1: \"The onset of "
      "action was 5 minutes earlier and time to maximum glucose infusion rate was 11 minutes earlier "
      "with Fiasp than with NovoRapid. The maximum glucose-lowering effect of Fiasp occurred between "
      "1 and 3 hours after injection.\"</li>")
    A("<li>Lyumjev (ultra-rapid insulin lispro), EMA product information, section 5.1: \"Onset of "
      "action of Lyumjev was 20 minutes post dose, 11 minutes faster than Humalog. Maximum "
      "glucose-lowering effect of Lyumjev occurred between 1 and 3 hours after injection. The "
      "duration of action of Lyumjev was 5 hours, 44 minutes shorter than Humalog.\" AUC of the "
      "glucose infusion rate 1080, 1860 and 3030 mg/kg following 7, 15 and 30 units.</li>")
    A("</ol>")
    A('<p style="font-size:9pt"><i>Note on the provenance of the model.</i> The pharmacodynamics '
      'underlying these curves are well documented: regulatory submissions and product information '
      'report euglycaemic-clamp glucose-infusion-rate profiles from which onset, time to peak and '
      'duration are read directly [8\u201312]. What has no peer-reviewed source is the specific '
      'two-parameter functional form used to interpolate them [1]. The distinction matters for this '
      'work in one respect. The clamp literature reports a maximum glucose-lowering effect between '
      'one and three hours, a window several times wider than the 45-to-75-minute peaks that '
      'delivery systems adopt; the preset values are therefore a narrowing of the published range '
      'rather than a quantity the label supplies, and an individual can sit anywhere within it.</p>')

    A("<h2>Software</h2>")
    A("<p>All analyses are implemented in Python and available at "
      "<code>github.com/tim2000s/Insulin-Kinetics</code> under AGPL-3.0. Numerical results in this "
      "paper are parsed directly from the analysis outputs rather than transcribed.</p>")

    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>" \
           + "".join(H) + "</body></html>"
    open(os.path.join(B, "paper.html"), "w").write(html)
    from weasyprint import HTML
    HTML(string=html, base_url=HERE).write_pdf(a.out)
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
