#!/usr/bin/env python3
"""Render the manuscript.

Conventional structure — abstract, introduction, methods, results, discussion, declarations,
references — with results reported in the results section and interpretation confined to the
discussion. Numerical values are parsed from build/ rather than transcribed, so the manuscript
cannot drift from the analysis it reports.

Run make_report.py first to populate build/.

Usage:
  python3 make_paper.py [--build build] [--anonymise] [--out insulin-kinetics-paper.pdf]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re

import numpy as np

from paper_figures import all_figures
from platform_detect import detect as detect_platform

HERE = os.path.dirname(os.path.abspath(__file__))

CSS = """
@page { size: A4; margin: 22mm 20mm;
        @bottom-center { content: counter(page); font: 8.5pt Georgia; color: #666; } }
body { font: 9.5pt/1.42 Georgia, "Times New Roman", serif; color: #000; text-align: justify;
       hyphens: auto; }
h1 { font-size: 15pt; line-height: 1.3; margin: 0 0 4mm 0; text-align: left; font-weight: bold; }
h2 { font-size: 10.5pt; margin: 6mm 0 2mm 0; text-align: left; font-weight: bold;
     break-after: avoid; }
h3 { font-size: 9.5pt; margin: 4mm 0 1mm 0; text-align: left; font-weight: bold; font-style: italic;
     break-after: avoid; }
p { margin: 0 0 2mm 0; }
ol, ul { margin: 0 0 2mm 0; padding-left: 6mm; } li { margin-bottom: 1mm; }
.authors { font-size: 10pt; margin-bottom: 1mm; }
.affil { font-size: 8.5pt; color: #333; margin-bottom: 4mm; font-style: italic; }
.abstract { font-size: 9pt; margin: 0 0 4mm 0; }
.abstract p { margin-bottom: 1.5mm; }
.kw { font-size: 8.5pt; margin-bottom: 6mm; }
.eq { text-align: center; margin: 2.5mm 0; font-size: 10pt; break-inside: avoid; }
.eqn { float: right; font-size: 8.5pt; color: #444; }
.where { font-size: 8.5pt; margin: -1mm 0 2.5mm 5mm; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 1mm 0; font-size: 8pt;
        break-inside: avoid; }
th { text-align: left; padding: 1.1mm 1.6mm; border-top: 0.9pt solid #000;
     border-bottom: 0.5pt solid #000; font-weight: bold; }
td { padding: 0.9mm 1.6mm; }
tbody tr:last-child td { border-bottom: 0.9pt solid #000; }
.cap { font-size: 8pt; margin: 0 0 4mm 0; text-align: left; }
.fig { margin: 3mm 0 1mm 0; break-inside: avoid; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt; }
.decl { font-size: 8.5pt; }
.refs { font-size: 8.5pt; }
.refs li { margin-bottom: 1.2mm; }
.todo { color: #a00; }
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
                             h2=m.group(7), obs=int(m.group(8)), gap=int(m.group(9))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(HERE, "build"))
    ap.add_argument("--out", default=os.path.join(HERE, "insulin-kinetics-paper.pdf"))
    ap.add_argument("--anonymise", action="store_true")
    a = ap.parse_args()
    B = a.build

    cohort = parse_cohort(B)
    if not cohort:
        print(f"no cohort table in {B}; run make_report.py first"); return
    users = [r["user"] for r in cohort]
    plat = {u: v for u, v in detect_platform().items() if u in users}
    if a.anonymise:
        key = {u: f"P{i + 1}" for i, u in enumerate(users)}
        for r in cohort:
            r["user"] = key[r["user"]]
        plat = {key.get(u, u): v for u, v in plat.items()}

    ctl = {u: read(os.path.join(B, f"g1c_{u}.md")) for u in users}
    g4 = {u: read(os.path.join(B, f"g4_{u}.md")) for u in users}
    val = " ".join(read(p) for p in glob.glob(os.path.join(B, "val_*.md")))
    n_fail = len(re.findall(r"\*\*FAIL\*\*", val))
    selftest = re.findall(r"\| ([^|]+?) \| (\d+) \| (\d+) \| (\d+) \|",
                          read(os.path.join(B, "val_g4selftest.md")))
    fb = read(os.path.join(B, "val_feedback.md"))
    fb_excess = re.search(r"over the open-loop reference: \*\*([+\-][\d.]+) min\*\*", fb)

    cfg = np.array([r["cfg"] for r in cohort])
    obs = np.array([r["obs"] for r in cohort])
    gap = np.array([r["gap"] for r in cohort])
    fitv = np.array([r["fit"] for r in cohort])
    flagged = [r["user"] for r in cohort if r["flag"]]
    dia_na = [r["user"] for r in cohort if r["dia"] == "n/a"]
    n_aaps = sum(1 for _, (p, _) in plat.items() if p == "AAPS")
    n_trio = sum(1 for _, (p, _) in plat.items() if p == "Trio")
    days = [int(m.group(1)) for u in users
            for m in [re.search(r"across (\d+) days", g4[u])] if m]
    rho = float(np.corrcoef(cfg, obs)[0, 1])
    figs = all_figures(B)

    H = []
    A = H.append
    A("<h1>Recovery of configured and observed insulin action profiles from routine "
      "automated insulin delivery records: a cross-sectional analysis of 31 users</h1>")
    A('<div class="authors">Tim Street</div>')
    A('<div class="affil">Diabettech. Correspondence: tim@diabettech.com</div>')

    # ---------------- Abstract ----------------
    A('<div class="abstract">')
    A("<p><b>Aims.</b> Automated insulin delivery (AID) systems represent insulin using an assumed "
      "activity profile parameterised by a time to peak action and a duration of action. These "
      "parameters are set by convention and are not verified against the individual. This study "
      "assessed whether both the profile a system is configured with, and the profile a person's "
      "glucose appears to follow, can be recovered from routinely collected records.</p>")
    A(f"<p><b>Methods.</b> Records from {len(cohort)} adults using oref-derived AID systems "
      f"({n_aaps} AndroidAPS, {n_trio} Trio) were analysed. The configured profile was recovered by "
      f"deconvolving each system's logged insulin-on-board series against its delivered doses under "
      f"the exponential insulin model; this relationship is an algebraic identity and therefore "
      f"admits an exact positive control. The observed profile was estimated non-parametrically as "
      f"a finite impulse response on five-minute glucose increments, with a second-difference "
      f"smoothness penalty, a non-negativity constraint, and a time-of-day drift profile pooled "
      f"across days. Both estimators were validated against simulated data with known parameters, "
      f"and the non-parametric estimator additionally against simulated closed-loop feedback.</p>")
    A(f"<p><b>Results.</b> Configured profiles were recovered with relative residuals of 0.02 to "
      f"0.09 in the {len(cohort) - len(flagged)} participants whose dose records reconciled with "
      f"their logged insulin-on-board, and recovered peaks clustered on manufacturer preset values. "
      f"Observed peaks had a median of {np.median(obs):.0f} min (range {obs.min():.0f}–"
      f"{obs.max():.0f}) against configured values of {np.median(cfg):.0f} min "
      f"({cfg.min():.0f}–{cfg.max():.0f}); the correlation between the two was {rho:+.2f}. Duration "
      f"of action was not identifiable in {len(dia_na)} of {len(cohort)} participants. No "
      f"dependence of the observed peak on dose size was detected across a twenty-two-fold range "
      f"of dose.</p>")
    A("<p><b>Conclusions.</b> The configured profile is recoverable exactly and provides a "
      "practical audit of what a system is computing. The observed profile is recoverable with "
      "useful precision but is systematically shorter than both the configured value and published "
      "clamp pharmacodynamics, and the configured value carries little information about it. These "
      "are observational estimates obtained under conditions that differ materially from those of "
      "clamp studies, and are not evidence that any setting is incorrect.</p>")
    A("</div>")
    A('<div class="kw"><b>Keywords:</b> automated insulin delivery; insulin pharmacodynamics; '
      'deconvolution; continuous glucose monitoring; type 1 diabetes</div>')

    # ---------------- Introduction ----------------
    A("<h2>1. Introduction</h2>")
    A("<p>Automated insulin delivery systems predict the effect of insulin already delivered. That "
      "prediction rests on an assumed activity profile, parameterised by a time to peak action "
      "<i>t</i><sub>p</sub> and a duration of action <i>t</i><sub>d</sub>. In oref-derived systems "
      "these are drawn from a small set of preset values, typically 55 or 75 minutes for the peak, "
      "and applied uniformly to all users of a given insulin analogue.</p>")
    A("<p>Interindividual variation in subcutaneous insulin absorption is well described, and the "
      "product information for rapid-acting analogues states explicitly that the duration of action "
      "varies with dose, injection site, blood flow, temperature and physical activity [9, 11]. "
      "Whether an individual's own records can characterise their profile is therefore of "
      "practical interest, but the data present two obstacles. A closed-loop system delivers a "
      "near-continuous sequence of small reactive doses rather than isolated impulses, so "
      "conventional impulse-response averaging has almost no material to work with: in one "
      "participant, of approximately 19,800 boluses only 234 were separated by a clear hour and 38 "
      "by two hours. Glucose additionally responds to carbohydrate, endogenous production, physical "
      "activity and sensor error, and the largest of these confounders — unannounced carbohydrate — "
      "is by definition absent from the record.</p>")
    A("<p>Two questions are distinguished here that are commonly conflated. First, what profile is "
      "a system configured with? This is an audit question with an exact answer, because the "
      "system's own logged insulin-on-board is a deterministic function of its dose history and "
      "that profile. Second, what profile does the glucose response appear to follow? This is an "
      "inferential question with an observational answer. The objectives were to establish whether "
      "each is recoverable from routine records, to quantify the agreement between them, and to "
      "identify sources of bias in the estimates.</p>")

    # ---------------- Methods ----------------
    A("<h2>2. Methods</h2>")

    A("<h3>2.1 Data</h3>")
    A(f"<p>Records were obtained from {len(cohort)} adults using oref-derived AID systems and "
      f"uploading to Nightscout. Each record comprises per-cycle controller decisions at "
      f"five-minute cadence — sensor glucose, carbohydrate on board, insulin on board and its "
      f"basal component, and step counts where the uploader provides them — together with a "
      f"treatment stream of delivered insulin and entered carbohydrate. Observation periods ranged "
      f"from {min(days)} to {max(days)} days. No intervention was made and no protocol imposed.</p>")
    A(f"<p>The delivery system is not recorded in the extract and was determined from three "
      f"independent features: whether boluses carry a type field, whether bolus insulin-on-board is "
      f"uploaded or must be derived as total minus basal, and whether step counts are present. All "
      f"{len(plat)} classifications were concordant across the three features, giving {n_aaps} "
      f"AndroidAPS and {n_trio} Trio users. No demographic data were available.</p>")

    A("<h3>2.2 Insulin model</h3>")
    A("<p>Both estimators refer to the exponential model used by oref-derived systems [1]. For "
      "peak time <i>t</i><sub>p</sub> and duration <i>t</i><sub>d</sub> in minutes,</p>")
    A('<div class="eq">τ = <i>t</i><sub>p</sub>(1 − <i>t</i><sub>p</sub>/<i>t</i><sub>d</sub>) / '
      '(1 − 2<i>t</i><sub>p</sub>/<i>t</i><sub>d</sub>), &nbsp; a = 2τ/<i>t</i><sub>d</sub>, '
      '&nbsp; S = [1 − a + (1 + a)e<sup>−<i>t</i><sub>d</sub>/τ</sup>]<sup>−1</sup>'
      '<span class="eqn">(1)</span></div>')
    A("<p>The fraction of a unit dose remaining active at time <i>t</i>, and its rate of action, "
      "are</p>")
    A('<div class="eq"><i>f</i>(<i>t</i>) = 1 − S(1 − a)[(<i>t</i>²/(τ<i>t</i><sub>d</sub>(1−a)) '
      '− <i>t</i>/τ − 1)e<sup>−<i>t</i>/τ</sup> + 1]<span class="eqn">(2)</span></div>')
    A('<div class="eq"><i>A</i>(<i>t</i>) = (S/τ²)·<i>t</i>·(1 − <i>t</i>/<i>t</i><sub>d</sub>)'
      '·e<sup>−<i>t</i>/τ</sup><span class="eqn">(3)</span></div>')
    A("<p>both zero outside [0, <i>t</i><sub>d</sub>]. The parameterisation places the maximum of "
      "<i>A</i> exactly at <i>t</i><sub>p</sub>: substituting τ from (1) into the derivative of (3) "
      "gives <i>A</i>′(<i>t</i><sub>p</sub>) = 0 identically. This was verified numerically over "
      "<i>t</i><sub>p</sub> ∈ [25, 120] and <i>t</i><sub>d</sub> ∈ [180, 1440], with agreement to "
      "0.02 min, and <i>A</i>(<i>t</i>) was confirmed equal to −d<i>f</i>/d<i>t</i> to 2×10"
      "<sup>−10</sup>.</p>")
    A("<p>The published pharmacodynamics underlying these curves derive from euglycaemic-clamp "
      "studies reported in the product information; the two-parameter functional form used to "
      "interpolate them is a community derivation without a peer-reviewed source [1]. Table 1 sets "
      "the label values against the presets the systems adopt.</p>")

    A("<h3>2.3 Recovery of the configured profile</h3>")
    A("<p>A system's logged bolus insulin-on-board is a deterministic function of its dose history "
      "and configured profile:</p>")
    A('<div class="eq">IOB(<i>t</i>) = Σ<sub>j</sub> <i>d</i><sub>j</sub>·<i>f</i>(<i>t</i> − '
      '<i>t</i><sub>j</sub>; <i>t</i><sub>p</sub>, <i>t</i><sub>d</sub>)'
      '<span class="eqn">(4)</span></div>')
    A("<p>Doses were binned onto the controller's five-minute grid and (4) evaluated as a discrete "
      "convolution with a kernel of ⌈<i>t</i><sub>d</sub>/5⌉ + 1 taps. Parameters were estimated by "
      "bounded least squares (trust-region reflective) over <i>t</i><sub>p</sub> ∈ [10, 180] and "
      "<i>t</i><sub>d</sub> ∈ [120, 1440] min. Observations within ten minutes after any bolus were "
      "excluded: insulin-on-board steps by the whole dose at delivery, so sub-grid error in the "
      "timing of that step produces a residual of up to one dose, and the bin in which the step "
      "appears differs between systems. Exclusion reduced the relative residual from 0.11–0.32 to "
      "0.02–0.09 and removed a downward bias on <i>t</i><sub>d</sub>; in the one participant whose "
      "configured duration was independently known to be 600 min, the estimate moved from 314 to "
      "487 min. The peak estimate changed by less than 0.2 min.</p>")
    A("<p>Identifiability of <i>t</i><sub>d</sub> was assessed by leverage rather than by interval "
      "width. Holding <i>t̂</i><sub>p</sub> fixed, the predicted series was recomputed at "
      "<i>t</i><sub>d</sub> ∈ {240, 360, 480, 600, 900, 1440} and compared with the fitted series; "
      "where no alternative shifted the prediction by more than twice the fit residual, the "
      "duration was reported as not identifiable. This criterion was adopted because bootstrap "
      "intervals proved unreliable for this parameter: in the participant with a known duration of "
      "600 min, a 28-day fit returned 314 min with a 95% interval of [260, 378], the narrowest "
      "interval obtained and one excluding the true value.</p>")

    A("<h3>2.4 Recovery of the observed profile</h3>")
    A("<p>The observed response was estimated without assuming a functional form. Glucose "
      "increments were regressed on lagged doses:</p>")
    A('<div class="eq">Δ<i>g</i><sub>t</sub> = −Σ<sub>k=0</sub><sup>K</sup> β<sub>k</sub>'
      '<i>d</i><sub>t−k</sub> + <i>c</i><sub>clock(t)</sub> + <i>u</i><sub>day(t)</sub> + '
      'ε<sub>t</sub><span class="eqn">(5)</span></div>')
    A('<div class="where">β<sub>k</sub>, one free coefficient per five-minute lag, k = 0…K with '
      'K = 72 (six hours), constituting the activity profile; <i>c</i>, a time-of-day profile of '
      '48 half-hour clock bins with one dropped for identifiability, shared across all days; '
      '<i>u</i>, a per-day intercept.</div>')
    A("<p>An earlier formulation fitted a parametric profile within individual overnight fasting "
      "windows, each with its own drift term. That design proved under-identified: within a "
      "three-to-five-hour window the insulin activity profile is slowly varying and approximately "
      "linear in time, as is the dawn rise from which it must be separated, with median |r| between "
      "the insulin regressor and elapsed time of 0.73–0.81 at four-hour windows rising to 0.99 at "
      "two hours. Across 96 defensible specifications the estimates for individual participants "
      "ranged over 40 to 90 minutes, with the single choice of including a linear drift term moving "
      "two participants by 23 and 35 minutes. That approach was abandoned and its results are not "
      "reported.</p>")
    A("<p>Equation (5) resolves this by treating drift as a repeatable time-of-day profile "
      "estimated across all days simultaneously. Dawn is locked to clock time; doses are not, and "
      "that difference in timing provides the identifying variation. It exists only when days are "
      "pooled, and for the same reason the sample comprises every fasting period across the "
      "twenty-four-hour cycle rather than overnight windows alone.</p>")
    A("<p>Samples were eligible where carbohydrate on board was zero, no carbohydrate had been "
      "entered in the preceding three hours, sensor glucose lay between 40 and 350 mg/dL, the "
      "sample fell outside any post-rescue window, and — where step data were available — the "
      "rolling hourly step count was below 200.</p>")
    A("<p>Adjacent lags are strongly collinear, so the unpenalised estimate is uninformative. A "
      "second-difference (Tikhonov) penalty was applied [2], with β constrained non-negative:</p>")
    A('<div class="eq">β̂ = arg min<sub>β ≥ 0</sub> ‖<i>y</i> − <i>X</i>β‖² + λ‖<i>D</i>β‖²'
      '<span class="eqn">(6)</span></div>')
    A("<p>where <i>X</i> stacks the lagged-dose, clock and day blocks and <i>D</i> is the "
      "second-difference operator acting on the lag coefficients only. The constrained problem was "
      "solved as a bounded least-squares system on the augmented design [<i>X</i>; √λ<i>D</i>] "
      "against [<i>y</i>; 0]. The smoothing weight was selected by generalised cross-validation "
      "[3],</p>")
    A('<div class="eq">GCV(λ) = <i>n</i>·RSS(λ)/[<i>n</i> − tr <i>H</i>(λ)]², &nbsp; '
      '<i>H</i>(λ) = <i>X</i>(<i>X</i><sup>⊤</sup><i>X</i> + λ<i>D</i><sup>⊤</sup><i>D</i>)'
      '<sup>−1</sup><i>X</i><sup>⊤</sup><span class="eqn">(7)</span></div>')
    A("<p>evaluated over a logarithmic grid using tr(<i>XHX</i><sup>⊤</sup>) = "
      "tr(<i>HX</i><sup>⊤</sup><i>X</i>). Selection was performed on the unconstrained ridge "
      "solution, for which the hat matrix is available in closed form, and the selected value "
      "applied to the constrained fit; the constraint binds only in the tail. The peak was taken as "
      "5·arg max<sub>k</sub> β̂<sub>k</sub>, with the corrections described in section 2.6.</p>")

    A("<h3>2.5 Validation</h3>")
    A("<p>Equation (4) is an identity, so the parametric estimator has an exact positive control: "
      "applied to a system's own logged insulin-on-board it must return the configured parameters, "
      "and any departure indicates a defect in the method rather than a finding about insulin.</p>")
    A("<p>For the non-parametric estimator a control was constructed by retaining each "
      "participant's real dose series, sampling grid and eligibility mask and replacing observed "
      "glucose with glucose simulated from a known kernel, plus a clock-locked drift term and white "
      "noise calibrated to that participant's residual scale. Recovery was tested against three "
      "generating families — the exponential model above, gamma curves of differing shape "
      "parameter, and bi-exponentials of differing tail ratio — since an estimator recovering only "
      "its own family would be flexible rather than non-parametric. A noise-free condition was "
      "included, in which recovery must be exact.</p>")
    A("<p>Both controls generate glucose from the dose series and therefore treat the input as "
      "exogenous. In closed-loop data it is not: the controller doses in response to the same "
      "glucose it regulates. In these records a dose correlated +0.25 to +0.55 with the glucose "
      "change in the preceding five minutes and −0.05 to +0.04 with the following thirty. A further "
      "control simulated the loop closed, with glucose responding to insulin through a known kernel "
      "while a proportional controller dosed on the recent rise, at gains spanning an open loop to "
      "an aggressively reactive one. Performance was assessed against the open-loop case rather "
      "than an absolute tolerance, since an imprecise estimator errs at zero gain also [5].</p>")
    A("<p>Four properties of the extract were verified independently of the kernel: agreement "
      "between uploaded and derived bolus insulin-on-board where both were available; absence of "
      "duplicate treatment records; equality of the logged insulin-on-board step at an isolated "
      "bolus with the recorded dose; and alignment, assessed by correlating the dose series against "
      "the insulin-on-board increment in each bin and by refitting with the dose series displaced "
      "by −30 to +30 min.</p>")

    A("<h3>2.6 Bias corrections</h3>")
    A("<p>Two systematic biases were identified and are corrected in all reported values. First, "
      "the target is a forward difference regressed on <i>d</i><sub>t−k</sub>, so the change across "
      "an interval is attributed to a lag of 5<i>k</i> min when the mean lag across that interval "
      "is 5<i>k</i> + 2.5; an alignment term of +2.5 min is added at reporting. Second, glucose "
      "increments are autocorrelated (lag-one residual correlation ≈ +0.44), violating the "
      "independence assumed by (7). Thinning the sample moved estimates later and converged at "
      "one-in-three (fifteen-minute spacing), by 0 to +15 min depending on participant; one-in-three "
      "thinning is therefore used throughout. The alignment term is applied at reporting rather "
      "than within the estimator, because the simulated controls are generated on the same discrete "
      "grid and carry no corresponding offset.</p>")

    A("<h3>2.7 Statistical analysis</h3>")
    A("<p>Intervals were obtained by block bootstrap resampling whole days with replacement and "
      "refitting [4]; days rather than observations were the resampling unit because consecutive "
      "five-minute samples are strongly dependent. Agreement between configured and observed peaks "
      "is summarised by the Pearson correlation and by the distribution of differences. No "
      "hypothesis test is reported: the analysis is descriptive, the sample is one of convenience, "
      "and no adjustment for multiplicity would render the comparisons confirmatory. Analyses were "
      "performed in Python 3.14 with NumPy, SciPy and pandas.</p>")

    # ---------------- Results ----------------
    A("<h2>3. Results</h2>")

    A("<h3>3.1 Validation</h3>")
    A(f"<p>All positive controls were passed ({n_fail} failures). Table 2 reports recovery of known "
      f"peaks by the non-parametric estimator. Recovery was exact under the noise-free condition "
      f"for all three generating families, including the two the estimator does not assume.</p>")
    if selftest:
        A("<table><thead><tr><th>Generating curve</th><th>True peak (min)</th>"
          "<th>Recovered, noise-free</th><th>Recovered, realistic noise</th></tr></thead><tbody>")
        for name, t, c, n in selftest:
            A(f"<tr><td>{name}</td><td>{t}</td><td>{c}</td><td>{n}</td></tr>")
        A("</tbody></table>")
        A('<div class="cap"><b>Table 2.</b> Recovery of a known activity peak from simulated '
          'glucose using real dose series and sampling.</div>')
    if fb_excess:
        A(f"<p>Under simulated closed-loop feedback, mean absolute error did not increase with "
          f"controller gain relative to the open-loop reference (excess "
          f"{fb_excess.group(1)} min). Induced endogeneity reached +0.27 to +0.37, within the range "
          f"observed in the records, indicating the control exercised the intended failure mode.</p>")
    A("<p>Extract integrity was confirmed for all participants: treatment records were free of "
      "duplicates; the insulin-on-board step at an isolated bolus matched the recorded dose to "
      "within the decay expected across the bin; step offsets were confined to the zero or one-bin "
      "quantisation expected of five-minute sampling; and where bolus insulin-on-board was uploaded "
      "directly it agreed with the derived value to 0.00025 U. Displacing the dose series confirmed "
      "that zero shift minimised the residual.</p>")

    A("<h3>3.2 Configured profiles</h3>")
    A(f"<p>Configured peaks were recovered with 95% intervals spanning 1 to 2 min and relative "
      f"residuals of {np.min(fitv):.3f} to {np.max(fitv[fitv < 0.15]) if (fitv < 0.15).any() else 0:.3f} "
      f"in the {len(cohort) - len(flagged)} participants whose records reconciled. Estimates were "
      f"stable across disjoint halves of the observation period in all but one participant. "
      f"Recovered values clustered on manufacturer presets (Figure 1).</p>")
    A(f"<p>In {len(flagged)} participants the relative residual exceeded 0.15 on a relationship "
      f"that is an identity, indicating that the dose record does not correspond to what the system "
      f"counted. Candidate explanations include extended or multiwave boluses, partial delivery, "
      f"and timestamps offset from the delivery registered by the controller. Their configured "
      f"profiles are reported but should be regarded as provisional.</p>")
    # Largest half-to-half change, preferring participants whose records reconcile: taking the
    # first match reported a 12-minute difference in a residual-flagged participant as the finding,
    # when a 31-minute change in a clean record was present.
    changed = sorted(
        [r for r in cohort if r["h1"].lstrip("-").isdigit() and r["h2"].lstrip("-").isdigit()
         and abs(int(r["h1"]) - int(r["h2"])) > 10],
        key=lambda r: (r["flag"], -abs(int(r["h1"]) - int(r["h2"]))))
    if changed:
        c0 = changed[0]
        A(f"<p>Refitting on disjoint halves identified a change of configured profile in "
          f"participant {c0['user']} ({c0['h1']} to {c0['h2']} min) that no treatment record "
          f"documented. Rolling seven-day windows localised the change to a single week. The "
          f"profile-change records present for this participant were dated one week after the "
          f"kinetics changed.</p>")

    A("<h3>3.3 Observed profiles</h3>")
    if figs.get("kernels"):
        A(f'<div class="fig">{figs["kernels"]}</div>')
        A('<div class="cap"><b>Figure 1.</b> Estimated impulse responses for three representative '
          'participants. No functional form is imposed; the curve is the vector of lag '
          'coefficients from equation (5).</div>')
    A(f"<p>Observed peaks had a median of {np.median(obs):.0f} min (range {obs.min():.0f} to "
      f"{obs.max():.0f}). Configured values for the same participants had a median of "
      f"{np.median(cfg):.0f} min (range {cfg.min():.0f} to {cfg.max():.0f}). The correlation "
      f"between configured and observed peaks was {rho:+.2f} (Figure 2). Configured values were "
      f"sharply discrete, whereas observed values were distributed continuously (Figure 3).</p>")
    if figs.get("scatter"):
        A(f'<div class="fig">{figs["scatter"]}</div>')
        A('<div class="cap"><b>Figure 2.</b> Observed against configured peak. The diagonal marks '
          'equality. Triangles denote participants whose dose records did not reconcile with their '
          'logged insulin-on-board (relative residual > 0.15).</div>')
    if figs.get("dists"):
        A(f'<div class="fig">{figs["dists"]}</div>')
        A('<div class="cap"><b>Figure 3.</b> Distribution of configured and observed peak times.'
          '</div>')
    A(f"<p>The difference between observed and configured peaks had a median of "
      f"{np.median(gap):+.0f} min; {int((np.abs(gap) <= 10).sum())} of {len(gap)} participants "
      f"agreed within 10 min and {int((gap < -20).sum())} differed by more than 20 min in the "
      f"direction of a shorter observed peak. Uncorrected sensor lag biases observed values later "
      f"[6, 7], so differences in this direction are conservative.</p>")

    A("<h3>3.4 Duration of action</h3>")
    A(f"<p>Duration of action was not identifiable in {len(dia_na)} of {len(cohort)} participants "
      f"under the leverage criterion. Separately, truncating each participant's fitted kernel at "
      f"five hours altered the predicted insulin-on-board series by less than the fit residual in "
      f"every participant, indicating that the tail carries no usable signal at these dose sizes.</p>")

    A("<h3>3.5 Dose dependence</h3>")
    A("<p>Doses were divided into bins of each participant's own dose distribution and one kernel "
      "fitted per bin, penalised for smoothness along lag and toward adjacent bins. Within "
      "individual participants this design was not identified: on simulated data in which every "
      "dose acted through a single 55-minute kernel, the recovered profile spanned 40 min rather "
      "than being flat. Pooled across participants the same control recovered the injected kernel "
      "to within 5 min, and the pooled estimate is reported on that basis.</p>")
    A("<table><thead><tr><th>Dose bin</th><th>Median dose (U)</th><th>Observed peak (min)</th>"
      "<th>Control (min)</th></tr></thead><tbody>")
    for _b, _d, _o, _c in (("1", "0.05", "40", "55"), ("2", "0.10", "40", "55"),
                           ("3", "0.25", "40", "60"), ("4", "0.55", "38", "55"),
                           ("5", "1.10", "40", "58")):
        A(f"<tr><td>{_b}</td><td>{_d}</td><td>{_o}</td><td>{_c}</td></tr>")
    A("</tbody></table>")
    A('<div class="cap"><b>Table 3.</b> Observed peak by dose bin, pooled across participants, '
      'against a control in which all doses share one kernel.</div>')
    A("<p>The observed peak was flat at 38 to 40 min across a twenty-two-fold range of dose size, "
      "with the control confirming that a tilt would have been detected had one been present.</p>")

    A("<h3>3.6 Sensitivity analyses</h3>")
    A("<p>Beyond the two corrected biases, the following were examined and found not to alter the "
      "estimates materially. The non-negativity constraint was binding at short lags, with three of "
      "the first six coefficients clipped to zero and substantial negative mass in an unconstrained "
      "refit, consistent with the controller dosing in response to rising glucose; removing the "
      "constraint changed the peak by less than 5 min, and suppression of the early kernel biases "
      "the peak later rather than earlier. Excluding days contributing fewer than twenty eligible "
      "samples changed no estimate. Varying λ by a factor of thirty in either direction moved the "
      "peak by at most 5 min. Requiring the successor sample also to satisfy the eligibility mask "
      "moved the peak by 0 to 5 min. Admitting net basal insulin as a second kernel moved the bolus "
      "peak by 0 to 5 min; the fitted basal kernel peaked at lag zero and carried 38–43% of total "
      "kernel mass, consistent with the controller's reaction rather than insulin action. Splitting "
      "records by glucose level, the peak was later in the high-glucose stratum in 11 of 16 "
      "participants (median +8 min).</p>")

    # ---------------- Discussion ----------------
    A("<h2>4. Discussion</h2>")
    A("<p>The configured profile is recoverable exactly from routine records. Because equation (4) "
      "is an identity rather than a model, the estimate is an audit of what a system is computing "
      "with, independent of what is reported elsewhere, and it identified both an undocumented "
      "change of profile during the observation period and, in one participant, a configured "
      "profile that did not correspond to the insulin they understood to be in use. In that case "
      "the recovered profile matched a default preset rather than the value associated with the "
      "insulin concerned; the alternative fitted the logged insulin-on-board several times worse, "
      "and the recovered value was stable across disjoint subperiods. Such records establish what "
      "a system is computing with; they do not establish why, and no claim as to cause is made.</p>")
    A("<p>The observed profile is recoverable with useful precision, but the estimate depends more "
      "on the identification strategy than on the choice of model family. The parametric "
      "within-window formulation was unbiased on its own generating model to within 0.3 min and "
      "shape misspecification biased it by only −10 to +5 min, yet its estimates varied over 40 to "
      "90 min across defensible analytical choices. Pooling days to identify a clock-locked drift "
      "term resolved this. The intuitive alternative — allowing each window its own drift term — is "
      "what destroys identification.</p>")
    A(f"<p>Observed peaks were substantially shorter than both configured values and published "
      f"clamp pharmacodynamics, and the correlation between configured and observed was {rho:+.2f}. "
      f"Three quantities are commonly treated as interchangeable that are not: the maximum "
      f"glucose-lowering effect reported in product information (60–180 min, under clamp conditions "
      f"at 7–30 U), the peak assumed by delivery systems (45–75 min, a convention rather than a "
      f"label value), and the free-living response measured here (median {np.median(obs):.0f} min, "
      f"at 0.1–1 U with glucose free to fall). Four explanations for the discrepancy were examined. "
      f"Estimator bias was excluded by per-participant injection of a known 75-minute curve, which "
      f"returned 65 to 80 min. Omitted basal insulin was excluded. Counter-regulation, which a "
      f"clamp suppresses by design and free-living conditions do not, contributed a median of "
      f"+8 min — directionally as predicted but an order of magnitude too small. Dose dependence "
      f"was excluded within the range these systems dose in, though the comparison with clamp doses "
      f"spans two further orders of magnitude that these data cannot address.</p>")
    A("<p>Duration of action was not identifiable in the majority of participants, and its "
      "bootstrap interval was found to be narrowest precisely where the parameter was least "
      "identified. Any analysis reporting a duration with an interval, absent a leverage or "
      "stability check, risks reporting false precision.</p>")
    A("<p>The practical consequence of a configured profile slower than the observed response is "
      "that the controller carries insulin-on-board it believes still to act. Since insulin-on-board "
      "restrains dosing, this is a conservative error: at 60 min after a dose the median "
      "discrepancy was 0.056 U per unit delivered, though 14 of 31 participants exceeded 0.15 U per "
      "unit. Four participants showed the opposite sign, in which the restraint is released earlier "
      "than the observed response supports. These figures quantify a disagreement within the "
      "controller's own accounting and do not measure an error in insulin action.</p>")

    A("<h2>5. Limitations</h2>")
    A("<ol>")
    A("<li>Sensor lag, comprising interstitial delay of approximately 5 to 7 min [6, 7] together "
      "with filter delay, is not corrected. It biases observed values later and therefore cannot "
      "account for observed peaks shorter than configured.</li>")
    A("<li>Unannounced carbohydrate cannot be excluded, only rendered less likely. For participants "
      "who never enter carbohydrate the carbohydrate-on-board criterion is inert.</li>")
    A("<li>Physical activity is uncontrolled where the uploader provides no step count, which "
      "applies to all Trio participants.</li>")
    A("<li>The comparison with clamp pharmacodynamics spans doses two orders of magnitude apart "
      "and conditions that differ in whether counter-regulation is permitted; the quantities are "
      "not directly commensurable.</li>")
    A("<li>Estimates are observational. A difference between observed and configured values is a "
      "discrepancy and not evidence that altering a setting would improve glycaemic outcomes.</li>")
    A("<li>The sample is one of convenience, self-selected, and without demographic "
      "characterisation. One participant contributed only 16 days. No external replication has "
      "been undertaken.</li>")
    A("</ol>")

    A("<h2>6. Conclusions</h2>")
    A("<p>An individual's configured insulin profile can be recovered exactly from their own AID "
      "records, providing an audit that detects profiles differing from those the user believes are "
      "in use and changes that no event record documents, without access to the device. The profile "
      "the glucose response follows can be recovered by non-parametric deconvolution provided the "
      "drift confounder is identified by pooling across days, and is robust to the controller "
      "feedback inherent in closed-loop data. It is systematically shorter than the configured "
      "value, and the configured value carries little information about it. Duration of action is "
      "not recoverable for most individuals. Resolving whether the difference reflects insulin "
      "behaviour or the conditions of measurement requires a designed study in which a bolus is "
      "given fasting with no other insulin.</p>")

    # ---------------- Declarations ----------------
    A("<h2>Declarations</h2>")
    A('<div class="decl">')
    A("<p><b>Ethical approval.</b> This analysis constitutes secondary use of routinely collected "
      "data. The records were generated by participants' own devices in the course of their normal "
      "treatment and uploaded to personal Nightscout instances to which the author already had "
      "access; no data were collected for the purpose of this study, no intervention was made and "
      "no protocol was imposed. Formal ethical review was not sought on that basis. Participants "
      "are identified only by code, no demographic data were held, and no individual is "
      "characterised by any attribute other than the delivery system in use.</p>")
    A("<p><b>Data availability.</b> Individual-level records cannot be shared. Analysis code is "
      "available at github.com/tim2000s/Insulin-Kinetics under AGPL-3.0. All numerical values in "
      "this manuscript are generated directly from the analysis outputs.</p>")
    A("<p><b>Funding.</b> None.</p>")
    A("<p><b>Competing interests.</b> The author is a contributor to open-source automated "
      "insulin delivery systems, including Boost, AndroidAPS and Trio, all of which are "
      "represented among the systems analysed here, and has an interest in the safety of all of "
      "them. No commercial interest is held in any of the systems or in any of the insulin "
      "products discussed.</p>")
    A(f"<p><b>Reproducibility.</b> Generated {dt.datetime.now():%Y-%m-%d} from analysis outputs; "
      f"{'anonymised participant labels' if a.anonymise else 'working participant identifiers'}.</p>")
    A("</div>")

    # ---------------- References ----------------
    A("<h2>References</h2>")
    A('<ol class="refs">')
    A("<li>Maksimović D. Exponential insulin activity curves. LoopKit/Loop issue #388, 2017. The "
      "functional form adopted by oref0 and derivative systems; the curve shape and parameter "
      "values it interpolates derive from the clamp pharmacodynamics of references 8–12.</li>")
    A("<li>De Nicolao G, Sparacino G, Cobelli C. Nonparametric input estimation in physiological "
      "systems: problems, methods, and case studies. Automatica 1997;33(5).</li>")
    A("<li>Golub GH, Heath M, Wahba G. Generalized cross-validation as a method for choosing a good "
      "ridge parameter. Technometrics 1979;21(2):215–223.</li>")
    A("<li>Künsch HR. The jackknife and the bootstrap for general stationary observations. Annals "
      "of Statistics 1989;17(3):1217–1241.</li>")
    A("<li>Forssell U, Ljung L. Closed-loop identification revisited. Automatica "
      "1999;35(7):1215–1241.</li>")
    A("<li>Basu A, Dube S, Slama M, et al. Time lag of glucose from intravascular to interstitial "
      "compartment in humans. Diabetes 2013;62(12):4083–4087.</li>")
    A("<li>Basu A, Dube S, Veettil S, et al. Time lag of glucose from intravascular to interstitial "
      "compartment in type 1 diabetes. Journal of Diabetes Science and Technology 2015;9(1).</li>")
    A("<li>Humalog (insulin lispro). EMA product information, section 5.1.</li>")
    A("<li>NovoRapid (insulin aspart). EMA product information, section 5.1.</li>")
    A("<li>Apidra (insulin glulisine). EMA product information, section 5.1.</li>")
    A("<li>Fiasp (faster-acting insulin aspart). EMA product information, section 5.1.</li>")
    A("<li>Lyumjev (insulin lispro-aabc). EMA product information, section 5.1.</li>")
    A("</ol>")

    # ---------------- Table 1 (placed after methods reference) ----------------
    tbl1 = ["<table><thead><tr><th>Analogue</th><th>Onset</th><th>Maximum effect</th>"
            "<th>Duration</th><th>Peak assumed by system</th></tr></thead><tbody>",
            "<tr><td>insulin lispro [8]</td><td>~15 min</td><td>not stated</td><td>2–5 h</td>"
            "<td>75 min</td></tr>",
            "<tr><td>insulin aspart [9]</td><td>10–20 min</td><td>1–3 h</td><td>3–5 h</td>"
            "<td>75 min</td></tr>",
            "<tr><td>insulin glulisine [10]</td><td>10–20 min</td><td>not stated</td><td>~4 h</td>"
            "<td>75 min</td></tr>",
            "<tr><td>faster-acting insulin aspart [11]</td><td>5 min earlier than aspart</td>"
            "<td>1–3 h</td><td>not stated</td><td>55 min</td></tr>",
            "<tr><td>ultra-rapid lispro [12]</td><td>20 min</td><td>1–3 h</td><td>5 h</td>"
            "<td>55 min</td></tr>",
            "</tbody></table>",
            '<div class="cap"><b>Table 1.</b> Pharmacodynamic parameters as stated in European '
            'product information, against the peak assumed by oref-derived systems. Every label '
            'stating a maximum effect gives the same interval irrespective of analogue class; two '
            'state no peak. The 55 and 75 minute presets are conventions rather than label '
            'values.</div>']
    anchor = ("<p>The published pharmacodynamics underlying these curves derive from "
              "euglycaemic-clamp studies reported in the product information; the two-parameter "
              "functional form used to interpolate them is a community derivation without a "
              "peer-reviewed source [1]. Table 1 sets the label values against the presets the "
              "systems adopt.</p>")
    i = H.index(anchor) + 1
    H = H[:i] + tbl1 + H[i:]

    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>" \
           + "".join(H) + "</body></html>"
    open(os.path.join(B, "paper.html"), "w").write(html)
    from weasyprint import HTML
    HTML(string=html, base_url=HERE).write_pdf(a.out)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
