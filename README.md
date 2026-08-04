# Insulin Kinetics

Estimating a person's insulin **action peak time** — and, when the data allows it, their **duration
of insulin action** — from ordinary closed-loop data. No test protocol, no clinic, no clamp study:
just the dosing and CGM history a loop already records.

**[METHOD.md](METHOD.md) is the document to read.** It covers the identification argument, the
model, what each script does and why, and the constraints on believing any of it.

## The short version

Glucose falls at a rate proportional to insulin *activity*, so the first derivative of CGM traces
the shape of the insulin curve. Peak time is a **shape** parameter and ISF is an **amplitude**
parameter, and they separate — which means the peak is estimable without knowing ISF, and survives
things that change only amplitude, such as a change of insulin concentration.

DIA is a different matter. It lives in the tail of the curve, and at small doses with a long
configured duration the tail is far below the noise. Whether DIA is identifiable is a property of
the person's curve and dose sizes, not of the method. One of the two scripts tells you which case
you are in before you try.

## The two scripts

| script | what it does |
|---|---|
| `gate1_recover_known_curve.py` | **Positive control.** Deconvolves the loop's own logged IOB against the delivered doses. The answer is known by construction — it must return the configured curve. Establishes that the peak is identifiable under this user's real dose spacing, and whether their DIA is identifiable at all. |
| `gate2_peak_from_glucose.py` | **The physiological estimate.** Fits the action peak to what glucose actually did, over isolated overnight fasting windows, with per-window amplitude and drift profiled out. Also screens for a mid-period insulin or settings change. |
| `gate3_dose_split.py` | **Does the peak differ between large and small doses?** Two kernels, one per dose stratum, with separate amplitudes. Read it against the two-kernel Gate 1 control, which has a known answer of zero. |

Run Gate 1 first. Without it, an unidentifiable fit still returns a confident-looking number.

The fits are **pooled, not per-dose** — a closed loop never delivers an isolated dose, so there is
no per-dose response to measure. One kernel is fitted to the whole series by convolution. METHOD.md
§2a explains why that is forced by the data rather than chosen for convenience.

## Quick start

```bash
pip install -r requirements.txt

python3 gate1_recover_known_curve.py --user <id> --expect-peak 38 --expect-dia 600
python3 gate2_peak_from_glucose.py  --user <id> --tz Europe/London --dia 600
python3 gate3_dose_split.py         --user <id> --tz Europe/London --dia 600
```

Each prints a summary and writes a Markdown report next to the script.

## What it needs

A PostgreSQL database holding per-cycle loop decisions (timestamp, CGM, COB, IOB, basal IOB, steps
where available) and a treatments table of delivered insulin and carbs. Built against a
Nightscout-derived schema; porting is mostly a matter of rewriting the two queries in `load()`.
Set the connection string at the top of each script.

## Status and caveats

This is analysis tooling, not a therapy device. Nothing here doses, and nothing here should be used
to change a pump setting without a clinician. The estimates are **observational**: a gap between the
observed peak and the configured one is a discrepancy worth understanding, not evidence that
changing the setting improves outcomes.

Two constraints are worth knowing before you start, both detailed in METHOD.md:

- **Sensor lag biases every estimate late** by a few minutes, and is not corrected.
- **Power binds before bias does.** Detecting a 10–15 minute change observationally takes longer
  than it is worth waiting. For a before/after question, a designed fasting test answers in days
  what this cannot answer in months.

## Licence

AGPL-3.0-or-later, matching the AndroidAPS work this came out of.
