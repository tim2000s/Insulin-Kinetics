# Estimating insulin peak time and duration of action from loop data

How these scripts work, what they can determine, and — at least as importantly — what they cannot.

---

## 1. The question, and why it is harder than it looks

An AID system needs two numbers to model insulin: the **peak time** of insulin action and the
**duration of insulin action (DIA)**. Both are usually set from the label or from folklore, and both
vary between people.

The obvious approach — find a bolus, watch glucose fall, note when it falls fastest — fails on real
closed-loop data for a reason worth stating up front:

> **A closed loop does not deliver impulses.** It delivers a near-continuous stream of small doses,
> each one a reaction to the last. In one cohort user, out of ~19,800 boluses only **234** had a
> clear hour either side and **38** had two hours. Impulse-response averaging starves.

There is also a deeper problem. Glucose moves because of insulin, carbohydrate, endogenous
production, exercise, stress and sensor error. To see the insulin curve you need windows where
insulin is the *only* thing moving glucose — and unannounced meals, the single worst contaminant,
are by definition not in the data.

## 2. What is identifiable, and what is not

Under isolation, the relationship is:

```
dBG/dt  ≈  −ISF · insulin_activity(t)
```

The **first derivative of glucose traces the activity curve**. Three consequences follow, and they
determine everything the scripts do:

**Peak time is a SHAPE parameter; ISF is an AMPLITUDE parameter.** They are separable. You can
estimate *when* insulin acts without knowing *how strongly* it acts, and without knowing ISF at all.
This is what makes the problem tractable, and it is why a concentration change of insulin concentration
does not disturb a peak estimate: concentration is pure amplitude.

**DIA is conditionally identifiable.** The tail of the curve is where DIA lives, and the tail is
tiny. For a user with a 10 h DIA and 0.1–0.3 U doses, the remaining insulin beyond ~5 h is a
fraction of a percent of a dose — below rounding, so the data carries almost no information about
where the curve ends. For a user with a ~4.75 h DIA and doses ~2.4× larger, the tail clears the
noise and DIA *is* identifiable. **Whether you can estimate DIA is a property of the curve and the
dose size, not of the method.** Both cases are shown in §7.

**Superposition rescues the sample size.** For a linear kernel, overlapping doses add. So every dose
is usable data rather than contamination, and the ~19,800 boluses come back into play. The binding
constraint moves from sample size to **collinearity**: if "insulin 30 minutes ago" and "insulin 35
minutes ago" are nearly the same regressor — which regular five-minute dosing threatens — the kernel
is not identifiable no matter how much data there is. Gate 1 exists to test exactly this.

## 3. The insulin model

Both scripts use the exponential ("free-peak") model that AndroidAPS and oref use, parameterised by
peak time `tp` and duration `td` in minutes:

```
tau          = tp · (1 − tp/td) / (1 − 2·tp/td)
a            = 2·tau/td
S            = 1 / (1 − a + (1 + a)·exp(−td/tau))

IOBfrac(t)   = 1 − S·(1 − a)·( (t²/(tau·td·(1−a)) − t/tau − 1)·exp(−t/tau) + 1 )
activity(t)  = (S/tau²)·t·(1 − t/td)·exp(−t/tau)
```

`IOBfrac` is the fraction of a dose still on board at `t`; `activity` is its rate of action — the
curve whose peak the loop actually cares about. Both are zero outside `[0, td]`.

## 4. Gate 1 — recovering a curve that is already known

`gate1_recover_known_curve.py`

**The idea.** A loop logs its own IOB every cycle, and that IOB is a *deterministic* function of the
dose history and the configured curve:

```
bolus_IOB(t) = Σ_j  dose_j · IOBfrac(t − t_j)
```

So deconvolving logged IOB against the delivered doses **must** return the configured curve. The
answer is known by construction. If the estimator cannot recover it, the method is broken and
nothing downstream is worth building.

**Why this is the right positive control**, rather than testing on a simulator: it runs on the real
dose series, so it exercises the actual spacing and burst structure of a live closed loop — which is
precisely where the identifiability risk lives. A simulator with tidy, well-separated dosing would
hide the failure it is meant to detect.

**Implementation.** Doses are binned onto the loop's own five-minute grid and the predicted IOB is
computed as a single convolution with a DIA-length kernel. The naive per-dose superposition is
O(doses × timepoints) — order 10⁸ per residual evaluation, and the optimiser calls it hundreds of
times. `(tp, td)` are fitted by least squares, with a **day-block bootstrap** for confidence
intervals so the correlation structure within a day is respected.

**Two data notes.** Only the *bolus* channel is used: total IOB folds in basal and temp-basal
delivery, and temp basals are recorded as a rate rather than an insulin amount, so they cannot be
reconstructed. And `iob_bolusiob` is logged but always NULL in this dataset (Nightscout does not
carry it), so bolus IOB is derived as `iob_iob − iob_basaliob`, which is the loop's own
decomposition — `basaliob` is net of scheduled basal and is routinely negative under zero-temps.

## 5. Gate 2 — the peak from observed glucose

`gate2_peak_from_glucose.py`

The physiological estimate. In an isolated window:

```
ΔBG_i  =  −k · conv(dose, activity(·; peak, DIA))_i  +  c  +  drift_i  +  e_i
```

**`k`, `c` and the drift term are fitted PER WINDOW and profiled out analytically** (an ordinary
least squares of ΔBG on the convolved-activity regressor, an intercept and a time trend). Only
`peak` is estimated globally, by a one-dimensional search over the total residual sum of squares.

This structure is doing specific work:

- **Per-window amplitude `k`** means ISF level differences between nights — dynamic ISF, site,
  hormones, illness — cannot bias the shape estimate. It also means a **concentration change lands
  entirely in `k`**, which is what makes a before/after peak comparison valid across a dilution.
- **A per-window time trend, not just an intercept.** This is not cosmetic. Overnight windows sit
  across dawn, and an unmodelled rising trend late in a window looks exactly like insulin action
  ending early. In testing this biased one user's estimate from 35.3 min down to 30.9 and produced a
  false signal that the configured peak was too late. A pre-dawn-only cut independently confirmed
  the trend-controlled figure.
- **DIA is held fixed, not fitted**, unless you have established it is identifiable for that user
  (Gate 1 will tell you). Fitting it when it carries no signal lets it absorb residual belonging
  elsewhere and drags the peak with it.

**Window selection is the whole ballgame.** A window qualifies only if, throughout:

| criterion | why |
|---|---|
| COB = 0, and no carbs recorded in the window or the 3 h before | announced carbs destroy isolation |
| overnight, in the **user's own timezone** | unannounced eating is least likely |
| steps below a threshold | exercise moves glucose and changes absorption |
| not inside a post-rescue window | rescue carbs are unannounced by definition |
| no CGM gap > 15 min, BG in a plausible range | interpolation across gaps invents signal |
| some insulin actually delivered | no input, no identification |

Windows are non-overlapping so that the bootstrap resamples independent units.

## 6. Constraints — read this before believing any number

**Sensor lag biases every estimate LATE**, by the interstitial delay (~4 min) plus filter delay.
This is not corrected. Two consequences:
- An absolute estimate should be read as "action peak *as seen through the sensor*".
- For comparison against a loop's configured value this is arguably correct, because the loop also
  acts on sensor glucose and carries the same lag.
- The direction matters when interpreting a discrepancy: lag cannot explain an observed peak that is
  *later* than configured — correcting for it would widen such a gap, not close it.

**Peak is dose-dependent.** Rapid analogues peak later and last longer at larger doses. Averaging
0.1 U micro-boluses with 6 U meal boluses estimates a curve that fits neither. An estimate from
SMB-dominated data will sit earlier than the same person's meal-bolus reality. Stratify if you have
the sample for it.

**Unannounced meals cannot be excluded, only made unlikely.** COB and an overnight window are
proxies. Any residual contamination biases toward an apparently weaker and later insulin effect.

**Exercise is only controlled where step data exists.** Some uploaders send none; the scripts then
proceed and say so explicitly in the report rather than silently returning nothing.

**Power is usually the binding constraint, not bias.** Between-window scatter is large — one user's
per-window SD was 37 min around a pooled estimate of 35. Pooled estimates are far better than any
single night, but detecting a *change* needs many windows:

| post-change windows | detectable shift (80% power, α 0.05) |
|---|---|
| 5 | ~48 min |
| 10 | ~35 min |
| 20 | ~26 min |
| 40 | ~20 min |

At roughly one usable window per night, that column is also days. **A shift of 10–15 minutes is not
detectable observationally on any useful timescale.** If you need to measure a change that size, a
designed test — a controlled fasting bolus before and after — will answer in days what this cannot
answer in months.

**Model misspecification.** The exponential family is assumed. If a person's true kinetics have a
different shape, the fit returns the closest exponential, not the truth. Fit RMSE relative to the
signal is reported so this is at least visible.

**These are observational estimates.** No counterfactual is claimed anywhere. A discrepancy between
observed and configured is a discrepancy, not a demonstration that changing the setting improves
anything.

## 7. Worked results

Two users, both real, illustrating the two regimes:

| | user 1 | user 2 |
|---|---|---|
| configured peak | 38 min | ~34 min (recovered) |
| Gate 1 recovered peak | **36.3** [35.8, 36.7] | **33.9** [33.2, 35.1] |
| Gate 1 recovered DIA | 375 [298, **1440**] — *not identifiable* | 285 [242, 340] — *identifiable* |
| Gate 2 observed peak | **35.3** [29.5, 46.1] | **54.6** [42.8, 62.3] |
| verdict | consistent with configured | **~20 min later than configured** |

User 1 has a 10 h DIA and ~0.1–0.3 U doses, so the tail is unobservable. User 2 has a ~4.75 h DIA
and ~2.4× the insulin, so it is not. Gate 1 tells you which regime you are in *before* you interpret
a DIA number.

User 2's gap survived every robustness check — with and without the drift control, on a pre-dawn
subset, and under a deliberately wrong DIA prior — and lag bias points the wrong way to explain it.
It remains **provisional**: one user, observational, exercise uncontrolled.

## 8. Running them

```bash
pip install -r requirements.txt

# Gate 1 first, always. If it fails, stop.
python3 gate1_recover_known_curve.py --user <id> --expect-peak 38 --expect-dia 600

# Then the physiological estimate, in the user's timezone.
python3 gate2_peak_from_glucose.py --user <id> --tz Europe/London --dia 600

# Robustness worth running every time (same --tz throughout):
python3 gate2_peak_from_glucose.py --user <id> --tz <tz> --no-linear-drift     # dawn-ramp sensitivity
python3 gate2_peak_from_glucose.py --user <id> --tz <tz> --night 0,4 --hours 3 # pre-dawn subset
python3 gate2_peak_from_glucose.py --user <id> --tz <tz> --dia <other>         # DIA prior sensitivity
```

`--tz` is required rather than defaulted. A wrong timezone silently selects the wrong hours; in
testing it returned an empty window set for a user six hours from the assumed default, which reads
as "no data" rather than as a configuration error.

Both write a Markdown report alongside the script.

**Data expected.** A PostgreSQL database (`DSN` at the top of each script) with per-cycle loop
decisions — timestamp, CGM, COB, IOB, basal IOB, steps where available — and a treatments table of
delivered insulin. The scripts were built against a Nightscout-derived schema; adapting them is
mostly a matter of rewriting the two queries in `load()`.

## 9. Order of operations

1. **Gate 1.** Identifiability under this user's real dose spacing, with a known answer. Also tells
   you whether DIA is estimable for them.
2. **Gate 2.** The observed peak, with the drift control on.
3. **Robustness.** Drift on/off, pre-dawn subset, DIA prior. If the answer moves a lot, it is the
   model talking, not the insulin.
4. **Power.** Check the detectable-shift table before promising anyone a before/after comparison.

Skipping (1) is the mistake worth naming: without it, an unidentifiable fit still returns a
confident-looking number.
