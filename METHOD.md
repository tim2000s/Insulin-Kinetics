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

**DIA is conditionally identifiable, and its confidence interval lies.** The tail of the curve is
where DIA lives, and the tail is tiny. For a user with a 10 h DIA and 0.1–0.3 U doses, the remaining
insulin beyond ~5 h is a fraction of a percent of a dose — below rounding, so the data carries
almost no information about where the curve ends. For a user with a ~8 h DIA and doses ~2.4× larger,
the tail clears the noise and DIA *is* identifiable.

The trap is that a weakly-identified DIA does not announce itself with a wide interval. For the one
user whose configured DIA is known independently (600 min), Gate 1 returned:

| window | DIA | 95% CI |
|---|---|---|
| 28 days | 314 | [260, 378] — **excludes the truth** |
| 45 days | 428 | [317, 1440] |
| 90 days | 1440 | [319, 1440] |

The 28-day interval is the tightest and the most wrong. The day-block bootstrap under-covers here,
because it resamples the noise around a likelihood that is nearly flat in DIA — a flat ridge with a
spurious local minimum reproduces across resamples.

**The diagnostic that works is LEVERAGE, computed directly.** Hold the peak, move DIA to 240, 360,
480, 600, 900, 1440, and measure how much the predicted IOB series moves in RMS terms against the
fit residual. If no alternative shifts the prediction by more than about twice the residual, the
data cannot tell those DIAs apart and the recovered value is arbitrary. Gate 1 now prints this
table and waives the DIA half of its verdict when DIA is unidentified — failing a user on a
parameter the data cannot constrain reports a defect where there is none.

Across the cohort this separates cleanly. For one user every alternative from 4 h to 24 h moved the
prediction by at most 0.044 U against a 0.179 U residual — nothing, which is exactly why their
estimate wandered 314 → 428 → 1440 with window length. For another, dropping DIA to 240 min moved
it by 0.635 U against a 0.162 U residual, so short durations are firmly excluded even though 480
and 600 are not separable from each other. Four of eleven users have no identified DIA at all.

**Window-length stability is the secondary check.** Refit at 28, 45 and 90 days.
A user whose DIA is genuinely identified returns the same number (one did: 487 / 482 / 463, and
peak 75.0 / 75.1 / 77.2); a user whose DIA is not returns whatever the window happens to support.
Do this before quoting a DIA. It is also worth doing for the peak, which was stable to ~1 min for
the identified user and wandered 38.5 → 36.6 → 32.9 for the other.

**Superposition rescues the sample size.** For a linear kernel, overlapping doses add. So every dose
is usable data rather than contamination, and the ~19,800 boluses come back into play. The binding
constraint moves from sample size to **collinearity**: if "insulin 30 minutes ago" and "insulin 35
minutes ago" are nearly the same regressor — which regular five-minute dosing threatens — the kernel
is not identifiable no matter how much data there is. Gate 1 exists to test exactly this.

## 2a. This is a pooled fit, not a per-dose one

Worth stating plainly, because it is the natural thing to assume: **neither script estimates a peak
for each dose and averages them.** There is one curve, fitted to the whole series at once.

That is forced by the data, not chosen for convenience. A closed loop delivers a dose every few
minutes, so any given rise or fall in glucose is the superposition of a dozen overlapping doses —
there is no window in which one dose acts alone and no per-dose response to measure. What the fit
does instead is convolve the entire dose series with a candidate kernel and ask which kernel
reproduces the observed series. Every dose contributes to every observation it can still be acting
on.

Two things do get estimated per unit rather than globally, and the distinction matters:

- **Gate 2 fits an amplitude and a drift ramp per WINDOW** (not per dose). That isolates the shape
  from night-to-night differences in ISF and dawn phenomenon.
- **Per-window peaks are computed**, but only to measure the between-window spread that sets the
  power table. They are far too noisy individually to report — SD 24–41 min across this cohort.

Gate 3 relaxes the single-kernel assumption in the one direction pharmacology suggests, by dose
size. See §7a.

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

**Observations within 10 minutes of a bolus are masked**, and this turned out to matter more than
expected. IOB steps up by the whole dose at delivery, so a small error in when that step is placed
produces a residual as large as the dose itself. Unmasked, the residual on an *exact identity* ran
11–32% of signal. Masking cuts it to 7–10% for the clean users, and — the real point — removes a
downward bias on DIA: the user whose configured DIA is known moved from 314 to 487 min against a
truth of 600. The peak barely moves (38.5 → 38.4, 75.0 → 74.9), so the mask is close to free.

A residual still above ~15% after masking is a flag, not noise. It means this user's dose records
are not what the app saw — extended or multiwave boluses, partial delivery, or timestamps offset
from the delivery the loop counted. Five of eleven cohort users flagged.

**A note on grid binning, because a partial fix makes it worse.** Doses and observations are both
floor-binned to the 5-minute grid, which places each about 2–3.5 min early on average. Those two
errors are in the same direction and largely cancel in the lag that the kernel sees, which is why
the estimate is sound. Switching only the dose side to rounding — the obvious "fix" — breaks the
cancellation and moves the recovered peak by 1.6–2 min in a cohort where the whole signal is 35–75
min. Change both or neither.

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

## 6c. Gate 4 — dropping the parametric form (the method that works)

Gate 2's failure was a design failure, so the fix is a different design rather than a better fit.
Stop assuming a curve shape; treat the record as a train of impulses and estimate the response.

    dBG_t = - SUM_k beta_k * dose_{t-k}  +  tod_{clock(t)}  +  day_{n(t)}  +  e_t

One free `beta_k` per five-minute lag out to six hours, no shape assumed, with a second-difference
(Tikhonov) smoothness penalty chosen by GCV and `beta >= 0`. This is standard non-parametric
deconvolution — the same machinery used for physiological input estimation, and identical to a
distributed-lag/FIR regression in econometrics.

**The identification change is what matters, not the flexibility.** Gate 2 gave every window its own
free drift ramp, which within a single window is the same slow shape as insulin action. Gate 4
models drift as a **repeatable time-of-day profile estimated from every day at once**. Dawn is
locked to the clock; insulin doses are not. That difference in timing is the identifying variation,
and it only exists when you pool across days. For the same reason the sample is every fasting
stretch around the clock, not only overnight — 12,000 to 86,000 samples per user rather than ~100
windows.

**Positive control, run before looking at any real answer:**

| generating curve | true peak | recovered, no noise | recovered, realistic noise |
|---|---|---|---|
| AAPS exponential | 35 | 35 | 35 |
| AAPS exponential | 55 | 55 | 55 |
| AAPS exponential | 75 | 75 | 70 |
| gamma, shape 3 (wrong family) | 45 | 45 | 50 |
| gamma, shape 6 (wrong family) | 35 | 35 | 35 |

It recovers the peak **even when the generating family is wrong**, which the parametric estimator
could not (that biased −10.4 to +5.3 min). Specification curve across max-lag and day-effect
choices: range **0–5 min**, against Gate 2's 23–45 min. Dropping the time-of-day term moves it 0–15
min.

**Two bugs this exposed, both caught by the control rather than by reading the code.** A carb record
timestamped before the grid start produced a negative slice bound, and `ok[0:negative]` silently
blanked 112 of 123 days. And the design regressed on `+dose` while constraining `beta >= 0`, which
pins every true (negative) coefficient at zero — at *zero noise* the estimator returned 355 min for
a true 45. Neither would have been visible from the output alone.

## 6b. Why Gate 2 is not validated, and what was withdrawn

Gate 1 always had a positive control. Gate 2 did not, and when one was finally built the method did
not survive it. Three tests, in the order they were run.

**1. Is the estimator biased?** No. `gate2_selftest.py` keeps a user's real windows — real dose
times, real dose sizes, real sampling — and replaces the observed glucose with glucose simulated
from a known peak, calibrated to that user's own amplitude, drift and noise. Recovery across true
peaks of 35, 45, 55 and 75 min was accurate to **±0.3 min**. Adding an unmodelled basal-action term
shifted it by at most 0.4 min; adding unannounced meals to a quarter of windows, at most 0.5 min.
The estimator does what it says on its own generating model.

**2. Is the exponential family too rigid?** Only mildly, and in the wrong direction to matter.
Generating glucose from non-exponential activity curves — gammas of several shapes, bi-exponentials
of several tail ratios — and fitting with the exponential estimator gives a bias of **−10.4 to +5.3
min**, mostly negative. A misspecified shape makes the reported peak *earlier*, so it cannot explain
a peak that comes out later than configured.

**3. Does the answer depend on the analyst?** Yes, ruinously. Refitting across 96 defensible
specifications (window length, night hours, minimum insulin, drift on or off, DIA prior):

| user | configured | median across specs | range | swing from the drift choice alone |
|---|---|---|---|---|
| K | 38 | 31.6 | [24.4, 75.9] | 2 min |
| G | 34 | 45.8 | [37.9, 54.9] | 4 min |
| J | 75 | 48.9 | [33.5, 76.6] | **23 min** |
| B | 55 | 67.4 | [56.5, 149.8] | **35 min** |

**The cause is collinearity.** Over a 3–5 hour fasting window the insulin activity profile is slowly
varying and largely monotone — that is, nearly a straight line in time. So is the dawn ramp it must
be separated from. Median |r(insulin regressor, time)| is **0.73–0.81** at four-hour windows and
rises to **0.99** at two hours. The two terms compete for the same slow component of the data, and
whichever one is permitted to absorb it decides the answer. A bootstrap on one specification cannot
see this, because it holds the choice fixed.

Restricting to windows where the two are close to orthogonal (|r| < 0.5) leaves 17–28% of windows and does
**not** fix it: the drift swing stayed at 13 min for one user and grew to 45 for another.

**What is withdrawn.** Every per-user physiological peak reported from Gate 2, specifically the
claim that one user's peak sits ~20 min later than their configured value, and the claim that
another's is consistent with their configured 75. Neither survives the specification curve.

**What survives.** Gate 1, which is an algebraic identity rather than an inference: it recovers
known configured curves to a fraction of a minute, matches vendor presets across the cohort, detects
a mid-period insulin change nobody recorded, and passes a two-kernel dose-size negative control. It
measures what the *loop believes*, which is a genuinely useful and separate thing from what the
insulin does.

**What would be needed to answer the physiological question.** Not more of this data. The confounder
is structural: overnight is chosen because it is quiet, and quiet is exactly when insulin action and
endogenous drift look alike. It needs a design where insulin action is not monotone across the
observation window — a fasting bolus given deliberately, with no other insulin, so the activity
profile has curvature the drift term cannot mimic.

## 6a. Detecting a change nobody recorded

A pooled estimate silently averages across an insulin brand switch, a dilution, or a settings
change. Three checks, in descending order of usefulness:

1. **Gate 1 on disjoint sub-windows** (`--days 14 --offset-days 14`, then `--days 14`). This is by
   far the sharpest instrument, because Gate 1's peak intervals are narrow. In a 10-user cohort it
   cleanly caught one user moving from a ~75–82 min curve to a ~55 min curve mid-period; rolling
   7-day windows localised the switch to a single week. Nothing in the treatment stream named it.
2. **Gate 2's change screen** — a rank correlation of per-window peak against time, plus a
   split-half refit. Reported automatically. At 28 days it is nearly powerless: the half-difference
   standard error is 12–21 min, which is the same size as the effects worth finding.
3. **Event flags** — `Insulin Change` and `Profile Switch` records. Necessary but not sufficient:
   `Insulin Change` usually means a routine cartridge refill, and a real formulation change often
   generates no event at all.

The split-half flag must be judged against its **own standard error**, not a fixed minute threshold.
A flat "> 20 min" rule fires about a fifth of the time on noise alone when the per-window SD is near
35 min — it produced three false flags in a 10-user cohort before this was corrected.

## 7. Worked results

Two users, both real, illustrating the two regimes:

| | user 1 | user 2 |
|---|---|---|
| configured peak | 38 min | ~34 min (recovered) |
| Gate 1 recovered peak | **36.3** [35.8, 36.7] | **33.9** [33.2, 35.1] |
| Gate 1 recovered DIA | 375 [298, **1440**] — *not identifiable* | 285 [242, 340] — *identifiable* |
| Gate 2 observed peak | ~~35.3~~ | ~~54.6~~ — **both withdrawn, see §6b** |
| verdict | Gate 1 agrees with configured | Gate 1 agrees with configured |

User 1 has a 10 h DIA and ~0.1–0.3 U doses, so the tail is unobservable. User 2 has a ~4.75 h DIA
and ~2.4× the insulin, so it is not. Gate 1 tells you which regime you are in *before* you interpret
a DIA number.

User 2's apparent gap did NOT survive the specification curve added later (§6b): across 96
defensible analyses the estimate ranged 37.9-54.9 min, and for two other users the drift choice
alone moved the answer by 23 and 35 minutes. The gap is withdrawn. The robustness checks originally
run — drift on/off, a pre-dawn subset, a wrong DIA prior — were too narrow a set to catch this, which
is the lesson: vary the specification, not just the data.

## 7a. Should large and small doses be split? (Gate 3)

Pharmacology says a bigger subcutaneous depot absorbs more slowly, so large doses should peak later.
If true, one fitted curve fits neither end. Two separate questions came out of testing it.

**Does the tail justify a split?** No. The proposed criterion was whether the post-5 h tail is above
noise. Truncating each user's own recovered kernel at 5 h and measuring what that removes from the
predicted IOB series gives **0.000–0.051 U RMS, against fit residuals of 0.15–0.55 U** — below the
noise floor for all eleven users, at p90 doses as well as median ones. No dose stratification
rescues that, because the tail is small relative to the residual in *both* strata. DIA is identified
where it is identified through the rising limb and mid-curve shape, not the tail.

**Does the peak differ by dose size?** Not demonstrably, and the estimator is poorly conditioned for
the question.

First the negative control, which is the part worth keeping. Run the two-kernel fit through Gate 1,
where the app applies one configured curve to every dose regardless of size, so the true difference
is exactly zero. Four users returned −0.9 to +3.1 min — the estimator is unbiased with respect to
dose size. One returned **+48.7 min**, which is impossible for a single configured curve and
identifies a data problem with that user's large doses rather than a discovery.

Then the real test, on glucose, full history:

| user | small (<1 U) | large (≥1 U) | difference | 95% CI | in-window large doses |
|---|---|---|---|---|---|
| B | 57.0 | 56.0 | −1.0 | [−8.0, +24.5] | 12.4% |
| F | 24.5 | 22.5 | −2.0 | [−23.9, +89.4] | 15.7% |
| A | 35.5 | 23.5 | −12.0 | [−16.0, +9.3] | 22.7% |
| J | 38.5 | 54.5 | +16.0 | [+2.1, +81.3] | 4.5% |

One of four excludes zero, in the predicted direction, in the user with the *least* large-dose
content and a CI 79 min wide — and whose small-dose estimate disagrees with their own single-kernel
result by 26 min. That is a conditioning failure, not a finding. The reason is collinearity: within
a window the small-dose and large-dose regressors have median |r| of 0.56, 0.69 and **0.94** for
the three users tested. At 0.94 the two peaks are simply not separable.

There is also a structural limit. Gate 2's windows are chosen to exclude meals, so they are
dominated by automatic micro-boluses — 0.9% to 22.7% of in-window doses clear 1 U, and for several
users no window contains both classes. **The overnight-fasting design can characterise the
micro-bolus curve and essentially cannot reach the meal-bolus curve.** A dose-dependent peak
probably is real in pharmacology; this data cannot see it, and answering it properly needs a
designed test with matched large and small doses under the same conditions.

**Verdict: do not split the dose path on size on this evidence.** Unproven, not disproven.

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

1. **Gate 1.** Identifiability under this user's real dose spacing, with a known answer.
2. **Gate 1 at three window lengths** (28/45/90 d). Stability, not the CI, is what tells you whether
   a recovered DIA — or peak — is real.
3. **Gate 1 on disjoint halves.** Did the configured curve change mid-period?
4. **Gate 2.** The observed peak, with the drift control on. Budget ~3 months of data: at 28 days
   the intervals came out 40–95 min wide across a 10-user cohort, which is no answer at all.
5. **Robustness.** Drift on/off, pre-dawn subset, DIA prior. If the answer moves a lot, it is the
   model talking, not the insulin.
6. **Power.** Check the detectable-shift table before promising anyone a before/after comparison.

Skipping (1) is the mistake worth naming: without it, an unidentifiable fit still returns a
confident-looking number.
