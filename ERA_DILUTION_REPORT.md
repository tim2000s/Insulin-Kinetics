# Detecting a change of insulin preparation in the glucose record

One participant changed preparation on 5 August 2026, from U200 to the same analogue diluted to
U100. The same mass is now delivered in twice the volume and recorded as twice the units. A larger
depot has more surface area, so dilution plausibly moves absorption earlier, and the question is
whether the record since the change can detect that.

Three independent measures put the change on the stated date and at close to the expected size.
Total daily dose taken from the device rose from a median of 15.9 U/day over 10 July to 4 August
to 33.6 U/day from 5 August. Bolus and automated microbolus delivery alone rose from 8.8 to
18.4 U/day, and reconstructed basal from 5.8 to 12.8 U/day. All three give a factor near 2.1.

## Why the bolus-only design does not answer it

`era_shift.py` fits one dose kernel before the boundary and one after, sharing a time-of-day
profile and per-day intercepts, and calibrates the result against boundaries placed at earlier
dates where nothing changed. Its dose regressor is built from bolus records alone. For a
closed-loop participant that omits roughly two fifths of the insulin, and at a concentration change
it omits it asymmetrically, because basal units double exactly as bolus units do. The omitted term
therefore has one scale before the boundary and another after it, which is the structure that
biases a before-and-after comparison of the very quantity under test.

The existing guard against the omission, `gate4_with_basal.py`, reads net basal relative to
schedule. That field is null for every one of this participant's 140,988 decision rows, so the
check has never covered them. The rate the loop set at each cycle is populated instead, covering
99% of five-minute bins. Summed with bolus delivery it reproduces the device's own total daily dose
at a median ratio of 0.90, the residual scatter being expected because the device figure is a
rolling 24-hour total rather than a calendar-day one.

`era_shift_basal.py` carries basal as a second pair of kernels, split at the same boundary and left
unconstrained in sign, since basal is cut in anticipation of a fall and a negative departure acting
through a positive kernel has to be able to raise glucose. Its bolus-only path reproduces
`era_shift.py` exactly, so any difference below is the basal channel rather than a rewrite.
Smoothing is selected by generalised cross-validation on the design actually fitted: the
four-kernel design lands at 10 and the bolus-only design at 1000, and applying the latter to the
former over-smooths it and produces a degenerate, bimodal null.

## What 26 days of record support

Fitted over the 180 days before the boundary plus 25.7 days after it, with 24 placebo boundaries
carrying a post-window of the same length.

| quantity | before | after | placebo null | verdict |
|---|---|---|---|---|
| peak of the bolus kernel (min) | 38 | 62 | median 42, 10th-90th 38 to 68 | 75th percentile, p about 0.50 |
| kernel area per unit delivered | 32.5 | 21.7 | ratio median 1.10, 10th-90th 0.83 to 1.47 | ratio 0.67, below all 24 |

The amplitude is the informative half. The kernel is expressed per unit delivered, so if the same
mass is now recorded as twice the units the per-unit response must halve. The observed ratio of
0.67 falls below every one of the 24 placebo values and inside the range a true halving would
produce once carried through the same short-window bias, 0.41 to 0.74. It holds between 0.61 and
0.67 across a hundred-fold range of smoothing. The design is seeing the change.

The timing is not resolved. A peak in the high fifties against 38 min before looks like a
substantial move, but a value at least that far from the null median arises about half the time
when nothing has changed, and the estimate is not stable in any case. `era_smoothing_check.py`
refits the model across a hundred-fold range of the smoothing penalty and the post-boundary peak
moves across roughly the width of the shift being claimed, while the area ratio holds between 0.61
and 0.67. The peak also moves as days accumulate, which the area ratio does not. No shift in
absorption timing is established by this window, in either direction.

## How much record would be needed

The same placebo procedure at a range of window lengths, summarised as the proportion of windows
landing within 15 min of the long-run estimate of 38 min. A proportion rather than a percentile
spread, because the null is right-skewed with a tail of degenerate fits at the end of the lag
range, and a binomial proportion has known uncertainty and must rise with window length.

| post-window (days) | fits | median peak (min) | within 15 min of long-run | 10th-90th | boundary range available (days) |
|---|---|---|---|---|---|
| 26 | 30 | 42 | 70% (+/-8) | 37 to 68 | 117 |
| 40 | 30 | 38 | 87% (+/-6) | 38 to 58 | 103 |
| 60 | 30 | 38 | 100% (+/-0) | 38 to 42 | 83 |
| 90 | 30 | 38 | 100% (+/-0) | 38 to 38 | 53 |

Sixty days is the point at which the estimator returns the long-run answer reliably enough for a
departure from it to carry meaning, which falls on 4 October 2026 for this boundary. The 90-day row
reads the same but rests on 53 days of remaining boundary range, so those placebo windows overlap
heavily, share most of their days and understate the spread.

## Limitation

The basal kernel peaks at 2 min and carries about 78% of the total absolute kernel mass. That is
the controller reacting to glucose rather than insulin acting on it, since basal is cut when a fall
is expected. Admitting basal is still the right specification, because the alternative is an
omitted term that changes scale at the boundary, and the placebo fits share the same reactive
structure so the calibration remains fair. The basal kernel itself should not be read as
pharmacology.
