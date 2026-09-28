# Plan — make `synthesize_card` produce a true parallelogram

Fixes open item #1 in `talk.md` §8.12. The `NORMAL_FULL_BARREL` card currently
traces a single line, not the familiar parallelogram, and the reason it has not
been fixed is a genuine collision between two fault signatures. This plan adds
the missing geometry and a discriminator that separates the two.

---

## 1. Objective

1. `synthesize_card(NORMAL_FULL_BARREL)` must return a card whose upstroke and
   downstroke are **distinct traces**, so the dashboard draws a parallelogram.
2. All six labels must remain separable — by `analytic_diagnosis()` *and* by the
   trained CNN — at the same accuracy as today.
3. No public key is renamed, added or removed: `card_features()` must keep
   returning exactly the ten documented keys, because
   `backend/tests/test_api_stream.py::FEATURE_KEYS` and
   `frontend/src/components/DynoCardCanvas.tsx` read them by name.

Non-goal: making the model reproduce any particular field card trace-for-trace.
This is a *parametric synthetic* generator whose job is to span the classes
cleanly enough to train and test against.

---

## 2. Current state

### 2.1 Why the card is a line

`backend/app/ai/dyno_classifier.py:147-150`:

```python
if label is CardLabel.NORMAL_FULL_BARREL:
    load += fluid_load_n * np.clip(normalised, 0.0, 1.0)
```

`normalised` is a pure function of position, so `load` is too. Measured: the
upstroke and downstroke loads differ by **0.888 N at the same position** against
a 15 000 N span. Both strokes trace one curve. The comment above it claims "the
familiar parallelogram"; the code cannot produce one.

Only this branch is affected. `ROD_FLOATING` (line 172) and `FLUID_POUND`
(line 164) already key off `upstroke`, so they retain stroke-dependence.
`GAS_INTERFERENCE`, `PUMP_TAGGING` and `UNANCHORED_TUBING` are deformations of
the same monotone line and inherit the defect visually, but each is defined by
its own perturbation rather than by stroke width.

### 2.2 The collision that blocked the first fix attempt

Adding the physically-correct stroke offset — a travelling valve makes the
upstroke and downstroke carry different column weights — immediately breaks
detection. Measured over 40 seeds, feature `upstroke_reversal` against its
0.10 fluid-pound threshold:

| stroke offset δ | analytic result | `upstroke_reversal` max |
|---|---|---|
| 0.00 (today) | `NORMAL` 40/40 | 0.022 |
| 0.10 | `NORMAL` 0/40 | 0.102 |
| 0.20 | `NORMAL` 0/40 | 0.170 |
| 0.30 | `NORMAL` 0/40 | 0.227 |
| 0.45 | `NORMAL` 0/40 | 0.298 |

So **even a 10 % stroke offset reads as fluid pound**. Cause: `upstroke_reversal`
took the smoothed derivative *across* the top-dead-centre turnaround, where a
real card legitimately steps. Geometry, not a fault.

> **Already fixed in this session.** `card_features()` now excludes a window
> either side of both turnarounds and measures only the interior of the upstroke.
> After that fix all six labels diagnose 25/25 from `synthesize_card`, and
> `test_ai_models.py` + `test_gibbs_solver.py` + `test_closed_loop.py` are 110
> passed. **The δ-sweep above was measured before that fix and no longer
> reproduces.** It is retained here because it is the evidence for why the
> turnaround exclusion is load-bearing — reverting it silently re-breaks every
> healthy card.

### 2.3 The remaining blocker: width vs. depression

With the turnaround fixed, a *symmetric* parallelogram is diagnosed
`ROD_FLOATING` 40/40. The cause is `downstroke_asymmetry` (line 386):

```python
asymmetry = float(np.mean(unit[1:][falling]) - np.mean(unit[1:][rising]))
```

This is the mean downstroke load minus the mean upstroke load. A normal card's
stroke width contributes to it; so does a genuine rod-float downstroke
depression. The feature cannot tell the two apart, and `analytic_diagnosis`
(line 465) thresholds it at `< -0.08`.

The existing test `test_ai_models.py:101` asserts
`downstroke_asymmetry == approx(0.0, abs=0.05)` for a full barrel — i.e. it
encodes the line, not the parallelogram, as correct behaviour. **That assertion
must change**; it is the defect stated as a requirement.

---

## 3. The discriminator

Measured, not assumed. Bin the card into 16 position slices; in each, compute
`w(u) = mean(downstroke load) − mean(upstroke load)`; fit a straight line
`a·u + b`; take the RMS residual, normalised by card span.

**A healthy parallelogram has a *straight* width profile. Rod float has a
bumpy one** — a Gaussian drag depression localised at mid-downstroke, not a
constant offset. That is a physically meaningful difference, not a tuned one.

Normalised RMS residual of the width profile, 60 randomised draws per label
across the full training envelope (stroke 1.8–3.0 m, rod 15–30 kN, fluid
12–45 kN, noise 0.002–0.03):

| label | min | max |
|---|---|---|
| `NORMAL_FULL_BARREL` | 0.0022 | 0.0254 |
| `PUMP_TAGGING` | 0.0299 | 0.0465 |
| `FLUID_POUND` | 0.0897 | 0.1767 |
| `GAS_INTERFERENCE` | 0.1128 | 0.1366 |
| `UNANCHORED_TUBING` | 0.1301 | 0.2127 |
| `ROD_FLOATING` | 0.1921 | 0.2288 |

`NORMAL` separates cleanly from all five faults with a **0.0045 margin at the
worst case** (0.0254 vs 0.0299). That margin is real but thin, so the threshold
is placed at 0.045 — mid-gap rather than hugging either cluster — and step 6
re-measures under noise before anything is committed.

---

## 4. Proposed architecture

No new module and no new public key. The discriminator is computed inside
`card_features()` and **replaces the meaning of the existing
`downstroke_asymmetry`** rather than being added as an eleventh key.

That choice is deliberate. Adding a key would break two pinned consumers:

- `backend/tests/test_api_stream.py::FEATURE_KEYS` asserts the exact ten-key
  set, and fails on any undocumented extra.
- `DynoCardCanvas.tsx:189` reads `features["downstroke_asymmetry"]` by name to
  caption the rod-float overlay.

Redefining the key keeps the contract stable and is the smaller diff. The
trade-off is that the name becomes slightly less literal — it is now "how much
the downstroke deviates from a constant-width parallelogram", not "mean load
difference". The docstring must say so explicitly.

### 4.1 The generator change

Add a stroke-differential term shared by every label, so all six cards are
parallelograms and differ only by their own deformation:

```python
# Travelling-valve differential.  The standing valve admits fluid on the
# upstroke and the travelling valve expels it on the downstroke, so the two
# strokes never carry the same column weight.  This width is what makes the
# card a parallelogram rather than a line.
valve_width = valve_width_fraction * fluid_load_n * np.clip(normalised, 0.0, 1.0)
load += np.where(upstroke, valve_width, -valve_width)
```

`valve_width_fraction` is drawn per-sample in `build_training_set()` from
`uniform(0.10, 0.45)`, and defaults to ~0.25 in `synthesize_card()`. The width
scales with `normalised` rather than being constant, because the column weight
difference grows with the volume swept.

Every branch keeps its own deformation on top, so `ROD_FLOATING` still shows a
Gaussian drag bump and `FLUID_POUND` still collapses its upstroke.

### 4.2 The feature change

```python
# Rod float vs. a healthy parallelogram.  A normal card has a *constant-width*
# loop: the downstroke sits a fixed fraction below the upstroke at every
# position.  Buoyant drag instead produces a localised depression at
# mid-downstroke, so the width profile is straight for a healthy card and
# bumpy for a floating one.  Measure the departure from straight.
```

Implementation: 16 position bins, mean load per stroke in each, least-squares
line fit, RMS residual, divided by card span so it is amplitude-independent.
Return in place of the current `asymmetry`.

---

## 5. Implementation steps

**Step 1 — add the valve width to the generator.**
`backend/app/ai/dyno_classifier.py`, `synthesize_card()`.
Add a `valve_width_fraction: float = 0.25` keyword (after `seed`, so existing
positional callers are unaffected) and the `np.where(upstroke, ...)` term above
the per-label branches. Update the `NORMAL_FULL_BARREL` comment, which currently
makes the false "monotone in position, therefore parallelogram" claim.
*No dependencies.*

**Step 2 — randomise it in the training set.**
Same file, `build_training_set()`. Draw `valve_width_fraction` per sample from
`uniform(0.10, 0.45)` and pass it through, so the network sees the whole
plausible range rather than one width.
*Depends on 1.*

**Step 3 — redefine `downstroke_asymmetry` as the width residual.**
Same file, `card_features()`. Replace the mean-difference computation with the
binned width-profile residual. Keep the key name and its place in the returned
dict. Rewrite the docstring entry for `downstroke_asymmetry` to state the new
meaning. Bins must be skipped where a bin has no samples of one stroke, and the
function must degrade to `0.0` when fewer than 4 usable bins exist rather than
raising.
*Depends on 1.*

**Step 4 — re-tune the thresholds that were fitted to the old meaning.**
Same file, `analytic_diagnosis()`. The `downstroke_asymmetry < -0.08` branch
(line 465) no longer means "downstroke depressed" and must be replaced by
`> 0.045` → `ROD_FLOATING`. Re-check the `> 0.15` unanchored branch (line 467):
it was signed to catch the *opposite* skew, and with an unsigned residual that
sense is gone. `UNANCHORED_TUBING` must be caught some other way — the candidate
is `skew`, which is already computed (line 445) and already dedicated to this
label, and whose test (`test_ai_models.py:126`) currently leans on the
asymmetry sign. Verify empirically before choosing; do not guess.
*Depends on 3.*

**Step 5 — update the tests that encode the defect.**
`backend/tests/test_ai_models.py`:
- line 101, `downstroke_asymmetry == approx(0.0, abs=0.05)` — still passes
  (a healthy card has a straight width profile, residual ≈ 0) but the *reason*
  changes. Add the new assertion.
- add `test_full_barrel_is_actually_a_parallelogram()` asserting the
  upstroke/downstroke load gap at matched position is a meaningful fraction of
  the span, not ~0.9 N. This is the regression guard for the whole change.
- line 114, rod float `< -0.05` — becomes `> 0.045`.
- line 126, unanchored `> 0.1` — retarget per step 4.
*Depends on 3, 4.*

**Step 6 — re-verify the separation before committing.**
Re-run the §3 sweep with noise to 0.03 over the full envelope, all six labels,
≥ 200 draws each. Confirm `NORMAL` max stays below the threshold and every
fault min stays above it. **If the margin collapses, stop and report** — the
threshold is then wrong and this plan needs a different discriminator. Do not
ship a threshold that only works at low noise.
*Depends on 3.*

**Step 7 — retrain and check the network.**
`train_and_save()` with the same `damping_rates=[0.05…0.30]`. Assert validation
accuracy and per-class recall still meet the bar in `test_ai_models.py`, and that
`test_network_agrees_with_the_analytic_cross_check` still passes. The CNN sees
raw normalised cards, not features, so it should adapt — but this is the claim
that needs proving, not assuming.
*Depends on 2, 4, 5, 6.*

**Step 8 — confirm the dashboard.**
Rebuild the frontend container, confirm the card renders as a parallelogram at
1280 and 1024 px, and that the rod-float overlay caption still reads sensibly
(it displays `asym`, whose scale has changed).
*Depends on 7.*

**Step 9 — update `talk.md`.**
Record the generator change, the discriminator and its measured separation, the
retrained accuracy, and the honest caveat that this is a parametric model, not a
field trace.
*Depends on all.*

---

## 6. Testing

**New unit tests** (step 5): the parallelogram is real; the width residual
separates healthy from floating across the randomised envelope; a degenerate
card (fewer than 4 usable bins) returns `0.0` instead of raising.

**Existing suites that must stay green**, all from the repo root:
```
./venv/bin/python -m pytest backend/tests/test_ai_models.py -q   # accuracy + features
./venv/bin/python -m pytest backend/tests/test_gibbs_solver.py -q
./venv/bin/python -m pytest backend/tests/test_closed_loop.py -q # 6-fault round trip
./venv/bin/python -m pytest -q                                  # 339 + 1 skipped
```

**Edge cases**:
- `samples=32` (the enforced minimum) — bin count drops; must not divide by zero.
- `noise_fraction=0.0` — exact parallelogram, residual exactly 0.
- A card whose position is not monotonic in time (the Gibbs solver re-phases
  onto BDC) — the binning keys off *position*, not sample index, so it must
  handle both orders. **This is the most likely place for the change to break
  something**, since every closed-loop card arrives through the solver.
- `downstroke_asymmetry` consumers: `test_api_stream.py::FEATURE_KEYS` (key set
  unchanged) and `DynoCardCanvas.tsx:189` (reads by name).

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Rebalancing the training distribution drops CNN accuracy | Medium | Steps 6–7 measure before and after; the old weights still exist in `simulation/trained_weights/`, so a failed retrain is a one-file revert. Do not delete them until step 7 passes. |
| Threshold 0.045 is fitted to the current generator | Medium | Step 6 tests 200 draws/label at max noise. Report the margin rather than asserting "it works". |
| `UNANCHORED_TUBING` loses its discriminator | **High** | The unanchored branch is the one that relied on asymmetry *sign*. Step 4 requires empirically validating the replacement before adopting it — this is where the plan is most likely to need a second pass. |
| Gibbs re-phasing breaks the position binning | Medium | Explicit edge case in §6; the closed-loop suite covers it, since every twin card comes through the solver. |
| Changing a feature's meaning without changing its name misleads a future reader | Certain | The docstring must be rewritten, and `talk.md` §3 records it. Renaming is *not* an option — two consumers pin the name. |
| A "successful" retrain that has merely memorised the new generator | Low | The test suite generates its cards with an independent seed from training, and `test_ai_models.py` uses a held-out split. |

**Standing rules from `talk.md` §7 apply**: no index-slicing patches, raw
docstrings, and no physics constant or threshold may be loosened to make a test
pass. Every threshold in this plan is set from measured separation, and that
measurement is reported.

---

## 8. Files to change

| File | Change |
|---|---|
| `backend/app/ai/dyno_classifier.py` | `synthesize_card()` valve-width term; `build_training_set()` sampling; `card_features()` width residual; `analytic_diagnosis()` thresholds |
| `backend/tests/test_ai_models.py` | Parallelogram regression test; retarget rod-float and unanchored assertions |
| `talk.md` | Record the change, the measured separation and the retrained accuracy |
| `frontend/src/components/DynoCardCanvas.tsx` | **Only if** the `asym` caption needs a scale note. Read-only otherwise. |

Not touched: `backend/app/physics/**` (the Gibbs solver is a transmission and
carries whatever shape it is given), `backend/app/core/config.py`, the legacy
prototype.

---

## 9. Execution order

1. **Step 1–3** — generator, sampling, feature. Land together; the feature is
   meaningless without the generator producing a width.
2. **Step 6** — measure the separation *before* writing any threshold. Cheap,
   and it can send the plan back to the drawing board while it is still small.
3. **Step 4–5** — thresholds, then tests.
4. **Step 7** — retrain, verify accuracy.
5. **Step 8–9** — dashboard, then `talk.md`.

Steps 6 and 7 are the two that can fail, and both are cheap to run in isolation.
If step 6 shows the margin is too thin, revert to the current behaviour — the
line-shaped card is a cosmetic defect with a correct diagnosis behind it, which
is a much better position than a wrong diagnosis with a pretty card.
