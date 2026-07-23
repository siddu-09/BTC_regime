# Regime Detection — Research Findings & Improvement Plan

Audit date: 2026-07-23. All numbers below were reproduced from the code and
artifacts in this folder, not taken from the README.

---

## TL;DR

The project contains a **genuinely strong signal (AUC 0.90)** that the current
setup **cannot see and does not report**, because the 4-class target mixes a
highly-predictable axis (volatility) with a completely unpredictable one
(direction), and then collapses both through `argmax` into an accuracy number.

| Claim | Status |
|---|---|
| "Model gets 37% vs 33% persistence, +4pt lift" | True but misleading — see below |
| The 74 features add predictive value | **False on accuracy.** A zero-feature Markov model scores higher |
| The post-prediction filter improves results | **False.** It contains look-ahead; causally it makes things worse |
| Volatility state is predictable | **True, and much more strongly than reported — AUC 0.90** |
| Direction is predictable | **False. AUC 0.515, straddles 0.50 in every fold** |

---

## 1. What the model actually learned

### 1.1 It collapsed to the majority class

Out-of-sample predicted vs actual distribution (n=3275):

| Regime | Predicted | Actual |
|---|---|---|
| bearish | **1.9%** | 28.0% |
| bullish | **6.4%** | 27.5% |
| chop_calm | 10.9% | 9.6% |
| chop_volatile | **80.7%** | 34.8% |

The model emits `chop_volatile` for 4 out of every 5 blocks. Per-class recall
confirms it: bearish 2.3%, bullish 5.9%. **It has essentially stopped
predicting direction at all.**

### 1.2 This collapse is not a bug — it is Bayes-optimal

The train transition matrix `P(next | current)`:

| current \ next | bearish | bullish | chop_calm | chop_volatile |
|---|---|---|---|---|
| bearish | 26.0 | 33.1 | 2.4 | **38.5** |
| bullish | 30.5 | 29.4 | 2.8 | **37.2** |
| chop_calm | 13.3 | 16.7 | **43.5** | 26.5 |
| chop_volatile | 25.9 | 30.0 | 4.9 | **39.2** |

The row-wise max is `chop_volatile` in 3 of 4 rows. Under `argmax`, **the
optimal decision rule *is* "always say chop_volatile except after chop_calm."**
The model found it. There is nothing more to squeeze from this formulation:
the ceiling if you knew `P(next|current)` exactly is **38.88%**.

### 1.3 Three trivial baselines that beat or match it

| Model | Out-of-sample accuracy |
|---|---|
| Always `chop_volatile` | 34.81% |
| Persistence (`next = current`) | 32.95% |
| **Zero-feature Markov (transition matrix only)** | **37.28%** |
| **One threshold on `median_range_pct`** | **36.98%** |
| XGBoost, 74 features, 265 trees | 36.85% |

The two-line rule is literally:

```python
pred = 'chop_calm' if median_range_pct < 0.0011 else 'chop_volatile'
```

And the Markov model beats XGBoost in **every single rolling-origin fold**:

| OOS year | XGB 4-class | Markov-only |
|---|---|---|
| 2022 | 37.1% | **38.2%** |
| 2023 | 37.3% | **39.5%** |
| 2024 | 35.7% | **37.4%** |
| 2025 | 37.0% | **37.4%** |
| 2026 | 36.8% | **37.0%** |

### 1.4 ~65 of the 74 features are noise

From `feature_importance.csv`, gain scores:

```
median_range_pct   39.1  ← real
avg_body_pct       23.6  ← real
avg_wick_pct       14.4  ← real
cnt6_chop_calm     12.5  ← real
realized_vol        9.9  ← real
session_04-08       7.9  ← calendar
is_weekend          6.5  ← calendar
...
[65 features flat in the 3.2 – 3.9 band = indistinguishable from noise]
prev1_chop_calm     0.0  ← never used by any tree
```

Every hand-crafted structural feature — `abs_efficiency` (3.44),
`wick_symmetry` (3.31), `control_strength_30m` (3.40),
`control_vote_strength` (3.85), `q4_agrees_with_block` (3.63) — sits in the
noise band. The exhaustion/control/wick feature families, which are the bulk of
`jane.py`'s 1132 lines, contributed nothing measurable.

Note also that pure calendar features (`session_*`, `is_weekend`,
`day_of_week`) outrank every structural feature. The model is partly fitting
intraday volatility seasonality.

---

## 2. Bugs found

### 2.1 CRITICAL — look-ahead in the post-prediction filter

`test.py:808`:

```python
for j in range(i - pending_count + 1, i + 1):
    filtered[j] = confirmed          # retroactively rewrites PAST predictions
```

When a transition is confirmed at block `i`, the filter goes back and rewrites
block `i-1`'s already-emitted prediction. At block `i-1` that information did
not exist. This is a look-ahead leak in the evaluation.

Measured impact, applied to the real raw model output:

| Variant | Accuracy |
|---|---|
| Raw model, no filter | 36.85% |
| Filter **as implemented** (retroactive) | **48.12%** ← fake |
| Filter made causal (no back-rewrite) | **31.57%** ← honest |

**If anyone reruns `test.py` today they will see ~48% and believe they found a
large edge.** They have not. Honestly evaluated, the filter is *worse* than no
filter, and worse than the 32.95% persistence baseline.

The reason is structural: hysteresis assumes regimes are sticky, but **67% of
blocks change regime**. Smoothing a target that churns two-thirds of the time
can only hurt.

### 2.2 HIGH — the README's numbers are not backed by the saved artifacts

`aa_results_test/predictions.csv` matches the **raw** argmax output at
**100.00%** of rows. The saved artifacts were produced by a version of
`test.py` that predates the §8 filter. So README §8 and its
"+4 to +15 points (raw vs. filtered)" claim correspond to no result on disk.

### 2.3 HIGH — `relabel.py` is missing

The README describes it as the source of ground truth. It is not in the folder.
`final_labels.csv` cannot be regenerated, thresholds cannot be audited or
tuned, and the labeling logic cannot be verified against the description.

### 2.4 MEDIUM — the scripts don't reproduce the saved results

`main.py` defaults to `OUT="hirst_results_main"`; `test.py` defaults to
`MODEL_DIR="hirst_results_main"`, `OUT="hirst_results_test"`. The actual
artifacts live in `latest_results_main/` and `aa_results_test/`. A plain
`python main.py && python test.py` writes to different directories than the
ones the README cites.

### 2.5 MEDIUM — label non-stationarity from fixed thresholds

`chop_calm` is **5.9%** of train but **9.6%** of test — a 63% relative jump.
The labeler uses hard-coded thresholds (range < 0.50%, body < 0.15%). As BTC's
realized volatility fell across 2025–2026, more blocks drifted under a fixed
bar. The label definition itself is drifting with the market.

### 2.6 LOW — 57–59% of the codebase is commented-out dead code

`main.py` 350/591 lines, `test.py` 552/957, `jane.py` 648/1132. Three
generations of superseded implementations are retained inline, which makes it
genuinely hard to tell which code produced which artifact — 2.2 above is a
direct consequence.

---

## 3. What IS predictable (the useful finding)

Binary reformulations, trained on 2020–2024, tested on 2025–2026. AUC 0.50 = no
signal.

| Question | AUC | Verdict |
|---|---|---|
| **Is the next block `chop_calm` (dead)?** | **0.9029** | **very strong** |
| **Is next-block range in the bottom tercile?** | **0.8205** | **very strong** |
| **Is next-block range in the top tercile?** | **0.8073** | **very strong** |
| Is the next block directional (bull *or* bear)? | 0.5804 | weak |
| Is the next block `chop_volatile`? | 0.5335 | ~nothing |
| **Bullish vs bearish, given directional?** | **0.5149** | **nothing** |

Stability across 5 rolling-origin folds:

| OOS year | calm AUC | direction AUC |
|---|---|---|
| 2022 | 0.9384 | 0.5340 |
| 2023 | 0.8522 | 0.5145 |
| 2024 | 0.9425 | 0.4985 |
| 2025 | 0.9080 | 0.5128 |
| 2026 | 0.8899 | 0.5356 |

**Volatility is strongly and stably predictable. Direction is not predictable
at all** — the direction AUC straddles 0.50 and dips *below* it in 2024.

Underlying driver is plain volatility clustering:

```
corr(range_t, range_t+1)          = 0.509
corr(log range_t, log range_t+1)  = 0.611
```

Single features already carry most of it (`avg_body_pct` alone: AUC 0.875;
`median_range_pct` alone: 0.872). All 74 features together reach 0.903 — so the
feature engineering adds about **+0.03 AUC** over one column.

### Why the 4-class accuracy hides all of this

`chop_volatile` is defined as *"high energy **AND** no directional winner."* The
"no winner" half inherits the unpredictable direction axis, which is why the
predictable energy signal (AUC 0.81–0.90) degrades to AUC 0.53 once it is
packaged as `chop_volatile`. The label design mixes signal with noise, and then
`argmax` discards what survives.

Corroborating evidence that the features *do* carry information that accuracy
cannot show: XGBoost log-loss **1.1977** vs Markov-only **1.2653**. The
probabilities are genuinely better. Only the decision rule throws it away.

---

## 4. Recommendations, ranked

### Tier 1 — Fix the measurement (do this first; no modeling required)

1. **Delete the retroactive rewrite** at `test.py:808`, or remove hysteresis
   entirely. It is look-ahead and it is not helping.
2. **Add the two real baselines** to `test.py`: zero-feature Markov, and the
   single-threshold rule. Persistence and majority-class are too weak to be
   informative here; any model must clear Markov (37.28%) to have earned its
   complexity.
3. **Stop reporting accuracy as the headline.** Use per-class AUC, log-loss and
   Brier score. Accuracy under 4-way argmax is structurally incapable of
   showing what this model knows.
4. **Report rolling-origin folds with error bars**, not one split. The
   single 2025-01-01 split hides that Markov wins every year.
5. Restore or rewrite `relabel.py` and commit it.

### Tier 2 — Reformulate the target (the actual fix)

6. **Split the target into two independent heads:**
   - **Energy head** — 3-class (calm / normal / volatile) or a direct
     regression on next-block `log(range)`. This is where the AUC 0.81–0.90
     lives. Ship this.
   - **Direction head** — keep it separate, report its AUC honestly, and expect
     ~0.51. Do not let it contaminate the energy label.

   **This is a measurement win, not an accuracy win.** Recombined back into 4
   classes it scores 36.43% — slightly *worse* than the current 36.70% (see
   §6). Its value is that it stops averaging an AUC-0.90 signal together with
   an AUC-0.515 one, so the half that works becomes visible and usable.

7. **Make the energy label quantile-based on a trailing window** (e.g. rolling
   1-year terciles of block range) instead of fixed 0.50%/0.15% thresholds.
   This removes the 5.9% → 9.6% drift in 2.5 and makes the label stationary by
   construction.

8. **Prune the feature set from 74 to ~10.** The flat 3.2–3.9 gain band is
   noise; dropping it costs nothing and reduces variance.

### Tier 3 — Add genuinely new information

Every current feature is derived from the same 1-minute OHLCV series, which is
why 65 of them are redundant. Real improvement needs *exogenous* inputs:

9. Perpetual **funding rate** and **open interest** deltas (regime-change
   leading indicators; you already have Binance infrastructure for these).
10. **Longer-horizon realized vol** (EWMA / GARCH-style at 1d, 3d, 7d) and
    **time-since-last-high-vol-block**.
11. **Scheduled event calendar** (CPI, FOMC) — these drive 4h volatility
    directly and are known in advance, which is exactly the kind of feature
    that survives out-of-sample.
12. Monotonic constraints on the vol features (`monotone_constraints`) — you
    know a priori that higher current range → higher next range.

Do **not** bother with class weighting or focal loss to fix the collapse. The
collapse is Bayes-optimal under argmax; only reformulation or a cost-sensitive
decision rule changes it.

### Tier 4 — If the goal is trading

13. The one decision-relevant output this dataset supports is **"will the next
    4h block be dead?"**, answerable at AUC 0.90. That is a
    **position-sizing / premium-selling** signal, not a directional one.
14. Evaluate it **economically, not by accuracy**: does gating an existing
    strategy on predicted-calm improve annualized Sharpe against an
    exposure-matched random-entry null and buy-and-hold? Per house rules,
    beating zero is not an edge, and per-trade Sharpe must be annualized before
    comparison.
15. Do not build anything directional on this dataset. Five independent folds
    say the information is not there.

---

## 5. Suggested order of work

| # | Action | Effort | Payoff |
|---|---|---|---|
| 1 | Remove look-ahead in the filter | 10 min | Stops a fake 48% from being believed |
| 2 | Add Markov + threshold baselines | 30 min | Honest bar for every future model |
| 3 | Switch headline metric to AUC / log-loss | 1 hr | Makes the real signal visible |
| 4 | Split target into energy + direction heads | half day | **37% → 0.90 AUC** |
| 5 | Quantile-based rolling energy labels | half day | Kills label drift |
| 6 | Prune 74 → 10 features | 1 hr | Less variance, faster |
| 7 | Add funding / OI / event-calendar features | 1–2 days | Only real path to *new* signal |
| 8 | Economic evaluation of the calm signal | 1–2 days | Decides if any of it is tradeable |

Items 1–3 are pure correctness and should happen regardless of direction. Item
4 is the highest payoff-per-hour in the project.

---

## 6. Will any of this improve accuracy? No.

Measured directly, same split, same features:

| Change | 4-class accuracy |
|---|---|
| Current model | 36.70% |
| Prune 74 → 10 features | **35.21%** (worse) |
| Hierarchical 2-head, recombined | **36.43%** (worse) |
| Fix the look-ahead bug | 48% → 37% (goes *down*, correctly) |

Because accuracy on this target is capped:

| Bound | Value |
|---|---|
| Ceiling from regime context alone | 38.88% |
| Model's calibrated expectation, `mean(max prob)` | 41.81% |
| Current | 36.70% |

Oracle decomposition shows where the missing accuracy lives:

| Oracle | Accuracy |
|---|---|
| **Perfect direction**, model's energy | 43.21% (+6.5 pp) |
| **Perfect energy**, coin-flip direction | **72.24%** (+35.5 pp) |

55.5% of blocks are bullish/bearish and direction is a coin flip, so **~27.7%
of all blocks are unwinnable by construction.**

### What does move

Ask a question that has an answer. Next-block energy tercile, balanced 3-class,
same features and split:

| Metric | Value |
|---|---|
| Random baseline | 33.3% |
| **Accuracy** | **62.0%** |
| Balanced accuracy | 54.3% |

**+28.7 points over baseline**, versus the current +1.9. Or keep the 4-class
target and abstain: 42.8% accuracy at 22% coverage, 51.1% at 9.7%, 73.9% at
2.1%. High accuracy on a slice, or low accuracy on everything — not both.

---

## 7. Are the blocks labelled correctly? Consistent — but not fully verified.

Audited against all 3,415,680 raw 1-minute candles, re-aggregated to 14,232
4-hour IST blocks (100% matched to `final_labels.csv`).

### First, a correction on method

An earlier version of this section presented the table below as proof the
labels are correct. **Most of those checks are tautological and could not have
failed.** `bullish` is *defined* as `net_pct >= +0.24%`, so "0 bullish blocks
closed down" and "0 violations of |net| >= 0.24%" restate the definition rather
than test it. They are worth running as bug-detectors, not as evidence of
correctness.

| Check | Result | What it actually proves |
|---|---|---|
| bullish blocks that closed DOWN | 0 / 4,175 | tautological |
| bearish blocks that closed UP | 0 / 3,827 | tautological |
| bullish/bearish violating \|net\| ≥ 0.24% | 0 / 8,002 | tautological |
| chop_volatile meeting the full directional test | 0 / 5,268 | tautological |
| chop_calm violating range < 0.50% | 0 / 962 | tautological |
| chop_calm violating 30m body < 0.15% | 0 / 962 | tautological |

What these **do** establish, legitimately: my block reconstruction is aligned
with the labeler's (wrong timezone, wrong block boundaries or a bad merge would
have produced large violation counts, not zero), and there are no sign errors
or off-by-one bugs.

Median statistics per class are exactly what the definitions describe:

| regime | range% | \|net\|% | efficiency | 30m body% | vol ratio |
|---|---|---|---|---|---|
| bearish | 1.668 | 0.874 | 0.540 | 0.257 | 1.197 |
| bullish | 1.626 | 0.864 | 0.559 | 0.252 | 1.095 |
| chop_calm | 0.384 | 0.102 | 0.284 | 0.069 | 0.412 |
| chop_volatile | 1.223 | 0.181 | 0.158 | 0.208 | 0.948 |

`chop_volatile` = high range, low net, low efficiency. `chop_calm` = low
everything. (The "clean separation gap" at efficiency 0.35 is also an artifact
of the threshold, not independent evidence.)

### The one test that could have failed — and mostly passed

Reimplementing the README's Steps 1–3 from scratch and comparing to
`final_labels.csv`:

| | |
|---|---|
| **exact match** | **94.37%** |

| actual \ README rules | bearish | bullish | chop_calm | chop_volatile |
|---|---|---|---|---|
| bearish | 3454 | 0 | 0 | **373** |
| bullish | 0 | 3834 | 0 | **341** |
| chop_calm | 9 | 7 | 877 | **69** |
| chop_volatile | 0 | 0 | 2 | 5266 |

So **801 blocks (5.63%) come from a branch the README does not specify well
enough to reproduce.** The mismatch is one-directional and structured: the
documented rules under-produce directional labels, calling 714 blocks
`chop_volatile` that the real labeler called `bullish`/`bearish`. That is the
shape you would expect from the Step-4 "control vote" fallback — and those 714
blocks do have a median |net| of 0.459% versus 0.181% for true `chop_volatile`,
so they are genuinely more directional. But an attempt to confirm the vote
mechanism directly (are they more one-sided across 30m candles?) came back
**identical across all three groups**, so the mechanism is unconfirmed. With
only ~8 30-minute candles per block that metric is coarse and the test is
underpowered — it fails to support the story rather than refuting it.

### RESOLVED — `relabel.py` was supplied 2026-07-23

All three open questions are now closed by direct inspection, and the labels
reproduce **exactly**.

#### Look-ahead: ruled out

`relabel.py:147`

```python
vol_med = df4h['volume'].rolling(VOL_WIN, min_periods=30).median()   # VOL_WIN = 540
```

`pandas.rolling()` is right-aligned, so the window is `[t-539 .. t]` — trailing
~90 days, never the future. Using the current block's own volume in its own
baseline is correct for a *descriptive* label. Re-running the calm-volume check
with the real 540-block window: **0 / 962 violations (0.00%)**.

My earlier probe tested windows of 20/30/50 and concluded the test "had no
power." It had no power because every candidate window was an order of
magnitude too short. **No future data reaches any label.**

#### The 5.6% gap: it was Step 5, not Step 4 — my earlier explanation was wrong

`relabel.py:113-120` contains a step the README does not document:

```python
# Step 5: Not directional but energetic -> chop_volatile
if energetic:
    if eff >= 0.30 and abs(net_ret) >= NET_TH and tfs_agree:   # <- 0.30, not 0.35
        return 'bullish' if net > 0 else 'bearish'
```

The effective efficiency threshold is **0.30, not the documented 0.35**. That
is the entire discrepancy. The "strength-weighted vote" story in the previous
draft was incorrect — Step 4 cannot produce these blocks, because Step 4 only
runs when `directional` is already True, which requires `eff >= 0.35`.

#### Exact reproduction

| rule set | match with `final_labels.csv` |
|---|---|
| README rules only (`eff >= 0.35`) | 94.98% |
| **including undocumented Step 5 (`eff >= 0.30`)** | **100.00%** |

All 14,232 labels are reproduced by three lines on the 4-hour candle:

```
calm       if range < 0.50% and vol_ratio < 0.80 and avg 30m body < 0.15%
bull/bear  elif efficiency >= 0.30 and |net| >= 0.24%
chop_vol   else
```

### NEW FINDING — the multi-timeframe cross-check does nothing

This is the headline feature of the labeling design, and it is inoperative.

`relabel.py:57`, inside `tf_control()`:

```python
net = c[-1] - o[0]
...
return int(np.sign(net)), float(eff * (abs(bd)*0.5 + abs(cd)*0.5))
```

For the eight 30-minute candles of a block, `c[-1]` is the block close and
`o[0]` is the block open. For the two 2-hour candles, likewise. So `d30`, `d2h`
and `d4h` are all `sign(block close − block open)` — **the same number computed
three times from the same two prices.**

Measured across the whole dataset: **blocks where `d30 != d4h`: 0 / 14,232.**

Consequences:

- `tfs_agree` is `True` for every block with a non-zero net move.
- Step 3's `directional and tfs_agree` collapses to just `directional`.
- **Step 4 — the "strength-weighted vote" — is unreachable dead code.** It has
  never executed once.
- The `strength` value `tf_control` carefully computes from body-dominance
  (`bd`) and candle-count-dominance (`cd`) is only ever consumed by Step 4, so
  it is computed 28,464 times and never used.
- In Step 5, `and tfs_agree` is a no-op, which is why the threshold is
  effectively 0.30.

This also explains a §1.4 result that previously had no explanation: the
`jane.py` features built to mirror this mechanism — `control_strength_30m`
(gain 3.40), `control_strength_2h` (3.44), `control_vote_strength` (3.85) —
sit in the noise band because **they mirror a mechanism that never fires.**

The fix is one line: `tf_control` should derive direction from the per-candle
aggregates it already computes (`bd`, the body-dominance, or `cd`, the
candle-count dominance) rather than from the aggregate `net`. That would make
the three timeframes genuinely independent and Step 4 reachable. It would
change some labels — but see the caveat below.

### But three of the four "regimes" are not regimes

This is the most important number in the whole audit:

| | |
|---|---|
| observed P(next == current) | 33.02% |
| expected if blocks were **completely independent** | 29.99% |
| **excess persistence** | **+3.02 pp** |

Broken out per class:

| regime | P(stay) | unconditional base rate | excess |
|---|---|---|---|
| bearish | 27.0% | 26.9% | **+0.1 pp** |
| bullish | 29.2% | 29.3% | **−0.1 pp** |
| chop_volatile | 38.4% | 37.0% | **+1.4 pp** |
| **chop_calm** | **43.9%** | **6.8%** | **+37.1 pp** |

**Knowing the current block is bullish tells you nothing about the next one** —
29.2% vs a 29.3% base rate. A regime is by definition a persistent state.
Bullish, bearish and chop_volatile have no memory whatsoever; they are accurate
*descriptions* of independent draws, not *states* the market occupies.

Only `chop_calm` behaves like a real regime (+37.1 pp), and that is precisely
the AUC 0.90 signal from §3. Every result in this audit is consistent with
that one fact.

### Secondary issue: boundary discretization

The directional/chop split hinges on a hard efficiency cut at 0.35:

| within ±X of the 0.35 cut | blocks | share |
|---|---|---|
| ±0.02 | 731 | 5.1% |
| ±0.05 | 1,812 | 12.7% |
| ±0.10 | 3,635 | **25.5%** |

A quarter of all blocks sit close enough to the boundary that an invisible
price difference flips their label. That is categorical noise stacked on top of
an already memoryless process.

### Does label uncertainty threaten the persistence finding? No.

Random label error *destroys* measured persistence, it cannot manufacture the
appearance of none. So the measured excess is a **lower** bound:

| if this share of labels were noise | true excess persistence |
|---|---|
| 0% | 3.02 pp |
| 10% | 3.73 pp |
| 25% | 5.38 pp |
| 50% | 12.10 pp |

Even under an absurd 50% noise assumption the excess stays far below
`chop_calm`'s +37 pp, and bullish's +0.1 pp only reaches +0.4 pp. Since the
labels are ~94% reproducible from deterministic rules, real noise is nowhere
near that. **The persistence conclusion is the load-bearing result of this
audit, it is computed from the label sequence alone, and it does not depend on
resolving any of the open questions above.**

### Conclusion — stated at the confidence the evidence supports

| Claim | Confidence |
|---|---|
| Labels reproduce exactly from audited source | **Certain** — 100.00% / 14,232 |
| The labeler contains no look-ahead | **Certain** — trailing 540-block window, verified 0/962 violations |
| The 5.6% gap is the undocumented Step 5 (`eff >= 0.30`) | **Certain** |
| The multi-timeframe cross-check is inoperative; Step 4 is dead code | **Certain** — 0/14,232 blocks where `d30 != d4h` |
| Direction has no memory at 4h | **High** — robust to label noise by construction |

The labels are **correct** — now established by exact reproduction from audited
source rather than by the tautological checks in the earlier draft. Every value
feeding a label is causal, and no future information enters the pipeline at any
point.

But the labeler is **not what it is documented to be**. It is advertised as a
three-timeframe agreement system with a strength-weighted tie-breaker; it is
actually a two-threshold rule on the 4-hour candle, with the tie-breaker
unreachable and the timeframe agreement always trivially true.

### Does fixing the labeler rescue the project? Almost certainly not.

Making `tf_control` genuinely multi-timeframe would change some labels, and it
is worth doing for honesty. But it will not create forecastability:

- The direction axis would still be `sign(net)` on a mostly-unpredictable
  quantity — §3 measured direction AUC at 0.515 across five independent folds.
- The energy axis is already the part that works (AUC 0.90), and it is
  unaffected by the timeframe bug, since Step 1's calm test never consults
  `tf_control` at all.
- A genuine Step 4 would only re-adjudicate blocks near the efficiency
  boundary — the same 25.5% band already identified as discretization noise.

So: fix the labeler because it does not do what it says, and remove the dead
code. Do not expect accuracy to move. The reason this project is stuck is §7's
persistence table, and no labeling change touches that.

---

## 8. "These aren't regimes — how can bull follow bear with no transition?"

Half right, and the half that is wrong is the expensive one.

### Right: this is a bar classifier, not a regime detector

A regime is a *persistent latent state*. §7 measured `bullish -> bullish` at
29.2% against a 29.3% base rate — literally zero memory. What `relabel.py`
produces is a **description of each 4-hour candle in isolation**, computed only
from that candle. Calling it regime detection oversells it, and the expectation
of transitions comes from the word "regime," not from anything in the code.

### But "bull directly followed by bear" is not an error — it is what BTC does

Measured on raw 4-hour returns, with no labels involved at all:

| lag | autocorr(return) | autocorr(sign) |
|---|---|---|
| 1 | −0.0295 | −0.0448 |
| 2 | −0.0231 | −0.0211 |
| 3 | +0.0520 | +0.0039 |
| 6 | −0.0515 | −0.0223 |
| **1, on \|return\|** | **+0.2977** | — |

Direction has no memory; **magnitude does**. And a runs test on the up/down
sequence gives observed 7,431 runs versus 7,113 expected under randomness,
**z = +5.33** — *more* runs than chance, meaning streaks are **shorter** than
random. BTC 4h alternates slightly more than a coin flip.

So an up bar followed immediately by a down bar is not only possible, it
happens **more often than chance**. There is no mechanism requiring price to
pause between them. The absence of transitions is a property of the market at
this horizon, not a defect in the labeling.

### The trap: forcing persistence manufactures a prettier number with less information

A causal EMA(20) state (`close > EMA` = up) gives exactly the shape the
intuition asks for:

| | |
|---|---|
| P(state persists) | **88.0%** |
| average run length | 8.4 blocks (**1.4 days**) |

That looks like a proper regime with clean transitions. Now the payoff:

| state | n | mean next 4h return | P(next block up) |
|---|---|---|---|
| up | 7,498 | +0.0351% | **49.6%** |
| down | 6,733 | −0.0081% | **52.5%** |
| unconditional | 14,232 | +0.0146% | 51.0% |

The regime is highly persistent and tells you **nothing** about the next
return — if anything the down-state is marginally more likely to be followed by
an up block, which is the same mild mean-reversion the runs test found.

And critically, **it would corrupt the scoreboard**: with sticky labels the
persistence baseline rises to 88%, so a model scoring "88% accuracy" would have
exactly zero edge. Smoothing raises accuracy and the baseline by the same
amount. Net information gain: zero — while the headline number looks 2.4x
better than today's 37%.

### What a genuine regime looks like in this data

Causal volatility state (current block's range vs its own trailing 540-block
terciles):

| state | n | median next-block range | P(next is high-vol) |
|---|---|---|---|
| low | 4,883 | 1.005% | **16.2%** |
| mid | 4,469 | 1.335% | 30.8% |
| high | 4,819 | 1.823% | **55.0%** |
| all | 14,171 | 1.357% | 34.0% |

Only 49.8% persistent — *less* sticky than the EMA direction state — but it
moves the next-block distribution by **3.4x** (16.2% -> 55.0%), where EMA
direction moved it by 1.06x (49.6% -> 52.5%).

**Persistence and information are different things.** The sticky label is
worthless; the informative label is only moderately sticky. Optimising for the
appearance of clean regime transitions selects for the wrong one.

### Conclusion

Do not relabel to add transitions to the directional classes. Relabel to
**delete the directional classes**. The volatility axis is the only one in this
dataset that behaves like a regime and carries information, and it is the same
conclusion every other section of this audit reached from a different
direction.

---

## 9. The stated goal: gate trend-strategies vs chop-strategies at 4h

Goal as given: *"find out whether the next four hours will be trending or
choppy, to gate strategies that work well in each."*

This is the **efficiency** axis (how much of the range ends up as net
displacement), not the direction axis. It had not been tested before now.

### 9.1 Trendiness has no memory

| quantity | lag-1 autocorrelation |
|---|---|
| efficiency `\|net\|/range` | **+0.0213** |
| path efficiency `\|net\|/distance walked` | **+0.0129** |
| range_pct (reference) | **+0.5092** |

Volatility persists. Trendiness does not. `corr(efficiency, range) = +0.215`,
so they are genuinely different quantities, not two views of one thing.

### 9.2 Nothing predicts it

Trained 2020–2024, tested 2025–2026, all 74 features:

| target | base | AUC |
|---|---|---|
| next efficiency > median (trending) | 53.4% | **0.4991** |
| next efficiency top third (strong trend) | 38.2% | 0.5049 |
| next efficiency bottom third (choppy) | 31.5% | 0.5098 |
| next path efficiency > median | 51.2% | 0.5092 |
| next path efficiency top third | 34.9% | 0.5238 |
| *next range top third (volatility)* | *17.8%* | ***0.8073*** |
| *next range bottom third (volatility)* | *51.5%* | ***0.8205*** |

Five definitions of "trending," all at chance. The volatility axis on identical
data and features reaches 0.81–0.82.

### 9.3 Even a perfect oracle would not deliver the gate

Filtering to blocks where the next block *actually is* trending or choppy, and
measuring momentum P&L:

| oracle | n | momentum Sharpe |
|---|---|---|
| knows next block is TRENDING | 4,813 | **+0.39** |
| knows next block is CHOPPY | 4,675 | −1.91 |

A perfect trend/chop oracle buys a momentum Sharpe of 0.39. The reason is
structural: **"trending" is directionless.** Knowing the market will move
decisively does not say which way, and a trend-following position must be on
the right side. This caps the value of the gate independently of whether it can
be predicted — and it cannot.

(Caveat: this applies to *momentum-continuation* strategies, which set position
from the prior block's direction. An *intrabar breakout* strategy would benefit
from directionless trendiness — but 9.2 shows the forecast is unavailable
either way.)

### 9.4 Both candidate gates die on costs

Two causal gates looked promising gross and in-sample. Neither survives.

**Gate A — trade momentum when the previous block was trending** (efficiency in
top third):

| cost | in-sample Sharpe | out-of-sample Sharpe |
|---|---|---|
| gross | +0.61 (t=1.37) | +0.30 (t=0.37) |
| 2 bp/side | +0.04 | −0.64 |
| 5 bp/side | −0.82 | −2.05 |

gross edge +1.02 bp/block, 0.571 position flips per block →
**breakeven 1.80 bp/side.**

**Gate B — trade momentum when volatility is low** (the best cell in the
state table, momentum Sharpe 0.86 gross):

| cost | in-sample Sharpe | out-of-sample Sharpe |
|---|---|---|
| gross | +0.38 (t=0.86) | +1.09 (t=1.33) |
| 2 bp/side | −0.44 | −0.02 |
| 5 bp/side | −1.67 | −1.69 |

gross edge +0.60 bp/block, 0.513 flips per block →
**breakeven 1.16 bp/side.**

Binance taker is 4–5 bp/side before slippage. Both gates are under the floor by
**2–4x**, and neither reaches statistical significance gross (all |t| < 1.4).
The eye-catching "Sharpe 0.95" from the first pass was in-sample, gross, and
measured only on the subset rather than as a deployable always-on strategy.

### 9.5 Verdict on the stated goal

**Trending-vs-choppy at 4h cannot be forecast, and would be worth less than
expected even if it could.** This is not a labeling problem or a model problem —
the quantity has an autocorrelation of 0.02.

What survives from the whole audit is one thing, and it is robust across every
test thrown at it: **next-block volatility is strongly predictable** (AUC 0.90
calm, 0.82 tercile, macro AUC 0.74 for a 3-state regime, stable across five
rolling-origin folds).

That is a real regime predictor at 4h. It is not a trend/chop gate. Its natural
uses are decisions whose payoff depends on **variance rather than direction** —
position sizing, stop and liquidation-distance planning, and selecting
volatility-sensitive strategies such as short-premium books. Whether it
improves any specific live strategy is a separate question that requires that
strategy's P&L, and is not answered here.

---

## 10. Fine-tooth-comb audit: OHLCV handling and feature construction

### 10.1 The data is pristine

| check | result |
|---|---|
| rows | 3,415,680 |
| duplicate timestamps | **0** |
| OHLC violations (high < low, high < open, …) | **0** |
| missing minutes vs a complete 1-min calendar | **0 (0.000%)** |
| gaps > 1 minute | **none** |
| zero/negative-volume bars | 369 (0.011%) |
| 1-min candles per 4h block | **exactly 240, every block** |
| 15-min candles per block | **exactly 16, every block** |

No exchange downtime, no partial blocks, no resampling artifacts. The 369
zero-volume minutes are genuine no-trade minutes, and all divisions by volume
are guarded with `max(..., 1e-10)`.

### 10.2 The features are computed correctly and causally

Ten features were recomputed from the raw 1-minute CSV by an independent
implementation, using only data available at each block's close:

| feature | max abs diff | verdict |
|---|---|---|
| block_range_pct | 9.97e-17 | **EXACT** |
| median_range_pct | 1.00e-16 | **EXACT** |
| avg_body_pct | 1.00e-16 | **EXACT** |
| abs_efficiency | 1.11e-16 | **EXACT** |
| abs_net_pct | 1.00e-16 | **EXACT** |
| close_position | 1.11e-16 | **EXACT** |
| color_uniformity | 0.00e+00 | **EXACT** |
| realized_vol | 1.00e-16 | **EXACT** |
| body_imbalance | 1.11e-16 | **EXACT** |
| counter_energy | 9.71e-17 | **EXACT** |

`vol_ratio` also matches `relabel.py`'s definition to 2.66e-15. All differences
are float rounding. **Block alignment is correct and there is no look-ahead in
the feature layer** — a purely causal reimplementation reproduces the shipped
file bit-for-bit. The only `shift(-1)` in `jane.py` builds the target.

Also verified: **zero NaN across all 74 features**, so the `fillna(0)` in
`main.py:514` is a no-op rather than a silent corruption; and no constant
features.

### 10.3 Defect 1 — jane.py uses 15-minute candles, the labeler uses 30-minute

`jane.py:983` resamples to **15min** and feeds `features_15m()`.
`relabel.py:139` resamples to **30min** for the calm test's `avg_body_pct`.

`jane.py`'s docstring says the features exist to "mirror the labeler's
decision." They do not mirror the quantity the labeler actually uses:

| | |
|---|---|
| jane `avg_body_pct` (15m) mean | 0.2003% |
| true 30m avg body (what the label tests) | 0.2806% |
| ratio | **1.401x** |
| correlation | 0.9495 |
| **blocks disagreeing on the 0.15% calm test** | **2,612 (18.4%)** |

The feature is a decent proxy (r = 0.95) but it is systematically 1.4x smaller
than the threshold quantity, so the model must relearn a shifted cut on a
noisier variable. Either resample to 30min in `jane.py`, or add the true 30m
body as its own feature.

### 10.4 Defect 2 — `body_imbalance` is an exact duplicate of `counter_energy`

```
body_imbalance = |bull_body - bear_body| / total_body
counter_energy = (losing-side body) / total_body
```

These satisfy `body_imbalance = 1 - 2 * counter_energy` whenever the net
direction matches the body-dominant side. Measured correlation: **−0.999995**.
One of the two carries no information the other lacks.

More broadly, six pairs exceed |r| > 0.95:

| pair | \|r\| |
|---|---|
| body_imbalance ~ counter_energy | **1.0000** |
| body_imbalance ~ path_efficiency | 0.9929 |
| counter_energy ~ path_efficiency | 0.9929 |
| control_strength_2h ~ control_vote_strength | 0.9773 |
| avg_body_pct ~ realized_vol | 0.9629 |
| avg_wick_pct ~ median_range_pct | 0.9501 |

`body_imbalance`/`counter_energy`/`path_efficiency` are a single 3-way
redundant cluster. This is consistent with §1.4: the effective feature count is
closer to 8–10 than 74.

### 10.5 Defect 3 — `control_vote_strength` mirrors dead code, with a different formula

`jane.py:1034` computes `vote = d30*s30 + d2h*s2h`.
`relabel.py:108` computes `votes = d30*s30 + d2h*s2h + d4h*eff`.

The feature omits the `d4h*eff` term, so it is not the labeler's vote even
nominally — and per §7 the labeler's Step 4 never executes, so it mirrors a
branch that has never run. Since `d30 == d2h` always (same §7 bug, as
`tf_control` is copied verbatim into `jane.py:820`), the feature reduces to
`s30 + s2h`. That explains its 0.977 correlation with `control_strength_2h`.

### 10.6 Defect 4 — the "unsigned single-block feature" invariant is violated

`jane.py:742` states single-block features are unsigned so the model cannot
build a "current direction = next direction" bias. Correlating every feature
against `sign(close − open)` of its own block:

| feature | corr with direction | status |
|---|---|---|
| **close_position** | **+0.6435** | **violates** |
| cur_bullish / cur_bearish | +0.63 / −0.62 | signed by design, OK |
| dir_balance_6 / _12 | +0.29 / +0.21 | signed by design, OK |
| **winner_rejection** | **−0.1166** | **violates** |
| q4_winner_rejection | −0.0490 | within tolerance |

`close_position = (close − low) / range` is strongly directional: a block
closing near its high is a bullish block. The unsigned form would be
`|close_position − 0.5| × 2` ("how extreme was the close, either side").

Since direction is unpredictable (§3), these two features act as a pure
overfitting channel for the directional classes. Both sit in the noise band of
feature importance, so the damage was limited — but the stated invariant is not
enforced anywhere. A one-line static test would catch it.

(`momentum_decay` looked like a violation on code reading — it is a signed
difference of signed returns — but measures |r| < 0.05 against direction. It is
an acceleration term, not a direction term. Flagged and cleared.)

### 10.7 Verdict

**The OHLCV data is used correctly.** Parsing, timezone handling, resampling
origin, block assignment, aggregation and label joins are all exact, and the
feature layer is fully causal. Nothing in this section changes any conclusion in
§1–§9 — the model's failure is not caused by a data or feature bug.

The four defects are quality issues, not correctness bugs:

| # | defect | severity | fix |
|---|---|---|---|
| 1 | 15m features vs 30m label quantity | medium | resample to 30min, or add true 30m body |
| 2 | `body_imbalance` ≡ `counter_energy`; 6 redundant pairs | medium | drop duplicates |
| 3 | `control_vote_strength` mirrors dead code, wrong formula | low | delete with Step 4 |
| 4 | `close_position` / `winner_rejection` leak direction | low | use unsigned forms; add a static invariant test |

---

## 11. Separating real signal from noise — the harness

This project generated four convincing-looking results that were all noise.
That is the actual problem to solve, and it is solvable mechanically. Each gate
below is the one that would have caught one of them. Implemented in
`validate.py`.

### 11.1 The single most important number

Running the same model and split against a **block-shuffled** null (shuffle the
target in contiguous 30-block chunks, preserving its autocorrelation):

| claim | observed | null | p | z | verdict |
|---|---|---|---|---|---|
| calm detection | 0.9037 | 0.5473 ± 0.0236 | **0.000** | **+15.09** | **REAL SIGNAL** |
| direction | 0.5140 | 0.5002 ± 0.0141 | **0.167** | +0.97 | **NOISE** |

Direction is now formally dead: p = 0.167. Not a judgement call.

But look at the calm null: **0.5473, not 0.50.** Compare the two null
constructions on the identical task:

| null construction | null mean |
|---|---|
| naive row shuffle | **0.4996** |
| block shuffle (preserves autocorrelation) | **0.5435** |

**An AUC of 0.54 on an autocorrelated target is what pure noise produces.**
A naive shuffle would score 0.54 at z = +1.9 and call it a discovery. The
honest null is 4.4 points higher, and 0.54 is exactly its centre.

This is why the null must match the target's structure. `chop_calm` clusters in
time (+37 pp persistence, §7), so block-shuffling leaves the model able to
exploit that clustering even with scrambled labels — and that free AUC must be
subtracted before claiming anything. Direction has no autocorrelation, so its
block-shuffled null collapses back to 0.5002.

**Rule: never compare an AUC to 0.50. Compare it to its own permuted null.**

### 11.2 Multiple testing — "best of N" needs its own null

I earlier tested 10 conditional slices for direction and reported the best,
weekend at 0.5435. Simulating the maximum of 10 independent noise tests at
SE = 0.028:

| | |
|---|---|
| median of max-of-10 | **0.5421** |
| 90th percentile | 0.5645 |
| 99th percentile | 0.5862 |
| **observed 0.5435** | **53rd percentile** |

The "best" slice landed at the *median* of what noise produces. Report
best-of-N against the max-statistic distribution, never against the
single-test null.

### 11.3 The five gates

| # | gate | catches | this project's false positive |
|---|---|---|---|
| 1 | **permutation test**, structure-preserving null | "the model found something" | direction AUC 0.515 (p=0.167); 65 noise features |
| 2 | **max-statistic null** | "the best of N configs worked" | weekend slice 0.5435 (53rd pct of noise) |
| 3 | **causal recomputation diff** | look-ahead | filter reported 48.12% vs true 31.57% |
| 4 | **cost floor, computed before OOS** | "great gross Sharpe" | gate breakeven 1.80bp vs 4–5bp taker |
| 5 | **ablation** — delete the line that arms it | dead code masquerading as a feature | Step 4 never executes; `control_vote_strength` |

Gate 5 is the one with no statistics in it and it caught the largest defect:
`d30 == d2h == d4h` by construction, so the entire multi-timeframe cross-check
was inoperative. Deleting it would have changed nothing — which is the test.

### 11.4 Reporting standard

Stop reporting bare metrics. Every claim gets five numbers:

```
observed | null mean | null sd | p | z
```

A bare "AUC 0.58" is unfalsifiable. `obs=0.5804 null=0.5473+/-0.0236 p=0.09
z=+1.40` is a result — and in that case, a negative one.

Same for P&L: annualised Sharpe **with its standard error** (`|t| < 2 is not a
result`) and breakeven cost per side, computed before anyone looks at the
out-of-sample curve.

---

## 12. relabel.py v2 — results

Rewritten to spec: 15m/1h/2h/4h ladder, 78/22 evidence-vs-memory blend, memory
over the previous 3 labels at 50/30/20, conditional inertia, high-conviction
override, raised thresholds, path-efficiency gate. Original preserved at
`relabel.py.bak.pre_v2_rewrite.20260723`.

Added constraint not in the spec: **a directional label's sign always equals
`sign(close − open)`.** Memory gates *whether* a block is directional, never
*which way*. Without this, a strong bullish memory plus a weak bearish block
could label a down-closing block bullish, destroying the descriptive validity
verified in §7.

### 12.1 Label-level results

| | v1 | v2 |
|---|---|---|
| transition rate | 67.0% | **45.1%** |
| one-block round trips | 2,888 | **2,547** (−12%) |
| directional labels | 56.2% of blocks | **27.2%** |
| chop_volatile | 37.0% | 68.9% |
| **excess persistence** | **+3.0 pp** | **+3.8 pp** |
| agreement with v1 | — | 68.0% |

Transitions fell 22 points. **Excess persistence rose 0.8 points.** Of the
21.9 pp gain in raw persistence, 21.1 pp is class concentration (the
independence baseline rose from 30.0% to 51.1%) and 0.8 pp is added structure.
This is the §8 prediction, confirmed on live output.

### 12.2 Model-level results — the headline is a mirage

| | v1 labels | v2 labels |
|---|---|---|
| **accuracy** | 37.07% | **66.96%** |
| majority baseline | 34.81% | 65.31% |
| **lift over majority** | **+2.26 pp** | **+1.65 pp** |
| lift over persistence | +4.12 pp | +15.63 pp |
| lift over Markov | −0.21 pp | +1.65 pp |
| **balanced accuracy** | **38.08%** | **35.30%** |
| predicts chop_volatile on | 66.8% | **94.2%** |

Accuracy nearly doubled and information content fell. Lift over majority
dropped, balanced accuracy dropped 2.8 points, and the model collapsed harder —
predicting `chop_volatile` on 94.2% of blocks against a 65.3% base rate. The
"+15.63 pp lift over persistence" is the number that looks best and means least.

**Do not use the 4-class model on v2 labels.**

### 12.3 What genuinely improved — a real directional signal

Testing "bullish vs bearish, given the block is directional", with a
block-shuffled null:

| labels | features | n | observed | null | p | verdict |
|---|---|---|---|---|---|---|
| v1 | all | 1819 | 0.5140 | 0.5002 ± 0.0141 | 0.167 | **NOISE** |
| v2 | all | 891 | 0.5899 | 0.5033 ± 0.0200 | <0.033 | REAL |
| **v2** | **price-only** | **891** | **0.5504** | **0.4905 ± 0.0230** | **<0.033** | **REAL (z=+2.6)** |

The third row is the one that counts. Dropping all 22 label-derived features
(`cur_*`, `prev1_*`, `prev2_*`, `cnt6_*`, `dir_*`, `regime_streak`,
`move_shrinking`) removes the label-memory recursion entirely. Direction
survives at **AUC 0.5504 against a 0.4905 null, z = +2.6**.

**This is the first real directional signal found anywhere in this audit.**
Restricting "directional" to genuinely strong moves — efficiency ≥ 0.45,
|net| ≥ 0.35%, path efficiency ≥ 0.25, vote ≥ 0.30, timeframe agreement — made
direction predictable on that subset where it was pure noise across all blocks
(§3, §9). Higher precision bought predictability, which is exactly the trade
the spec asked for.

### 12.4 Caveats before anyone acts on 12.3

1. **n = 891 test blocks**, and z = +2.6. Suggestive, not settled.
2. **Multiple-testing exposure is real.** This session ran dozens of tests.
   Per §11.2, a +2.6σ result found after extensive search needs confirmation
   on a split not used during the search.
3. **AUC 0.5504 is 6 points over null** — small. Whether it clears the cost
   floor (§9.4: breakeven must exceed 4–5 bp/side) is untested.
4. It applies to **27% of blocks**, not all.

Next step is confirmation, not deployment: re-run on a held-out period never
touched here (e.g. train ≤2023, test 2024 only), then compute the cost floor
before looking at any P&L curve.

### 12.5 Net assessment

The rewrite did what was specified and fixed a real bug. On the stated
objective — "allow XGBoost to learn persistent market structure" — the 4-class
framing got **worse**, because memory adds stickiness rather than information.
But the higher-precision directional definition surfaced a genuine, if small,
directional edge that survives removal of its own recursion.

Recommendation: **drop the 4-class model; keep the v2 labels as a definition of
a high-precision directional subset, and model that subset as a binary
problem** — with §12.4's confirmation run first.

Two fixes follow, and only one of them is a relabeling job:

- **Fixable by relabeling** — replace the hard 0.35 efficiency cut with a
  quantile-based or continuous target, removing the 25.5% boundary noise.
- **Not fixable by relabeling** — direction has no memory. No label definition
  can create persistence that isn't in the price. Drop direction from the
  target and model the energy axis, which is the only axis that persists.
