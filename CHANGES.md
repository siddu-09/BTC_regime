# Changes — BTC 4h Regime Detection (v1 → v2)

> **SUPERSEDED.** This file records the v1 → v2 rewrite only. The current
> pipeline is v2.1, and the labeling logic, feature set, change log and
> recommendations are all maintained in **`PIPELINE-SPEC.md`** — read that
> instead. This file is kept for the v2 detail it contains.

Session date: 2026-07-23. Record of the labeling logic, the feature set, and
every change made in the v1 → v2 rewrite.

Companion documents:
- `PIPELINE-SPEC.md` — **current** logic, changes and recommendations
- `RESEARCH-FINDINGS.md` — the full audit and evidence behind these decisions
- `HANDOFF.md` — short version for whoever picks this up next
- `config.py` — every tunable, with the measurement behind each default
- `validate.py` — the signal-vs-noise harness referenced throughout

---

## 1. Files changed

| file | status |
|---|---|
| `relabel.py` | **rewritten** (v2). Original at `relabel.py.bak.pre_v2_rewrite.20260723` |
| `jane.py` | 5 bug fixes in the feature layer |
| `test.py` | look-ahead removed; default dirs corrected |
| `main.py` | default output dir corrected |
| `validate.py` | **new** — permutation / null-testing harness |
| `RESEARCH-FINDINGS.md` | **new** — audit report |
| `HANDOFF.md` | **new** — handoff note |
| `CHANGES.md` | **new** — this file |

Data files untouched: `BTC_IST.csv`, `final_labels.csv`, `latest_features.csv`.
New outputs: `labels_v2.csv`, `features_v2_fixed.csv`.

---

## 2. The labeling logic (relabel.py v2)

### 2.1 What a label means

Each 4-hour IST block gets exactly one of `bullish` / `bearish` / `chop_calm` /
`chop_volatile`. The label answers **"who controlled this block?"**, not "what
did one candle do".

Blocks are the six fixed IST sessions: 00-04, 04-08, 08-12, 12-16, 16-20, 20-00.

### 2.2 Timeframe ladder

v1 used 30m / 2h / 4h. v2 uses **15m / 1h / 2h / 4h**:

| TF | candles per block | role |
|---|---|---|
| 15m | 16 | detects an impulse starting mid-block |
| 1h | 4 | early structure |
| 2h | 2 | main structural witness |
| 4h | 1 | confirms net displacement |

### 2.3 Per-timeframe control — `tf_control()`

For each timeframe, from its own candles:

```
body_dom  = (bull_body - bear_body) / total_body        in [-1, +1]
count_dom = (n_bull - n_bear) / n                        in [-1, +1]
direction = sign(0.6 * body_dom + 0.4 * count_dom)
efficiency= |last_close - first_open| / (max_high - min_low)
strength  = efficiency * (0.5*|body_dom| + 0.5*|count_dom|)
```

`direction` comes from **internal candle structure**, not from the aggregate net
move. 0.6/0.4 favours body over count so three tiny green candles do not outvote
one large red one. `strength` is unsigned, in [0, 1].

For the 4h view (one candle) `body_dom` and `count_dom` collapse to ±1, so
`strength` reduces to efficiency — exactly the confirming role intended.

> **This is the v1 bug.** v1 set `direction = sign(c[-1] - o[0])`. For every
> sub-timeframe of a block that equals (block close − block open), so 30m, 2h
> and 4h returned the same number. Measured: `d30 != d4h` on **0 of 14,232**
> blocks. `tfs_agree` was therefore always True and the Step-4 weighted vote was
> unreachable dead code.

### 2.4 Weighted current evidence

```
current = 0.15*d15*s15 + 0.25*d1h*s1h + 0.35*d2h*s2h + 0.25*d4h*s4h
```

2h carries the most weight: it is the coarsest view that still exposes internal
structure, so it best answers "did control hold across the whole block". 4h
sees displacement but no path, so it confirms rather than decides. 15m is
noisiest, hence smallest, but is the only view that catches a mid-block impulse.

### 2.5 Memory

The previous **3** labels, exponentially weighted:

```
memory = 0.50*label(t-1) + 0.30*label(t-2) + 0.20*label(t-3)
where bullish=+1, bearish=-1, chop_calm=0, chop_volatile=0
```

Three blocks = 12 hours: long enough to represent the recent state, short
enough not to drag a stale regime across a session.

```
blended = 0.78 * current + 0.22 * memory
```

At 22%, memory shifts the effective efficiency requirement by at most 0.10
(0.45 → 0.55) — enough to reject a marginal counter-trend block, never enough
to reject a genuine reversal, which clears 0.60 by definition.

### 2.6 HARD INVARIANT

**A directional label's sign always equals `sign(close − open)`.**

Memory and multi-timeframe evidence decide *whether* a block is directional,
never *which way*. Without this, strong bullish memory plus a weak down block
could label a down-closing block bullish. Added during the rewrite; not in the
original spec.

### 2.7 Decision order

```
1. DEAD BLOCK
   range < 0.45% AND vol_ratio < 0.85 AND avg 15m body < 0.135%
   -> chop_calm

2. CURRENT EVIDENCE
   per-TF direction + strength; weighted score; all_agree flag; quality metrics

3. HIGH-CONVICTION OVERRIDE  (memory fully bypassed)
   all 4 TFs agree AND efficiency >= 0.60 AND |blended| >= 0.70
   AND vol_ratio >= 1.50 AND |net| >= 0.80% AND path_eff >= 0.25
   -> bullish / bearish, labelled on this exact block

4. MEMORY-ADJUSTED DIRECTIONAL TEST
   opposing memory:  required_eff = 0.45 + 0.10*|memory|
   agreeing memory:  required_eff = 0.45 - 0.02*|memory|
   all TFs agree:    floor lowered to 0.40 (+ inertia penalty if opposing)

   directional if:  efficiency >= required_eff
                AND |net| >= 0.35%
                AND |blended| >= 0.30
                AND sign(blended) == sign(net)
                AND path_eff >= 0.25
   -> bullish / bearish

5. OTHERWISE -> chop_volatile
```

**Memory never assigns a label.** If evidence fails, the block becomes
`chop_volatile` — not a repeat of the previous regime.

### 2.8 Thresholds and why

| constant | v1 | v2 | reasoning |
|---|---|---|---|
| `EFF_DIR` | 0.30 eff. | **0.45** | at 0.30 a block retracing two thirds of its range still counted as a trend — the largest source of false trends |
| `EFF_BORDERLINE` | 0.30 | **0.40** | reachable only on full TF agreement, a far stronger condition now that TFs are independent |
| `EFF_HIGH` | — | **0.60** | ~80th percentile of BTC 4h block efficiency; selects the clearly-trending tail |
| `NET_MIN` | 0.24% | **0.35%** | filters moves too small to matter against 4–9 bp round-trip costs |
| `NET_HIGH` | — | **0.80%** | ~2.3x the minimum |
| `VOTE_TH` | 0.15 | **0.30** | on the blended score in [-1,+1] |
| `VOTE_HIGH` | — | **0.70** | high-conviction gate |
| `CALM_RANGE` | 0.50% | **0.45%** | slightly stricter dead-block test |
| `CALM_VOL` | 0.80 | **0.85** | slightly looser volume gate, offsetting the tighter range |
| `PATH_EFF_MIN` | — | **0.25** | new; rejects blocks that walked > 4x their net displacement |
| `INERTIA_MAX` | — | **0.10** | max extra efficiency demanded of a reversal |
| `CONT_DISCOUNT` | — | **0.02** | small bar reduction when the block agrees with memory |
| `HC_VOL_RATIO` | — | **1.50** | volume confirmation for the override |
| `VOL_WIN` | 540 | 540 | ~90 days of 4h blocks, trailing (unchanged) |

### 2.9 Quality measures — `block_quality()`

Computed from the 15m candles. Only `path_eff` is a hard gate; the rest are
diagnostics, kept few to stay explainable.

| measure | formula | meaning |
|---|---|---|
| `path_eff` | \|net\| / Σ\|Δclose\| | walked there vs zigzagged there |
| `overlap` | mean overlap of consecutive candles | high = rotation |
| `persistence` | share of movement in the net direction | directional conviction |
| `wick_imbalance` | \|up_wick − dn_wick\| / total wick | rejection asymmetry |

### 2.10 Causality

Every input is available at the block's close. Memory reads only
already-assigned previous labels. The volume baseline is
`rolling(540).median()` — pandas rolling is right-aligned, window `[t-539, t]`.
**No future data enters any label.** Verified: an independent causal
reimplementation reproduces the shipped feature file bit-for-bit (max diff
1e-16).

### 2.11 Results

| | v1 | v2 |
|---|---|---|
| transition rate | 67.0% | **45.1%** |
| one-block round trips | 2,888 | **2,547** (−12%) |
| directional labels | 56.2% of blocks | **27.2%** |
| chop_volatile | 37.0% | 68.9% |
| chop_calm | 6.8% | 5.2% |
| **excess persistence** | **+3.0 pp** | **+3.8 pp** |
| agreement with v1 | — | 68.0% |

Decision paths taken: `no_control` 68.9%, `directional` 25.5%, `calm` 5.2%,
`high_conviction` 0.4%.

> **Read excess persistence, not transition rate.** Of the 21.9 pp gain in raw
> persistence, 21.1 pp is class concentration (the independence baseline rose
> from 30.0% to 51.1%) and only 0.8 pp is added structure. Because `label(t)`
> depends on `label(t-1..t-3)`, stickiness inflates accuracy and the persistence
> baseline by the same amount.

---

## 3. The feature set (jane.py)

**75 features**, one row per 4h block, all computed from 15m candles of that
block plus causal context. Target is `next_regime` = `regime.shift(-1)`.

### 3.1 Design invariant

Single-block features are **unsigned** — they describe how strong a move was,
never which direction. Direction enters only via `cur_*` one-hots and the
explicitly signed multi-block `dir_balance_6` / `dir_balance_12`.

### 3.2 Energy / volatility (6)

`block_range_pct`, `median_range_pct`, `avg_body_pct`, `avg_wick_pct`,
`big_candle_freq`, `realized_vol`

**This group carries essentially all the predictive signal in the dataset.**

### 3.3 Control / conviction (8)

`abs_efficiency`, `path_efficiency`, `abs_net_pct`, `color_uniformity`,
`dir_consistency`, `structure_strength`, `longest_run`, `close_position`

### 3.4 Wicks / rejection (5)

`wick_symmetry`, `wick_to_body`, `winner_rejection`, `counter_energy`,
`max_pullback_ratio`

### 3.5 Overlap (1)

`avg_overlap`

### 3.6 Exhaustion — final quarter (6)

`q4_range_expansion`, `q4_range_share`, `q4_agrees_with_block`,
`q4_wick_to_body`, `q4_winner_rejection`, `q4_volume_ratio`

### 3.7 Decay / late-block (4)

`momentum_decay`, `counter_tail_count`, `last_candle_wick_pct`, `late_giveback`

### 3.8 Halves (5)

`first_half_magnitude`, `second_half_magnitude`, `halves_agree`,
`half_momentum_shift`, `reversal_point`

### 3.9 Multi-timeframe (5) — rebuilt this session

`control_strength_15m`, `control_strength_1h`, `control_strength_2h`,
`control_vote_strength`, `tf_all_agree`

Mirrors relabel v2 exactly: same `tf_control`, same `TF_WEIGHTS`.
`control_vote_strength` = |weighted 4-TF evidence score|.

`control_strength_4h` is **not** emitted — for a single candle strength
collapses to efficiency, measuring r = 1.0000 with `abs_efficiency`. Still
computed internally for the evidence score and the agreement flag.

### 3.10 Volume (1)

`vol_ratio` — block volume / trailing 540-block median. Matches relabel.py
exactly (verified to 2.7e-15).

### 3.11 Regime context (19)

`cur_*` (4), `prev1_*` (4), `prev2_*` (4), `cnt6_*` (4), `regime_streak`,
`dir_streak`, `move_shrinking`

### 3.12 Signed trend context (2)

`dir_balance_6`, `dir_balance_12` — the only deliberately signed features.

### 3.13 Multi-block price context (5)

`range_3b`, `efficiency_3b`, `prev_range`, `range_change`,
`prev_momentum_decay`

### 3.14 Calendar (8)

`session_*` (6), `day_of_week`, `is_weekend`

---

## 4. Bugs fixed

| # | bug | file | verification |
|---|---|---|---|
| 1 | `tf_control` returned `sign(c[-1]-o[0])` — identical on all sub-TFs | `relabel.py`, `jane.py` | `tf_all_agree` now varies (89.4%), was constant |
| 2 | **Look-ahead**: hysteresis retroactively rewrote emitted predictions | `test.py:808` | 48.12% (fake) → **31.57%** (causal) |
| 3 | `body_imbalance` ≡ `1 − 2·counter_energy` | `jane.py` | removed; exact duplicates **1 → 0** |
| 4 | `control_vote_strength` omitted the 4h term, used broken direction | `jane.py` | now mirrors relabel v2 |
| 5 | `close_position` leaked direction | `jane.py` | r **+0.6435 → +0.0558** |
| 6 | Default `OUT`/`MODEL_DIR` pointed at non-existent dirs | `main.py`, `test.py` | now match artifacts on disk |
| 7 | 15m features vs 30m label quantity (18.4% disagreement) | `relabel.py` | resolved — v2's finest TF is 15m |

### Bug 2 in detail — the dangerous one

```python
# BEFORE: rewrites already-emitted predictions when a transition confirms
for j in range(i - pending_count + 1, i + 1):
    filtered[j] = confirmed
```

At block `j` that information did not exist; it arrives only at block `i > j`.
Anyone running the old `test.py` saw **~48% accuracy** and a large apparent edge
that was entirely look-ahead. Honestly evaluated the filter scores **below** the
raw model (31.57% vs 36.85%) and below the 32.95% persistence baseline —
hysteresis assumes sticky regimes, but 67% of v1 blocks change regime.

### Bug 1 in detail — the instructive one

The old `jane.py` comment said `tf_all_agree` was deleted because *"the
30m/2h/4h net directions almost always agree (constant features, zero
information)."* The symptom was real; the diagnosis was wrong. They agreed
because they were the same number computed three times. **A feature that looks
constant may be a broken calculation rather than an uninformative one.**

### Feature-count changes

74 → 75. Added `control_strength_15m`, `control_strength_1h`, `tf_all_agree`.
Removed `body_imbalance`, `control_strength_30m`.

### Introduced and caught during the fix

The first pass emitted `control_strength_4h`, which measured r = 1.0000 with
`abs_efficiency`. Removed from output before shipping.

---

## 5. Left alone deliberately

- **Path defaults** — `relabel.py` writes `labels.csv`, `jane.py` reads
  `final_labels.csv`. Unifying either direction would overwrite the v1 baseline
  or break reproduction of existing artifacts. Pass env vars explicitly.
- **`winner_rejection`** direction correlation (−0.1166) — direction-neutral by
  construction; the residual is a real market asymmetry, not a coding error.
- **~57% commented-out dead code** in `main.py` / `test.py` / `jane.py` — not a
  bug, but it is why it was hard to tell which code produced which artifact.
- **4 remaining |r| > 0.95 feature pairs** — empirical correlations, not
  algebraic identities. Dropping them is a modelling choice, not a fix.

---

## 6. What the changes did and did not achieve

### Did not: the 4-class model got worse

| | v1 labels | v2 labels |
|---|---|---|
| accuracy | 37.07% | **66.96%** |
| majority baseline | 34.81% | 65.31% |
| **lift over majority** | **+2.26 pp** | **+1.65 pp** |
| balanced accuracy | 38.08% | **35.30%** |
| predicts `chop_volatile` on | 66.8% | **94.2%** |

Accuracy nearly doubled while information content fell. **Do not report bare
accuracy on v2 labels** — report lift over majority and balanced accuracy.
Retire the 4-class model.

### Did: a real directional signal in a binary framing

"bullish vs bearish, given directional", block-shuffled null, price-only
features (all 22 label-derived features dropped so the memory recursion cannot
leak):

| labels | n | observed | null | z |
|---|---|---|---|---|
| v1 | 1819 | 0.5140 | 0.5002 ± 0.0141 | +0.97 (**noise**, p=0.167) |
| **v2 post-fix** | **891** | **0.5361** | **0.4910 ± 0.0207** | **+2.19 (real)** |

Tightening "directional" to genuinely strong moves made direction predictable
where it was pure noise across all blocks.

**Caveats before acting on it:** the effect is roughly **2σ**; repeat runs of
the same test gave z = +2.08, +2.19 and +2.6, which is permutation noise from
using only 30 shuffles. `p = 0.033` is the resolution floor of a
30-permutation test (1/30), not a measured value. It was found after dozens of
tests in one session. Re-run with 500+ shuffles, confirm on a split not used
during the search (train ≤2023, test 2024 only), and compute the cost floor
(breakeven must exceed 4–5 bp/side) before looking at any P&L curve.

### Unchanged by any of this: volatility

Next-block volatility remains the strongest signal in the dataset — calm
detection AUC **0.9186** on v2, 0.9038 on v1, both p < 0.001 against a
block-shuffled null, stable across five rolling-origin folds. It does not
depend on the labeling scheme.

---

## 7. Reproduce

```bash
# 1. labels
LABELS_CSV=labels_v2.csv python relabel.py

# 2. features
LABELS_CSV=labels_v2.csv OUTPUT_PATH=features_v2_fixed.csv python jane.py

# 3. train + evaluate
FEATURES_CSV=features_v2_fixed.csv python main.py
FEATURES_CSV=features_v2_fixed.csv python test.py
```

`relabel.py` prints excess persistence, the decision-path breakdown, and a v1
comparison on every run. Use `validate.py` for permutation tests, max-statistic
nulls, causal diffs, cost floors and ablation before believing any new result.
