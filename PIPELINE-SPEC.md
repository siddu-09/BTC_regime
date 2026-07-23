# BTC 4h Regime Detection — Pipeline Spec & Roadmap

Authoritative current-state document. Last updated 2026-07-23.

**Document map** — read this one first:

| file | purpose |
|---|---|
| **`PIPELINE-SPEC.md`** | **this file — current logic, changes, recommendations** |
| `RESEARCH-FINDINGS.md` | the full audit and the evidence behind every claim |
| `HANDOFF.md` | short version for a new engineer |
| `CHANGES.md` | earlier v1→v2 record, superseded by §3 here |
| `config.py` | every tunable, with the measurement behind each default |
| `validate.py` | signal-vs-noise harness (permutation, ablation, cost floor) |

---

## 1. What runs today

```bash
LABELS_CSV=labels_v21.csv                              python relabel.py
LABELS_CSV=labels_v21.csv OUTPUT_PATH=features_v21.csv python jane.py
FEATURES_CSV=features_v21.csv OUT=v21_results_main     python main.py
FEATURES_CSV=features_v21.csv MODEL_DIR=v21_results_main \
    OUT=v21_results_test                               python test.py
```

Data: `BTC_IST.csv`, 3,415,680 1-minute Binance BTCUSDT candles, IST,
2020-01-02 → 2026-06-30. Verified clean: **0 duplicates, 0 OHLC violations,
0 missing minutes, exactly 240 candles in every one of the 14,232 blocks.**

Blocks are the six fixed IST sessions: 00-04, 04-08, 08-12, 12-16, 16-20, 20-00.

### Current output

| | full | in-sample | **out-of-sample** |
|---|---|---|---|
| chop_volatile | 67.3% | 68.3% | **63.8%** |
| bullish | 15.0% | 15.1% | **14.6%** |
| bearish | 12.5% | 12.0% | **14.1%** |
| chop_calm | 5.2% | 4.5% | **7.5%** |

OOS window 2025-01-01 → 2026-06-30, n = 3,275.
Transition rate 46.0%. **Excess persistence +4.59 pp.**

### Tweaking anything

All knobs live in `config.py`, in three sections. Three ways to change a value:

```bash
# 1. edit config.py

# 2. env var, prefixed by section
RL_VOTE_TH=0.35 RL_CALM_ADAPTIVE=1 python relabel.py
XGB_CLASS_WEIGHT=balanced XGB_MAX_ESTIMATORS=400  python main.py
EV_USE_FILTER=1 python test.py

# 3. programmatic, for sweeps — loads the 462MB CSV once
python -c "
from config import label_config
from relabel import load_1min, label_blocks
df = load_1min('BTC_IST.csv')
for vt in (0.25, 0.30, 0.35):
    out, diag = label_blocks(df, cfg=label_config(vote_th=vt), verbose=False)
    print(vt, out.regime.value_counts(normalize=True).round(3).to_dict())
"
```

Unknown keys raise `KeyError`; the three weight groups must each sum to 1.0 or
you get a `ValueError`. Each script prints its resolved config on startup.

---

## 2. Labeling logic (relabel.py v2.1)

The label answers **"who controlled this block?"**, not "what did one candle
do". Priority: high precision on directional labels; low transition count from
a higher evidence bar, never from repeating the previous label; zero lag on
genuine reversals.

### 2.1 Timeframe ladder

| TF | candles/block | role |
|---|---|---|
| 15m | 16 | detects an impulse starting mid-block |
| 1h | 4 | early structure |
| 2h | 2 | main structural witness |
| 4h | 1 | confirms net displacement |

### 2.2 Per-timeframe control — `tf_control()`

```
body_dom  = (bull_body - bear_body) / total_body          in [-1, +1]
count_dom = (n_bull - n_bear) / n                          in [-1, +1]
direction = sign(0.6 * body_dom + 0.4 * count_dom)
efficiency= |last_close - first_open| / (max_high - min_low)
strength  = efficiency * (0.5*|body_dom| + 0.5*|count_dom|)
```

Direction comes from **internal candle structure**, not the aggregate net move.
0.6/0.4 favours body over count so three tiny green candles cannot outvote one
large red one. `strength` is unsigned, in [0, 1].

For the 4h view (one candle) both dominance terms collapse to ±1, so strength
reduces to efficiency — its intended confirming role. `control_strength_4h` is
therefore **not** emitted as a feature; it measured r = 1.0000 with
`abs_efficiency`.

### 2.3 Weighted evidence and memory

```
current  = 0.15*d15*s15 + 0.25*d1h*s1h + 0.35*d2h*s2h + 0.25*d4h*s4h
memory   = 0.50*label(t-1) + 0.30*label(t-2) + 0.20*label(t-3)
           where bullish=+1, bearish=-1, chop=0
blended  = 0.78 * current + 0.22 * memory
```

2h carries the most weight: coarsest view that still exposes internal
structure. 4h sees displacement but no path. 15m is noisiest but the only view
catching a mid-block impulse. Memory over 3 blocks = 12 hours: recent state
without dragging a stale regime across a session.

### 2.4 Hard invariant

**A directional label's sign always equals `sign(close − open)`.** Memory and
multi-timeframe evidence decide *whether* a block is directional, never *which
way*. Every label stays factually true of its own block.

### 2.5 Decision order

```
1. DEAD BLOCK
   range < 0.45% AND vol_ratio < 0.85 AND avg 15m body < 0.135%
   -> chop_calm

2. CURRENT EVIDENCE
   per-TF direction + strength, weighted score, all_agree flag

3. HIGH-CONVICTION OVERRIDE  (memory bypassed entirely)
   all 4 TFs agree AND eff >= 0.60 AND |blended| >= 0.70
   AND vol_ratio >= 1.50 AND |net| >= 0.80%
   -> bullish / bearish, on this exact block

4. DIRECTIONAL TEST
   required_eff = 0.40 if all TFs agree else 0.45
   directional if  eff >= required_eff
               AND |net| >= 0.35%
               AND |blended| >= 0.30
   -> bullish / bearish

5. OTHERWISE -> chop_volatile
```

**Memory never assigns a label.** If evidence fails, the block is
`chop_volatile`, not a repeat of the previous regime.

### 2.6 Active thresholds

| key | value | note |
|---|---|---|
| `eff_dir` | 0.45 | ~p58 of the efficiency distribution |
| `eff_borderline` | 0.40 | only on full TF agreement |
| `eff_high` | 0.60 | ~p80, clearly-trending tail |
| `net_min` | 0.35% | above typical 4–9 bp round-trip cost |
| `net_high` | 0.80% | high-conviction move |
| **`vote_th`** | **0.30** | **the only binding secondary gate** |
| `vote_high` | 0.70 | high-conviction gate |
| `calm_range` | 0.45% | absolute — see §5.2 |
| `calm_vol` | 0.85 | relative to trailing 540-block median |
| `hc_vol_ratio` | 1.50 | volume confirmation |
| `vol_win` | 540 | ~90 days, trailing |

### 2.7 Causality

Every input is available at the block's close. Memory reads only
already-assigned previous labels. The volume baseline is
`rolling(540).median()`; pandas rolling is right-aligned, window `[t-539, t]`.
**No future data enters any label.** Verified: an independent causal
reimplementation reproduces the shipped feature file bit-for-bit (max diff
1e-16).

---

## 3. Change log

### 3.1 v1 → v2 — seven bugs fixed

| # | bug | verification |
|---|---|---|
| 1 | `tf_control` returned `sign(c[-1]-o[0])` — identical on all sub-TFs | `d30 != d4h` on **0 / 14,232** blocks; now varies |
| 2 | **Look-ahead**: hysteresis retroactively rewrote emitted predictions | 48.12% (fake) → **31.57%** (causal) |
| 3 | `body_imbalance` ≡ `1 − 2·counter_energy` | exact duplicates **1 → 0** |
| 4 | `control_vote_strength` omitted the 4h term, used broken direction | now mirrors the labeler |
| 5 | `close_position` leaked direction | r **+0.6435 → +0.0558** |
| 6 | Default `OUT`/`MODEL_DIR` pointed at non-existent dirs | now match artifacts |
| 7 | 15m features vs 30m label quantity (18.4% disagreement) | resolved by the 15m ladder |

**Bug 2 was dangerous** — anyone running the old `test.py` saw ~48% accuracy
and a large apparent edge that was entirely look-ahead.

**Bug 1 is the instructive one.** The old `jane.py` comment said `tf_all_agree`
was deleted because "the directions almost always agree (constant features,
zero information)." The symptom was real, the diagnosis wrong: they agreed
because they were the same number computed three times. *A feature that looks
constant may be a broken calculation rather than an uninformative one.*

### 3.2 v2 → v2.1 — three gates removed on ablation evidence

| gate | sole rejections | effect of removing | verdict |
|---|---|---|---|
| **sign agreement** | **0** | byte-identical labels | **dead — removed** |
| **memory inertia** | — | byte-identical labels | **dead — removed** |
| **path efficiency** | 228 | excess +3.79 → **+4.59** pp, AUC 0.5361 → **0.5431** | **harmful — removed** |
| vote | **1,550** | AUC 0.5361 → 0.5010, z +1.48 → +0.38 | **KEPT** |

The inertia never binds because `vote` fails first. Memory still acts through
`blended`; only the *threshold* adjustment was removed. All three remain as
config switches so the ablation reproduces.

Same failure mode as v1's Step-4 dead code. **Every gate needs an ablation test
at build time, or it silently does nothing.**

### 3.3 v2.2 experiment — Q4 breakout rescue (built, measured, default OFF)

Hypothesis: a block that ranges for three quarters then breaks out in the last
hour is chop *on average* but trending *at the boundary the next block
inherits*. Rescue those from `chop_volatile` to a trend label.

Implemented in `relabel.py` step 5, gated by `use_q4_rescue`. Fires on 1,020
blocks (7.2%), cutting chop_volatile 67.3% → 60.0%.

**Result — the premise half-holds:**

| this block labelled via | n | next directional | **next SAME dir** | **opposite** |
|---|---|---|---|---|
| ordinary directional | 3,852 | 37.2% | **22.7%** | **14.5%** |
| **q4_breakout** | 1,020 | **41.0%** | **22.9%** | **18.0%** |
| base rate | 14,231 | 34.7% | — | — |

Q4 breakouts genuinely predict that the next block will be **directional**
(41.0% vs 34.7%, +6.3 pp). They do **not** predict *which way*: same-direction
22.9% vs 22.7% for ordinary trend blocks — zero gain — while opposite-direction
is higher (18.0% vs 14.5%). The same:opposite ratio is **1.27 vs 1.57**, so a
Q4 breakout is *less* directionally persistent than an ordinary trend block,
consistent with late-block breakouts often being exhaustion or stop-run moves.

A threshold sweep confirms it — same-dir stays flat at 22.9–23.6% across five
settings while next-dir falls 43.9% → 36.7%. Tightening selects fewer blocks,
not better-directed ones.

**Downstream cost:**

| | v2.1 | v2.2 |
|---|---|---|
| **lift over Markov** | **+1.68 pp** | **+1.04 pp** |
| balanced accuracy | 34.76% | 34.76% |
| bullish AUC | 0.5986 | **0.5743** |
| chop_volatile AUC | 0.6043 | **0.5718** |

Three of four classes got worse. Raw accuracy also fell (65.47% → 59.39%) but
that is not the comparison — the majority baseline fell with it
(63.79% → 57.77%).

**The salvage:** the impulse is real, it just signals *energy*, not
*direction*. As a **feature** it lifts "next block is directional" AUC
0.5491 → 0.5509, z +3.60 → **+5.88**. Keep the measurement, drop the
relabelling. Default `use_q4_rescue=False`; set `RL_USE_Q4_RESCUE=1` to
reproduce.

---

## 4. Feature set (jane.py) — 75 features

One row per 4h block, computed from that block's 15m candles plus causal
context. Target is `next_regime = regime.shift(-1)`.

**Design invariant:** single-block features are unsigned — they describe how
strong a move was, never which direction. Direction enters only via `cur_*`
one-hots and the explicitly signed `dir_balance_6` / `dir_balance_12`.

| family | n | members |
|---|---|---|
| **Energy / volatility** | 6 | `block_range_pct`, `median_range_pct`, `avg_body_pct`, `avg_wick_pct`, `big_candle_freq`, `realized_vol` |
| Control / conviction | 8 | `abs_efficiency`, `path_efficiency`, `abs_net_pct`, `color_uniformity`, `dir_consistency`, `structure_strength`, `longest_run`, `close_position` |
| Wicks / rejection | 5 | `wick_symmetry`, `wick_to_body`, `winner_rejection`, `counter_energy`, `max_pullback_ratio` |
| Overlap | 1 | `avg_overlap` |
| Exhaustion (final quarter) | 6 | `q4_range_expansion`, `q4_range_share`, `q4_agrees_with_block`, `q4_wick_to_body`, `q4_winner_rejection`, `q4_volume_ratio` |
| Decay / late-block | 4 | `momentum_decay`, `counter_tail_count`, `last_candle_wick_pct`, `late_giveback` |
| Halves | 5 | `first_half_magnitude`, `second_half_magnitude`, `halves_agree`, `half_momentum_shift`, `reversal_point` |
| Multi-timeframe | 5 | `control_strength_15m/1h/2h`, `control_vote_strength`, `tf_all_agree` |
| Volume | 1 | `vol_ratio` |
| Regime context | 19 | `cur_*`, `prev1_*`, `prev2_*`, `cnt6_*`, `regime_streak`, `dir_streak`, `move_shrinking` |
| Signed trend context | 2 | `dir_balance_6`, `dir_balance_12` |
| Multi-block price | 5 | `range_3b`, `efficiency_3b`, `prev_range`, `range_change`, `prev_momentum_decay` |
| Calendar | 8 | `session_*` (6), `day_of_week`, `is_weekend` |

---

## 5. Recommendations

Ordered by measured payoff. Every number here was produced this session; none
is a rule of thumb.

### 5.1 Drop the 29 structure features — measured improvement

| family | n | calm AUC | dir AUC |
|---|---|---|---|
| ENERGY only | 17 | 0.8959 | 0.5316 |
| **STRUCTURE only** | 29 | **0.8014** | 0.5043 |
| CONTEXT only | 22 | 0.8149 | 0.5152 |
| **CALENDAR only** | **8** | **0.8562** | 0.5520 |
| **ENERGY + CONTEXT + CALENDAR** | **47** | **0.9210** | 0.5496 |
| ALL | 75 | 0.9178 | 0.5475 |

Adding the structure block to the other three **costs −0.0033 calm AUC**.

Why: **efficiency has lag-1 autocorrelation 0.021.** Features measuring
structure describe a quantity with no memory, so they can only add variance.
Note 8 calendar features beat 29 structure features (0.8562 vs 0.8014) —
intraday volatility seasonality is real signal, block structure is not.

**Prune by what a feature measures, not by importance rank.**

### 5.2 Make the absolute thresholds volatility-relative

Yearly pass rates:

| threshold | range across years | spread |
|---|---|---|
| **`net ≥ 0.35%`** | **46.0% – 76.5%** | **30.5 pp** |
| `range < 0.45%` | 0.0% – 12.7% | 12.7 pp |
| `eff ≥ 0.45` | 38.3% – 45.2% | **6.9 pp** |

**Ratios are stable, absolutes drift.** `eff_dir` is a ratio and is by far the
steadiest. In 2021, 76.5% of blocks cleared "a meaningful move"; in 2023,
46.0%.

- **`net_min`** is the worst offender. Replace with `k × trailing median range`
  or a trailing quantile of |net|. Biggest stationarity fix available.
- **`calm_range`** — set `calm_adaptive=True` (already implemented). Holds the
  chop_calm share at ~15–20% per year instead of 0.0% (2021) to 12.7% (2023),
  and 5.4% to 15.6% between adjacent 2025 quarters. This matters most because
  `chop_calm` carries the only strong signal (AUC 0.9175) and its prevalence
  currently triples and collapses between quarters.
- **`eff_dir`** — leave at 0.45. Scale-free and stable.
- **`vote_th`** — the binding gate, but do not tune it on the sweep run here;
  see §5.5.

### 5.3 Stop adding OHLCV-derived features — the source is exhausted

15 features built specifically for the predictable axis (EWMA realized vol at
1d/3d/7d/30d, vol-of-vol, short/long vol ratio, session-relative range,
blocks-since-high-vol, inter-block gap, volume trend):

| set | calm AUC | dir AUC |
|---|---|---|
| current 75 | 0.9178 | 0.5475 |
| **+ 15 new vol features** | **0.9176** | 0.5493 |

Incremental by group, all within ±0.007 — noise against a permutation SD of
~0.02:

| group | Δ calm | Δ dir |
|---|---|---|
| long-horizon RV | +0.0004 | +0.0033 |
| vol-of-vol + ratios | +0.0011 | +0.0074 |
| session-relative range | +0.0002 | −0.0030 |
| blocks_since_highvol | +0.0000 | −0.0067 |

**The information in 1m BTCUSDT OHLCV has been extracted.** The only remaining
lever is exogenous data: perpetual funding rate, open-interest deltas,
orderbook imbalance, cross-asset, and a scheduled-event calendar (CPI, FOMC)
— which is known in advance and drives 4h volatility directly.

### 5.4 Add the Q4 impulse as a feature, not a label

See §3.3. `q4_breakout` as a boolean feature lifts "next block is directional"
AUC 0.5491 → 0.5509 with z +3.60 → +5.88. Small but it sharpens a real signal,
and it does not manufacture directional labels that fail to persist.

### 5.5 Move to a three-way split before tuning anything further

Everything this session used train/test, and the test period has now been
looked at dozens of times. A `vote_th` sweep peaked at 0.30 with z = +3.64 and
neighbours at +1.41 and +1.11 — a spike flanked by nothing is the shape of
test-set selection, not a robust optimum.

Tune thresholds on a **validation** period; touch the **test** period once.
Without this, §5.2's `vote_th` question cannot be answered honestly.

### 5.6 Report the right numbers

With one class near 64%, raw accuracy is close to meaningless.

- **Lead with lift over Markov** (zero-feature transition matrix) and
  **balanced accuracy**. `test.py` now prints both.
- Every claimed signal gets five numbers: `observed | null mean | null sd | p | z`
  against a **block-shuffled** null.
- **Never compare an AUC to 0.50.** For an autocorrelated target the honest
  null sits well above it — measured at **0.5435** for `chop_calm`, versus
  0.4996 for a naive row shuffle. An AUC of 0.54 on that target is *nothing*.
- Use **500+ permutations**. At 25–30 shuffles the z estimate carries ±0.5
  noise: identical labels returned z = +1.48, +1.92 and +2.43 on three draws.

---

## 6. What is real and what is not

| claim | status |
|---|---|
| **Next-block volatility / calm detection** | **REAL** — AUC 0.9175, p<0.001, stable across 5 rolling-origin folds |
| Direction, v1 labels, all blocks | **NOISE** — AUC 0.5140 vs 0.5002 null, p = 0.167 |
| Direction, v2 high-precision subset | **REAL but weak** — AUC 0.5361 vs 0.4910 null, ~2σ, needs confirmation |
| Trending-vs-choppy forecast | **IMPOSSIBLE** — efficiency autocorr 0.021, five definitions all AUC 0.499–0.524 |
| 4-class accuracy as a metric | **MISLEADING** — capped ~39% on v1; on v2 it tracks the majority baseline |
| Momentum gated on trend/vol state | **DIES ON COSTS** — breakeven 1.16–1.80 bp/side vs 4–5 bp taker |

**Three of the four regimes are not regimes.** Excess persistence: bullish
−0.1 pp, bearish +0.1 pp, chop_volatile +1.4 pp, **chop_calm +37.1 pp**. Only
`chop_calm` behaves like a persistent state.

---

## 7. Prioritized roadmap

| # | action | effort | expected payoff |
|---|---|---|---|
| 1 | Drop the 29 structure features | 1 h | +0.0033 calm AUC, 75 → 46 features |
| 2 | `calm_adaptive=True` | 1 h | fixes chop_calm swinging 0%–15.6% |
| 3 | Make `net_min` volatility-relative | half day | fixes a 30.5 pp yearly drift |
| 4 | Add `q4_breakout` as a feature | 1 h | z +3.60 → +5.88 on next-directional |
| 5 | Three-way split; re-tune `vote_th` on validation | 1 day | makes every future tuning claim honest |
| 6 | Confirm the directional finding (500+ perms, train ≤2023 / test 2024) | 1 day | decides whether it is real |
| 7 | Add exogenous data (funding, OI, event calendar) | 1–2 weeks | **the only route to new signal** |
| 8 | Economic evaluation of the calm signal | 1–2 days | decides whether any of it is tradeable |

Items 1–4 are cheap and measured. Item 5 is the precondition for trusting
anything after it. Item 7 is the only one that can move the ceiling.

**Do not** pursue: more OHLCV-derived features (§5.3), directional prediction
across all blocks (§6), a trend/chop gate (§6), or 4-class accuracy as an
optimisation target (§5.6).
