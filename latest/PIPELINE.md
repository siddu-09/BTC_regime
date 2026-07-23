# BTC 4h Regime Detection — Full Pipeline (latest)

Self-contained snapshot of the working pipeline. Every number in this document
was produced by running the scripts in **this folder** on `features.csv`
(the v2.1 label set). Reproduce with the commands in §2.

**Headline result**

| metric | in-sample | out-of-sample |
|---|---|---|
| **Accuracy** | **69.80%** | **65.47%** |
| Balanced accuracy | 36.0% | 34.76% |
| Macro F1 | 0.342 | 0.317 |
| Log-loss | 0.784 | 0.934 |

Overfit gap 4.33 pp. But raw accuracy is **not** the story — see §7 and §9.

---

## 0. What's in this folder

| file | purpose |
|---|---|
| `relabel.py` | builds the regime labels from raw 1-min candles (the labeling logic) |
| `jane.py` | turns each labelled block into 75 features |
| `main.py` | trains the XGBoost model (train window 2020–2024) |
| `test.py` | evaluates the frozen model out-of-sample (2025-01 → 2026-06) |
| `config.py` | every tunable weight/threshold/parameter, one place |
| `features.csv` | **final feature matrix** — one row per 4h block, 75 features + label (= `features_v21.csv`) |
| `labels.csv` | **final labels** — `date, block, regime` (= `labels_v21.csv`) |
| `results_main/` | trained model + feature importances (produced by `main.py`) |
| `results_test/` | confusion matrix, per-class report, calibration, monthly acc (produced by `test.py`) |

`relabel.py` and `jane.py` need the raw `BTC_IST.csv` (3.4M 1-min candles, ~462 MB,
git-ignored) to regenerate `labels.csv` / `features.csv` from scratch. Those two
CSVs are shipped here already built, so `main.py` and `test.py` run without it.

---

## 1. The project in one paragraph

Every day is split into six fixed 4-hour IST blocks (00-04, 04-08, …, 20-00).
Each block is labelled with one of four **regimes** describing who controlled it:
`bullish`, `bearish`, `chop_calm`, `chop_volatile`. From each block we compute 75
features and ask a single question: **given block _t_ (its features + its regime),
what is the regime of block _t+1_?** The model is XGBoost. The honest finding is
that only *one* of the four regimes (`chop_calm`, i.e. next-block volatility) is
actually predictable; direction is not. Everything below explains why.

Data: `BTC_IST.csv`, Binance BTCUSDT 1-minute, IST, 2020-01-02 → 2026-06-30.
Verified clean: 0 duplicates, 0 OHLC violations, 0 missing minutes, exactly 240
candles in every one of the 14,232 blocks.

---

## 2. How to run

```bash
# 1. (optional) rebuild labels from raw 1-min data — needs BTC_IST.csv
DATA_1MIN_CSV=BTC_IST.csv LABELS_CSV=labels.csv python relabel.py

# 2. (optional) rebuild features from labels — also needs BTC_IST.csv
LABELS_CSV=labels.csv OUTPUT_PATH=features.csv python jane.py

# 3. train  (produces results_main/)
FEATURES_CSV=features.csv OUT=results_main python main.py

# 4. evaluate out-of-sample  (produces results_test/)
FEATURES_CSV=features.csv MODEL_DIR=results_main OUT=results_test python test.py
```

Steps 3–4 are all you need to reproduce the headline numbers; the CSVs are
already here. Every knob can be overridden by env var (`RL_` for relabel,
`XGB_`/`TRAIN_` for main, `EV_` for test) or edited in `config.py`.

---

## 3. Labeling — how a block gets its regime

The label answers **"who controlled this block?"**, not "what did one candle do".
Design priority: high precision on the directional labels; low transition count
from a *higher evidence bar*, never from repeating the previous label; zero lag on
genuine reversals.

### 3.1 The hard invariant (this is what "TradingView-confirming" means)

> **A directional label's sign always equals `sign(close − open)` of its own block.**

Memory and multi-timeframe evidence decide *whether* a block is directional, never
*which way*. So if you pull the block up on TradingView, a `bullish` label is
**always** a block that closed above its open, and `bearish` always closed below.
The label can never contradict the candle you see on the chart. What the extra
machinery does is decide whether an up-block was *convincing enough* to call
`bullish`, or whether it was just noise and should be `chop_volatile`.

### 3.2 The timeframe ladder

Each 4h block is viewed at four resolutions, each an independent witness:

| TF | candles/block | role |
|---|---|---|
| 15m | 16 | catches an impulse starting mid-block (noisiest) |
| 1h | 4 | early structure |
| 2h | 2 | **main structural witness (highest weight)** |
| 4h | 1 | confirms net displacement, sees no path |

For each TF, `tf_control()` computes a **direction** and an unsigned **strength**
from the *internal candle structure* (not the aggregate move):

```
body_dom  = (bull_body − bear_body) / total_body       in [−1, +1]
count_dom = (n_bull − n_bear) / n                       in [−1, +1]
direction = sign(0.6·body_dom + 0.4·count_dom)
efficiency= |last_close − first_open| / (max_high − min_low)
strength  = efficiency · (0.5·|body_dom| + 0.5·|count_dom|)
```

The 0.6/0.4 split favours body over count, so three tiny green candles cannot
outvote one large red one. (Historical note: an earlier version returned
`sign(close−open)` for every sub-TF, making them identical on all 14,232 blocks —
"multi-timeframe agreement" was vacuous. Deriving direction from body+count
dominance makes each TF a real, independent witness.)

### 3.3 Weighted evidence and memory

```
current  = 0.15·d15·s15 + 0.25·d1h·s1h + 0.35·d2h·s2h + 0.25·d4h·s4h
memory   = 0.50·label(t−1) + 0.30·label(t−2) + 0.20·label(t−3)   (bull=+1, bear=−1, chop=0)
blended  = 0.78·current + 0.22·memory
```

`current` is the live evidence; `memory` is the recent regime over the last 3
blocks (= 12 hours). `blended` mixes them 78/22.

### 3.4 Decision order ([relabel.py `classify()`](relabel.py))

```
1. DEAD BLOCK  → chop_calm
   range < calm_range  AND  vol_ratio < 0.85  AND  avg 15m body < calm_range·0.30

2. CURRENT EVIDENCE  (compute current / blended / all_agree; no label yet)
   if net_dir == 0  → chop_volatile   (flat block)

3. HIGH-CONVICTION OVERRIDE  → bullish / bearish   (memory bypassed entirely)
   all 4 TFs agree  AND  eff4 ≥ 0.60  AND  |blended| ≥ 0.70
   AND  vol_ratio ≥ 1.50  AND  |net| ≥ 0.80%

4. DIRECTIONAL TEST  → bullish / bearish
   required_eff = 0.40 if all TFs agree else 0.45
   eff4 ≥ required_eff  AND  |net| ≥ 0.35%  AND  |blended| ≥ 0.30 (vote_th)

5. OTHERWISE  → chop_volatile
```

**Memory never assigns a label.** If the evidence fails, the block is
`chop_volatile` — *not* a repeat of the previous regime. The direction of every
directional label still comes from `sign(net)` (§3.1).

### 3.5 Causality (no look-ahead)

Every input is available at the block's close. Memory reads only *already-assigned*
previous labels. The volume baseline is `rolling(540).median()` (pandas rolling is
right-aligned, window `[t−539, t]`). No future data enters any label. An
independent causal reimplementation reproduces the shipped feature file
bit-for-bit (max diff 1e-16).

### 3.6 Resulting label distribution (`labels.csv`)

| regime | full | in-sample | out-of-sample |
|---|---|---|---|
| chop_volatile | 67.3% | 68.3% | 63.8% |
| bullish | 15.0% | 15.1% | 14.6% |
| bearish | 12.5% | 12.0% | 14.1% |
| chop_calm | 5.2% | 4.5% | 7.5% |

Transition rate 46.0% (full). Excess persistence +4.59 pp — and critically, that
persistence is almost entirely `chop_calm`: per-regime excess is bullish −0.1,
bearish +0.1, chop_volatile +1.4, **chop_calm +37.1 pp**. Only `chop_calm` behaves
like a genuinely persistent *state*; the other three are closer to independent
draws. This is the single most important fact about the whole project.

---

## 4. Finalized weights & thresholds

All live in `config.py`. The three weight groups are each validated to sum to 1.0
(or `label_config()` raises `ValueError`).

**Weights** — why each value:

| group | key | value | rationale |
|---|---|---|---|
| Timeframe | `w_15m` / `w_1h` / `w_2h` / `w_4h` | 0.15 / 0.25 / **0.35** / 0.25 | 2h is the coarsest view that still exposes internal structure → highest weight; 4h sees displacement but no path; 15m is noisiest but the only mid-block-impulse catcher |
| Current vs memory | `w_current` / `w_memory` | **0.78** / 0.22 | evidence leads; memory only nudges |
| Memory decay | `mem_w1` / `mem_w2` / `mem_w3` | 0.50 / 0.30 / 0.20 | recent state without dragging a stale regime across 12h |

**Thresholds:**

| key | value | note |
|---|---|---|
| `eff_dir` | 0.45 | ~p58 of the efficiency distribution; scale-free & stable across years |
| `eff_borderline` | 0.40 | only reachable on full TF agreement |
| `eff_high` | 0.60 | ~p80, clearly-trending tail |
| `net_min` | 0.35% | above typical 4–9 bp round-trip cost; **most non-stationary knob** (yearly pass 46–76.5%) |
| `net_high` | 0.80% | high-conviction move |
| **`vote_th`** | **0.30** | **the only binding secondary gate** (sole reason 1,550 blocks stay chop) |
| `vote_high` | 0.70 | high-conviction gate |
| `calm_range` | 0.45% | absolute → non-stationary; `calm_adaptive=True` switches to a trailing quantile |
| `calm_vol` | 0.85 | vs trailing 540-block median volume |
| `hc_vol_ratio` | 1.50 | volume confirmation for the override |
| `vol_win` | 540 | ~90 days, trailing |

**Dead gate switches (all default OFF, kept only so the ablation reproduces):**
`use_sign_gate` (0 sole rejections), `use_inertia` (byte-identical labels),
`use_path_gate` (removing it *improved* excess persistence +3.79 → +4.59 pp and
directional AUC 0.5361 → 0.5431), `use_q4_rescue` (rescues 1,142 blocks out of
chop but the impulse signals *energy* not *direction*, so it belongs as a feature,
not a label). **Lesson baked into the config:** every gate needs an ablation test
or it silently does nothing.

---

## 5. Features — what jane.py computes (75)

One row per 4h block, computed from that block's 15m candles plus causal context.
Target column is `next_regime = regime.shift(-1)`.

**Design invariant:** single-block features are **unsigned** — they describe how
*strong* a move was, never which direction. Direction enters only through the
`cur_*` regime one-hots and the two explicitly-signed `dir_balance_6/12`. This is
deliberate: it stops the model learning a false "this block went up → next block
goes up" bias.

| family | n | examples | what it measures |
|---|---|---|---|
| **Energy / volatility** | 6 | `block_range_pct`, `median_range_pct`, `avg_body_pct`, `realized_vol` | how *big* the block was — the predictable axis |
| Control / conviction | 8 | `abs_efficiency`, `abs_net_pct`, `structure_strength`, `longest_run` | how one-sided / trending |
| Wicks / rejection | 5 | `wick_to_body`, `winner_rejection`, `counter_energy` | rejection & exhaustion |
| Overlap | 1 | `avg_overlap` | candle overlap (chop signature) |
| Exhaustion (Q4) | 6 | `q4_range_share`, `q4_agrees_with_block`, `q4_volume_ratio` | is the last hour accelerating or fading |
| Decay / late-block | 4 | `momentum_decay`, `late_giveback` | end-of-block momentum |
| Halves | 5 | `first/second_half_magnitude`, `halves_agree`, `reversal_point` | within-block shape |
| Multi-timeframe | 5 | `control_strength_15m/1h/2h`, `control_vote_strength`, `tf_all_agree` | the labeler's own evidence, as features |
| Volume | 1 | `vol_ratio` | vs 90-day median |
| Regime context | 19 | `cur_*`, `prev1_*`, `prev2_*`, `cnt6_*`, `regime_streak`, `dir_streak` | recent regime history |
| Signed trend context | 2 | `dir_balance_6`, `dir_balance_12` | the *only* signed multi-block features |
| Multi-block price | 5 | `range_3b`, `efficiency_3b`, `prev_range` | short price history |
| Calendar | 8 | `session_*` (6), `day_of_week`, `is_weekend` | intraday/weekly seasonality |

### 5.1 What the features actually reveal (measured, not assumed)

- **Energy features are the signal.** `median_range_pct`, `avg_wick_pct`,
  `avg_body_pct`, `realized_vol` are the top four by model gain (§6). They answer
  "will the next block be calm or active?" — and that question *is* answerable.
- **Calendar is real, small signal.** 8 calendar features out-predict the 29
  structure features on their own (calm AUC 0.856 vs 0.801). Intraday volatility
  seasonality exists; block micro-structure does not carry over.
- **Structure/control features are near-dead weight.** Efficiency has lag-1
  autocorrelation of only **0.021** — a block being "efficient/trending" tells you
  essentially nothing about the next block. Adding the 29-feature structure block
  on top of energy+context+calendar *costs* −0.0033 calm AUC. In the trained model
  only **2 of 7** labeler-aligned control features even reach the top-25.
- **The OHLCV source is exhausted.** A separate test adding 15 hand-built vol
  features (multi-horizon EWMA RV, vol-of-vol, session-relative range, …) moved
  calm AUC by 0.0002 — pure noise. There is no more juice in 1-minute OHLCV.

---

## 6. The model — design, learning, parameters

**Task:** multi-class (`multi:softprob`, 4 classes) — predict `next_regime` from
the 75 features of the current block.

**Split:** train `< 2025-01-01` (10,956 blocks), test `2025-01-01 → 2026-06-30`
(3,275 blocks). Time-ordered, no shuffling.

**How the tree count is chosen:** 4-fold **expanding-window** `TimeSeriesSplit`
with early stopping (patience 80). Each fold trains on the past and validates on
the next slice; the median best-iteration is taken → **280 trees** here. This is
what stops the model from memorizing: too few trees underfit, too many memorize
specific blocks, and the walk-forward CV finds the turn.

**Fixed hyperparameters** (`config.py → TRAIN_DEFAULTS`, all deliberately
conservative for a noisy, imbalanced target):

| param | value | why |
|---|---|---|
| `max_depth` | 4 | shallow trees, capture interactions without memorizing |
| `learning_rate` | 0.02 | slow learning to find subtle multi-TF patterns |
| `subsample` | 0.8 | row subsampling → regularization |
| `colsample_bytree` | 0.7 | column subsampling → regularization |
| `min_child_weight` | 30 | large → refuses to split on thin, noisy leaves |
| `reg_lambda` / `reg_alpha` | 3.0 / 0.5 | L2 + L1 shrinkage |
| `n_estimators` | 280 | chosen by the CV above |
| `class_weight` | **flat** | see §6.1 |

### 6.1 How the weights are "finalized based on the labeling logic"

Two connections:

1. **Feature alignment.** `jane.py` recomputes `control_vote_strength`,
   `control_strength_*` and `tf_all_agree` using the *exact same* `tf_control()`
   and timeframe weights (0.15/0.25/0.35/0.25) the labeler uses, so the model sees
   the labeler's own evidence score as inputs and can, in principle, re-derive the
   decision boundary.
2. **Class weighting was tested against the label imbalance and left flat.**
   Because ~67% of blocks are `chop_volatile`, `balanced` (inverse-frequency)
   weighting was measured: it lifts balanced accuracy 34.8 → 43.2% but misleads
   early stopping (runs to 1,762 trees, overfit gap blows out to 31.6 pp) and lands
   **20.7 pp below the Markov baseline**. Per-class AUCs barely move between flat
   and balanced (chop_calm 0.9175 vs 0.9138) — the model *knows the same things*
   either way, only the decision rule changes. So `flat` is the default and we
   judge by AUC, not accuracy.

---

## 7. Results (this run, reproduced exactly)

### 7.1 Baselines & accuracy — out-of-sample

| baseline | value |
|---|---|
| persistence (next = current) | 50.17% |
| majority class | 63.79% |
| **Markov (zero features) — the real bar** | **63.79%** |
| **model accuracy** | **65.47%** |
| lift over Markov | **+1.68 pp** |
| balanced accuracy | 34.76% (random 25%) |
| macro F1 | 0.317 · log-loss 0.934 |

In-sample accuracy 69.80%, overfit gap 4.33 pp.

### 7.2 Per-class AUC (imbalance-proof — the metric that matters)

| class | AUC | verdict |
|---|---|---|
| **chop_calm** | **0.9175** | genuinely predictable (volatility clustering) |
| chop_volatile | 0.6043 | mostly "not calm" |
| bullish | 0.5986 | ≈ honest null (~0.54 for this autocorrelated target) → noise |
| bearish | 0.5617 | below that → noise |

### 7.3 Confusion matrix — out-of-sample (rows = actual, cols = predicted)

```
               bearish  bullish  chop_calm  chop_volatile
bearish              0        0         13            450
bullish              0        0         17            461
chop_calm            0        0        101            144
chop_volatile        0        0         46           2043
```

**The bearish and bullish columns are entirely zero.** Across 3,275 blocks the
model *never once* predicts a directional regime. It predicts `chop_volatile`
94.6% of the time and `chop_calm` 5.4%. Every actual bull/bear block is called
chop. This is not a bug — it is the rational response to §7.2: since direction is
noise, betting on it is never the arg-max, so predicting the 64% majority
minimizes expected loss. **The model's entire +1.68 pp edge is the 101 correct
`chop_calm` calls** that a zero-feature Markov model would have missed.

### 7.4 Per-class report

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| bearish | 0.00 | 0.00 | 0.00 | 463 |
| bullish | 0.00 | 0.00 | 0.00 | 478 |
| chop_calm | 0.57 | 0.41 | 0.48 | 245 |
| chop_volatile | 0.66 | 0.98 | 0.79 | 2089 |
| **macro avg** | 0.31 | 0.35 | **0.32** | 3275 |

### 7.5 Top features the model actually used (by gain)

`median_range_pct` (32.4) · `avg_wick_pct` (19.9) · `avg_body_pct` (15.1) ·
`realized_vol` (11.4) · `cur_bullish` (10.8) · `cnt6_chop_calm` (10.0) ·
`is_weekend` (9.2) · `cur_chop_volatile` (8.1) · `session_04-08` (7.8) ·
`dir_balance_6` (7.2). → **Energy + calendar + regime context.** Confirms §5.1: the
model leans on exactly the features that carry signal and ignores the structure
block.

---

## 8. Do the features give the model everything it needs?

Split by what's being predicted:

- **Calm / volatility axis → YES, complete.** AUC 0.9175 from the energy features
  alone; extra vol features add noise-level 0.0002. Nothing to add.
- **Direction (bull/bear) axis → NO — and no feature could fix it.** This is a
  *data* gap, not a feature-engineering gap. Next-block direction is unpredictable
  from price (efficiency autocorr 0.021), so you cannot engineer a feature for
  information that isn't in 1-minute OHLCV. The only lever is **exogenous data**:
  perpetual funding rate, open-interest deltas, orderbook imbalance, cross-asset,
  and a scheduled-event calendar (CPI, FOMC).

---

## 9. Why accuracy is high but F1 is low (and the confidence note you asked for)

**They are measuring different things, and the gap is diagnostic, not
contradictory.**

- **Accuracy (65.47%) is high because it rewards the majority guess.** 64% of test
  blocks are `chop_volatile`; a model that shouts "chop" every time scores ~64% for
  free. Our model does almost exactly that, so it lands at 65%. High accuracy here
  means "the data is imbalanced", not "the model is good".
- **Macro F1 (0.317) is low because it averages per-class F1 with equal weight, and
  two classes have F1 = 0.** bearish and bullish get precision = recall = 0 (the
  model never predicts them), so their F1 is 0. Averaging 0.00, 0.00, 0.48, 0.79
  gives 0.32. F1 refuses to give credit for a class you never get right — which is
  the honest view.
- **The reconciliation:** accuracy sees a 65% winner; F1 sees a model that has
  solved 1 of 4 classes. **F1 is telling the truth about direction; accuracy is
  being flattered by the imbalance.** This is exactly why `test.py` leads with
  **lift over Markov (+1.68 pp)** and **balanced accuracy (34.76%)** instead of raw
  accuracy — with one class near 64%, raw accuracy is close to meaningless.

**Note on confidence.** The selective-prediction table
(`results_test/confidence_calibration.csv`) shows accuracy rising with the
confidence threshold — 65.5% at conf ≥ 0.30 up to **70.9% at conf ≥ 0.70** (which
still covers 40% of blocks). That is real and usable: the model *knows when it is
confident*, and its confident predictions are its calm calls. But note transition
accuracy stays ~44–46% regardless of threshold — confidence buys you accuracy on
the *persistent* (calm) blocks, not on the *turns*. So confidence gating is a lever
for the one signal that exists (calm), and does nothing for direction.

---

## 10. Bottom line

| claim | status |
|---|---|
| Next-block **calm / volatility** detection | **REAL** — AUC 0.9175, stable across folds |
| Next-block **direction** | **NOISE** — AUC ≈ 0.56–0.60 vs ~0.54 honest null |
| 4-class **accuracy** as a metric | **MISLEADING** — tracks the majority baseline |
| Feature set for the **calm** question | **complete** — OHLCV is exhausted |
| Feature set for the **direction** question | **cannot be completed from OHLCV** — needs exogenous data |

The model does the one honest thing available: it predicts volatility well, refuses
to guess direction, and its accuracy is high only because the labels are
imbalanced. The real, tradeable signal — if any — is the `chop_calm` / volatility
axis, and the only route to *new* signal is exogenous data (funding, OI,
event calendar).
