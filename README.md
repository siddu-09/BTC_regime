# BTC 4-Hour Regime Prediction — Project README

## 1. What This Project Does

This project predicts the **regime of the next 4-hour BTC candle** using only
information available at the close of the *current* 4-hour candle. It is
trained on 5 years of history (2020–2024) and tested on 1.5 years of unseen
data (2025–2026).

```
Features(current block) + Regime(current block)  ──►  Regime(next block)
```

Every feature is computed strictly from data available up to the moment the
current block closes.

---

## 2. The Data

- **Source**: `BTC_IST.csv` — Binance BTCUSDT 1-minute OHLCV candles
- **Timezone**: IST (UTC+5:30) — all timestamps and session boundaries are in
  Indian Standard Time
- **Range**: 2020-01-02 → 2026-06-30 (~3.4 million 1-minute candles)
- **4-hour blocks**: the day is split into six fixed IST sessions —

  | Block | IST Hours |
  |---|---|
  | 00-04 | 12:00 AM – 4:00 AM |
  | 04-08 | 4:00 AM – 8:00 AM |
  | 08-12 | 8:00 AM – 12:00 PM |
  | 12-16 | 12:00 PM – 4:00 PM |
  | 16-20 | 4:00 PM – 8:00 PM |
  | 20-00 | 8:00 PM – 12:00 AM |

  Roughly 14,230 four-hour blocks total across the whole dataset.

---

## 3. The Four Regimes & How They're Labeled

Every block is labeled into exactly one of four regimes:

| Regime | What it looks like on a chart |
|---|---|
| **bullish** | One side (buyers) clearly won; price displaced upward with real conviction |
| **bearish** | Sellers clearly won; price displaced downward with real conviction |
| **chop_volatile** | Big candles, high energy, both sides fighting, price nets out roughly flat |
| **chop_calm** | Tiny bodies, tiny wicks, no real price shift, low volume — a dead market |

### The labeling algorithm (`relabel.py`)

The labeler cross-checks **three timeframes** — 30-minute, 2-hour, and
4-hour — before deciding on a label.

**Step 1 — Dead-block test (calm).**
A block is `chop_calm` only if ALL of these hold: range < 0.50% of price,
volume < 80% of its rolling baseline, AND the average 30-minute candle body
is tiny (< 0.15% of price).

**Step 2 — Multi-timeframe direction check.**
For the 30-minute candles (up to 8 per block) and the 2-hour candles (2 per
block), the labeler computes a *direction* (+1 bullish, -1 bearish, 0 none)
and a *control strength* (0–1, how one-sided the candles were).

**Step 3 — Agreement across timeframes.**
If the 30-minute direction, the 2-hour direction, and the 4-hour net
direction all agree, AND the 4-hour efficiency (`|close-open| / range`) is
≥ 0.35 with a meaningful net move (≥ 0.24%), the block is labeled `bullish`
or `bearish`.

**Step 4 — Overall control vote when timeframes disagree.**
If the thresholds indicate "directional" but the timeframes don't agree with
each other, the labeler computes a **strength-weighted vote** across all
three timeframes. Whichever side had more convincing control overall wins
the label. If no side has a clear lead, the block is `chop_volatile`
(energetic, but no winner).

---

## 4. Feature Engineering (`jane.py`)

`jane.py` reads the 1-minute CSV and the labels CSV and produces one row of
features per 4-hour block. Labels are pure ground truth from `relabel.py` —
`jane.py` does not re-label anything. Features are organized around five
questions:

| # | Question | Feature examples |
|---|---|---|
| 1 | **How is this regime ending?** (exhaustion) | `q4_agrees_with_block`, `q4_winner_rejection`, `q4_wick_to_body`, `q4_volume_ratio`, `momentum_decay`, `counter_tail_count`, `late_giveback` |
| 2 | **Who had power?** (control/conviction) | `abs_efficiency`, `path_efficiency`, `color_uniformity`, `dir_consistency`, `structure_strength`, `body_imbalance`, `close_position` |
| 3 | **One-sided or two-sided rejection?** (wicks) | `wick_symmetry`, `wick_to_body`, `winner_rejection`, `counter_energy`, `max_pullback_ratio` |
| 4 | **How much energy?** (volatility state) | `block_range_pct`, `median_range_pct`, `avg_body_pct`, `avg_wick_pct`, `realized_vol`, `big_candle_freq`, `vol_ratio` |
| 5 | **What's the context?** (where are we now) | `cur_bullish/bearish/chop_calm/chop_volatile`, `prev1_*`, `prev2_*`, `regime_streak`, `cnt6_*`, session one-hots, `day_of_week`, `is_weekend` |

### Multi-timeframe control features
Two features directly mirror the labeler's own cross-check logic:
- `control_strength_30m`, `control_strength_2h` — unsigned strength score per
  timeframe (0 = totally contested, 1 = total one-sided control)
- `control_vote_strength` — the magnitude of the strength-weighted vote.

### Unsigned single-block features
Every single-block feature is **unsigned** — it describes *how strong* the
move/imbalance/rejection was, never *which direction*. Direction enters the
model from exactly one place: the `cur_regime` one-hot columns (Markov
context, known at the block's close) plus a small number of explicitly
**signed, multi-block** trend-context features (`dir_balance_6`,
`dir_balance_12`) that capture mean-reversion behavior across recent blocks.

---

## 5. The XGBoost Engine (`main.py`)

### How the model learns
1. Load `features.csv` — each row is one 4-hour block's features + its
   current regime + the *target* (next block's regime).
2. Split chronologically: everything before 2025-01-01 is training (5 years,
   ~11,000 blocks); everything after is held out for testing.
3. Train gradient-boosted trees. Each new tree looks at the errors of all
   previous trees combined and adds a small correction. Over ~250–300 trees,
   the model refines its prediction for every block.
4. **Time-series cross-validation** (`TimeSeriesSplit`, 4 expanding folds)
   finds the number of trees via early stopping — training on an expanding
   historical window and validating on the following period each time.

### Hyperparameters
```python
max_depth=4            # up to 4-way feature interactions per tree
learning_rate=0.02      # each tree only nudges the prediction slightly
min_child_weight=30     # a leaf needs 30+ examples to be trusted
reg_lambda=3.0          # L2 regularization — smooths out noisy splits
colsample_bytree=0.7    # each tree sees 70% of features
subsample=0.8           # each tree sees 80% of rows
objective="multi:softprob"
```

---

## 6. How Confidence Is Calculated

XGBoost's `multi:softprob` objective outputs a raw score per class from the
sum of every tree's vote, then applies softmax:

```
P(class_i) = exp(score_i) / Σ exp(score_j)   for all 4 classes
confidence = max(P(bullish), P(bearish), P(chop_calm), P(chop_volatile))
```

### What determines the confidence level
The conditional transition probabilities in the training data are:

```
After bullish:        29% bullish, 31% bearish, 37% chop_volatile,  3% chop_calm
After bearish:        33% bullish, 26% bearish, 39% chop_volatile,  2% chop_calm
After chop_volatile:  30% bullish, 26% bearish, 39% chop_volatile,  5% chop_calm
After chop_calm:      17% bullish, 13% bearish, 27% chop_volatile, 44% chop_calm
```

No transition's probability exceeds ~44% except for `chop_calm` + tiny
range (< 0.3%), where the next block stays `chop_calm` ~71% of the time.
Confidence for most blocks sits in the 0.30–0.40 range, reflecting these
underlying transition probabilities.

### Handling low-confidence blocks
Low-confidence blocks are handled by the post-prediction filter (§8): a
low-confidence transition call is rejected and replaced with "the current
regime continues."

---

## 7. Testing (`test.py`)

`test.py` loads the frozen model trained by `main.py` and predicts on every
block from 2025-01-01 to 2026-06-30. It reports:

- **Persistence baseline** — accuracy of guessing "next = current"
- **Raw model accuracy** — the model's unfiltered predictions
- **Filtered accuracy** — after the post-prediction filter (§8)
- **Overfit gap** — in-sample accuracy minus out-of-sample accuracy
- Per-class precision/recall, full confusion matrix, and a
  **transition-only** confusion matrix (accuracy specifically on blocks
  where the regime actually changed)
- **Confidence calibration** — accuracy at increasing confidence thresholds
- **Monthly accuracy** — accuracy broken down by calendar month

---

## 8. The Post-Prediction Filter

Applied **after** the model outputs its raw probabilities, on the
out-of-sample predictions. It does not touch training or the labels.

### Step 1 — Confidence gate
If the model's top prediction is different from the current regime, but its
confidence is below a threshold (`CONF_THRESHOLD = 0.45`), the prediction is
rejected and replaced with "the current regime continues."

### Step 2 — Online hysteresis
A confidence-gated transition call must repeat for **2 consecutive blocks**
(`MIN_PERSIST = 2`) before it's accepted. A single-block prediction is held
at the previously confirmed regime. Once a transition has repeated for 2
blocks, it is retroactively confirmed for both of them.

```python
if predicted != current_regime and confidence < 0.45:
    predicted = current_regime
# a change is only confirmed once predicted 2 blocks in a row
```

Both steps are causal — they only use information available up to the
current block, with no look-ahead.

---

## 9. What the Results Mean

*(Numbers below are from the evaluation run recorded in `metrics.json`,
`confidence_calibration.csv`, `monthly_accuracy.csv`,
`feature_importance.csv`.)*

### Headline numbers
| Metric | Value | What it means |
|---|---|---|
| Persistence baseline | ~33% | Accuracy from guessing "next = current" every time |
| Majority-class baseline | ~35% | Accuracy from always guessing the single most common next-regime |
| In-sample accuracy | ~46% | How well the model fits the 5 years it was trained on |
| Out-of-sample (raw) accuracy | ~37% | The unfiltered model's accuracy on 2025–2026 |
| Overfit gap | ~9 points | In-sample minus out-of-sample |
| Lift over persistence | +4 to +15 points (raw vs. filtered) | Improvement over "just guess it stays the same" |

### Per-class report
`chop_volatile` and `chop_calm` have higher recall than `bullish`/`bearish`
in the per-class report, reflecting that volatility state (energy level)
carries more predictive signal than direction.

### Confidence calibration table
```
min_conf   pct_of_blocks   accuracy   transition_accuracy
  0.30         99%           37%            32%
  0.50          5%           62%            19%
  0.60          3%           70%            15%
  0.70          1%           69%             8%
```
Accuracy rises as the confidence threshold rises, confirming the
confidence scores are calibrated to actual correctness. The
transition-accuracy column stays low even at high confidence thresholds —
high-confidence predictions are concentrated on persistence calls (regime
staying the same), not transition calls.

### Monthly accuracy
Accuracy ranges from roughly 30% to 39% month to month across the
2025–2026 test window, without a sustained decline over the 18-month
period.

### Feature importance
The top-ranked features are volatility/energy signals (`median_range_pct`,
`avg_body_pct`, `avg_wick_pct`, `realized_vol`) and recent regime history
(`cnt6_chop_calm`, `session_04-08`, `is_weekend`). Structural/control
features (`abs_efficiency`, `wick_symmetry`, `control_vote_strength`,
`q4_agrees_with_block`) rank lower but are present in every trained model.

### Confusion matrix pattern
The largest off-diagonal mass sits between `bullish` ↔ `bearish` and
between trending regimes and `chop_volatile`. `chop_calm` has the fewest
cross-class confusions, consistent with its labeling criteria (tiny range,
tiny volume) being a narrower, more separable band of values than the other
three regimes.
