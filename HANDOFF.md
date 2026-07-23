# Handoff — relabel.py v2

For whoever picks this up. Full evidence in `RESEARCH-FINDINGS.md`; this is the
short version of what changed, what to do, and what will mislead you.

Original preserved at `relabel.py.bak.pre_v2_rewrite.20260723`.

---

## 1. Run it

```bash
LABELS_CSV=labels_v21.csv                              python relabel.py
LABELS_CSV=labels_v21.csv OUTPUT_PATH=features_v21.csv python jane.py
FEATURES_CSV=features_v21.csv OUT=v21_results_main     python main.py
FEATURES_CSV=features_v21.csv MODEL_DIR=v21_results_main \
    OUT=v21_results_test                               python test.py
```

Tested end-to-end on the full 3,415,680-candle dataset.

Note the default label output is `labels.csv` while `jane.py` defaults to
reading `final_labels.csv`. Pass the env vars explicitly — this gap pre-existed
the rewrite and was left alone so the v1 baseline is not overwritten.

### Tweaking values

**Everything tunable lives in `config.py`**, in three sections, each with the
measurement behind the default. Three ways to change a value:

```bash
# 1. edit config.py

# 2. env var, prefixed by section
RL_VOTE_TH=0.35 python relabel.py
XGB_CLASS_WEIGHT=balanced XGB_MAX_ESTIMATORS=400 python main.py
EV_USE_FILTER=1 python test.py

# 3. programmatic, for sweeps
python -c "
from config import label_config
from relabel import load_1min, label_blocks
df = load_1min('BTC_IST.csv')          # load once
for vt in (0.25, 0.30, 0.35):
    out, diag = label_blocks(df, cfg=label_config(vote_th=vt), verbose=False)
    print(vt, out.regime.value_counts(normalize=True).round(3).to_dict())
"
```

Config is validated: unknown keys raise `KeyError`, and the three weight groups
must each sum to 1.0 or you get a `ValueError`. Each script prints its resolved
config on startup.

---

## 1b. v2.1 — three gates removed on ablation evidence

Ablation (delete one gate, regenerate labels, measure) found two of the four
secondary gates were doing nothing at all:

| gate | sole rejections | effect of removing | verdict |
|---|---|---|---|
| **sign agreement** | **0** | byte-identical labels | **dead — removed** |
| **memory inertia** | — | byte-identical labels | **dead — removed** |
| **path efficiency** | 228 | excess persist +3.79 → **+4.59** pp, AUC 0.5361 → **0.5431** | **harmful — removed** |
| vote | **1,550** | AUC 0.5361 → 0.5010, z +1.48 → +0.38 | **KEPT — the only one working** |

The inertia never binds because `vote` fails first. Memory still acts, through
`blended = 0.78*current + 0.22*memory`; what was removed is the *threshold*
adjustment. All three remain as config switches so the ablation is reproducible.

Result: transitions 46.0%, **excess persistence +3.02 → +4.59 pp**,
distribution 67.3 / 5.2 / 15.0 / 12.5.

Same failure mode as v1's Step-4 dead code, found the same way. **Every gate
needs an ablation test, or it silently does nothing.** `validate.py` has
`ablate()`.

## 2. THE ONE THING NOT TO CONCLUDE

Swap in the v2 labels, rerun the 4-class model, and accuracy goes from **37% to
67%**. That is not an improvement.

| | v1 labels | v2 labels |
|---|---|---|
| accuracy | 37.07% | **66.96%** |
| majority-class baseline | 34.81% | **65.31%** |
| **lift over majority** | **+2.26 pp** | **+1.65 pp** |
| balanced accuracy | 38.08% | **35.30%** |
| predicts `chop_volatile` on | 66.8% | **94.2%** |

The baseline moved with the accuracy, because `chop_volatile` grew to 68.9% of
blocks. Lift over majority **fell**, balanced accuracy **fell**, and the model
collapsed harder onto one class.

**Do not run the 4-class model on these labels and report the accuracy.** Report
lift over majority and balanced accuracy, or the result will read as a doubling
of performance when information content went down.

Same trap on the label side: transitions dropped 67.0% → 45.1%, but **excess
persistence** (persistence minus what independent draws at the same class
balance give) only moved **+3.0 → +3.8 pp**. Fewer transitions is not the win
condition. `relabel.py` prints excess persistence on every run — that is the
number to watch.

---

## 3. What the rewrite actually bought

**A real directional signal, in a binary framing.** Testing "bullish vs bearish,
given directional", against a block-shuffled null:

| labels | features | n | observed | null | verdict |
|---|---|---|---|---|---|
| v1 | all | 1819 | 0.5140 | 0.5002 ± 0.0141 | NOISE (p=0.167) |
| v2 (pre-fix) | price-only | 891 | 0.5504 | 0.5050 ± 0.0218 | REAL, z=+2.08 |
| **v2 (post-fix)** | **price-only** | **891** | **0.5361** | **0.4910 ± 0.0207** | **REAL, z=+2.19** |

The price-only rows drop all 22 label-derived features (`cur_*`, `prev1_*`,
`prev2_*`, `cnt6_*`, `dir_*`, `regime_streak`, `move_shrinking`) so the label's
own memory recursion cannot leak into the result. Direction survives.

Tightening "directional" to genuinely strong moves made direction predictable
where it was noise across all blocks.

Observed AUC fell 0.5504 → 0.5361 after the bug fixes, which is expected:
removing `close_position`'s direction leak took some apparent directional signal
with it. The effect still clears its null.

**Before acting on it:**
- n = 891, and the effect is roughly **2σ**. An earlier run of the same test
  gave z = +2.6, this one +2.19 — that spread is permutation noise from using
  only 30 shuffles. Re-run with 500+ before quoting a number.
- p = 0.033 is the **resolution floor** of a 30-permutation test (1/30), not a
  measured value.
- It was found after dozens of tests in one session. Per §11.2 of
  `RESEARCH-FINDINGS.md`, confirm on a split not used during the search —
  train ≤2023, test 2024 only.
- Compute the cost floor (breakeven must exceed 4–5 bp/side) *before* looking
  at any P&L curve. See `validate.py`.

---

## 4. Bugs fixed (all verified)

| # | bug | file | verification |
|---|---|---|---|
| 1 | `tf_control` returned `sign(c[-1]-o[0])` — identical across every sub-timeframe | `jane.py` | ported v2 fix; `tf_all_agree` now varies (89.4% agree, was constant) |
| 2 | **Look-ahead**: hysteresis retroactively rewrote emitted predictions | `test.py:808` | 48.12% (fake) → **31.57%** (causal) |
| 3 | `body_imbalance` ≡ `1 − 2·counter_energy` | `jane.py` | removed; exact duplicates now **0** |
| 4 | `control_vote_strength` omitted the 4h term, used broken direction | `jane.py` | now mirrors relabel v2's weighted 4-TF evidence score |
| 5 | `close_position` leaked direction (r = +0.6435) | `jane.py` | unsigned form: **+0.6435 → +0.0558** |
| 6 | Default `OUT`/`MODEL_DIR` pointed at non-existent dirs | `main.py`, `test.py` | now match the artifacts on disk |
| 7 | 15m features vs 30m label quantity (18.4% disagreement) | `relabel.py` | resolved — v2's finest TF is 15m, same as `jane.py` |

Rebuild after pulling: `LABELS_CSV=... OUTPUT_PATH=... python jane.py`.
Feature count 74 → 75 (removed `body_imbalance`; added `control_strength_15m`,
`control_strength_1h`, `tf_all_agree`; renamed `control_strength_30m`).

`control_strength_4h` is deliberately **not** emitted: for a single candle,
strength collapses to efficiency, so it measured r = 1.0000 with
`abs_efficiency`. It is still computed internally for the evidence score and
agreement flag.

### Note on bug 1

The comment at the old `jane.py:1036` said `tf_all_agree` was deleted because
"the 30m/2h/4h net directions almost always agree (constant features, zero
information)." That symptom was real and the diagnosis was wrong — the
directions agreed because they were the same number computed three times. With
the calculation fixed the flag varies and is a genuine discriminator. Worth
remembering as a pattern: a feature that looks constant may be a broken
calculation rather than an uninformative one.

## 4b. Left alone deliberately

- **Path defaults**: `relabel.py` writes `labels.csv`, `jane.py` reads
  `final_labels.csv`. Not unified, because either direction would overwrite the
  v1 baseline or break reproduction of the existing artifacts. Pass env vars.
- **`winner_rejection`** direction correlation (−0.1166). It is
  direction-neutral by construction; the residual is a real market asymmetry,
  not a coding error.
- **~57% commented-out dead code** in `main.py` / `test.py` / `jane.py`. Not a
  bug, but it is what made it hard to tell which code produced which artifact.
- **4 remaining |r| > 0.95 feature pairs.** Empirical correlations, not
  algebraic identities — dropping them is a modelling choice, not a fix.

---

## 5. Side effect worth knowing

v1 computed its calm-body test on **30m** candles while `jane.py` built features
from **15m** — the two disagreed on the calm test for 2,612 blocks (18.4%). v2's
finest timeframe is 15m, so labeler and features now use the same quantity.
That mismatch is resolved as a side effect of the rewrite, not by design.

---

## 6. Recommended next step

Drop the 4-class model. Keep v2 labels as the *definition* of a high-precision
directional subset and model that subset as a binary problem, with the
confirmation run in §3 done first.

The volatility axis remains the strongest signal in the dataset regardless of
labeling (calm detection AUC 0.9186 on v2, 0.9038 on v1, both p < 0.001) and
does not depend on any of this.
