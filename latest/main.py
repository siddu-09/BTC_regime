# """
# main.py — Train XGBoost to predict next 4hr block regime

#     Features(block t) + regime(block t) → regime(block t+1)

# How the model learns:
#     1. Sees current block features + current regime (from jane.py)
#     2. Predicts next block regime
#     3. Compares with your TradingView label (ground truth)
#     4. Adjusts tree weights on what it got wrong
#     5. Repeats across 10,000+ training blocks (5 years)

#     Time-series CV finds the right number of boosting rounds —
#     too few = underfits, too many = memorizes noise.

# No HMM, no boundary correction, no confidence gating.
# Those added complexity without improving out-of-sample accuracy.

# Usage:
#     python main.py
#     FEATURES_CSV=features.csv SPLIT_DATE=2025-01-01 python main.py
# """
# import os, json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import (accuracy_score, log_loss, f1_score,
#                              classification_report)
# from sklearn.model_selection import TimeSeriesSplit
# import matplotlib; matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# # ============================================================================
# # CONFIG
# # ============================================================================
# FEATURES_CSV = os.environ.get("FEATURES_CSV", "fz_features.csv")
# SPLIT_DATE   = os.environ.get("SPLIT_DATE", "2025-01-01")
# OUT          = Path(os.environ.get("OUT", "kresults_main"))
# OUT.mkdir(parents=True, exist_ok=True)

# REGIMES = ['bearish', 'bullish', 'chop_calm', 'chop_volatile']

# # Columns that are NOT features (identifiers, labels, raw price levels)
# ID_COLS = {'date', 'block', 'block_id', 'regime', 'next_regime', 'target',
#            'block_open', 'block_high', 'block_low', 'block_close', 'block_volume'}

# # XGBoost: moderate depth, slow learning, proper regularization
# XGB_BASE = dict(
#     max_depth=4,
#     learning_rate=0.03,
#     subsample=0.8,
#     colsample_bytree=0.7,
#     min_child_weight=15,
#     reg_lambda=3.0,
#     reg_alpha=0.5,
#     objective="multi:softprob",
#     eval_metric="mlogloss",
#     tree_method="hist",
#     random_state=42,
# )


# # ============================================================================
# # DATA
# # ============================================================================
# def load_data(path):
#     df = pd.read_csv(path)
#     df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
#     df = df.sort_values(['date', 'block']).reset_index(drop=True)

#     # Detect target column (jane.py outputs 'next_regime')
#     if 'next_regime' in df.columns:
#         target_col = 'next_regime'
#     elif 'target' in df.columns:
#         target_col = 'target'
#     else:
#         raise ValueError("No target column found ('next_regime' or 'target')")

#     # Feature columns = everything except identifiers and labels
#     feat_cols = [c for c in df.columns if c not in ID_COLS]
#     return df, feat_cols, target_col


# def sample_weights(y, le):
#     """Flat weights — let the model learn the natural class boundaries."""
#     return np.ones(len(y), dtype=float)


# # ============================================================================
# # TRAINING
# # ============================================================================
# def find_best_n_trees(X, y, w, params, n_classes):
#     """
#     Time-series CV with early stopping.

#     Why this matters:
#         XGBoost adds trees one at a time. Each tree corrects the previous ones'
#         mistakes. But after enough trees, it starts memorizing specific blocks
#         instead of learning general patterns. Early stopping finds the point
#         where it stops improving on unseen future data.

#         We use 4 expanding-window folds:
#             fold 1: train 2020-2021 → validate 2021-2022
#             fold 2: train 2020-2022 → validate 2022-2023
#             fold 3: train 2020-2023 → validate 2023-mid-2024
#             fold 4: train 2020-mid2024 → validate mid2024-2025

#         Each fold's best iteration count is recorded.
#         We take the median → that's our safe stopping point.
#     """
#     tscv = TimeSeriesSplit(n_splits=4)
#     best_iters = []

#     for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
#         m = xgb.XGBClassifier(
#             **params, n_estimators=1000, num_class=n_classes,
#             early_stopping_rounds=50)
#         m.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx],
#               eval_set=[(X[va_idx], y[va_idx])], verbose=False)

#         bi = m.best_iteration
#         va_acc = accuracy_score(y[va_idx], m.predict(X[va_idx]))
#         va_persist = accuracy_score(y[va_idx], y[va_idx])  # placeholder
#         print(f"    fold {fold+1}: best_iter={bi:4d}  val_acc={va_acc*100:.1f}%")
#         best_iters.append(bi)

#     chosen = int(np.median(best_iters))
#     # 10% safety margin (slightly more trees than median → slightly less underfitting)
#     chosen = min(int(chosen * 1.1), 1000)
#     return chosen


# # ============================================================================
# # MAIN
# # ============================================================================
# def main():
#     print("=" * 68)
#     print("  MAIN — Train XGBoost on next-block regime prediction")
#     print("=" * 68)

#     # ── Load ──
#     df, feat_cols, target_col = load_data(FEATURES_CSV)
#     print(f"\n  Blocks:   {len(df):,}  ({df.date.min()} → {df.date.max()})")
#     print(f"  Features: {len(feat_cols)}")
#     print(f"  Target:   '{target_col}'")

#     le = LabelEncoder().fit(REGIMES)
#     X = df[feat_cols].fillna(0).values
#     y = le.transform(df[target_col].values)

#     # Training window: everything before SPLIT_DATE
#     tr = df['date'].values < SPLIT_DATE
#     X_tr, y_tr = X[tr], y[tr]
#     w_tr = sample_weights(y_tr, le)

#     print(f"\n  Training: {tr.sum():,} blocks  ({df[tr].date.min()} → {df[tr].date.max()})")

#     # Class distribution
#     print("\n  Next-regime distribution (train):")
#     for cls in le.classes_:
#         n = (df[tr][target_col] == cls).sum()
#         pct = n / tr.sum() * 100
#         bar = '█' * int(pct)
#         print(f"    {cls:15s} {n:5,d}  ({pct:.0f}%)  {bar}")

#     persist = (df[tr]['regime'].values == df[tr][target_col].values).mean()
#     print(f"\n  Persistence baseline (train): {persist*100:.1f}%")

#     # ── Find optimal tree count via time-series CV ──
#     print("\n  Finding optimal n_estimators via time-series CV ...")
#     n_trees = find_best_n_trees(X_tr, y_tr, w_tr, XGB_BASE, len(REGIMES))
#     print(f"  → using n_estimators = {n_trees}")

#     # ── Train final model on full training window ──
#     print(f"\n  Training final model ...")
#     params = dict(XGB_BASE)
#     params['n_estimators'] = n_trees
#     model = xgb.XGBClassifier(**params, num_class=len(REGIMES))
#     model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

#     # ── In-sample diagnostics ──
#     proba = model.predict_proba(X_tr)
#     pred = proba.argmax(1)
#     acc = accuracy_score(y_tr, pred)
#     ll = log_loss(y_tr, proba, labels=list(range(len(REGIMES))))
#     f1 = f1_score(y_tr, pred, average='macro')

#     print(f"\n  In-sample results:")
#     print(f"    Accuracy:    {acc*100:.1f}%")
#     print(f"    Macro F1:    {f1:.3f}")
#     print(f"    Log-loss:    {ll:.3f}")
#     print(f"    Persistence: {persist*100:.1f}%")
#     print(f"    Lift:        +{(acc-persist)*100:.1f} pts")

#     # ── Feature importance ──
#     booster = model.get_booster()
#     booster.feature_names = feat_cols
#     imp = pd.DataFrame({"feature": feat_cols})
#     for typ in ["gain", "weight", "cover"]:
#         scores = booster.get_score(importance_type=typ)
#         imp[typ] = imp.feature.map(scores).fillna(0)
#     imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)

#     print(f"\n  Top-15 features (what the model learned to use):")
#     for _, row in imp.head(15).iterrows():
#         bar = '█' * int(row.gain / max(imp.gain.max(), 1) * 25)
#         print(f"    {row.feature:30s} {row.gain:8.1f}  {bar}")

#     # Check: are the labeling-aligned features being used?
#     labeling_feats = {'abs_efficiency', 'abs_net_pct', 'block_range_pct',
#                       'vol_ratio', 'wick_symmetry', 'counter_energy',
#                       'net_to_range', 'net_pct', 'color_uniformity'}
#     used_in_top30 = set(imp.head(30).feature) & labeling_feats
#     print(f"\n  Labeling-aligned features in top-30: {len(used_in_top30)}/{len(labeling_feats)}")
#     print(f"    {sorted(used_in_top30)}")

#     # ── Save ──
#     model.save_model(str(OUT / "xgb_next_block.json"))
#     imp.to_csv(OUT / "feature_importance.csv", index=False)

#     json.dump({
#         "classes": list(le.classes_),
#         "feat_cols": feat_cols,
#         "target_col": target_col,
#     }, open(OUT / "label_classes.json", "w"), indent=2)

#     json.dump({
#         "n_train": int(tr.sum()),
#         "n_features": len(feat_cols),
#         "n_estimators": n_trees,
#         "insample_acc": round(float(acc), 4),
#         "insample_f1": round(float(f1), 4),
#         "insample_logloss": round(float(ll), 4),
#         "persistence_baseline": round(float(persist), 4),
#         "xgb_params": params,
#     }, open(OUT / "metrics.json", "w"), indent=2)

#     # ── Plot ──
#     fig, ax = plt.subplots(figsize=(10, 10))
#     top = imp.head(25)
#     ax.barh(top.feature[::-1], top.gain[::-1], color="#3B82F6")
#     ax.set_xlabel("Gain (avg loss reduction per split)")
#     ax.set_title("Top-25 Features — what the model learned", fontweight="bold")
#     plt.tight_layout()
#     plt.savefig(OUT / "feature_importance.png", dpi=150)
#     plt.close()

#     print(f"\n  Saved → {OUT}/")
#     print("=" * 68)


# if __name__ == "__main__":
#     main()
# """
# main.py — Train XGBoost to predict next 4h block regime

# Aligned to jane.py's feature set and the multi-TF labeling logic.

# The model sees: features(block t) + regime(block t) → predicts regime(block t+1)
# Learns over 2020-2024, validated via time-series CV, tested on 2025-2026.

# The most important features (mirroring the labeler's decision):
#   - tf_all_agree / tf_disagreement  → trend vs chop discriminator
#   - control_strength_30m/2h          → how convincing was control
#   - block_range_pct / avg_body_pct   → calm vs active
#   - cur_regime + dir_balance         → direction context (mean-reverting)

# Usage:
#     python main.py
#     FEATURES_CSV=features.csv SPLIT_DATE=2025-01-01 python main.py
# """
# import os, json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, log_loss, f1_score
# from sklearn.model_selection import TimeSeriesSplit
# import matplotlib; matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# FEATURES_CSV = os.environ.get("FEATURES_CSV", "latest_features.csv")
# SPLIT_DATE   = os.environ.get("SPLIT_DATE", "2025-01-01")
# OUT          = Path(os.environ.get("OUT", "latest_results_main"))
# OUT.mkdir(parents=True, exist_ok=True)

# REGIMES = ['bearish', 'bullish', 'chop_calm', 'chop_volatile']

# # Non-feature columns
# ID_COLS = {'date', 'block', 'block_id', 'regime', 'next_regime', 'target'}

# # XGBoost: slow learning to find subtle multi-TF patterns, strong regularization
# XGB_BASE = dict(
#     max_depth=4,
#     learning_rate=0.02,
#     subsample=0.8,
#     colsample_bytree=0.7,
#     min_child_weight=30,
#     reg_lambda=3.0,
#     reg_alpha=0.5,
#     objective="multi:softprob",
#     eval_metric="mlogloss",
#     tree_method="hist",
#     random_state=42,
# )


# def load_data(path):
#     df = pd.read_csv(path)
#     df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
#     df = df.sort_values(['date', 'block']).reset_index(drop=True)
#     target_col = 'next_regime' if 'next_regime' in df.columns else 'target'
#     feat_cols = [c for c in df.columns if c not in ID_COLS]
#     return df, feat_cols, target_col


# def find_best_n_trees(X, y, params, n_classes):
#     """Time-series CV to find how many trees before overfitting."""
#     tscv = TimeSeriesSplit(n_splits=4)
#     best_iters = []
#     for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
#         m = xgb.XGBClassifier(**params, n_estimators=2000, num_class=n_classes,
#                               early_stopping_rounds=80)
#         m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
#         bi = m.best_iteration
#         va_acc = accuracy_score(y[va_idx], m.predict(X[va_idx]))
#         print(f"    fold {fold+1}: best_iter={bi:4d}  val_acc={va_acc*100:.1f}%")
#         best_iters.append(bi)
#     return int(np.median(best_iters))


# def main():
#     print("=" * 68)
#     print("  MAIN — Train XGBoost on next-block regime")
#     print("=" * 68)

#     df, feat_cols, target_col = load_data(FEATURES_CSV)
#     print(f"\n  Blocks:   {len(df):,}  ({df.date.min()} -> {df.date.max()})")
#     print(f"  Features: {len(feat_cols)}")
#     print(f"  Target:   '{target_col}'")

#     le = LabelEncoder().fit(REGIMES)
#     X = df[feat_cols].fillna(0).values
#     y = le.transform(df[target_col].values)

#     tr = df['date'].values < SPLIT_DATE
#     X_tr, y_tr = X[tr], y[tr]
#     print(f"\n  Training: {tr.sum():,} blocks ({df[tr].date.min()} -> {df[tr].date.max()})")

#     print("\n  Next-regime distribution (train):")
#     for cls in le.classes_:
#         n = (df[tr][target_col] == cls).sum()
#         pct = n / tr.sum() * 100
#         print(f"    {cls:15s} {n:5,d} ({pct:4.1f}%) {'#'*int(pct)}")

#     persist = (df[tr]['regime'].values == df[tr][target_col].values).mean()
#     print(f"\n  Persistence baseline (train): {persist*100:.1f}%")

#     print("\n  Time-series CV for n_estimators ...")
#     n_trees = find_best_n_trees(X_tr, y_tr, XGB_BASE, len(REGIMES))
#     print(f"  -> n_estimators = {n_trees}")

#     print("\n  Training final model ...")
#     params = dict(XGB_BASE); params['n_estimators'] = n_trees
#     model = xgb.XGBClassifier(**params, num_class=len(REGIMES))
#     model.fit(X_tr, y_tr, verbose=False)

#     proba = model.predict_proba(X_tr); pred = proba.argmax(1)
#     acc = accuracy_score(y_tr, pred)
#     ll = log_loss(y_tr, proba, labels=list(range(len(REGIMES))))
#     f1 = f1_score(y_tr, pred, average='macro')
#     print(f"\n  In-sample: acc {acc*100:.1f}%  F1 {f1:.3f}  log-loss {ll:.3f}")
#     print(f"  (in-sample should be ~40-48%, NOT 55%+ — that would be memorizing)")

#     # Feature importance
#     booster = model.get_booster(); booster.feature_names = feat_cols
#     imp = pd.DataFrame({"feature": feat_cols})
#     for typ in ["gain", "weight", "cover"]:
#         imp[typ] = imp.feature.map(booster.get_score(importance_type=typ)).fillna(0)
#     imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)

#     print(f"\n  Top-15 features:")
#     for _, r in imp.head(15).iterrows():
#         print(f"    {r.feature:26s} {r.gain:8.1f}  {'#'*int(r.gain/max(imp.gain.max(),1)*25)}")

#     # Check labeler-aligned features are being used
#     aligned = {'control_vote_strength',
#                'control_strength_30m', 'control_strength_2h', 'control_vote_strength',
#                'abs_efficiency', 'block_range_pct', 'avg_body_pct', 'wick_symmetry'}
#     used = set(imp.head(25).feature) & aligned
#     print(f"\n  Labeler-aligned features in top-25: {len(used)}/{len(aligned)}")
#     print(f"    {sorted(used)}")

#     # Save
#     model.save_model(str(OUT / "xgb_next_block.json"))
#     imp.to_csv(OUT / "feature_importance.csv", index=False)
#     json.dump({"classes": list(le.classes_), "feat_cols": feat_cols,
#                "target_col": target_col}, open(OUT / "label_classes.json", "w"), indent=2)
#     json.dump({"n_train": int(tr.sum()), "n_features": len(feat_cols),
#                "n_estimators": n_trees, "insample_acc": round(float(acc), 4),
#                "insample_f1": round(float(f1), 4),
#                "persistence_baseline": round(float(persist), 4),
#                "xgb_params": params}, open(OUT / "metrics.json", "w"), indent=2)

#     fig, ax = plt.subplots(figsize=(10, 10))
#     top = imp.head(25)
#     ax.barh(top.feature[::-1], top.gain[::-1], color="#3B82F6")
#     ax.set_xlabel("Gain"); ax.set_title("Top-25 Features", fontweight="bold")
#     plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()

#     print(f"\n  Saved -> {OUT}/")
#     print("=" * 68)


# if __name__ == "__main__":
#     main()
"""
main.py — Train XGBoost to predict next 4h block regime

Aligned to jane.py's feature set and the multi-TF labeling logic.

The model sees: features(block t) + regime(block t) → predicts regime(block t+1)
Learns over 2020-2024, validated via time-series CV, tested on 2025-2026.

The most important features (mirroring the labeler's decision):
  - tf_all_agree / tf_disagreement  → trend vs chop discriminator
  - control_strength_30m/2h          → how convincing was control
  - block_range_pct / avg_body_pct   → calm vs active
  - cur_regime + dir_balance         → direction context (mean-reverting)

Usage:
    python main.py
    FEATURES_CSV=features.csv SPLIT_DATE=2025-01-01 python main.py
"""
import os, json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, log_loss, f1_score,
                             balanced_accuracy_score)
from sklearn.model_selection import TimeSeriesSplit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import train_config, describe

FEATURES_CSV = os.environ.get("FEATURES_CSV", "latest_features.csv")
SPLIT_DATE   = os.environ.get("SPLIT_DATE", "2025-01-01")
# Defaults match the artifact directories actually present in this repo, so a
# plain `python main.py && python test.py` reproduces the documented results.
OUT          = Path(os.environ.get("OUT", "latest_results_main"))
OUT.mkdir(parents=True, exist_ok=True)

REGIMES = ['bearish', 'bullish', 'chop_calm', 'chop_volatile']

# Non-feature columns
ID_COLS = {'date', 'block', 'block_id', 'regime', 'next_regime', 'target'}

# All tunables live in config.py. Override with XGB_<KEY>=value, e.g.
#   XGB_MAX_DEPTH=6 XGB_CLASS_WEIGHT=flat python main.py
CFG = train_config()

XGB_BASE = dict(
    max_depth=CFG['max_depth'],
    learning_rate=CFG['learning_rate'],
    subsample=CFG['subsample'],
    colsample_bytree=CFG['colsample_bytree'],
    min_child_weight=CFG['min_child_weight'],
    reg_lambda=CFG['reg_lambda'],
    reg_alpha=CFG['reg_alpha'],
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=CFG['n_jobs'],
)


def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df.sort_values(['date', 'block']).reset_index(drop=True)
    target_col = 'next_regime' if 'next_regime' in df.columns else 'target'
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    return df, feat_cols, target_col


def find_best_n_trees(X, y, w, params, n_classes):
    """Time-series CV to find how many trees before overfitting."""
    tscv = TimeSeriesSplit(n_splits=CFG['cv_splits'])
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        m = xgb.XGBClassifier(**params, n_estimators=CFG['max_estimators'],
                              num_class=n_classes,
                              early_stopping_rounds=CFG['early_stopping_rounds'])
        m.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx],
              eval_set=[(X[va_idx], y[va_idx])], verbose=False)
        bi = m.best_iteration
        va_acc = accuracy_score(y[va_idx], m.predict(X[va_idx]))
        print(f"    fold {fold+1}: best_iter={bi:4d}  val_acc={va_acc*100:.1f}%")
        best_iters.append(bi)
    return int(np.median(best_iters))


def main():
    print("=" * 68)
    print("  MAIN — Train XGBoost on next-block regime")
    print("=" * 68)
    print(describe(CFG, "config (override with XGB_<KEY>=value):"))

    df, feat_cols, target_col = load_data(FEATURES_CSV)
    print(f"\n  Blocks:   {len(df):,}  ({df.date.min()} -> {df.date.max()})")
    print(f"  Features: {len(feat_cols)}")
    print(f"  Target:   '{target_col}'")

    le = LabelEncoder().fit(REGIMES)
    X = df[feat_cols].fillna(0).values
    y = le.transform(df[target_col].values)

    tr = df['date'].values < SPLIT_DATE
    X_tr, y_tr = X[tr], y[tr]
    print(f"\n  Training: {tr.sum():,} blocks ({df[tr].date.min()} -> {df[tr].date.max()})")

    print("\n  Next-regime distribution (train):")
    for cls in le.classes_:
        n = (df[tr][target_col] == cls).sum()
        pct = n / tr.sum() * 100
        print(f"    {cls:15s} {n:5,d} ({pct:4.1f}%) {'#'*int(pct)}")

    persist = (df[tr]['regime'].values == df[tr][target_col].values).mean()
    print(f"\n  Persistence baseline (train): {persist*100:.1f}%")

    # Class weighting. v2 labels are ~67% chop_volatile; with flat weights the
    # model collapses onto the majority class (predicted it on 94.2% of blocks
    # in testing). 'balanced' weights by inverse frequency, trading raw accuracy
    # for balanced accuracy — the metric that actually moves on imbalanced
    # labels. Switch with XGB_CLASS_WEIGHT=flat.
    if CFG['class_weight'] == 'balanced':
        counts = np.bincount(y_tr, minlength=len(REGIMES)).astype(float)
        inv = len(y_tr) / (len(REGIMES) * np.maximum(counts, 1))
        w_tr = inv[y_tr]
        print("  Class weights: balanced (inverse frequency)")
        for i, cls in enumerate(le.classes_):
            print(f"    {cls:15s} n={int(counts[i]):5d}  weight={inv[i]:.3f}")
    else:
        w_tr = np.ones(len(y_tr), dtype=float)
        print("  Class weights: flat")

    print("\n  Time-series CV for n_estimators ...")
    n_trees = find_best_n_trees(X_tr, y_tr, w_tr, XGB_BASE, len(REGIMES))
    print(f"  -> n_estimators = {n_trees}")

    print("\n  Training final model ...")
    params = dict(XGB_BASE); params['n_estimators'] = n_trees
    model = xgb.XGBClassifier(**params, num_class=len(REGIMES))
    model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

    proba = model.predict_proba(X_tr); pred = proba.argmax(1)
    acc = accuracy_score(y_tr, pred)
    ll = log_loss(y_tr, proba, labels=list(range(len(REGIMES))))
    f1 = f1_score(y_tr, pred, average='macro')
    bal = balanced_accuracy_score(y_tr, pred)
    majority = pd.Series(y_tr).value_counts(normalize=True).max()

    # Raw accuracy is close to meaningless when one class is ~67% of the data.
    # Report it against the majority baseline, and lead with balanced accuracy.
    print(f"\n  In-sample:")
    print(f"    accuracy           {acc*100:5.1f}%")
    print(f"    majority baseline  {majority*100:5.1f}%   lift {(acc-majority)*100:+5.2f} pp")
    print(f"    balanced accuracy  {bal*100:5.1f}%   (random = {100/len(REGIMES):.1f}%)")
    print(f"    macro F1           {f1:.3f}")
    print(f"    log-loss           {ll:.3f}")
    print(f"\n  Predicted vs actual class share (in-sample):")
    for i, cls in enumerate(le.classes_):
        print(f"    {cls:15s} predicted {(pred==i).mean()*100:5.1f}%   "
              f"actual {(y_tr==i).mean()*100:5.1f}%")
    print("\n  If 'predicted' is far above 'actual' for one class, the model has")
    print("  collapsed onto it — check lift over majority, not raw accuracy.")

    # Feature importance
    booster = model.get_booster(); booster.feature_names = feat_cols
    imp = pd.DataFrame({"feature": feat_cols})
    for typ in ["gain", "weight", "cover"]:
        imp[typ] = imp.feature.map(booster.get_score(importance_type=typ)).fillna(0)
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)

    print(f"\n  Top-15 features:")
    for _, r in imp.head(15).iterrows():
        print(f"    {r.feature:26s} {r.gain:8.1f}  {'#'*int(r.gain/max(imp.gain.max(),1)*25)}")

    # Check labeler-aligned features are being used
    aligned = {'control_vote_strength',
               'control_strength_30m', 'control_strength_2h', 'control_vote_strength',
               'abs_efficiency', 'block_range_pct', 'avg_body_pct', 'wick_symmetry'}
    used = set(imp.head(25).feature) & aligned
    print(f"\n  Labeler-aligned features in top-25: {len(used)}/{len(aligned)}")
    print(f"    {sorted(used)}")

    # Save
    model.save_model(str(OUT / "xgb_next_block.json"))
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    json.dump({"classes": list(le.classes_), "feat_cols": feat_cols,
               "target_col": target_col}, open(OUT / "label_classes.json", "w"), indent=2)
    json.dump({"n_train": int(tr.sum()), "n_features": len(feat_cols),
               "n_estimators": n_trees, "insample_acc": round(float(acc), 4),
               "insample_balanced_acc": round(float(bal), 4),
               "insample_f1": round(float(f1), 4),
               "majority_baseline": round(float(majority), 4),
               "lift_over_majority": round(float(acc - majority), 4),
               "persistence_baseline": round(float(persist), 4),
               "class_weight": CFG['class_weight'],
               "xgb_params": params}, open(OUT / "metrics.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(10, 10))
    top = imp.head(25)
    ax.barh(top.feature[::-1], top.gain[::-1], color="#3B82F6")
    ax.set_xlabel("Gain"); ax.set_title("Top-25 Features", fontweight="bold")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()

    print(f"\n  Saved -> {OUT}/")
    print("=" * 68)


if __name__ == "__main__":
    main()