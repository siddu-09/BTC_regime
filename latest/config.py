"""
config.py — single place to tweak every knob in the pipeline.

Three ways to change a value, in increasing precedence:

  1. Edit the defaults below.
  2. Environment variable, prefixed by section:
         RL_VOTE_TH=0.35 python relabel.py
         XGB_MAX_DEPTH=6 python main.py
         EV_USE_FILTER=1 python test.py
  3. Programmatic override, for sweeps:
         from config import label_config
         from relabel import label_blocks
         for vt in [0.25, 0.30, 0.35]:
             labels = label_blocks(cfg=label_config(vote_th=vt))

Every gate flag records what the ablation measured, so turning one back on is
an informed choice rather than a guess.
"""
import os


# ============================================================================
# LABELING (relabel.py) — env prefix RL_
# ============================================================================
LABEL_DEFAULTS = {
    # --- timeframe evidence weights (must sum to 1.0) ----------------------
    # 2h dominates: coarsest view that still exposes internal structure.
    # 4h is one candle — confirms displacement, sees no path.
    # 15m is noisiest but the only view catching a mid-block impulse.
    'w_15m': 0.15,
    'w_1h': 0.25,
    'w_2h': 0.35,
    'w_4h': 0.25,

    # --- current evidence vs memory (must sum to 1.0) ----------------------
    'w_current': 0.78,
    'w_memory': 0.22,

    # --- memory decay over the previous 3 labels (must sum to 1.0) ---------
    'mem_w1': 0.50,
    'mem_w2': 0.30,
    'mem_w3': 0.20,

    # --- directional thresholds --------------------------------------------
    # eff_dir 0.45 sits at ~p58 of the block-efficiency distribution.
    'eff_dir': 0.45,
    'eff_borderline': 0.40,     # reachable only on full TF agreement
    'eff_high': 0.60,           # ~p80, the clearly-trending tail
    'net_min': 0.0035,          # 0.35% — above typical 4-9bp round-trip cost
    'net_high': 0.0080,         # high-conviction move

    # vote_th is THE binding gate: sole reason for rejecting 1,550 blocks.
    # WARNING: a 6-point sweep peaked here at 0.30 with z=+3.64 and neighbours
    # at +1.41 / +1.11. That spike is the shape of test-set selection, not a
    # robust optimum. Confirm on a fresh split before moving it.
    'vote_th': 0.30,
    'vote_high': 0.70,

    # --- calm (dead block) --------------------------------------------------
    # calm_range is ABSOLUTE and therefore non-stationary: chop_calm was 0.00%
    # of blocks in 2021 (median 4h range 2.22%) and 12.7% in 2023 (0.99%).
    # Set calm_adaptive=True to use a trailing quantile instead, which holds
    # the share between ~14% and ~22% across every year.
    'calm_range': 0.0045,
    'calm_vol': 0.85,
    'calm_body_frac': 0.30,     # avg 15m body < calm_range * this
    'calm_adaptive': False,
    'calm_quantile': 0.15,      # used only when calm_adaptive=True

    # --- high-conviction override (bypasses memory entirely) ---------------
    'hc_vol_ratio': 1.50,

    # --- Q4 breakout rescue -------------------------------------------------
    # Applies ONLY to blocks that failed the whole-block directional test.
    # Whole-block efficiency averages over the entire 4 hours, so a block that
    # ranges for three quarters and then breaks out cleanly in the last hour
    # scores as chop even though it ENDS with directional momentum. What
    # carries into the next block is the state at the boundary, not the average.
    #
    # A block is rescued to bullish/bearish when its final quarter:
    #   - moves in the same direction as the block as a whole   (q4 confirms)
    #   - travels at least q4_net_min of price                  (significant)
    #   - is efficient in its own right                         (a real impulse)
    #   - accounts for q4_share_min of the block's total range  (the shift matters)
    # MEASURED: the rule fires on 1,020 blocks (7.2%) and cuts chop_volatile
    # from 67.3% to 60.0%. The premise half-holds and half-fails:
    #
    #   next block is DIRECTIONAL    41.0%  vs 34.7% base   <- TRUE, +6.3pp
    #   next block SAME direction    22.9%  vs 22.7% for
    #                                        ordinary trend blocks  <- ZERO gain
    #   next block OPPOSITE          18.0%  vs 14.5%        <- WORSE
    #
    # same:opposite is 1.27 for rescued blocks vs 1.57 for ordinary trend
    # blocks, so a Q4 breakout is LESS directionally persistent than an
    # ordinary trend block - consistent with late-block breakouts often being
    # exhaustion or stop-run moves. A threshold sweep confirms it: same-dir
    # stays flat at 22.9-23.6% across five settings while next-dir falls
    # 43.9% -> 36.7%. Tightening selects fewer blocks without improving
    # direction.
    #
    # Downstream cost: lift over Markov +1.68 -> +1.04 pp, and per-class AUC
    # fell for three of four classes (bullish -0.024, chop_volatile -0.033).
    #
    # The impulse is real but it signals ENERGY, not DIRECTION. Its correct
    # home is a feature, not a label - as a feature it lifts "next block is
    # directional" AUC 0.5491 -> 0.5509 (z +3.60 -> +5.88).
    #
    # Default OFF. Set use_q4_rescue=True to reproduce the measurement.
    'use_q4_rescue': False,
    'q4_net_min': 0.0025,      # |q4 net| / price
    'q4_eff_min': 0.50,        # |q4 net| / q4 range
    'q4_share_min': 0.35,      # q4 range / block range

    # --- secondary gates ----------------------------------------------------
    # ABLATION RESULTS (14,232 blocks, out-of-sample directional AUC vs a
    # block-shuffled null). "identical" = byte-identical label output.
    #
    #   sign gate   : 0 sole rejections, identical labels          -> DEAD
    #   inertia     : identical labels                             -> DEAD
    #   path gate   : removing it improved excess persistence
    #                 +3.79 -> +4.59 pp and AUC 0.5361 -> 0.5431   -> HARMFUL
    #   vote gate   : removing it collapsed AUC 0.5361 -> 0.5010   -> KEEP
    #
    # All three are kept as switches so the ablation can be reproduced.
    'use_sign_gate': False,
    'use_inertia': False,
    'use_path_gate': False,
    'path_eff_min': 0.25,
    'inertia_max': 0.10,
    'cont_discount': 0.02,

    # --- misc ---------------------------------------------------------------
    'vol_win': 540,             # ~90 days of 4h blocks, trailing
}


# ============================================================================
# TRAINING (main.py) — env prefix XGB_ / TRAIN_
# ============================================================================
TRAIN_DEFAULTS = {
    'max_depth': 4,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 30,
    'reg_lambda': 3.0,
    'reg_alpha': 0.5,
    'n_jobs': 4,                # throttled: unbounded all-core loops crash this box
    'cv_splits': 4,
    'early_stopping_rounds': 80,
    'max_estimators': 2000,

    # MEASURED TRADE-OFF on v2.1 labels (~67% chop_volatile), same split:
    #
    #                        flat      balanced
    #   trees chosen by CV    280        1762
    #   OOS accuracy        65.47%      43.11%
    #   lift over Markov    +1.68pp    -20.67pp
    #   balanced accuracy   34.76%      43.19%
    #   overfit gap          4.33pp     31.60pp
    #   log-loss             0.934       1.171
    #   chop_calm AUC       0.9175      0.9138
    #
    # 'balanced' does fix the majority-class collapse (balanced accuracy
    # 34.76 -> 43.19) but the weighted objective misleads early stopping: it
    # ran to 1762 trees and blew the overfit gap out to 31.6pp, landing 20.67pp
    # BELOW a zero-feature Markov baseline. 'flat' is therefore the default.
    #
    # Note the per-class AUCs barely move between the two (chop_calm 0.9175 vs
    # 0.9138). The model knows the same things either way — only the decision
    # rule changes. Judge by AUC, not by accuracy.
    #
    # If experimenting with 'balanced', cap max_estimators (~400) as well, or
    # the tree search will run away again.
    'class_weight': 'flat',   # 'flat' | 'balanced'
}


# ============================================================================
# EVALUATION (test.py) — env prefix EV_
# ============================================================================
EVAL_DEFAULTS = {
    # The hysteresis filter is OFF by default. Honestly evaluated it scores
    # BELOW the raw model (31.57% vs 36.85% on v1 labels) and below the
    # persistence baseline — it assumes sticky regimes, but most blocks change.
    # The look-ahead that made it look good (48.12%) has been removed.
    'use_filter': False,
    'conf_threshold': 0.45,
    'min_persist': 2,

    # Confidence thresholds for the selective-prediction table.
    'conf_grid': '0.3,0.35,0.4,0.5,0.6,0.7',
}


# ============================================================================
# PLUMBING
# ============================================================================
def _coerce(default, raw):
    if isinstance(default, bool):
        return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


def _build(defaults, prefix, overrides):
    cfg = dict(defaults)
    for k, v in cfg.items():
        env = os.environ.get(f'{prefix}{k.upper()}')
        if env is not None:
            cfg[k] = _coerce(v, env)
    for k, v in overrides.items():
        if k not in defaults:
            raise KeyError(f'unknown config key {k!r}; valid: {sorted(defaults)}')
        cfg[k] = v
    return cfg


def label_config(**overrides):
    """Config for relabel.py. Env prefix RL_."""
    cfg = _build(LABEL_DEFAULTS, 'RL_', overrides)
    tf = cfg['w_15m'] + cfg['w_1h'] + cfg['w_2h'] + cfg['w_4h']
    if abs(tf - 1.0) > 1e-9:
        raise ValueError(f'timeframe weights must sum to 1.0, got {tf}')
    ev = cfg['w_current'] + cfg['w_memory']
    if abs(ev - 1.0) > 1e-9:
        raise ValueError(f'w_current + w_memory must sum to 1.0, got {ev}')
    mw = cfg['mem_w1'] + cfg['mem_w2'] + cfg['mem_w3']
    if abs(mw - 1.0) > 1e-9:
        raise ValueError(f'memory weights must sum to 1.0, got {mw}')
    return cfg


def train_config(**overrides):
    """Config for main.py. Env prefix XGB_."""
    cfg = _build(TRAIN_DEFAULTS, 'XGB_', overrides)
    if cfg['class_weight'] not in ('flat', 'balanced'):
        raise ValueError("class_weight must be 'flat' or 'balanced'")
    return cfg


def eval_config(**overrides):
    """Config for test.py. Env prefix EV_."""
    return _build(EVAL_DEFAULTS, 'EV_', overrides)


def describe(cfg, title):
    lines = [f'  {title}']
    for k in sorted(cfg):
        lines.append(f'    {k:22s} {cfg[k]}')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(describe(label_config(), 'LABEL'))
    print(describe(train_config(), 'TRAIN'))
    print(describe(eval_config(), 'EVAL'))
