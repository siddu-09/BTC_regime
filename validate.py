"""
validate.py — separate real signal from noise.

Every claimed result in this project must pass these gates before it is
believed. Four convincing-looking findings here turned out to be noise; each
gate below is the one that would have caught one of them.

  gate 1  permutation test    -> "the model found something"      (direction AUC 0.515)
  gate 2  max-statistic null  -> "the best of N slices worked"    (weekend AUC 0.5435)
  gate 3  causal diff         -> "the filter improved accuracy"   (48% was look-ahead)
  gate 4  cost floor          -> "the gate has Sharpe 0.95"       (breakeven 1.8bp vs 4-5bp)
  gate 5  ablation            -> "the multi-TF check discriminates" (Step 4 was dead code)

Usage:
    from validate import permutation_test, max_stat_null, cost_floor, ablate
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')   # throttled: all-core loops crash this box
import numpy as np


# ---------------------------------------------------------------- gate 1
def block_shuffle(y, rng, size=30):
    """Shuffle in contiguous blocks so the target keeps its autocorrelation.

    This matters more than it looks. A naive row shuffle destroys temporal
    clustering, giving a null centred at 0.50. The honest null for an
    autocorrelated target sits well above that -- measured at 0.5435 for
    chop_calm on this dataset. Test against 0.50 and you will call noise a
    signal.
    """
    idx = np.arange(len(y))
    chunks = [idx[i:i + size] for i in range(0, len(y), size)]
    order = rng.permutation(len(chunks))
    return y[np.concatenate([chunks[i] for i in order])[:len(y)]]


def permutation_test(fit_score, y, n_perm=60, seed=7, block=30):
    """fit_score(y) -> scalar metric. Refits the WHOLE pipeline on shuffled y.

    Returns (observed, null_mean, null_std, p_value, z). Signal is real only
    if p < 0.05 against a structure-preserving null.
    """
    rng = np.random.default_rng(seed)
    obs = fit_score(y)
    null = np.array([fit_score(block_shuffle(y, rng, block)) for _ in range(n_perm)])
    p = float((null >= obs).mean())
    return obs, null.mean(), null.std(), p, (obs - null.mean()) / null.std()


# ---------------------------------------------------------------- gate 2
def max_stat_null(n_tried, se, n_sim=20000, seed=1):
    """Distribution of the BEST metric across n_tried independent noise tests.

    Report the best of N slices/configs against THIS, never against 0.50.
    Example: best-of-10 AUCs with SE 0.028 has a median of 0.542 under pure
    noise -- so an observed 0.5435 is the 53rd percentile of nothing.
    """
    rng = np.random.default_rng(seed)
    sims = rng.normal(0.5, se, size=(n_sim, n_tried)).max(axis=1)
    return {'median': np.median(sims), 'p90': np.quantile(sims, .9),
            'p99': np.quantile(sims, .99), 'sims': sims}


# ---------------------------------------------------------------- gate 3
def causal_diff(shipped, recompute_causally, tol=1e-9):
    """Recompute a column using ONLY past+present data and diff it.

    Any mismatch is look-ahead. This is how the retroactive-rewrite bug in the
    hysteresis filter was found: it reported 48.12% against a true 31.57%.
    """
    d = np.abs(np.asarray(shipped, float) - np.asarray(recompute_causally, float))
    return {'max_abs_diff': float(np.nanmax(d)),
            'n_mismatched': int((d > tol).sum()), 'clean': bool(np.nanmax(d) <= tol)}


# ---------------------------------------------------------------- gate 4
def cost_floor(pnl_gross_bp, trades_per_period, live_cost_bp):
    """Breakeven cost per side. Compute BEFORE looking at out-of-sample Sharpe.

    A strategy whose breakeven is below the live fee is dead no matter how good
    the gross curve looks.
    """
    be = pnl_gross_bp / trades_per_period if trades_per_period else np.inf
    return {'gross_bp': pnl_gross_bp, 'trades': trades_per_period,
            'breakeven_bp_per_side': be, 'live_cost_bp': live_cost_bp,
            'survives': bool(be > live_cost_bp)}


def sharpe_se(sharpe, n_periods, periods_per_year):
    """Annualised Sharpe with its standard error. |t| < 2 is not a result."""
    yrs = n_periods / periods_per_year
    se = 1 / np.sqrt(yrs) if yrs > 0 else np.inf
    return {'sharpe': sharpe, 'se': se, 't': sharpe / se if se else 0.0,
            'years': yrs, 'significant': abs(sharpe / se) > 2 if se else False}


# ---------------------------------------------------------------- gate 5
def ablate(score_fn, baseline_score, feature_groups, tol=0.005):
    """Delete each feature group and re-score. If the metric does not move, the
    group was never load-bearing.

    This is what exposed control_strength_30m / control_strength_2h /
    control_vote_strength: they mirror a Step-4 vote that never executes,
    because d30 == d2h == d4h by construction in tf_control().
    """
    out = {}
    for name, cols in feature_groups.items():
        s = score_fn(drop=cols)
        out[name] = {'score_without': s, 'delta': s - baseline_score,
                     'load_bearing': abs(s - baseline_score) > tol}
    return out


# ---------------------------------------------------------------- reporting
def verdict(name, obs, null_mean, null_std, p):
    tag = 'REAL SIGNAL' if p < 0.05 else 'INDISTINGUISHABLE FROM NOISE'
    return (f'{name:34s} obs={obs:.4f}  null={null_mean:.4f}+/-{null_std:.4f}  '
            f'p={p:.3f}  z={(obs-null_mean)/null_std:+6.2f}   {tag}')


if __name__ == '__main__':
    print(__doc__)
    m = max_stat_null(10, 0.028)
    print(f'best-of-10 under pure noise: median {m["median"]:.4f}  '
          f'p90 {m["p90"]:.4f}  p99 {m["p99"]:.4f}')
