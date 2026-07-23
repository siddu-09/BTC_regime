"""
relabel.py - High-precision multi-timeframe regime labeling for BTCUSDT 4h blocks.

The label answers "who controlled this block?", not "what did one candle do".
Design priority: HIGH PRECISION on directional labels, low transition count
achieved by raising the evidence bar, and zero lag on genuine reversals.

All tunable values live in `config.py`. Change them by editing that file, by
env var (`RL_VOTE_TH=0.35 python relabel.py`), or programmatically:

    from config import label_config
    from relabel import label_blocks
    for vt in (0.25, 0.30, 0.35):
        out = label_blocks(cfg=label_config(vote_th=vt))

HARD INVARIANT
--------------
A directional label's SIGN always equals sign(close - open). Memory and
multi-timeframe evidence decide WHETHER a block is directional, never WHICH
WAY, so every label stays factually true of its own block.

CAUSALITY
---------
Every input is available at the block's close. Memory reads only already-
assigned previous labels. The volume baseline is a trailing rolling median
(pandas rolling is right-aligned). No future data enters any label.

CHANGES IN v2.1 - three gates removed on ablation evidence
----------------------------------------------------------
  sign-agreement : 0 sole rejections, byte-identical labels        -> removed
  memory inertia : byte-identical labels; it never binds because
                   the vote gate fails first                       -> removed
  path efficiency: removing it improved excess persistence
                   +3.79 -> +4.59 pp and directional AUC
                   0.5361 -> 0.5431                                -> removed
  vote           : removing it collapsed AUC to 0.5010             -> KEPT

Memory still acts, through `blended = w_current*current + w_memory*memory`.
What was removed is the *threshold* adjustment, which never changed an outcome.
All three remain available as config switches so the ablation is reproducible.

KNOWN TRADE-OFF
---------------
Because label(t) depends on label(t-1..t-3), predicting label(t+1) is partly
predicting a recursion whose inputs are known. Transition counts fall and
accuracy rises, but the PERSISTENCE BASELINE rises with it. The win condition
is EXCESS PERSISTENCE, which main() prints. Fewer transitions alone means
nothing.

Usage:
    DATA_1MIN_CSV=BTC_IST.csv LABELS_CSV=labels.csv python relabel.py
"""
import os
import numpy as np
import pandas as pd

from config import label_config, describe

DATA_1MIN_CSV = os.environ.get("DATA_1MIN_CSV", "BTC_IST.csv")
OUTPUT_PATH = os.environ.get("LABELS_CSV", "labels.csv")
COMPARE_TO = os.environ.get("COMPARE_TO", "final_labels.csv")

BLOCK_HOURS = {
    "00-04": (0, 4), "04-08": (4, 8), "08-12": (8, 12),
    "12-16": (12, 16), "16-20": (16, 20), "20-00": (20, 24),
}
REGIMES = ['bearish', 'bullish', 'chop_calm', 'chop_volatile']
DIR_VALUE = {'bullish': 1.0, 'bearish': -1.0, 'chop_calm': 0.0,
             'chop_volatile': 0.0}


# ============================================================================
# HELPERS
# ============================================================================
def assign_block(dt):
    h = dt.hour
    for label, (s, e) in BLOCK_HOURS.items():
        if s <= h < e:
            return label
    return "20-00"


def resample(df, rule):
    return df.resample(rule, origin='start_day').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open'])


def _safe(n, d, fallback=0.0):
    return n / d if d else fallback


def tf_control(o, h, l, c):
    """Direction and control strength for one timeframe's candles.

    direction: +1 / -1 / 0, from the INTERNAL structure of the candles - which
               side owned more body, and more candles.
    strength : [0, 1], how one-sided that control was, scaled by how
               efficiently price actually travelled.

    BUGFIX (v2): this previously returned sign(c[-1] - o[0]), which is identical
    across every sub-timeframe of a block, making multi-timeframe agreement
    vacuous - measured identical on 14,232 / 14,232 blocks. Deriving direction
    from body and count dominance makes each timeframe an independent witness.

    For a single candle (the 4h view) body_dom and count_dom collapse to
    sign(net) and strength reduces to efficiency - exactly its confirming role.
    """
    n = len(c)
    if n < 1:
        return 0, 0.0
    rng = h.max() - l.min()
    if rng <= 0:
        return 0, 0.0
    bodies = c - o
    bull_body = bodies[bodies > 0].sum()
    bear_body = -bodies[bodies < 0].sum()
    total_body = bull_body + bear_body
    body_dom = _safe(bull_body - bear_body, total_body)
    count_dom = (int((c > o).sum()) - int((c < o).sum())) / n
    # 0.6/0.4 favours body over count: three tiny green candles should not
    # outvote one large red one.
    direction = int(np.sign(0.6 * body_dom + 0.4 * count_dom))
    efficiency = abs(c[-1] - o[0]) / rng
    strength = efficiency * (0.5 * abs(body_dom) + 0.5 * abs(count_dom))
    return direction, float(min(strength, 1.0))


def path_efficiency(c15):
    """|net| / distance walked. Low = zigzag. Orthogonal to efficiency, which
    normalises by range rather than by path length."""
    steps = np.abs(np.diff(c15))
    return _safe(abs(c15[-1] - c15[0]), steps.sum())


def q4_impulse(o15, h15, l15, c15, block_range, price):
    """Measure the final quarter of the block as an impulse in its own right.

    Whole-block efficiency averages over four hours, so a block that ranges for
    three quarters then breaks out in the last hour looks like chop. What
    carries into the NEXT block is the state at the boundary, so the last
    quarter is measured separately.

    Returns (direction, net_ret, efficiency, range_share).
    """
    n = len(c15)
    q = max(n // 4, 1)
    qo, qh, ql, qc = o15[-q:], h15[-q:], l15[-q:], c15[-q:]
    q_net = qc[-1] - qo[0]
    q_rng = qh.max() - ql.min()
    return (int(np.sign(q_net)),
            q_net / price if price else 0.0,
            _safe(abs(q_net), q_rng),
            _safe(q_rng, block_range))


# ============================================================================
# CLASSIFIER
# ============================================================================
def classify(sub, row4h, vol_ratio, memory_score, cfg, calm_range):
    """Label one 4h block.

    1. dead-block test           -> chop_calm
    2. per-timeframe control     -> weighted current evidence
    3. high-conviction override  -> label now, memory bypassed
    4. directional test          -> bullish / bearish, else chop_volatile
    """
    o4, h4, l4, c4 = row4h['open'], row4h['high'], row4h['low'], row4h['close']
    price = o4 if o4 > 0 else 1.0
    net = c4 - o4
    rng = h4 - l4
    net_ret = net / price
    rng_pct = rng / price
    eff4 = _safe(abs(net), rng)

    g15 = sub['15m']
    o15, h15, l15, c15 = (g15['open'].values, g15['high'].values,
                          g15['low'].values, g15['close'].values)
    avg_body_pct = _safe(np.mean(np.abs(c15 - o15)), price) if len(c15) else rng_pct

    # ---- 1. dead block ----------------------------------------------------
    if (rng_pct < calm_range and vol_ratio < cfg['calm_vol']
            and avg_body_pct < calm_range * cfg['calm_body_frac']):
        return 'chop_calm', {'reason': 'calm'}

    # ---- 2. current evidence ---------------------------------------------
    tf_w = {'15m': cfg['w_15m'], '1h': cfg['w_1h'],
            '2h': cfg['w_2h'], '4h': cfg['w_4h']}
    d, s = {}, {}
    for tf in ('15m', '1h', '2h'):
        g = sub[tf]
        if len(g) >= 1:
            d[tf], s[tf] = tf_control(g['open'].values, g['high'].values,
                                      g['low'].values, g['close'].values)
        else:
            d[tf], s[tf] = 0, 0.0
    d['4h'], s['4h'] = tf_control(np.array([o4]), np.array([h4]),
                                  np.array([l4]), np.array([c4]))

    current = sum(tf_w[tf] * d[tf] * s[tf] for tf in tf_w)
    all_agree = (len(set(d.values())) == 1 and d['4h'] != 0)
    blended = cfg['w_current'] * current + cfg['w_memory'] * memory_score
    net_dir = int(np.sign(net))
    diag = {'eff': eff4, 'vote': blended, 'current': current,
            'memory': memory_score, 'all_agree': all_agree}

    if net_dir == 0:
        diag['reason'] = 'flat'
        return 'chop_volatile', diag

    p_eff = path_efficiency(c15) if len(c15) > 1 else 1.0
    diag['path_eff'] = p_eff
    path_ok = (p_eff >= cfg['path_eff_min']) if cfg['use_path_gate'] else True

    # ---- 3. high-conviction override --------------------------------------
    # Strict by design: this path guarantees zero lag on a real reversal, it is
    # not a way to add trend labels. Memory is bypassed entirely.
    if (all_agree and eff4 >= cfg['eff_high']
            and abs(blended) >= cfg['vote_high']
            and vol_ratio >= cfg['hc_vol_ratio']
            and abs(net_ret) >= cfg['net_high'] and path_ok):
        diag['reason'] = 'high_conviction'
        return ('bullish' if net_dir > 0 else 'bearish'), diag

    # ---- 4. directional test ----------------------------------------------
    required_eff = cfg['eff_borderline'] if all_agree else cfg['eff_dir']
    if cfg['use_inertia']:
        # Retained for reproducibility only: ablation showed byte-identical
        # labels with this on or off, because the vote gate binds first.
        opposes = (memory_score != 0 and np.sign(memory_score) != net_dir)
        adj = (cfg['inertia_max'] if opposes else -cfg['cont_discount']) * abs(memory_score)
        required_eff += adj

    checks = {
        'eff': eff4 >= required_eff,
        'net': abs(net_ret) >= cfg['net_min'],
        'vote': abs(blended) >= cfg['vote_th'],
        'path': path_ok,
    }
    if cfg['use_sign_gate']:
        # Retained for reproducibility only: 0 sole rejections in ablation.
        checks['sign'] = (np.sign(blended) == net_dir)

    diag['required_eff'] = required_eff
    if all(checks.values()):
        diag['reason'] = 'directional'
        return ('bullish' if net_dir > 0 else 'bearish'), diag

    # ---- 5. Q4 breakout rescue --------------------------------------------
    # Only reachable once the whole-block test has already failed. Catches the
    # block that ranged for three quarters and then broke out cleanly into the
    # close: chop on average, but trending at the boundary the next block
    # inherits. Direction still comes from sign(net), so the invariant holds.
    if cfg['use_q4_rescue'] and len(c15) >= 4:
        q_dir, q_net, q_eff, q_share = q4_impulse(o15, h15, l15, c15, rng, price)
        diag.update({'q4_dir': q_dir, 'q4_net': q_net, 'q4_eff': q_eff,
                     'q4_share': q_share})
        if (q_dir == net_dir
                and abs(q_net) >= cfg['q4_net_min']
                and q_eff >= cfg['q4_eff_min']
                and q_share >= cfg['q4_share_min']
                and abs(net_ret) >= cfg['net_min']):
            diag['reason'] = 'q4_breakout'
            return ('bullish' if net_dir > 0 else 'bearish'), diag

    diag['reason'] = 'no_control'
    diag['failed'] = '+'.join(k for k, ok in checks.items() if not ok)
    return 'chop_volatile', diag


# ============================================================================
# PIPELINE
# ============================================================================
def label_blocks(df1m=None, cfg=None, verbose=True):
    """Label every 4h block. Importable so sweeps do not reload the CSV.

    Returns (labels_df, diagnostics_df).
    """
    cfg = cfg or label_config()
    if df1m is None:
        df1m = load_1min(DATA_1MIN_CSV, verbose)

    tfs = {'15m': resample(df1m, '15min'), '1h': resample(df1m, '1h'),
           '2h': resample(df1m, '2h'), '4h': resample(df1m, '4h')}
    for d in tfs.values():
        d['block_id'] = (d.index.strftime('%Y-%m-%d') + '_' +
                         d.index.map(assign_block))
    if verbose:
        print("  " + "  ".join(f"{k}:{len(v):,}" for k, v in tfs.items()))

    df4h = tfs['4h']
    vol_med = df4h['volume'].rolling(cfg['vol_win'], min_periods=30).median()
    df4h['vol_ratio'] = (df4h['volume'] / vol_med).fillna(1.0)

    # Adaptive calm threshold: a trailing quantile of block range. The absolute
    # threshold is non-stationary - chop_calm was 0.00% of blocks in 2021
    # (median 4h range 2.22%) and 12.7% in 2023 (0.99%).
    if cfg['calm_adaptive']:
        rng_pct = (df4h['high'] - df4h['low']) / df4h['open']
        calm_series = rng_pct.rolling(cfg['vol_win'], min_periods=60) \
                             .quantile(cfg['calm_quantile']).shift(1) \
                             .fillna(cfg['calm_range'])
    else:
        calm_series = pd.Series(cfg['calm_range'], index=df4h.index)

    groups = {tf: {b: g for b, g in tfs[tf].groupby('block_id', sort=False)}
              for tf in ('15m', '1h', '2h')}
    empty = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    mem_w = (cfg['mem_w1'], cfg['mem_w2'], cfg['mem_w3'])

    rows, hist = [], []
    for idx, row in df4h.iterrows():
        bid = row['block_id']
        sub = {tf: groups[tf].get(bid, empty) for tf in ('15m', '1h', '2h')}
        if len(sub['15m']) < 4:
            regime, diag = 'chop_volatile', {'reason': 'insufficient_data'}
        else:
            memory = sum(w * v for w, v in
                         zip(mem_w, list(reversed(hist[-3:]))))
            regime, diag = classify(sub, row, row['vol_ratio'], memory, cfg,
                                    calm_series.loc[idx])
        rows.append({'date': idx.strftime('%Y-%m-%d'),
                     'block': assign_block(idx), 'regime': regime,
                     'reason': diag.get('reason', '?'),
                     'failed': diag.get('failed', ''),
                     'eff': diag.get('eff', np.nan),
                     'vote': diag.get('vote', np.nan)})
        hist.append(DIR_VALUE[regime])

    full = pd.DataFrame(rows)
    return full[['date', 'block', 'regime']], full


def load_1min(path, verbose=True):
    df = pd.read_csv(path, usecols=['open_time', 'open', 'high', 'low',
                                    'close', 'volume'])
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.set_index('open_time').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if verbose:
        print(f"  {len(df):,} 1min candles")
    return df


def report(out, diag, compare_to=COMPARE_TO):
    r = out.regime.values
    print("\n  Distribution:")
    vc = out.regime.value_counts()
    for k in REGIMES:
        print(f"    {k:15s} {vc.get(k,0):6,d} ({vc.get(k,0)/len(out)*100:5.1f}%)")

    print("\n  Decision path taken:")
    for k, v in diag.reason.value_counts().items():
        print(f"    {k:22s} {v:6,d} ({v/len(diag)*100:5.1f}%)")

    failed = diag.loc[diag.failed != '', 'failed']
    if len(failed):
        print("\n  Gate that blocked a directional label (sole reason in brackets):")
        from collections import Counter
        anyf, solo = Counter(), Counter(f for f in failed if '+' not in f)
        for f in failed:
            for g in f.split('+'):
                anyf[g] += 1
        for g in ('eff', 'net', 'vote', 'path', 'sign'):
            if anyf.get(g):
                print(f"    {g:6s} {anyf[g]:6,d}   [{solo.get(g,0):,}]")

    trans = (r[:-1] != r[1:]).mean()
    p = out.regime.value_counts(normalize=True)
    indep = (p ** 2).sum()
    flips = sum(1 for i in range(1, len(r) - 1)
                if r[i] != r[i - 1] and r[i + 1] == r[i - 1])
    print(f"\n  Transition rate       {trans*100:5.1f}%")
    print(f"  Persistence           {(1-trans)*100:5.1f}%")
    print(f"  If independent        {indep*100:5.1f}%")
    print(f"  EXCESS persistence    {((1-trans)-indep)*100:+5.2f} pp"
          "   <- THE number that matters")
    print(f"  One-block round trips {flips:6,d} ({flips/len(r)*100:.1f}%)")

    if compare_to and os.path.exists(compare_to):
        old = pd.read_csv(compare_to)
        m = old.merge(out, on=['date', 'block'], suffixes=('_old', '_new'))
        ro = m.regime_old.values
        to = (ro[:-1] != ro[1:]).mean()
        po = old.regime.value_counts(normalize=True)
        print(f"\n  vs {compare_to}:")
        print(f"    agreement           {(ro==m.regime_new.values).mean()*100:5.1f}%")
        print(f"    transitions    old {to*100:5.1f}%  ->  new {trans*100:5.1f}%")
        print(f"    excess persist old {((1-to)-(po**2).sum())*100:+5.2f} pp"
              f"  ->  new {((1-trans)-indep)*100:+5.2f} pp")

    print("\n  Fewer transitions is NOT the win condition. Excess persistence")
    print("  is. If transitions fall and excess persistence does not rise, the")
    print("  labels only became stickier, and the persistence baseline the")
    print("  model must beat rose by exactly as much as its accuracy will.")


def main():
    cfg = label_config()
    print("=" * 74)
    print("  RELABEL v2.1 - high-precision multi-TF regime labels")
    print("=" * 74)
    print(describe(cfg, "config (override with RL_<KEY>=value):"))
    print()
    out, diag = label_blocks(cfg=cfg)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved -> {OUTPUT_PATH}  ({len(out):,} blocks)")
    report(out, diag)
    print("=" * 74)


if __name__ == '__main__':
    main()
