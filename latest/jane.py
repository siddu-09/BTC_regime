# """
# jane.py — Feature engineering for next-block regime prediction

# The model sees: features(current block) + regime(current block)
# The model predicts: regime(next block)

# Every feature answers one of these questions:
#   1. HOW IS THIS REGIME ENDING?     — exhaustion, decay, Q4 behavior
#   2. WHO HAD POWER?                 — control, dominance, conviction
#   3. WHAT IS IT SLIPPING INTO?      — transition hints from the tail
#   4. HOW MUCH ENERGY WAS THERE?     — volatility state (clusters strongly)
#   5. WHAT'S THE CONTEXT?            — current regime, recent history, session

# Reads:  1min CSV (IST) + labels CSV (TradingView ground truth)
# Output: features.csv — ready for XGBoost

# Usage:
#     DATA_1MIN_CSV=BTC_IST.csv LABELS_CSV=labels.csv python jane.py
# """

# import os
# import numpy as np
# import pandas as pd
# from scipy import stats

# # ============================================================================
# # CONFIG
# # ============================================================================
# DATA_1MIN_CSV = os.environ.get("DATA_1MIN_CSV", "BTC_IST.csv")
# LABELS_CSV    = os.environ.get("LABELS_CSV", "final_labels.csv")
# OUTPUT_PATH   = os.environ.get("OUTPUT_PATH", "fz_features.csv")

# BLOCK_HOURS = {
#     "00-04": (0, 4),  "04-08": (4, 8),   "08-12": (8, 12),
#     "12-16": (12, 16), "16-20": (16, 20), "20-00": (20, 24),
# }
# REGIMES = ['bearish', 'bullish', 'chop_calm', 'chop_volatile']


# # ============================================================================
# # DATA LOADING
# # ============================================================================
# def load_1min(path):
#     df = pd.read_csv(path, usecols=['open_time','open','high','low','close','volume'])
#     df['open_time'] = pd.to_datetime(df['open_time'])
#     df = df.set_index('open_time').sort_index()
#     df = df[~df.index.duplicated(keep='first')]
#     for c in ['open','high','low','close','volume']:
#         df[c] = pd.to_numeric(df[c], errors='coerce')
#     print(f"  {len(df):,} 1min candles  ({df.index.min().date()} → {df.index.max().date()})")
#     return df


# def load_labels(path):
#     lbl = pd.read_csv(path)
#     lbl['date'] = pd.to_datetime(lbl['date']).dt.strftime('%Y-%m-%d')
#     lbl['block_id'] = lbl['date'] + '_' + lbl['block'].astype(str)
#     lbl = lbl.set_index('block_id')['regime']
#     bad = ~lbl.isin(REGIMES)
#     if bad.any():
#         print(f"  WARNING: {bad.sum()} unrecognized labels dropped")
#         lbl = lbl[~bad]
#     print(f"  {len(lbl):,} labeled blocks  {dict(lbl.value_counts())}")
#     return lbl


# def resample(df, rule):
#     return df.resample(rule, origin='start_day').agg({
#         'open':'first','high':'max','low':'min','close':'last','volume':'sum'
#     }).dropna(subset=['open'])


# def assign_block(dt):
#     h = dt.hour
#     for label, (s, e) in BLOCK_HOURS.items():
#         if s <= h < e: return label
#     return "20-00"


# def add_block_columns(df):
#     df = df.copy()
#     df['block'] = df.index.map(assign_block)
#     df['date'] = df.index.strftime('%Y-%m-%d')
#     df['block_id'] = df['date'] + '_' + df['block']
#     return df


# # ============================================================================
# # HELPER
# # ============================================================================
# def _sd(n, d, fill=0.0):
#     """Safe divide."""
#     return n / d if d != 0 else fill


# # ============================================================================
# # FEATURE GROUP 1: HOW IS THIS REGIME ENDING?
# #
# # The last quarter (Q4) of the block tells you whether the current regime
# # is sustaining or exhausting. This is the transition signal.
# # ============================================================================
# def features_exhaustion(o, h, l, c, v, n, net, net_dir, rng, price):
#     f = {}
#     q = max(n // 4, 2)  # last quarter = last 4 candles of 16

#     # Q4 slices
#     q4_o, q4_h, q4_l, q4_c, q4_v = o[-q:], h[-q:], l[-q:], c[-q:], v[-q:]
#     # Q1 slices (for comparison)
#     q1_o, q1_h, q1_l, q1_c = o[:q], h[:q], l[:q], c[:q]

#     # --- Q4 range vs Q1 range: expanding = climax, contracting = fading ---
#     q1_rng = q1_h.max() - q1_l.min()
#     q4_rng = q4_h.max() - q4_l.min()
#     f['q4_range_expansion'] = _sd(q4_rng, max(q1_rng, 1e-10))  # >1 expanding, <1 contracting
#     f['q4_range_share']     = _sd(q4_rng, max(rng, 1e-10))      # Q4's share of total range

#     # --- Q4 direction: going WITH or AGAINST the block? ---
#     q4_net = q4_c[-1] - q4_o[0]
#     f['q4_net_pct'] = _sd(q4_net, price)
#     if net_dir != 0:
#         f['q4_agrees_with_block'] = 1.0 if np.sign(q4_net) == net_dir else -1.0
#     else:
#         f['q4_agrees_with_block'] = 0.0

#     # --- Q4 wicks: is rejection GROWING at the end? ---
#     q4_bodies = np.abs(q4_c - q4_o)
#     q4_wicks  = (q4_h - q4_l) - q4_bodies
#     f['q4_wick_to_body'] = _sd(q4_wicks.sum(), max(q4_bodies.sum(), 1e-10))

#     # winner's side wicks in Q4 (exhaustion = winner getting rejected at close)
#     q4_uw = q4_h - np.maximum(q4_o, q4_c)
#     q4_lw = np.minimum(q4_o, q4_c) - q4_l
#     if net_dir > 0:
#         f['q4_winner_rejection'] = _sd(q4_uw.sum(), max(q4_rng, 1e-10))
#     elif net_dir < 0:
#         f['q4_winner_rejection'] = _sd(q4_lw.sum(), max(q4_rng, 1e-10))
#     else:
#         f['q4_winner_rejection'] = _sd((q4_uw + q4_lw).sum(), max(2 * q4_rng, 1e-10))

#     # --- Q4 volume: climax (high) = potential reversal, fade (low) = continuation ---
#     block_avg_v = v.mean() if v.mean() > 0 else 1
#     f['q4_volume_ratio'] = _sd(q4_v.mean(), block_avg_v)

#     # --- Momentum decay: first half returns vs second half returns ---
#     rets = np.diff(c) / c[:-1]
#     if len(rets) >= 4:
#         mid = len(rets) // 2
#         f['momentum_decay'] = rets[:mid].mean() - rets[mid:].mean()
#     else:
#         f['momentum_decay'] = 0

#     # --- Counter-trend tail: how many of last 4 candles went AGAINST block? ---
#     if net_dir != 0 and n >= 5:
#         tail_dirs = np.sign(c[-4:] - np.concatenate([[c[-5]], c[-4:-1]]))
#         f['counter_tail_count'] = int((tail_dirs != net_dir).sum()) / 4.0
#     else:
#         f['counter_tail_count'] = 0.5

#     # --- Last candle: wick dominance (heavy wick = rejection at close) ---
#     lc_rng = h[-1] - l[-1]
#     lc_body = abs(c[-1] - o[-1])
#     f['last_candle_wick_pct'] = _sd(lc_rng - lc_body, max(lc_rng, 1e-10))

#     # --- Late giveback: did price reach a peak/trough early then give it back? ---
#     q3_idx = (3 * n) // 4
#     if net > 0:
#         peak = c[:q3_idx+1].max()
#         f['late_giveback'] = _sd(peak - c[-1], max(rng, 1e-10))
#     elif net < 0:
#         trough = c[:q3_idx+1].min()
#         f['late_giveback'] = _sd(c[-1] - trough, max(rng, 1e-10))
#     else:
#         f['late_giveback'] = 0

#     return f


# # ============================================================================
# # FEATURE GROUP 2: WHO HAD POWER?
# #
# # Which side controlled the block? This tells us if the regime label
# # (bullish/bearish/chop) was convincing or marginal.
# # ============================================================================
# def features_control(o, h, l, c, v, n, net, net_dir, rng, price):
#     f = {}
#     bodies = np.abs(c - o)
#     bull = c > o
#     bear = c < o
#     bull_body = bodies[bull].sum() if bull.any() else 0
#     bear_body = bodies[bear].sum() if bear.any() else 0
#     tot_body = bull_body + bear_body

#     # --- Body imbalance: who had BIGGER candles? ---
#     f['body_imbalance'] = _sd(bull_body - bear_body, max(tot_body, 1e-10))

#     # --- Color dominance: who had MORE candles? ---
#     f['bull_candle_share'] = bull.sum() / n
#     # Color uniformity: how lopsided is the count? (1=all same, 0=50-50 split)
#     # The labeler checks: are most 15m candles the same color? Uniform = trending.
#     f['color_uniformity'] = abs(bull.sum() - bear.sum()) / n

#     # --- Net return: the labeler's threshold for "meaningful move" ---
#     # net_pct (signed): tells direction + magnitude
#     # abs_net_pct: the labeler checks |net_return| >= 0.24%
#     f['net_pct']     = _sd(net, price)
#     f['abs_net_pct'] = _sd(abs(net), price)

#     # --- Volume concentration: which side had volume? ---
#     bull_vol = v[bull].sum() if bull.any() else 0
#     bear_vol = v[bear].sum() if bear.any() else 0
#     tot_vol = bull_vol + bear_vol
#     f['bull_volume_share'] = _sd(bull_vol, max(tot_vol, 1e-10))
#     f['volume_imbalance']  = _sd(bull_vol - bear_vol, max(tot_vol, 1e-10))

#     # --- Efficiency: how much of the range did the winner capture? ---
#     f['path_efficiency'] = _sd(abs(net), max(np.sum(np.abs(np.diff(c))), 1e-10))
#     f['net_to_range']    = _sd(net, max(rng, 1e-10))
#     f['abs_efficiency']  = _sd(abs(net), max(rng, 1e-10))

#     # --- Directional consistency: % of returns in winner's direction ---
#     rets = np.diff(c) / c[:-1]
#     if abs(net) > 0 and len(rets) > 0:
#         d = np.sign(net)
#         with_trend = np.abs(rets[np.sign(rets) == d]).sum()
#         f['dir_consistency'] = _sd(with_trend, np.abs(rets).sum())
#     else:
#         f['dir_consistency'] = 0.5

#     # --- Structure: HH+HL vs LH+LL progression ---
#     hh = sum(h[i] > h[i-1] for i in range(1, n))
#     hl = sum(l[i] > l[i-1] for i in range(1, n))
#     lh = sum(h[i] < h[i-1] for i in range(1, n))
#     ll = sum(l[i] < l[i-1] for i in range(1, n))
#     f['structure_bias'] = (hh + hl - lh - ll) / (2 * (n - 1))

#     # --- Longest run of same-direction candles ---
#     dirs = np.sign(np.diff(c))
#     mx = cur = 1
#     for i in range(1, len(dirs)):
#         if dirs[i] == dirs[i-1] and dirs[i] != 0:
#             cur += 1; mx = max(mx, cur)
#         else:
#             cur = 1
#     f['longest_run'] = mx / n

#     # --- Close position in block range (near high = bull conviction) ---
#     f['close_position'] = _sd(c[-1] - l.min(), max(rng, 1e-10))

#     return f


# # ============================================================================
# # FEATURE GROUP 3: WICKS & PULLBACKS
# #
# # One-sided wicks = directional control (trending)
# # Two-sided wicks = rejection from both sides (chop)
# # Deep pullbacks = contested (chop-like)
# # Shallow pullbacks = trend intact
# # ============================================================================
# def features_wicks_pullbacks(o, h, l, c, v, n, net, net_dir, rng, price):
#     f = {}
#     bodies = np.abs(c - o)
#     bull = c > o; bear = c < o
#     upper_wicks = h - np.maximum(o, c)
#     lower_wicks = np.minimum(o, c) - l
#     tot_upper = upper_wicks.sum()
#     tot_lower = lower_wicks.sum()
#     tot_wick = tot_upper + tot_lower
#     tot_body = bodies.sum()

#     # --- Wick symmetry: 1=both sides rejected equally, 0=one-sided ---
#     f['wick_symmetry'] = 1 - abs(tot_upper - tot_lower) / max(tot_wick, 1e-10)

#     # --- Wick asymmetry (signed): positive=upper-heavy, negative=lower-heavy ---
#     # The labeler checks: wicks on bull candles (upper) vs bear candles (lower)
#     # Upper-heavy = sellers rejecting highs, lower-heavy = buyers rejecting lows
#     f['wick_asymmetry'] = _sd(tot_upper - tot_lower, max(tot_wick, 1e-10))

#     # --- Wick-to-body ratio: high = lots of rejection ---
#     f['wick_to_body'] = _sd(tot_wick, max(tot_body, 1e-10))

#     # --- Winner's rejection: wicks on the winning side's extremes ---
#     if net_dir > 0:
#         f['winner_wick_share'] = _sd(tot_upper, max(tot_wick, 1e-10))
#     elif net_dir < 0:
#         f['winner_wick_share'] = _sd(tot_lower, max(tot_wick, 1e-10))
#     else:
#         f['winner_wick_share'] = 0.5

#     # --- Counter-move energy: how much total body went against the winner ---
#     if net_dir != 0:
#         with_body = bodies[bull].sum() if net_dir > 0 else bodies[bear].sum()
#         ctr_body  = bodies[bear].sum() if net_dir > 0 else bodies[bull].sum()
#         f['counter_energy'] = _sd(ctr_body, max(tot_body, 1e-10))
#         # max single counter-candle vs max with-trend candle
#         with_b = bodies[bull] if net_dir > 0 else bodies[bear]
#         ctr_b  = bodies[bear] if net_dir > 0 else bodies[bull]
#         f['max_pullback_ratio'] = _sd(
#             ctr_b.max() if len(ctr_b) > 0 else 0,
#             max(with_b.max() if len(with_b) > 0 else 1e-10, 1e-10))
#     else:
#         f['counter_energy'] = 0.5
#         f['max_pullback_ratio'] = 1.0

#     # --- Candle overlap: high overlap = choppy, low overlap = trending ---
#     overlaps = []
#     for i in range(1, n):
#         lo = max(l[i], l[i-1])
#         hi = min(h[i], h[i-1])
#         un = max(h[i], h[i-1]) - min(l[i], l[i-1])
#         overlaps.append(_sd(max(0, hi - lo), max(un, 1e-10)))
#     f['avg_overlap'] = np.mean(overlaps) if overlaps else 0

#     return f


# # ============================================================================
# # FEATURE GROUP 4: ENERGY / VOLATILITY STATE
# #
# # Volatility clusters: current vol → next vol. Strongest single predictor.
# # ============================================================================
# def features_energy(o, h, l, c, v, n, rng, price):
#     f = {}
#     ranges = h - l

#     # Keep ONLY 3 non-redundant volatility measures:
#     # 1. block_range_pct — total range (the labeler's threshold feature)
#     f['block_range_pct'] = _sd(rng, price)
#     # 2. big_candle_freq — counts spikes (different from average range)
#     med = np.median(ranges) if len(ranges) > 0 else 0
#     f['big_candle_freq'] = np.mean(ranges > 1.5 * med) if med > 0 else 0
#     # 3. block_volume — raw, converted to vol_ratio later (the labeler's threshold feature)
#     f['block_volume'] = v.sum()

#     # REMOVED (all correlated >0.80 with block_range_pct → caused chop_calm over-prediction):
#     #   median_range_pct, max_range_pct, total_travel_pct, realized_vol, detrended_vol
#     # detrended_vol was especially misleading: clean trends have LOW noise = same as calm

#     return f


# # ============================================================================
# # FEATURE GROUP 5: SEQUENCE (first half vs second half)
# #
# # Which phase came first? Continuation or reversal?
# # ============================================================================
# def features_sequence(o, h, l, c, n, net, net_dir, price):
#     f = {}
#     mid = n // 2
#     net1 = c[mid] - o[0]
#     net2 = c[-1] - c[mid]

#     f['first_half_net']  = _sd(net1, price)
#     f['second_half_net'] = _sd(net2, price)
#     f['halves_agree']    = 1 if (np.sign(net1) == np.sign(net2) and net1 != 0) else 0

#     # reversal point: where the block hit its extreme against the final direction
#     if net > 0:
#         f['reversal_point'] = float(np.argmin(c)) / (n - 1)
#     elif net < 0:
#         f['reversal_point'] = float(np.argmax(c)) / (n - 1)
#     else:
#         f['reversal_point'] = 0.5

#     return f


# # ============================================================================
# # FEATURE GROUP 6a: 1HR CONTROL CHECK (4 candles per block)
# #
# # THIS WAS OUR STRONGEST LABELING SIGNAL:
# #   "on 1hr we can see who had control most of the time"
# #   "1hr all red = bearish, no debate"
# #
# # 4 hourly candles per block. Each candle represents a full hour of
# # price discovery — not micro-noise like 15m. When all 4 agree,
# # that's sustained control. When wicks appear on 1hr, that's REAL
# # rejection, not a 1-minute spike.
# # ============================================================================
# def features_1h(grp):
#     o = grp['open'].values; h = grp['high'].values
#     l = grp['low'].values; c = grp['close'].values
#     v = grp['volume'].values
#     n = len(c)
#     if n < 2:
#         return {}
#     f = {}
#     bull = (c > o); bear = (c < o)
#     nb = bull.sum(); nr = bear.sum()
#     bodies = np.abs(c - o)

#     # ── Color uniformity: "are all 1hr candles the same color?" ──
#     # This was THE decisive check: all same = clean control, split = contested
#     f['1h_all_same_color'] = 1 if (nb == n or nr == n) else 0
#     f['1h_color_uniformity'] = abs(nb - nr) / n   # 1.0 = all same, 0 = 50-50
#     f['1h_bull_count'] = nb / n

#     # ── Body imbalance: which hourly candle was biggest? ──
#     bull_body = bodies[bull].sum() if nb > 0 else 0
#     bear_body = bodies[bear].sum() if nr > 0 else 0
#     tot_body = bull_body + bear_body
#     f['1h_body_imbalance'] = _sd(bull_body - bear_body, max(tot_body, 1e-10))

#     # ── Wick analysis: one-sided or two-sided on HOURLY scale? ──
#     # A 1hr wick = sellers/buyers rejected for a full hour. Real signal.
#     upper_w = (h - np.maximum(o, c)).sum()
#     lower_w = (np.minimum(o, c) - l).sum()
#     tot_wick = upper_w + lower_w
#     f['1h_wick_symmetry'] = 1 - abs(upper_w - lower_w) / max(tot_wick, 1e-10)
#     f['1h_wick_asymmetry'] = _sd(upper_w - lower_w, max(tot_wick, 1e-10))
#     f['1h_wick_to_body'] = _sd(tot_wick, max(tot_body, 1e-10))

#     # ── Structure: HH/HL vs LH/LL on hourly scale ──
#     # Each comparison = a full hour vs the previous hour. Meaningful.
#     if n >= 3:
#         hh = sum(h[i] > h[i-1] for i in range(1, n))
#         hl = sum(l[i] > l[i-1] for i in range(1, n))
#         lh = sum(h[i] < h[i-1] for i in range(1, n))
#         ll = sum(l[i] < l[i-1] for i in range(1, n))
#         f['1h_structure_bias'] = (hh + hl - lh - ll) / (2 * (n - 1))
#     else:
#         f['1h_structure_bias'] = 0

#     # ── Volume: which hourly candles had the volume? ──
#     bull_vol = v[bull].sum() if nb > 0 else 0
#     bear_vol = v[bear].sum() if nr > 0 else 0
#     tot_vol = bull_vol + bear_vol
#     f['1h_volume_imbalance'] = _sd(bull_vol - bear_vol, max(tot_vol, 1e-10))

#     # ── Last 1hr candle: momentum going into the next block ──
#     f['1h_last_candle_net'] = _sd(c[-1] - o[-1], max(o[-1], 1e-10))
#     # Last candle direction vs block direction
#     block_dir = np.sign(c[-1] - o[0])
#     last_dir = np.sign(c[-1] - o[-1])
#     f['1h_last_agrees_block'] = 1.0 if (block_dir == last_dir and block_dir != 0) else 0.0

#     return f


# # ============================================================================
# # FEATURE GROUP 6b: 2H HIGHER-TIMEFRAME CHECK
# #
# # 2 candles per block: do they agree? Which was bigger?
# # ============================================================================
# def features_2h(grp):
#     o, h, l, c, v = grp['open'].values, grp['high'].values, grp['low'].values, \
#                      grp['close'].values, grp['volume'].values
#     if len(c) < 2:
#         return {}
#     f = {}
#     bull = c > o
#     f['htf_both_same_color'] = 1 if (bull[0] == bull[1]) else 0
#     bodies = np.abs(c - o)
#     f['htf_body_ratio'] = _sd(bodies[0] - bodies[1], max(bodies.sum(), 1e-10))
#     uw = (h - np.maximum(o, c)).sum()
#     lw = (np.minimum(o, c) - l).sum()
#     f['htf_wick_symmetry'] = 1 - abs(uw - lw) / max(uw + lw, 1e-10)
#     f['htf_second_candle_net'] = _sd(c[1] - o[1], max(o[1], 1e-10))
#     return f


# # ============================================================================
# # FEATURE GROUP 7: 4H INDICATORS — REMOVED
# #
# # atr_pct_4h, bb_width_4h, rsi_4h, ema_slope_4h, vol_zscore_4h were all
# # multi-block lagging indicators (14-30 period lookback). They dominated
# # the model (atr_pct_4h was #1 feature with 2x gain of everything else)
# # and caused massive chop_calm over-prediction because:
# #   - After calm periods, ATR/BB are still low even when a breakout starts
# #   - The model sees "low ATR" and predicts calm, missing the transition
# #   - These features duplicate what block_range_pct + vol_ratio already provide
# #     but add lag that creates false "calm" signals
# #
# # The lookback context features (cur_*, cnt6_*, prev1_*, range_3b) already
# # capture recent history without the lag problem.
# # ============================================================================


# # ============================================================================
# # MASTER: COMPUTE ALL FEATURES FOR ONE BLOCK
# # ============================================================================
# def compute_block_features(grp_15m):
#     """Compute all features from a block's 15m candles."""
#     o = grp_15m['open'].values
#     h = grp_15m['high'].values
#     l = grp_15m['low'].values
#     c = grp_15m['close'].values
#     v = grp_15m['volume'].values
#     n = len(c)
#     if n < 4:
#         return {}

#     price = o[0] if o[0] > 0 else 1
#     rng = h.max() - l.min()
#     net = c[-1] - o[0]
#     net_dir = np.sign(net)

#     f = {}
#     f.update(features_exhaustion(o, h, l, c, v, n, net, net_dir, rng, price))
#     f.update(features_control(o, h, l, c, v, n, net, net_dir, rng, price))
#     f.update(features_wicks_pullbacks(o, h, l, c, v, n, net, net_dir, rng, price))
#     f.update(features_energy(o, h, l, c, v, n, rng, price))
#     f.update(features_sequence(o, h, l, c, n, net, net_dir, price))
#     return f


# # ============================================================================
# # MAIN PIPELINE
# # ============================================================================
# def run():
#     print("=" * 70)
#     print(" JANE — Feature pipeline for next-block regime prediction")
#     print("=" * 70)

#     # ── Load ──
#     print("\n[1] Loading data ...")
#     df1m = load_1min(DATA_1MIN_CSV)
#     labels = load_labels(LABELS_CSV)

#     # ── Resample ──
#     print("\n[2] Resampling ...")
#     df15 = add_block_columns(resample(df1m, '15min'))
#     df1h = add_block_columns(resample(df1m, '1h'))
#     df2h = add_block_columns(resample(df1m, '2h'))
#     print(f"  15m: {len(df15):,}  1h: {len(df1h):,}  2h: {len(df2h):,}")

#     # ── Per-block features ──
#     print("\n[3] Computing per-block features ...")

#     # 15m features (groups 1-5: exhaustion, control, wicks, energy, sequence)
#     rows_15m = []
#     for bid, g in df15.groupby('block_id', sort=False):
#         r = compute_block_features(g)
#         r['block_id'] = bid
#         rows_15m.append(r)
#     f15 = pd.DataFrame(rows_15m).set_index('block_id')

#     # 1h features (group 6a: sustained control, hourly wicks, structure)
#     rows_1h = []
#     for bid, g in df1h.groupby('block_id', sort=False):
#         r = features_1h(g)
#         r['block_id'] = bid
#         rows_1h.append(r)
#     f1h = pd.DataFrame(rows_1h).set_index('block_id')

#     # 2h features (group 6b: HTF agreement)
#     rows_2h = []
#     for bid, g in df2h.groupby('block_id', sort=False):
#         r = features_2h(g)
#         r['block_id'] = bid
#         rows_2h.append(r)
#     f2h = pd.DataFrame(rows_2h).set_index('block_id')

#     # ── Assemble ──
#     print("\n[4] Assembling ...")
#     final = f15.join(f1h, how='left').join(f2h, how='left')

#     # Join labels
#     final['regime'] = final.index.map(labels.to_dict())
#     n_before = len(final)
#     final = final.dropna(subset=['regime'])
#     print(f"  Labeled: {len(final)} blocks (dropped {n_before - len(final)})")

#     # Sort chronologically
#     final['_dt'] = pd.to_datetime(final.index.str[:10]) + \
#                     pd.to_timedelta(final.index.str[11:13].astype(int), unit='h')
#     final = final.sort_values('_dt').drop(columns='_dt')

#     # date / block from block_id
#     final['date']  = final.index.str[:10]
#     final['block'] = final.index.str[11:]

#     # ── Volume baseline (rolling median for vol_ratio) ──
#     vol_med = final['block_volume'].rolling(540, min_periods=30).median()
#     final['vol_ratio'] = (final['block_volume'] / vol_med).fillna(1.0)
#     final.drop(columns='block_volume', inplace=True)  # raw volume is scale-dependent

#     # ── LOOKBACK CONTEXT (group 5) ──
#     print("\n[5] Adding lookback context ...")

#     # Current regime one-hot (INPUT to model — not leakage, it's known at block close)
#     for r in REGIMES:
#         final[f'cur_{r}'] = (final['regime'] == r).astype(int)

#     # Previous block regimes (lag 1, 2)
#     for lag in [1, 2]:
#         for r in REGIMES:
#             final[f'prev{lag}_{r}'] = final[f'cur_{r}'].shift(lag).fillna(0).astype(int)

#     # Regime streak
#     grp = (final['regime'] != final['regime'].shift()).cumsum()
#     final['regime_streak'] = final.groupby(grp).cumcount() + 1

#     # Rolling regime counts (last 6 blocks = 24hrs)
#     for r in REGIMES:
#         final[f'cnt6_{r}'] = final[f'cur_{r}'].rolling(6, min_periods=1).sum()

#     # Rolling volatility (3-block)
#     final['range_3b']      = final['block_range_pct'].rolling(3, min_periods=1).mean()
#     final['efficiency_3b'] = final['abs_efficiency'].rolling(3, min_periods=1).mean()

#     # Previous block's key features
#     final['prev_range']      = final['block_range_pct'].shift(1).fillna(0)
#     final['prev_efficiency'] = final['abs_efficiency'].shift(1).fillna(0)
#     final['prev_vol_ratio']  = final['vol_ratio'].shift(1).fillna(1)

#     # Range change (expanding or contracting vs previous block)
#     final['range_change'] = (final['block_range_pct'] /
#                               final['prev_range'].replace(0, np.nan)).clip(0, 5).fillna(1)

#     # ── DIRECTIONAL STREAK & REVERSAL SIGNALS ──
#     # Data shows: after 2+ consecutive trending blocks in same direction,
#     # flip probability increases by 3-8pts. The biggest signal is
#     # trending→chop (~46% of blocks after a trend go to chop).

#     # Directional streak: consecutive bull or bear (not just regime streak)
#     dir_map = {'bullish': 1, 'bearish': -1, 'chop_calm': 0, 'chop_volatile': 0}
#     final['_dir'] = final['regime'].map(dir_map)
#     dir_streak = np.zeros(len(final))
#     dirs = final['_dir'].values
#     for i in range(1, len(dirs)):
#         if dirs[i] != 0 and dirs[i] == dirs[i-1]:
#             dir_streak[i] = dir_streak[i-1] + 1
#         elif dirs[i] != 0:
#             dir_streak[i] = 1
#     final['dir_streak'] = dir_streak    # 0=chop, 1=first trending, 2+=consecutive

#     # Move shrinkage: is current move smaller than previous in same direction?
#     # Shrinking moves → momentum fading → flip/chop more likely
#     prev_net = final['net_to_range'].shift(1).fillna(0)
#     same_dir = (np.sign(final['net_to_range']) == np.sign(prev_net)) & (final['_dir'] != 0)
#     shrinking = same_dir & (final['net_to_range'].abs() < prev_net.abs())
#     final['move_shrinking'] = shrinking.astype(int)
#     final['net_change_ratio'] = (final['net_to_range'].abs() /
#                                   prev_net.abs().replace(0, np.nan)).clip(0, 5).fillna(1)

#     # Previous block's net direction (for cross-block momentum)
#     final['prev_net_to_range'] = prev_net
#     final['prev_net_sign'] = np.sign(prev_net)

#     # Cross-block direction agreement: is current block same direction as previous?
#     final['cross_block_agrees'] = (np.sign(final['net_to_range']) == np.sign(prev_net)).astype(int)

#     # Previous block's exhaustion level (if prior was already exhausting, double warning)
#     final['prev_momentum_decay'] = final['momentum_decay'].shift(1).fillna(0)
#     final['prev_close_position'] = final['close_position'].shift(1).fillna(0.5)

#     final.drop(columns='_dir', inplace=True)

#     # Session (IST) one-hot
#     for s in BLOCK_HOURS:
#         final[f'session_{s}'] = (final['block'] == s).astype(int)

#     final['day_of_week'] = pd.to_datetime(final['date']).dt.dayofweek
#     final['is_weekend']  = (final['day_of_week'] >= 5).astype(int)

#     # ── TARGET ──
#     final['next_regime'] = final['regime'].shift(-1)
#     n_before = len(final)
#     final = final.dropna(subset=['next_regime'])
#     print(f"  Final: {len(final)} blocks")

#     # ── Reorder columns ──
#     id_cols = ['date', 'block', 'regime', 'next_regime']
#     feat_cols = [c for c in final.columns if c not in id_cols]
#     final = final[id_cols + feat_cols]

#     # ── Save ──
#     final.to_csv(OUTPUT_PATH, index=True)
#     print(f"\n  Saved → {OUTPUT_PATH}")
#     print(f"  Shape: {final.shape}")
#     print(f"  Features: {len(feat_cols)}")

#     # ── Summary ──
#     print(f"\n{'='*70}")
#     print(f" FEATURE GROUPS")
#     print(f"{'='*70}")
#     groups = {
#         '1. Exhaustion / Q4 decay':  [c for c in feat_cols if any(c.startswith(p) for p in
#                                        ['q4_','momentum_','counter_tail','last_candle','late_give'])],
#         '2. Control / power':        [c for c in feat_cols if c in
#                                        ['body_imbalance','bull_candle_share','bull_volume_share',
#                                         'volume_imbalance','path_efficiency','net_to_range',
#                                         'abs_efficiency','dir_consistency','structure_bias',
#                                         'longest_run','close_position']],
#         '3. Wicks / pullbacks':      [c for c in feat_cols if c in
#                                        ['wick_symmetry','wick_to_body','winner_wick_share',
#                                         'counter_energy','max_pullback_ratio','avg_overlap']],
#         '4. Energy / volatility':    [c for c in feat_cols if c in
#                                        ['block_range_pct','median_range_pct','max_range_pct',
#                                         'total_travel_pct','big_candle_freq','realized_vol',
#                                         'detrended_vol','vol_ratio']],
#         '5. Sequence':               [c for c in feat_cols if c in
#                                        ['first_half_net','second_half_net','halves_agree',
#                                         'reversal_point','net_pct','abs_net_pct']],
#         '6. 2h HTF':                 [c for c in feat_cols if c.startswith('htf_')],
#         '7. 4h indicators':          [c for c in feat_cols if c.endswith('_4h')],
#         '8. Current regime':         [c for c in feat_cols if c.startswith('cur_')],
#         '9. Lookback context':       [c for c in feat_cols if any(c.startswith(p) for p in
#                                        ['prev','cnt6_','regime_streak','range_3b','efficiency_3b',
#                                         'range_change'])],
#         '10. Session / time':        [c for c in feat_cols if c.startswith('session_') or
#                                        c in ['day_of_week','is_weekend']],
#     }
#     total = 0
#     for name, cols in groups.items():
#         print(f"  {name:28s}  {len(cols):3d}")
#         total += len(cols)
#     print(f"  {'TOTAL':28s}  {total:3d}")

#     print(f"\n  Target distribution:")
#     print(f"  {dict(final['next_regime'].value_counts())}")
#     persist = (final['regime'] == final['next_regime']).mean()
#     print(f"  Persistence baseline: {persist*100:.1f}%")
#     print("=" * 70)

#     return final


# if __name__ == '__main__':
#     run()

"""
jane.py — Feature engineering aligned to the multi-TF labeling logic

The labeler decides regime by:
  1. Dead-block test (calm): tiny range + low volume + tiny 30m bodies
  2. Multi-TF direction: 30m, 2h, 4h each vote on direction + strength
  3. AGREEMENT across timeframes → directional; disagreement → chop
  4. When uncertain → strength-weighted control vote

So the features mirror this decision:
  - Energy/deadness features (for calm detection)
  - Per-TF control strength (unsigned — how one-sided)
  - TF AGREEMENT features (the core discriminator: trend vs chop)
  - Exhaustion/Q4 features (for predicting transitions)
  - Signed trend context ONLY at multi-block level (for direction)

Direction (bull vs bear) comes ONLY from:
  - cur_regime one-hot
  - multi-block signed trend context (dir_balance, price_trend)
Single-block features are UNSIGNED (strength/magnitude, not direction)
to prevent the model building a false "current direction = next direction" bias.

Reads:  1min CSV (IST) + labels CSV
Output: features.csv

Usage:
    DATA_1MIN_CSV=BTC_IST.csv LABELS_CSV=labels.csv python jane.py
"""
import os
import numpy as np
import pandas as pd

DATA_1MIN_CSV = os.environ.get("DATA_1MIN_CSV", "BTC_IST.csv")
LABELS_CSV    = os.environ.get("LABELS_CSV", "final_labels.csv")
OUTPUT_PATH   = os.environ.get("OUTPUT_PATH", "latest_features.csv")

BLOCK_HOURS = {
    "00-04": (0, 4),  "04-08": (4, 8),   "08-12": (8, 12),
    "12-16": (12, 16), "16-20": (16, 20), "20-00": (20, 24),
}
REGIMES = ['bearish', 'bullish', 'chop_calm', 'chop_volatile']


# ============================================================================
# DATA
# ============================================================================
def load_1min(path):
    df = pd.read_csv(path, usecols=['open_time','open','high','low','close','volume'])
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.set_index('open_time').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    print(f"  {len(df):,} 1min candles ({df.index.min().date()} -> {df.index.max().date()})")
    return df


def load_labels(path):
    lbl = pd.read_csv(path)
    lbl['date'] = pd.to_datetime(lbl['date']).dt.strftime('%Y-%m-%d')
    lbl['block_id'] = lbl['date'] + '_' + lbl['block'].astype(str)
    lbl = lbl.set_index('block_id')['regime']
    bad = ~lbl.isin(REGIMES)
    if bad.any():
        print(f"  WARNING: {bad.sum()} unrecognized labels dropped")
        lbl = lbl[~bad]
    print(f"  {len(lbl):,} labeled blocks  {dict(lbl.value_counts())}")
    return lbl


def resample(df, rule):
    return df.resample(rule, origin='start_day').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna(subset=['open'])


def assign_block(dt):
    h = dt.hour
    for label, (s, e) in BLOCK_HOURS.items():
        if s <= h < e:
            return label
    return "20-00"


def add_block_id(df):
    df = df.copy()
    df['block_id'] = df.index.strftime('%Y-%m-%d') + '_' + df.index.map(assign_block)
    return df


def _sd(n, d, f=0.0):
    return n / d if d != 0 else f


# ============================================================================
# PER-TIMEFRAME CONTROL (mirrors the labeler's tf_direction)
# ============================================================================
def tf_control(o, h, l, c):
    """
    Returns (direction, strength) for a set of candles.
      direction: +1 bull, -1 bear, 0 none — from the INTERNAL structure of the
                 candles (which side owned more body, and more candles).
      strength:  0-1, how one-sided that control was.
    Same computation relabel.py v2 uses.

    BUGFIX: this previously returned sign(c[-1] - o[0]). For every sub-timeframe
    of a block that equals (block close - block open), so the 30m/2h/4h
    "directions" were one number computed three times — measured identical on
    14,232 / 14,232 blocks. That is why tf_all_agree looked constant and was
    deleted (see the note in run()): the features were not uninformative, the
    direction calculation was broken.
    """
    n = len(c)
    if n < 1:
        return 0, 0.0
    rng = h.max() - l.min()
    if rng <= 0:
        return 0, 0.0
    bull = int((c > o).sum()); bear = int((c < o).sum())
    bodies = np.abs(c - o)
    bull_body = bodies[c > o].sum(); bear_body = bodies[c < o].sum()
    tot_body = bull_body + bear_body
    eff = abs(c[-1] - o[0]) / rng
    body_dom = (bull_body - bear_body) / tot_body if tot_body > 0 else 0
    count_dom = (bull - bear) / n if n > 0 else 0
    # 0.6/0.4 favours body over count: three tiny green candles should not
    # outvote one large red one. Matches relabel.py v2.
    direction = int(np.sign(0.6 * body_dom + 0.4 * count_dom))
    strength = eff * (abs(body_dom) * 0.5 + abs(count_dom) * 0.5)
    return direction, float(min(strength, 1.0))


# ============================================================================
# 15M FEATURES: exhaustion, control, wicks, energy, sequence (all UNSIGNED)
# ============================================================================
def features_15m(o, h, l, c, v):
    n = len(c)
    if n < 4:
        return {}
    f = {}
    price = o[0] if o[0] > 0 else 1
    rng = h.max() - l.min()
    rng_safe = max(rng, 1e-10)
    net = c[-1] - o[0]
    net_dir = np.sign(net)
    bodies = np.abs(c - o)
    bull = c > o; bear = c < o
    tot_body = bodies.sum()

    # ── ENERGY (for calm/volatile boundary) ──
    ranges = h - l
    f['block_range_pct']    = _sd(rng, price)
    f['median_range_pct']   = _sd(np.median(ranges), price)
    f['avg_body_pct']       = _sd(np.mean(bodies), price)   # calm test uses this
    f['avg_wick_pct']       = _sd(np.mean((h-l) - bodies), price)
    f['big_candle_freq']    = np.mean(ranges > 1.5*np.median(ranges)) if np.median(ranges) > 0 else 0
    rets = np.diff(c) / c[:-1]
    f['realized_vol']       = np.std(rets) if len(rets) > 1 else 0

    # ── CONTROL STRENGTH (unsigned — how convincing, not which way) ──
    f['abs_efficiency']     = _sd(abs(net), rng_safe)       # |net|/range
    f['path_efficiency']    = _sd(abs(net), max(np.sum(np.abs(np.diff(c))), 1e-10))
    f['abs_net_pct']        = _sd(abs(net), price)
    f['color_uniformity']   = abs(bull.sum() - bear.sum()) / n
    bull_body = bodies[bull].sum(); bear_body = bodies[bear].sum()
    # REMOVED body_imbalance: it satisfies body_imbalance = 1 - 2*counter_energy
    # exactly, measured correlation -0.999995. It carried no information that
    # counter_energy (computed below) does not already carry.
    f['dir_consistency']    = _control_consistency(net, rets)
    hh = sum(h[i] > h[i-1] for i in range(1, n)); hl = sum(l[i] > l[i-1] for i in range(1, n))
    lh = sum(h[i] < h[i-1] for i in range(1, n)); ll = sum(l[i] < l[i-1] for i in range(1, n))
    f['structure_strength'] = abs((hh + hl - lh - ll) / (2 * (n - 1)))
    dirs = np.sign(np.diff(c)); mx = cur = 1
    for i in range(1, len(dirs)):
        if dirs[i] == dirs[i-1] and dirs[i] != 0: cur += 1; mx = max(mx, cur)
        else: cur = 1
    f['longest_run']        = mx / n
    # UNSIGNED form. (close - low)/range correlated +0.6435 with the block's own
    # direction, violating the "single-block features are unsigned" invariant
    # stated at the top of this file. |cp - 0.5|*2 keeps "how extreme was the
    # close within the range" and drops which side it closed on.
    f['close_position']     = abs(_sd(c[-1] - l.min(), rng_safe) - 0.5) * 2.0

    # ── WICKS (rejection: one-sided=trend, two-sided=chop) ──
    uw = (h - np.maximum(o, c)).sum(); lw = (np.minimum(o, c) - l).sum()
    tw = uw + lw
    f['wick_symmetry']      = 1 - abs(uw - lw) / max(tw, 1e-10)
    f['wick_to_body']       = _sd(tw, max(tot_body, 1e-10))
    if net_dir > 0:   f['winner_rejection'] = _sd(uw, max(tw, 1e-10))
    elif net_dir < 0: f['winner_rejection'] = _sd(lw, max(tw, 1e-10))
    else:             f['winner_rejection'] = 0.5

    # ── PULLBACKS (depth of counter-moves) ──
    if net_dir != 0:
        ctr = bear_body if net_dir > 0 else bull_body
        f['counter_energy'] = _sd(ctr, max(tot_body, 1e-10))
        with_b = bodies[bull] if net_dir > 0 else bodies[bear]
        ctr_b  = bodies[bear] if net_dir > 0 else bodies[bull]
        f['max_pullback_ratio'] = _sd(ctr_b.max() if len(ctr_b) else 0,
                                       max(with_b.max() if len(with_b) else 1e-10, 1e-10))
    else:
        f['counter_energy'] = 0.5; f['max_pullback_ratio'] = 1.0
    ov = []
    for i in range(1, n):
        lo = max(l[i], l[i-1]); hi = min(h[i], h[i-1])
        un = max(h[i], h[i-1]) - min(l[i], l[i-1])
        ov.append(_sd(max(0, hi-lo), max(un, 1e-10)))
    f['avg_overlap'] = np.mean(ov) if ov else 0

    # ── EXHAUSTION / Q4 (transition signals) ──
    q = max(n // 4, 2)
    q4o, q4h, q4l, q4c, q4v = o[-q:], h[-q:], l[-q:], c[-q:], v[-q:]
    q1h, q1l = h[:q], l[:q]
    q1_rng = q1h.max() - q1l.min(); q4_rng = q4h.max() - q4l.min()
    f['q4_range_expansion'] = _sd(q4_rng, max(q1_rng, 1e-10))
    f['q4_range_share']     = _sd(q4_rng, rng_safe)
    q4_net = q4c[-1] - q4o[0]
    f['q4_agrees_with_block'] = (1.0 if np.sign(q4_net) == net_dir and net_dir != 0
                                  else (-1.0 if net_dir != 0 else 0.0))
    q4b = np.abs(q4c - q4o); q4w = (q4h - q4l) - q4b
    f['q4_wick_to_body']    = _sd(q4w.sum(), max(q4b.sum(), 1e-10))
    q4uw = q4h - np.maximum(q4o, q4c); q4lw = np.minimum(q4o, q4c) - q4l
    if net_dir > 0:   f['q4_winner_rejection'] = _sd(q4uw.sum(), max(q4_rng, 1e-10))
    elif net_dir < 0: f['q4_winner_rejection'] = _sd(q4lw.sum(), max(q4_rng, 1e-10))
    else:             f['q4_winner_rejection'] = 0.5
    f['q4_volume_ratio']    = _sd(q4v.mean(), max(v.mean(), 1e-10))
    if len(rets) >= 4:
        mid = len(rets)//2; f['momentum_decay'] = rets[:mid].mean() - rets[mid:].mean()
    else: f['momentum_decay'] = 0
    if net_dir != 0 and n >= 5:
        td = np.sign(c[-4:] - np.concatenate([[c[-5]], c[-4:-1]]))
        f['counter_tail_count'] = int((td != net_dir).sum()) / 4.0
    else: f['counter_tail_count'] = 0.5
    lc_rng = h[-1] - l[-1]; f['last_candle_wick_pct'] = _sd(lc_rng - abs(c[-1]-o[-1]), max(lc_rng, 1e-10))
    q3 = (3*n)//4
    if net > 0:   f['late_giveback'] = _sd(c[:q3+1].max() - c[-1], rng_safe)
    elif net < 0: f['late_giveback'] = _sd(c[-1] - c[:q3+1].min(), rng_safe)
    else:         f['late_giveback'] = 0

    # ── SEQUENCE (unsigned magnitudes + structural) ──
    mid = n // 2
    n1 = c[mid] - o[0]; n2 = c[-1] - c[mid]
    f['first_half_magnitude']  = _sd(abs(n1), price)
    f['second_half_magnitude'] = _sd(abs(n2), price)
    f['halves_agree']          = 1 if (np.sign(n1) == np.sign(n2) and n1 != 0) else 0
    f['half_momentum_shift']   = _sd(abs(n2) - abs(n1), price)
    if net > 0:   f['reversal_point'] = float(np.argmin(c)) / (n-1)
    elif net < 0: f['reversal_point'] = float(np.argmax(c)) / (n-1)
    else:         f['reversal_point'] = 0.5

    # store internal signed net for lookback (dropped later)
    f['_signed_eff'] = _sd(net, rng_safe)
    return f


def _control_consistency(net, rets):
    if abs(net) > 0 and len(rets) > 0:
        d = np.sign(net)
        return _sd(np.abs(rets[np.sign(rets) == d]).sum(), np.abs(rets).sum())
    return 0.5


# ============================================================================
# MAIN
# ============================================================================
def run():
    print("=" * 70)
    print(" JANE — Features aligned to multi-TF labeling logic")
    print("=" * 70)

    print("\n[1] Loading ...")
    df1m = load_1min(DATA_1MIN_CSV)
    labels = load_labels(LABELS_CSV)

    # Ladder matches relabel.py v2: 15m / 1h / 2h / 4h
    print("\n[2] Resampling (15m / 1h / 2h / 4h) ...")
    df15 = add_block_id(resample(df1m, '15min'))
    df1h = add_block_id(resample(df1m, '1h'))
    df2h = add_block_id(resample(df1m, '2h'))
    df4h = add_block_id(resample(df1m, '4h'))
    print(f"  15m:{len(df15):,}  1h:{len(df1h):,}  2h:{len(df2h):,}  4h:{len(df4h):,}")

    print("\n[3] 15m features ...")
    rows = []
    for bid, g in df15.groupby('block_id', sort=False):
        r = features_15m(g['open'].values, g['high'].values, g['low'].values,
                         g['close'].values, g['volume'].values)
        r['block_id'] = bid; rows.append(r)
    f15 = pd.DataFrame(rows).set_index('block_id')

    print("[4] Multi-TF agreement features (the core discriminator) ...")
    # Per-block candle groups for each timeframe in the relabel.py v2 ladder.
    grp = {'15m': {b: g for b, g in df15.groupby('block_id', sort=False)},
           '1h':  {b: g for b, g in df1h.groupby('block_id', sort=False)},
           '2h':  {b: g for b, g in df2h.groupby('block_id', sort=False)}}
    g4h = {bid: row for bid, row in df4h.set_index('block_id').iterrows()}

    # Same weights the labeler uses to form its current-evidence score.
    TF_WEIGHTS = {'15m': 0.15, '1h': 0.25, '2h': 0.35, '4h': 0.25}

    mtf_rows = []
    for bid in f15.index:
        r = {'block_id': bid}
        d, s = {}, {}
        for tf in ('15m', '1h', '2h'):
            g = grp[tf].get(bid)
            if g is not None and len(g) >= 1:
                d[tf], s[tf] = tf_control(g['open'].values, g['high'].values,
                                          g['low'].values, g['close'].values)
            else:
                d[tf], s[tf] = 0, 0.0
        if bid in g4h:
            row = g4h[bid]
            d['4h'], s['4h'] = tf_control(
                np.array([row['open']]), np.array([row['high']]),
                np.array([row['low']]), np.array([row['close']]))
        else:
            d['4h'], s['4h'] = 0, 0.0

        # Control strengths (unsigned — how convincing was each TF's control).
        # These separate trending from chop_volatile: the labeler's core signal,
        # captured as a continuous feature.
        # 4h is deliberately NOT emitted: for a single candle body_dom and
        # count_dom are both +/-1, so strength collapses to efficiency and
        # control_strength_4h would be identical to abs_efficiency (measured
        # r = 1.0000). It is still computed above, for the evidence score and
        # the agreement flag.
        for tf in ('15m', '1h', '2h'):
            r[f'control_strength_{tf}'] = s[tf]

        # Magnitude of the labeler's weighted current-evidence score (unsigned).
        # Previously this omitted the 4h term and used the broken direction, so
        # it reduced to s30 + s2h. Now it mirrors relabel.py v2 exactly.
        evidence = sum(TF_WEIGHTS[tf] * d[tf] * s[tf] for tf in TF_WEIGHTS)
        r['control_vote_strength'] = abs(evidence)

        # RESTORED: with direction fixed, the four timeframes are independent
        # witnesses and agreement is a real discriminator. It was previously
        # constant only because tf_control returned the same number four times.
        r['tf_all_agree'] = int(len(set(d.values())) == 1 and d['4h'] != 0)

        mtf_rows.append(r)
    fmtf = pd.DataFrame(mtf_rows).set_index('block_id')

    print("[5] Assembling + labels ...")
    final = f15.join(fmtf, how='left')
    final['regime'] = final.index.map(labels.to_dict())
    final = final.dropna(subset=['regime'])

    # chronological order
    final['_dt'] = (pd.to_datetime(final.index.str[:10]) +
                    pd.to_timedelta(final.index.str[11:13].astype(int), unit='h'))
    final = final.sort_values('_dt').drop(columns='_dt')
    final['date'] = final.index.str[:10]
    final['block'] = final.index.str[11:]

    # Volume ratio from 4h
    vol4h = df4h.set_index('block_id')['volume']
    vmed = vol4h.rolling(540, min_periods=30).median()
    vratio = (vol4h / vmed).fillna(1.0)
    final['vol_ratio'] = final.index.map(vratio.to_dict())

    print("[6] Context + trend + target ...")
    # Current regime one-hot (Markov context — the ONLY per-block direction source)
    for r in REGIMES:
        final[f'cur_{r}'] = (final['regime'] == r).astype(int)
    # Previous 2 blocks
    for lag in [1, 2]:
        for r in REGIMES:
            final[f'prev{lag}_{r}'] = final[f'cur_{r}'].shift(lag).fillna(0).astype(int)
    # Regime streak
    grp = (final['regime'] != final['regime'].shift()).cumsum()
    final['regime_streak'] = final.groupby(grp).cumcount() + 1
    # Rolling regime counts (24h)
    for r in REGIMES:
        final[f'cnt6_{r}'] = final[f'cur_{r}'].rolling(6, min_periods=1).sum()

    # Directional streak (consecutive same-direction trending)
    dir_map = {'bullish': 1, 'bearish': -1, 'chop_calm': 0, 'chop_volatile': 0}
    final['_dir'] = final['regime'].map(dir_map)
    ds = np.zeros(len(final)); dv = final['_dir'].values
    for i in range(1, len(dv)):
        if dv[i] != 0 and dv[i] == dv[i-1]: ds[i] = ds[i-1] + 1
        elif dv[i] != 0: ds[i] = 1
    final['dir_streak'] = ds

    # Move shrinkage (using internal signed efficiency)
    prev_eff = final['_signed_eff'].shift(1).fillna(0)
    same = (np.sign(final['_signed_eff']) == np.sign(prev_eff)) & (final['_dir'] != 0)
    final['move_shrinking'] = (same & (final['_signed_eff'].abs() < prev_eff.abs())).astype(int)

    # ── SIGNED TREND CONTEXT (multi-block direction — mean-reverting on BTC) ──
    final['dir_balance_6'] = (final['cur_bullish'].rolling(6, min_periods=1).sum() -
                              final['cur_bearish'].rolling(6, min_periods=1).sum()) / 6
    final['dir_balance_12'] = (final['cur_bullish'].rolling(12, min_periods=1).sum() -
                               final['cur_bearish'].rolling(12, min_periods=1).sum()) / 12

    # Rolling volatility context
    final['range_3b'] = final['block_range_pct'].rolling(3, min_periods=1).mean()
    final['efficiency_3b'] = final['abs_efficiency'].rolling(3, min_periods=1).mean()
    final['prev_range'] = final['block_range_pct'].shift(1).fillna(0)
    final['range_change'] = (final['block_range_pct'] /
                             final['prev_range'].replace(0, np.nan)).clip(0, 5).fillna(1)
    final['prev_momentum_decay'] = final['momentum_decay'].shift(1).fillna(0)

    # Session
    for s in BLOCK_HOURS:
        final[f'session_{s}'] = (final['block'] == s).astype(int)
    final['day_of_week'] = pd.to_datetime(final['date']).dt.dayofweek
    final['is_weekend'] = (final['day_of_week'] >= 5).astype(int)

    # Target
    final['next_regime'] = final['regime'].shift(-1)
    final = final.dropna(subset=['next_regime'])

    # Drop internal columns
    final = final.drop(columns=['_dir', '_signed_eff'])

    # Reorder
    id_cols = ['date', 'block', 'regime', 'next_regime']
    feat_cols = [c for c in final.columns if c not in id_cols]
    final = final[id_cols + feat_cols]

    final.to_csv(OUTPUT_PATH, index=True)
    print(f"\n  Saved -> {OUTPUT_PATH}  shape={final.shape}  features={len(feat_cols)}")

    persist = (final['regime'] == final['next_regime']).mean()
    print(f"  Persistence baseline: {persist*100:.1f}%")
    print(f"  Target dist: {dict(final['next_regime'].value_counts())}")
    print("=" * 70)
    return final


if __name__ == '__main__':
    run()