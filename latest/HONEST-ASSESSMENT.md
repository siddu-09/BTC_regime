# Honest Assessment — BTC 4h Regime Pipeline

A candid engineering read on the whole project: what's genuinely good, what the
results actually mean, where you're stuck, and what's worth trying next. This is
an opinion memo, written to be useful rather than flattering. Numbers referenced
here were all produced by the scripts in this folder (see `PIPELINE.md`).

---

## 1. The one-paragraph verdict

**Your engineering is better than your signal, and that's the honest headline.**
The pipeline is unusually disciplined — causal-clean labels, real ablations,
permutation nulls, honest baselines, no look-ahead. That rigor is the actual asset
here. But the rigor was used to establish a **negative result**: of the four
regimes, only `chop_calm` (next-block volatility) is predictable (AUC 0.92), and
direction is noise (AUC 0.56–0.60 against a ~0.54 honest null). The model does the
correct thing — it predicts "calm vs active" and refuses to guess direction. So
the project succeeded as *research* (you now know what's real) even though it
mostly failed as *alpha* (three of four regimes aren't forecastable). Do not
confuse the 65% accuracy for success; it's the imbalance, not the model.

---

## 2. What's genuinely good (don't lose this)

- **Causality is airtight.** Right-aligned rolling windows, memory reads only
  past labels, an independent reimplementation reproduces the features
  bit-for-bit. Most retail quant projects silently leak the future; you didn't.
  This is the single most valuable habit in the repo.
- **You kill your own darlings.** The v2.1 ablation removed three gates
  (`sign`, `inertia`, `path`) once they were shown to be dead or harmful, and the
  Q4-rescue idea was measured, shown to signal energy-not-direction, and demoted
  to a feature. That is real scientific discipline.
- **The metrics are honest.** Leading with lift-over-Markov and balanced accuracy
  instead of raw accuracy, and comparing AUC to an autocorrelation-aware null
  (0.5435, not 0.50), is exactly right. It's why you *know* direction is noise
  instead of believing a 0.54 AUC is signal.
- **Config is disciplined.** Every knob in one place, every default annotated with
  the measurement behind it, weight-sum validation. Sweeps are cheap and safe.

Keep all of this. It's rarer and more valuable than any individual threshold.

---

## 3. The uncomfortable truths

### 3.1 The labeling logic is elegant and mostly not paying off
This is the honest message about the labels: **the fancy parts don't earn their
keep, and the part that works is simple.**

- The multi-timeframe ladder, the 0.6/0.4 body-vs-count direction, the
  memory-blended vote — all of it produces *directionally precise* labels
  (a `bullish` block really did close up, convincingly). But since next-block
  direction doesn't persist, that precision buys **nothing downstream**. You built
  a high-precision instrument to measure a quantity that has no memory.
- The one label that carries signal — `chop_calm` — comes from a **plain
  range + volume threshold** (the "dead block" test), not from any of the
  multi-TF machinery. A three-line rule is doing the real work; the other 150
  lines are producing labels the model then correctly ignores.
- The ablations already told you this: the binding gate is `vote_th`, and the
  sophisticated direction/memory apparatus around it is either dead
  (`sign`, `inertia`) or harmful (`path`). **The complexity has been audited and
  found mostly inert.** That's not a criticism of the craft — it's a signal that
  the problem, not the code, is the constraint.

**Blunt version:** if you deleted the directional half of the labeler and kept only
"calm vs not-calm", you would lose almost nothing the model can actually use.

### 3.2 You are partly predicting your own recursion
`label(t)` depends on `label(t-1..t-3)` through memory. So predicting `label(t+1)`
is partly predicting a recursion whose inputs are known. This inflates transition
persistence *and* the baseline the model must beat, by design. The code is honest
about this (it leads with *excess* persistence), but it means the target is a
slightly circular object — a subtle reason not to over-trust any accuracy gain.

### 3.3 The "regimes" are mostly not regimes
Excess persistence: bullish −0.1, bearish +0.1, chop_volatile +1.4,
**chop_calm +37.1 pp**. Only one of your four states behaves like a persistent
state. A "regime model" where three of four regimes are effectively IID draws is,
strictly, a volatility detector wearing a 4-class costume.

---

## 4. Limitations you actually hit

| # | limitation | severity | why it matters |
|---|---|---|---|
| 1 | **Direction is unpredictable from OHLCV** (efficiency autocorr 0.021) | fundamental | no feature engineering fixes this — it's a data-source ceiling |
| 2 | **`vote_th` was tuned on the test set** (spec admits it: peak z +3.64 flanked by +1.41/+1.11) | high | that spike is likely test-set selection; the "optimum" may not be real |
| 3 | **No clean validation split** — everything is train/test, test looked at dozens of times | high | every future tuning claim is compromised until fixed |
| 4 | **The ~2σ directional finding is unconfirmed** (AUC 0.5361 vs 0.4910 null) | medium | it's exactly the size of thing that evaporates on a fresh split |
| 5 | **Absolute thresholds drift** — `net_min` yearly pass 46–76.5%, `calm_range` makes chop_calm swing 0–15.6% | medium | labels mean different things in different years; `calm_adaptive` fixes half of it but is OFF |
| 6 | **Raw data not in repo** (`BTC_IST.csv` git-ignored, absent) | medium | you can't rebuild labels/features from scratch here; reproducibility depends on one external file |
| 7 | **No economic evaluation** — is the calm signal even tradeable? | high | momentum gated on regime already dies on costs (breakeven 1.16–1.80 bp vs 4–5 bp taker). Calm might too |
| 8 | **Single label definition** — no robustness check across reasonable label variants | low-med | AUC 0.92 on calm could be partly label-construction, not market structure |

Limitations 2, 3, and 7 are the ones that would actually change your conclusions.
1 is the ceiling you keep hitting. The rest are hygiene.

---

## 5. Where I'd improve it (prioritized, honest)

**Tier 1 — do these before trusting any more numbers**

1. **Reframe the problem to match the signal.** Stop optimizing 4-class accuracy.
   The real target is binary **calm vs active** (or a continuous next-block
   realized-vol regression). You'll get a cleaner metric, a stronger model, and an
   honest framing. Everything you've measured points here.
2. **Three-way split (train / validation / test).** Tune on validation, touch test
   once. This is the precondition for limitation 2/3. Without it you cannot answer
   "is vote_th=0.30 real?" honestly.
3. **Economic evaluation of the calm signal.** Before more modeling, answer: does
   knowing "next block is calm" make money (tighter stops? vol-scaled sizing?
   options?) after costs? If not, the AUC 0.92 is a true-but-useless fact.

**Tier 2 — cheap, measured, low-risk**

4. **Turn on `calm_adaptive=True`.** Holds chop_calm at ~14–22%/yr instead of
   0–15.6%. This is the one class with signal, and its prevalence currently triples
   and collapses between quarters — fixing that stabilizes the whole target.
5. **Make `net_min` volatility-relative** (`k × trailing median range` or a
   trailing quantile of |net|). Biggest stationarity fix available.
6. **Drop the 29 structure features.** Measured to *cost* −0.0033 calm AUC. Fewer
   features, same signal, less overfitting surface.
7. **Add `q4_breakout` as a feature (not a label).** Lifts "next block directional"
   AUC 0.5491 → 0.5509, z +3.60 → +5.88. Small but real and honest.

**Tier 3 — the only thing that raises the ceiling**

8. **Exogenous data.** Perp funding rate, open-interest deltas, orderbook
   imbalance, cross-asset (DXY/SPX), and a scheduled-event calendar (CPI, FOMC —
   known in advance, drives 4h vol directly). This is the *only* route to
   directional signal. Everything OHLCV-derived is exhausted (measured: +15 vol
   features moved calm AUC by 0.0002).

**What I would NOT do:** more OHLCV features, directional prediction across all
blocks, a trend/chop gate, or 4-class accuracy as an optimization target. All four
are measured dead ends.

---

## 6. Sweeps worth running (and the trap in each)

Run these **on a validation split**, not on test. Report five numbers per signal:
`observed | null mean | null sd | p | z` against a **block-shuffled** null, 500+
permutations.

### 6.1 Label-side sweeps (change the target — handle with care)

| sweep | grid | watch | the trap |
|---|---|---|---|
| `vote_th` | 0.20–0.40 step 0.025 | excess persistence, directional AUC | changing it changes the labels, so the model baseline moves too — you cannot compare AUC across settings without re-computing each setting's own null |
| `calm_quantile` (with `calm_adaptive=True`) | 0.10–0.25 | chop_calm share stability across years, calm AUC | too high floods calm and dilutes the one real signal |
| `net_min` as `k×median_range` | k ∈ 0.5–1.5 | yearly pass-rate spread (want it flat) | absolute vs relative changes directional share by year — compare spread, not level |
| `eff_dir` | 0.40–0.55 | directional precision | probably leave at 0.45 — it's already the steadiest knob (yearly 38–45%); sweeping mostly wastes runs |
| `use_q4_rescue` on/off | boolean | downstream lift-over-Markov | already measured: ON hurts lift +1.68 → +1.04. Confirm, don't re-litigate |

**The meta-trap for all label sweeps:** every setting produces a *different target*,
and both the model's accuracy and the baseline it must beat move together. The only
valid comparison is **excess persistence** and **AUC-vs-its-own-null**, never raw
accuracy across settings.

### 6.2 Model-side sweeps (fixed labels — cleaner)

| sweep | grid | watch | note |
|---|---|---|---|
| `class_weight` | flat vs balanced | balanced accuracy AND overfit gap | balanced fixes the collapse but blew the gap to 31.6 pp and ran to 1762 trees — **cap `max_estimators` ~400 if you try it** |
| `max_depth` | 3, 4, 5 | overfit gap, calm AUC | deeper won't help a noisy target; expect ≤4 to win |
| `min_child_weight` | 15, 30, 50 | in/out gap | higher = more regularization; current 30 is sensible |
| `learning_rate` × trees | 0.01–0.05 | CV-chosen tree count | slower LR + more trees rarely beats current on this target |
| **binary calm-vs-active** | reframe | calm AUC, precision@k | the sweep most likely to produce something *useful*, not just *significant* |
| decision threshold on calm prob | 0.3–0.7 | precision/recall trade for calm | you care about confident calm calls — tune this, not global accuracy |

**Expected outcome, honestly:** the model-side sweeps will move calm AUC by ±0.01
(noise) and won't create directional signal. The binary reframe and the
calm-probability threshold are the only two likely to change what you can *do* with
the model. Set expectations accordingly so you don't over-fit the test set chasing
0.005 AUC.

---

## 7. The honest message about the labeling logic

If you take one thing from this file, take this:

> **The labeling logic is a beautifully-built answer to a question the market
> mostly doesn't answer.** Its directional half is precise, causal, and inert —
> it labels *which* blocks trended, but trend doesn't carry forward, so the model
> can't use it. Its simple "dead block" half is the only part that produces a
> learnable target. You have already proven this to yourself through ablation; the
> code is just carrying the elaborate machinery forward out of momentum.

That is not wasted work — it's how you *earned* the negative result, and a
well-established negative result is worth more than a fragile positive one. But the
next honest move is to **stop refining the directional labels** and either (a)
collapse the problem to the volatility axis that works, or (b) go get the exogenous
data that's the only thing capable of making direction predictable. Continuing to
tune `vote_th` and the multi-TF weights is polishing a part of the machine that the
downstream model has already voted, unanimously and correctly, to ignore.

---

## 8. Why memory (the 3-block blend) was added — and did it earn it?

*(Written in plain terms.)*

### What you were trying to fix
Originally you labelled each block only from its own price action. Problem: it was
noisy. The regime kept flipping — bullish, chop, bearish, chop — block after block.
A jumpy timeline like that looks broken. So you added a smoother: when deciding the
current block, also look at the last 3 blocks' regimes (`blended = 0.78·now +
0.22·recent`). If things were bullish lately, lean bullish. Fewer flips. Cleaner
chart. Totally reasonable instinct.

### What you got right
Two things, and they're not small:
- **Memory can only soften, never lie.** It can turn a weak up-block from "bullish"
  into "chop", but it can *never* turn an up-block into "bearish". Direction always
  matches the candle you'd see on the chart. So smoothing never made a label false.
- **It only looks backward.** It uses past labels, never future ones. No cheating.

If you're going to smooth, this is the safe way to do it.

### The catch (the honest part)
Ask one question: **were the flips noise, or was that the market telling you
something?**

We measured it. Over 4-hour blocks, whether BTC goes up or down is basically a
**coin flip** — one block's direction tells you almost nothing about the next
(the number is 0.02 out of 1.0; a real signal would be much higher). So the
constant flipping wasn't your labels being twitchy. **It was the truth.** The market
genuinely doesn't keep going the same direction from one 4h block to the next.

So when you smoothed the flips away, here's what really happened:
- The timeline **looks** calmer and cleaner. ✅
- But you didn't add any real predictability. The smoothed labels are no more
  forecastable than the jumpy ones. You tidied the *picture*, not the *signal*.
- And smoothing makes the labels "stickier", which raises the bar the model has to
  beat — by about as much as it helped. That's why the code measures *excess*
  persistence, not just "how few flips" — fewer flips alone means nothing.

**Simple version:** you painted over the flickering, but the flickering was real.
The paint made it prettier, not smarter.

### You already had the better tool
There are two ways to fight label noise, and you're using the weaker one as a
backup:
1. **Raise the bar for calling a block directional** (the `vote_th` gate). This is
   the honest fix, and it's already your *main* gate — it's what stops a single
   wiggly candle from flipping the label. This does the real work.
2. **Copy forward the recent regime** (the 3-block memory). This is the softer,
   cosmetic smoother on top — and it's the one that adds the "circular" problem
   (the label now partly depends on earlier labels).

Tool 1 is doing the heavy lifting. Tool 2 is a light touch-up that mostly changes
how the chart *looks*.

### The real lesson
The flipping only ever looked like a problem because you were treating **direction**
as a regime. On the axis that *is* a real regime — calm vs active
(`chop_calm`) — the market stays put on its own (calm clusters strongly). You never
needed to smooth that; it's already stable. The jitter was only ever on the
direction axis, and direction *should* jitter, because direction really does flip.

So the deepest fix isn't a better smoother. It's to stop asking direction to behave
like a regime at all — and label the thing that actually is one (calm vs active).

### If you want to know exactly how much memory did
Run the labeler with the memory weight set to 0, 0.11, 0.22, 0.33 and compare —
but judge by **excess persistence and calm AUC**, not by "how few flips". My honest
prediction: it barely moves. (Needs the raw `BTC_IST.csv`, which isn't in this repo,
so I couldn't run it here.)

---

## 9. If you want a single next step

Do this, in order, and stop after step 2 if the answer is no:

1. **Three-way split**, re-fit, and check: is the calm AUC still ~0.92 on a
   never-touched test slice? (Almost certainly yes.)
2. **Economic eval**: does a calm/active call, after costs, improve *anything*
   (sizing, stops, straddle timing)? If no → the project's honest conclusion is
   "volatility clusters, which we already knew, and it isn't tradeable here."
3. If yes → reframe to binary calm-vs-active, turn on `calm_adaptive`, drop the
   structure features, and ship that narrow, real model. Then, and only then,
   consider exogenous data for a second, separate directional bet.

The worst outcome isn't a negative result — you already have a clean one. The worst
outcome is spending another month tuning label thresholds to move a test-set number
that was never real. Don't.
