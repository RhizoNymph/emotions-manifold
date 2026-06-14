# Testing Goodfire's manifold-steering framework on Anthropic's 171 emotion vectors

## TL;DR

We tested Goodfire's manifold-aware steering framework on
Anthropic's 171 emotion concepts in Gemma 3. The geometric machinery
does what it claims at the static level: pullback paths are more like
geodesics than like straight lines, and the curved-metric geodesic
length correlates with valence/arousal distance better than the
straight-line baseline does (Pearson r=+0.758 vs +0.710, edge +0.049
over 14,535 pairs). Goodfire's central "manifold-aware steering keeps
behavior on-manifold" claim is validated for the first time at
significance — but only with the curved-metric geodesic, only at 4-D,
and only on the on-manifold metric (off-M_y E gap +0.019, p=0.002,
n=40). M_y-line tracking remains null at both 4-D and 8-D for both
methods. A positive control reveals that this 4-D edge largely
**compensates for the on-manifold quality lost** when going from
8-D to 4-D linear (4-D linear off-M_y is +0.085 worse than 8-D
linear, p<0.0001) — the geodesic recovers ~22 % of the gap.
The curved metric helps when you've under-fit the manifold, not
in absolute terms over a well-chosen linear baseline. Goodfire's
central distinguishing claim — stepping through waypoints *during*
generation — was tested at n=40 with matched segmented constant-vector
controls and finds **no support**: the pre-registered primary endpoint
(pullback M_y-line, TV vs segmented-constant, two-sided Wilcoxon) is
null (p=0.49), time-variation's only real effect is an off-manifold
*cost* selective to the geodesic and linear paths (+0.09–0.10,
p≤0.0003), and the dramatic "flip" seen in an earlier n=12 run turns
out to be a segmentation artifact plus sampling noise.

The one-line thesis: curvature routes *where you end up on the
manifold*, not *which V/A point you arrive at*.

We characterize *when* manifold-aware steering helps via a novel
arousal-lift predictor A_lift (Pearson r=+0.276, CI [+0.081, +0.436],
p=0.007, n=96 for the 8-D pullback margin; the initial n=40 estimate
of +0.384 was ~30 % inflated). We also report a methodological result on
internal-state probing: differential probes for "what the model's
activation looks like under condition X" reliably *measure* state but
don't reliably *steer* behavior; contrastive classification yields
more interpretable directions but the gap remains. The pattern across
multiple probes (composition, refusal, eval-awareness) is that
**measuring conditions in the affect subspace is easier than
controlling behavior through it.**

## 0. Methodological note on the manifold framing

Our primary pipeline does **not** implement Goodfire's curved-metric
machinery directly: we use the 8-D PCA subspace as M_h (flat) and a
kernel-weighted barycenter pullback that is mathematically a
Nadaraya–Watson kernel regression from V/A space to PCA activation
space. The "curvature" that emerges comes from the fact that
different waypoints have different effective nearest neighbors in
V/A.

To close the gap we also implemented Goodfire's full density-aware
Riemannian metric `G_E(h) = (α·e^{-E(h)} + β)^{-1} · I` on the same
8-D subspace (JAX KDE energy, L-BFGS-B geodesic solver), precomputed
geodesics for all 14,535 emotion pairs, and re-ran the n=40 A-lift
behavioral comparison. Sections §7–§10 report the geometric-vs-
behavioral dissociation that this enables.

## 1. Background

Anthropic's emotion-vectors paper extracted 171 diff-in-means
activation directions at layer 40 of Gemma 3 and reported V/A
circumplex + cluster structure across them. Goodfire's
manifold-steering paper argued that activation steering should
follow the *geometry* of the activation manifold (geodesics under
a density-aware metric), demonstrated on 1-D ordered concepts
(days, ages) and a 2-D in-context grid, and flagged emotions /
refusals / sycophancy as natural next steps. This project does
that next step.

## 2. Construction

Stack: 8-D PCA subspace of the 171 centroids as M_h (no spline,
unlike Goodfire's parametric setup); Gaussian KDE on those
centroids with bandwidth = median nearest-neighbor distance;
density-aware metric G_E from Goodfire; two-manifold setup with
M_y = 2-D (valence, arousal) from Claude judge ratings. M_y covers
all 171 emotions, with centroids from 50 judged stories each (one
emotion has 49).

**Judging-pipeline note.** Some results were rated through the
synchronous Messages API and others through the asynchronous Batches
API (a credit outage mid-project forced the switch). These are the same
inference: the two code paths share the judge model
(`claude-sonnet-4-6`, a pinned snapshot), the prompt template, the
response parser, and temperature (0.0) — the batched module imports the
prompt and parser from the sequential one. The Batches API runs
identical models, so the split is transport-only and introduces no
systematic difference. A spot-check re-judging 113 identical passages
both ways confirms it: the sequential−batched *bias* is +0.006 (valence)
and +0.017 (arousal) on the 1–9 scale — far below any effect size in
this paper — with per-passage mean |Δ| of 0.03/0.09 reflecting
temperature-0 residual nondeterminism that is present between *any* two
passes (83% of passages rated identically). The paired-Wilcoxon tests
absorb that residual; the near-zero bias is what rules out a
pipeline-driven shift. (`results/judge_pipeline_equivalence_spotcheck.json`.)

The pullback construction at each M_y target point y\*:
weights w_i = softmax(−‖y_i − y*‖² / 2σ²); pullback point
h\* = Σ w_i h_i.

## 3. Static geometric results

- **Isometry at 171 scale**: 8-D PCA subspace beats raw activation
  by +0.055 in V/A distance correlation (Pearson r=+0.710 subspace
  vs +0.655 linear). At 30 emotions the edge was essentially zero;
  scaling up revealed the structure.
- **PCA-2 has the strongest geometric isometry** at +0.845, beating
  raw 5376-D activations by +0.190. See
  `results/manifold_alternatives/embedding_isometry.png`.
- **PC1 ≈ valence, PC2 ≈ arousal**: r(PC1, V) = +0.879, r(PC2, A) =
  −0.820. PC3–8 carry orthogonal structure (confusion/cognition,
  empathy/nostalgia, anger axis, etc.) with weak V/A correlations.
  See `results/figures/writeup/pc_loadings.png`.
- **Diffusion maps marginally beat PCA at low D**: diffusion-2
  at +0.868 vs PCA-2 at +0.845. UMAP under-performs both at every
  setting tested.
- **All-pairs A_lift atlas** (10,584 pairs — all 14,535 minus the
  3,951 with V/A separation < 1.0, skipped as degenerate): centered
  at zero, std 0.07,
  range [−0.29, +0.33]. Predict-win emotions cluster around
  high-arousal positive (proud, loving, refreshed); predict-loss
  emotions around calm-positive (compassionate, at ease, serene).
- **A_lift is uncorrelated with all existing pair_alignment metrics**
  (|r| < 0.06), so if A_lift predicts behavior it contributes new
  information.

## 4. The chord pullback experiment (n=40)

40 emotion pairs × pullback / geodesic / linear trajectories × 30
waypoints × 10 prompts × Claude judge. Two metrics: off-M_y E (mean
distance to nearest centroid in M_y) and M_y-line distance (Euclidean
to target chord).

### Results

| comparison | mean gap | 95 % CI | Wilcoxon p (1-sided) | wins |
|---|---:|---|---:|---:|
| pullback − linear (off-M_y E) | +0.023 | [+0.002, +0.045] | 0.051 | 22/40 |
| pullback − linear (M_y-line) | no consistent effect | | | |

Pullback consistently produces *somewhat* more on-manifold behavior;
no systematic M_y-line advantage.

A cautionary decomposition from the early 4-pair pilot (day 3):
happy→sad was the *only* pilot pair where pullback beat linear on
arousal tracking (+0.404); on the other three, linear was slightly
better (−0.03 to −0.15). The early "pullback wins" headline reduced
to one pair — the aggregate n=40 effect above is real but much
smaller than the pilot suggested, and single-pair driver checks are
mandatory at small n.

### A_lift as a conditioning variable

A_lift(pair) = mean over chord waypoints of (kernel-weighted arousal
at y\* − target arousal at y*). At n=40:
- Pearson r=+0.384, CI [+0.110, +0.605], p=0.015
- Of 14 predict-WIN pairs: 7 wins, 1 loss
- Of 14 predict-LOSS pairs: 7 losses, 3 wins
- Predict-TIE pairs lost 7/12 — A_lift is a **one-sided positive-effect
  detector**, not a symmetric margin predictor.

### A_lift at n=96 (n=100+ extension chain): effect holds, magnitude
modestly attenuated

We extended the original n=40 pairs by sampling 60 additional pairs
stratified by A_lift quintile and re-ran the chord experiment. The
new pairs were judged via the Batches API and the original 40
sequentially; that split is transport-only (see the judging-pipeline
note in §2), so it adds only the ~0.01–0.03 per-pair judge noise the
paired design absorbs, not a systematic shift.

| analysis | n=40 (original) | n=96 (extension) | Δ |
|---|---:|---:|---:|
| A_lift → margin Pearson r | +0.384 | +0.276 | −0.108 |
| 95 % CI | [+0.110, +0.605] | [+0.081, +0.436] | tighter |
| p (1-sided) | 0.015 | 0.007 | sharper |
| \|A_lift\| → off_gap Pearson r | +0.579 | +0.357 | −0.222 |
| 95 % CI | [+0.405, +0.730]\* | [+0.202, +0.515] | tighter |
| p (1-sided) | <0.001 | <0.001 | unchanged |

(*Original n=40 CI from `analyze_alift_correlates.py`.)

**Findings stand at n=96 but the original n=40 estimates were ~30 %
too large.** Both A_lift and |A_lift| remain significant predictors
of pullback advantage; the corrected magnitudes are still meaningful
and are the ones cited in the TL;DR.

Sampling caveat: the 60 extension pairs were stratified by A_lift
quintile, so the n=96 correlation is computed over a deliberately
A_lift-spread, non-random sample. Stratifying on the predictor
inflates its variance and can bias r in either direction relative to
the natural pair population; the n=96 numbers should be read as
"effect persists out-of-sample," not as a population estimate.

**Predict-TIE bucket (|A_lift| < 0.02, n=43 at the larger sample)**:
pullback wins 13/43 on M_y-line (vs 7/12 in the n=40 estimate, 30 %
vs 42 %). The asymmetry — A_lift as a one-sided positive-effect
detector — sharpens at the larger sample. See
`results/figures/writeup/alift_n100_extension.png`.

### V_lift control: A_lift's predictive power is arousal-specific

For comparison we computed V_lift the same way (kernel-weighted
valence at chord waypoints minus target valence). At n=40:
- V_lift → margin: r=−0.159 (p=0.326). Null.
- A_lift → margin: r=+0.383 (p=0.015).
- **|A_lift| → off-M_y E gap**: r=+0.579 (p<0.001).
- |V_lift| → off-M_y E gap: r=+0.139 (p=0.39). Null.

V_lift carries no predictive signal for either pullback metric;
A_lift's effect is arousal-specific. The |A_lift|→off-M_y E gap
relationship is even stronger than the directional A_lift→margin
one, sharpening the interpretation: **pairs where the arousal-kernel-
weighting is most active in either direction are the pairs where
pullback most clearly stays on-manifold.** See
`results/figures/writeup/vlift_predictor.png`.

## 5. The compositional pathology test (n=20)

The "linear (h_e1 + h_e2) produces incoherent off-manifold behavior"
claim from the source papers, tested at norm-matched magnitudes.

| condition | coherence gap | off-M_y E gap |
|---|---:|---:|
| raw magnitude (linear ~3× pullback norm) | +0.227 (p=0.0005) | confounded |
| **norm-matched** | +0.053 (p=0.049) | **−0.031 (p=0.058)** |

The dramatic "pathological compositions" effect of the source paper
is mostly a magnitude artifact. At controlled magnitudes the
asymmetry is small and on-manifold-ness favors pullback by a CI-excludes-zero margin.

Stratified: for **same-quadrant compositions, pullback wins both
metrics**. For stretched compositions (opposite-valence or
opposite-arousal), linear edges out coherence slightly but pullback
remains on-manifold-better.

## 6. Behavioral probes

### Tone modulation (positive control)
8 emotion targets across V/A quadrants. All 8 see behavior closer to
target than baseline. Linear vs pullback single-emotion: gap +0.16,
CI [−0.054, +0.410], p=0.156. **Framework's basic premise validated.**
Anxious is uniquely hard for Gemma to produce coherently (0% linear,
10% pullback) — RLHF artifact.

### Refusal modulation
Refusal rate locked at 6 % baseline across all 7 emotions × 2 steering
types. Engagement vs hedging shows weak directional response
(self-confident → engage, fear → hedge) but within sample noise.
**Gemma 3's policy gate is decoupled from affect-subspace perturbation
at production scales.**

### Eval-awareness
Differential direction h_eval − h_natural cleanly measurable but
cosine 0.97 with neutral-tag direction (mostly "has bracketed
prefix"). Contrastive classifier produces cleaner interpretation
(top-aligned: awestruck/blissful/refreshed under explicit framing;
calm/patient/docile under subtle framing). **Steering with measured
direction is behaviorally inert**: unit-normalized at scale 80 (10×
emotion steer), produces ≤ 0.2 shift in V.

## 7. The Riemannian-metric replication

A reviewer's first move is to ask whether our N-W pullback is a
strawman of the "real" Goodfire method. This section answers that
directly by running the proper curved-metric pipeline on the same
n=40 dataset.

### Geometric finding: real curvature, real isometry edge

Pairwise distance correlation with V/A distance (n=14,535):

| M_h distance | Pearson r |
|---|---:|
| 8-D PCA straight chord | +0.710 |
| Euclidean arclength of geodesic | +0.722 |
| **G_E length of geodesic** | **+0.758** |

The +0.049 edge over straight-line baseline is comparable to the
+0.055 isometry edge of subspace-vs-raw-residual — the curvature
is doing real work.

### Behavioral finding (8-D): no steering advantage

Same n=40 A-lift pairs:

| comparison | mean | 95 % CI | Wilcoxon p (1-sided) |
|---|---:|---|---:|
| pullback − linear (M_y-line) | −0.036 | [−0.085, +0.014] | 0.925 |
| **geodesic − linear (M_y-line)** | **−0.021** | [−0.042, −0.002] | **0.978** |
| pullback − linear (off-M_y E) | +0.023 | [+0.003, +0.044] | 0.051 |
| **geodesic − linear (off-M_y E)** | **+0.008** | [−0.005, +0.020] | **0.064** |

A_lift, our 8-D pullback predictor (r=+0.38, p=0.015), is **not** a
predictor for geodesic margin (r=+0.084, p=0.60).

### Head-to-head pullback vs geodesic

Direct comparison on the same 40 pairs: geodesic closer to M_y-line
22/40, pullback 18/40, Wilcoxon p=0.51. Per-pair margins agree
(Pearson r=+0.32, p=0.04). The simpler N-W kernel regression is
*not* a strawman of the proper curved-metric method.

### Ablations

**β ablation** (curvature strength): sweeping β ∈ {0.001, 0.01, 0.1,
1.0} across three orders of magnitude, isometry edge stays flat at
+0.037 to +0.041. The curvature regime is saturated — the +0.04 edge
is what's available in the data, not an artifact of metric tuning.

**Dimension ablation** (2, 4, 6, 8, 10, 12, 14, 16, 24, 32 PCA dims):
G_E vs chord edge peaks at **6-D (+0.085)** with d=4 (+0.063) and
d=8 (+0.050) as side peaks; goes **negative** both *below* the peak
(−0.016 at d=2) and above 12-D (−0.023 at 14, −0.032 at 16, −0.106 at
32). The d=2 negative point matters: at d=2 the subspace is essentially
the V/A plane itself (PC1≈valence, PC2≈arousal), so there is no
non-behavior-aligned structure for the curved metric to exploit — the
edge requires an *interior* dimensionality where discarded dimensions
still carry usable structure (see §8's d=2 falsification). Activation
manifold has roughly 4–10 effective dimensions for V/A alignment.

**Denser dim sweep** (6, 10, 12, 14, 24 added to the original
4/8/16/32 — see `results/figures/writeup/dim_ablation_denser.png`):
fills in the curve. The G_E edge *peaks at d=6 (+0.085)*, larger
than both the 4-D point (+0.063) and 8-D (+0.050). Above d=12 the
edge becomes negative (−0.023 at 14, −0.032 at 16, −0.085 at 24,
−0.106 at 32). Sharper claim than the original 4/8/16/32 grid:
**6-D is the sweet spot for the curved metric on V/A**, with the
edge falling off monotonically in either direction.

**Silverman vs clustered_NN bandwidth (behavioral re-run, n=40, 8-D).**
The geometric edge difference (Silverman +0.062 vs production
clustered_NN +0.050, Δ +0.012 over 800 pairs) actually *does* translate
behaviorally for the geodesic method:

| method | metric | clustered_NN | Silverman | Δ (silv−prod) | Wilcoxon p (1-sided silv<prod) |
|---|---|---:|---:|---:|---:|
| pullback | off-M_y E | +0.431 | +0.442 | +0.012 | n.s. (worse) |
| **geodesic** | **off-M_y E** | **+0.446** | **+0.421** | **−0.025** | **p=0.001** ✓ |
| linear | off-M_y E | +0.454 | +0.449 | −0.005 | n.s. |
| pullback | M_y-line | +2.180 | +2.236 | +0.055 | n.s. (worse) |
| geodesic | M_y-line | +2.165 | +2.153 | −0.012 | n.s. |
| linear | M_y-line | +2.144 | +2.173 | +0.028 | n.s. (worse) |

**Silverman bandwidth is significantly better than the production
clustered_NN bandwidth — but only for the curved-metric geodesic
on the on-manifold metric.** The +0.012 geometric edge translates
to a +0.025 behavioral improvement (more than 2× amplification),
and that improvement is geodesic-specific: pullback and linear see
no benefit (and slight cost for pullback). (This comparison's n=4
sequential + n=36 batched judge split is transport-only — see the
judging-pipeline note in §2 — so it adds no systematic bias beyond the
~0.01–0.03 per-pair judge noise the paired-Wilcoxon absorbs.) See
`results/figures/writeup/silverman_vs_clustered_nn.png`.

**Adaptive-bandwidth KDE** (per-centroid k-NN bandwidth at k=5, k=10):
on the same 800-pair sample at 8-D (chord baseline r=+0.703):

| KDE bandwidth | r vs V/A | edge |
|---|---:|---:|
| fixed Silverman (σ=2.504) | +0.766 | **+0.062** |
| fixed clustered-NN (σ=3.982, production) | +0.753 | +0.050 |
| adaptive k=5 | +0.690 | **−0.013** |
| adaptive k=10 | +0.691 | **−0.012** |

Adaptive bandwidth *hurts* the isometry. The geometric ratio between
on- and off-manifold regions matters more than the local density of
each centroid; variable bandwidth flattens the metric in dense
regions (where the framework's curvature is supposed to help) and
expands it in sparse regions (where there's nothing to attract the
geodesic toward). Fixed Silverman bandwidth gives a small (+0.012)
edge over the production heuristic. See
`results/figures/writeup/adaptive_kde_geodesic.png`.

## 8. Dimensionality and the compensation reframe

The §7 dimension ablation found the geometric edge peaking at d=6.
This section asks whether that translates behaviorally, and resolves
what the low-dimension geodesic gain actually means via a positive
control.

### Behavioral re-runs at d=4 and d=6 (n=40 each)

We re-ran the chord experiment at d=4 and (motivated by the new
geometric peak finding) at d=6, comparing against the production d=8.

**d=4 behavioral (n=40):**

| comparison | mean | 95 % CI | Wilcoxon p (1-sided) | wins |
|---|---:|---|---:|---:|
| pullback − linear (M_y-line) | −0.066 | [−0.101, −0.034] | 1.000 | 11/40 |
| geodesic − linear (M_y-line) | +0.002 | [−0.017, +0.020] | 0.251 | 22/40 |
| pullback − linear (off-M_y E) | +0.009 | [−0.006, +0.023] | 0.202 | 22/40 |
| **geodesic − linear (off-M_y E)** | **+0.019** | **[+0.008, +0.030]** | **0.002** | **28/40** |

**d=6 behavioral (n=40, new):**

| comparison | mean | 95 % CI | Wilcoxon p (1-sided) | wins |
|---|---:|---|---:|---:|
| pullback − linear (M_y-line) | −0.098 | [−0.139, −0.060] | 1.000 | 7/40 |
| geodesic − linear (M_y-line) | +0.001 | [−0.017, +0.018] | 0.508 | 19/40 |
| **pullback − linear (off-M_y E)** | **+0.022** | **[−0.001, +0.046]** | **0.023** | (see scatter) |
| **geodesic − linear (off-M_y E)** | **+0.017** | **[+0.004, +0.030]** | **0.008** | (see scatter) |

**Three-way summary (off-M_y E means by dim):**

| method | d=4 | d=6 | d=8 |
|---|---:|---:|---:|
| pullback | 0.530 | 0.443 | 0.431 |
| geodesic | 0.519 | 0.449 | 0.446 |
| linear   | 0.538 | 0.466 | 0.454 |
| **gap geodesic−linear** | **−0.019** | **−0.017** | **−0.008** |

The geodesic on-manifold edge over linear stays significant at both
4-D and 6-D (p=0.002 and 0.008), and shrinks to marginal at 8-D
(p=0.064). The **geometric peak at d=6 does *not* translate to a
larger behavioral edge** than d=4 — both dims give essentially the
same behavioral on-manifold gain (~+0.018), even though the geometric
edge is +0.085 at d=6 vs +0.063 at d=4. See
`results/figures/writeup/dim_behavioral_compare.png`.

**The two methods diverge in their preferred regime**: pullback
wants higher dim (gain at 8-D, disappears at 4-D); geodesic wants
lower (marginal at 8-D, significant at 4-D). A_lift is **not** a
predictor at 4-D.

### 4-D linear vs 8-D linear positive control

The 4-D geodesic off-M_y +0.019 result above could be either:
"the curved metric is essential" (geodesic vs linear gap) or
"the curved metric compensates for dimension loss" (4-D linear is
worse than 8-D linear, geodesic recovers some of it).

Comparing the same 40 pairs at d=4 vs d=8 with LINEAR steering only:

| metric | 4-D linear | 8-D linear | 4-D − 8-D | Wilcoxon (4-D worse) |
|---|---:|---:|---:|---:|
| **off-M_y E** | +0.538 | +0.454 | **+0.085** | **p<0.0001** |
| M_y-line dist | +2.183 | +2.144 | +0.039 | p=0.24 |

**4-D linear is substantially worse than 8-D linear on off-M_y E**
(9/40 wins for 4-D). The 4-D geodesic off-M_y +0.019 gain over 4-D
linear therefore recovers only ~22 % (0.019/0.085) of the on-manifold
quality that 4-D linear loses relative to 8-D linear.

Reframe: the 4-D curved-metric finding is best read as **"the curved
metric partially compensates for dimension loss when you go below
the natural ~8-D dimensionality"** rather than "the curved metric is
essential." This is a weaker, more honest version of Goodfire's
claim — the framework helps when you've under-fit the manifold,
which is a useful but more limited statement than "geodesics beat
straight lines absolutely." See
`results/figures/writeup/linear_4d_vs_8d_positive_control.png`.

### d=2 falsification: under-fit severity is not the mechanism

The compensation reframe makes a quantitative prediction: push the
under-fit further, to d=2, and the geodesic edge should *grow* (2-D
linear loses even more ground, leaving more for the geodesic to
recover). We pre-registered this against a competing hypothesis — that
because PC1 ≈ valence (r=+0.879) and PC2 ≈ arousal (r=−0.820), the 2-D
subspace *is* essentially the V/A plane, so the edge should instead
*collapse*: linear interpolation already moves along the behavior axes,
leaving nothing for curvature to route around. The d=2 n=40 run (same
pairs, same pipeline; the geodesic-favoring Silverman bandwidth, since
the production heuristic is numerically singular at d=2) settles it:

| dim | geodesic off-M_y edge | linear under-fit (vs 8-D) |
|---|---:|---:|
| d=2 | **−0.003** (p=0.79, 16/40) | **+0.045** (p=0.0001) |
| d=4 | +0.019 (p=0.002, 28/40) | +0.085 (p<0.0001) |
| d=8 | +0.008 (p=0.064, 22/40) | (baseline) |

We ran d=2 under both bandwidths to rule out a bandwidth artifact (the
primary table uses Silverman because the production clustered_nn
heuristic is numerically singular at d=2). The null is robust: the
geodesic edge is −0.003 under Silverman and +0.004 (p=0.18, 24/40)
under clustered_nn — the two straddle zero, neither approaches the d=4
+0.019/p=0.002. The linear under-fit is +0.045 / +0.053 respectively
(linear interpolation does not use the bandwidth, so these are the same
quantity measured twice; their agreement is a reproducibility check),
both below d=4's +0.085.

Both predictions of the compensation hypothesis fail. The geodesic edge
does not grow at d=2 — it sits at the noise floor under either bandwidth,
tracking the geometric edge, which is itself *negative* at d=2 (−0.016
Silverman / −0.050 clustered_nn, vs +0.063 at 4-D). And 2-D linear under-fits *less* than 4-D linear (+0.045 vs
+0.085), not more: going to d=2 makes linear steering *better* on
on-manifold-ness, because the dimensions discarded between d=4 and d=2
(PC3–PC4: cognition/empathy structure with weak V/A correlation) were
diluting linear steering with off-V/A perturbation. That off-V/A
dilution at d=4 is precisely what the d=4 geodesic was compensating for.

So the honest mechanism is narrower than "the curved metric helps when
you've under-fit the manifold." The geodesic edge needs *two* conditions
at once: the **discarded** dimensions must carry behaviorally-relevant
structure that linear steering misses, **and** the **retained** axes
must not already be the behavior axes. Only the d≈4–6 band satisfies
both — at d=2 the retained axes already are the V/A plane (nothing to
route around), and above d≈8 the curved metric bends in unhelpful
directions. The geometric and behavioral pictures, which dissociate at
d=6 (geometric +0.085, behavioral +0.017), reconcile at d=2: both null.
See `results/day8.md` and
`results/riemannian_analysis_2d/linear_2d_vs_8d.png`.

## 9. Time-varying steering (Goodfire's central distinguishing claim)

Goodfire's framework treats the geodesic as a *trajectory* the model
should follow temporally, not just a shape in activation space. To
test this directly we implemented **segmented generation**: divide
96-token output into K=8 segments of 12 tokens each, applying a
different waypoint vector per segment. (Per-token switching would
require modifying the vLLM fork; this 8-segment chunking is the
closest tractable approximation.)

An initial n=12 run (day 5) suggested a dramatic flip — pullback's
constant-vector advantage reversing to 1/12 wins under TV — but that
design confounded time-variation with the segmented *call structure*
itself (KV-rebuilt continuations, 96-token budget, stop-token
boundaries) and compared against the unsegmented chord baseline with
mismatched metric definitions. The day-7 n=40 rerun adds the controls
that separate these: four conditions on the same 40 pairs, all judged
through the batched pipeline — tv8 (hard switches, 8×12 tokens), tv16
(16×6: half-size switches, same budget), and cv8/cv16 (the *path-
midpoint waypoint held constant* through the identical segmented call
structure).

Pre-registered primary endpoint — pullback M_y-line, tv8 vs cv8,
two-sided Wilcoxon: **mean +0.026, CI [−0.084, +0.123], p=0.49. Null.**
Time-variation per se neither helps nor hurts pullback's behavioral
tracking once segmentation is held fixed.

TV − CV-segmented gaps (n=40, positive = time-variation worse):

| comparison | method | off-M_y E | M_y-line |
|---|---|---:|---:|
| tv8−cv8 | pullback | +0.004 (p=0.80) | +0.026 (p=0.49) |
| tv8−cv8 | geodesic | **+0.097 (p=0.0001)** | +0.038 (p=0.41) |
| tv8−cv8 | linear | **+0.091 (p=0.0003)** | +0.017 (p=0.82) |
| tv16−cv16 | pullback | +0.021 (p=0.35) | −0.009 (p=0.60) |
| tv16−cv16 | geodesic | **+0.062 (p=0.044)** | +0.100 (p=0.023) |
| tv16−cv16 | linear | **+0.062 (p=0.003)** | +0.032 (p=0.48) |

Three conclusions:

1. **Goodfire's temporal claim finds no support.** TV never beats its
   matched constant control on any method × metric (all 12 tv−cv means
   ≥ −0.009, none significantly negative). The one real time-variation
   effect is a *cost*, and a selective one: stepping through waypoints
   pushes geodesic and linear generations off-manifold (+0.09–0.10,
   they traverse the degenerate path interior) while pullback — whose
   waypoints are already behaviorally adapted barycenters — is
   unaffected (+0.004, null).
2. **The n=12 flip was a segmentation artifact plus sampling noise.**
   Pullback loses to linear under the constant-vector segmented
   controls too (cv8: 21/40 myl wins; cv16: 14/40), so the reversal was
   never TV-specific; and the 1/12 extremity does not replicate (tv8:
   18/40). Even cv8 — steered constantly at the very midpoint the
   metric targets — lands far from the unsegmented chord baseline,
   so segmented and unsegmented runs are not comparable; matched
   segmented controls are mandatory for any TV design.
3. **The discontinuity interpretation is also dead**: halving the
   switch size (tv16) shrinks the geodesic off-M_y penalty
   (+0.097 → +0.062) but behaves like tv8 everywhere else, and the
   geodesic's modest advantage over linear survives every segmented
   condition (23–29/40 wins) — consistent with the constant-vector
   n=40 chord result.

A genuine per-token implementation (requires vLLM fork changes)
remains open but is now poorly motivated: the predicted benefit fails
to appear at the segment scale under matched controls. Judging losses:
4/4,800 texts (judge refusals on degenerate steered text). See
`results/figures/writeup/tv8_vs_cv8_n40.png` and
`tv16_vs_cv16_n40.png`; full matrix in `results/day7.md` and
`results/time_varying_n40/analysis_*/`.

## 10. Interpretation: the geometric–behavioral dissociation

Curvature in the activation manifold is real both geometrically and
behaviorally — but it routes "where you end up on the manifold," not
"which V/A point you arrive at." The +0.063 isometry edge at 4-D
under G_E tells us the curved metric better captures V/A distance
structure than a straight line does. The +0.019 on-manifold
behavioral gain at p=0.002 tells us this geometric improvement
translates into steered outputs that stay closer to real emotion
centroids. But the null M_y-line margin tells us that "staying near
real emotion concepts" and "actually reaching the V/A target the
user asked for" are different things, and only the first is
manifold-routable.

## 11. Differential vs contrastive direction extraction

- **Differential**: Δh = h_condition − h_baseline. Captures average
  activation shift; weights every coordinate equally regardless of
  whether it discriminates.
- **Contrastive**: logistic regression weight vector. Weights along
  the direction of maximum class separation.
- For eval-awareness: contrastive direction is cleaner — top-aligned
  emotions are calm/patient/serene/peaceful; suppresses *all*
  high-arousal symmetrically (both excited and furious).
- **Both fail to steer behavior**: contrastive at unit-normalized scale
  ±80 produces only marginally larger effects than differential (V
  range 0.3 vs 0.2). Direction is cleaner but the causal coupling to
  behavior is similar.

> Differential probes show what's there in activations; contrastive
> classification shows what discriminates conditions. The latter is
> more interpretable but neither reliably steers behavior at
> production-typical scales.

## 12. The unified pattern: interpretability ≠ steering

Across the chord experiment, composition test, refusal probe, and
eval-awareness probe/steering, we can:
- **Measure** internal state directions cleanly under varied conditions.
- **Steer** behavior with native emotion vectors.
- **Not bridge** from measured-state direction to behavioral control.

The eval-awareness direction is real (clean contrastive classifier),
interpretable (calm cluster + hostility suppression), and steering-
inert (no dose-response at any scale tested).

This is the most important transferable lesson. Projects that
extract direction-of-X from activations and assume the direction
can be used to control X-related behavior should expect the
steering effect to be much weaker than the measurement signal, and
possibly absent entirely.

## 13. A note on endpoints and multiple comparisons

This writeup reports dozens of paired Wilcoxon tests, and several
numbers sit in the p=0.02–0.06 range. Rather than a post-hoc
correction, we designate endpoints explicitly:

**Primary** (decided by the experimental design, robust to any
reasonable correction):
- The 4-D geodesic on-manifold gain (+0.019, p=0.002) and the 4-D vs
  8-D linear positive control (+0.085, p<0.0001) that reframes it.
- The time-varying primary endpoint, pre-registered before the n=40
  run: pullback M_y-line, tv8 vs cv8, two-sided — null (p=0.49).
- The geometric isometry results (n=14,535 and n=800 pair samples;
  not borderline).

**Exploratory** (real-looking but at uncorrected p=0.02–0.06, would
not all survive correction): pullback's 8-D on-manifold edge
(p=0.051), the d=6 pullback edge (p=0.023), the Silverman geodesic
improvement (p=0.001 but selected post-hoc from a bandwidth
comparison), the norm-matched composition gaps (p≈0.05), and the
A_lift directional correlation (p=0.007 at n=96 but on a stratified
extension sample). These are reported with CIs so readers can weigh
them; none of them carries the paper's thesis.

## 14. Scope and limits

Manifold-aware steering helps with:
- Producing more on-manifold behavior on chord trajectories (off-M_y
  E gap +0.023 at n=40 for pullback at 8-D; +0.019 at p=0.002 for
  geodesic at 4-D)
- More on-manifold behavior in compositions at matched magnitude
- Identifying *which* chord trajectories will produce target-aligned
  wins via A_lift (positive cases only)

It does not:
- Produce compositional pathology of source-paper magnitude (dramatic
  version was magnitude-confounded)
- Modulate refusal-rate behavior through emotion-subspace perturbation
- Enable steering via measured-from-condition directions even at large
  scale
- Benefit from time-varying delivery: stepping through waypoints during
  generation is null for pullback against the matched segmented-constant
  control (n=40, p=0.49) and an off-manifold *cost* for geodesic and
  linear paths (+0.09–0.10)

## 15. Methodological contributions

1. **Two-manifold setup** for non-parametric concept spaces:
   M_h (PCA + density metric) + M_y (behavior space as parameterization
   for pullback). Removes Goodfire's parametric-fit assumption.
2. **A_lift** as a falsifiable pair-level predictor for chord pullback
   margins. Geometrically computable from M_y alone; uncorrelated with
   existing pair-alignment metrics.
3. **Contrastive over differential** for activation direction
   extraction in interpretability work.
4. **The "interpretability ≠ steering" distinction**: measurement-
   derived directions consistently underperform native concept vectors
   as steering targets, even when interpretability is excellent.

## 16. Limitations

- Single model (Gemma 3) — post-training-specific findings (refusal
  robustness, anxious-hard-to-produce) may not generalize.
- Single layer/hook (layer 40 post-MLP).
- 8-D PCA subspace may discard structure; we tested alternative
  embeddings (UMAP, diffusion maps) and adaptive-bandwidth KDE in
  the §7 ablations and found no improvement over PCA-8 for V/A
  isometry beyond +0.158 from diffusion-2 at the embedding level.
- n=40 chord, n=20 composition, n=8 tone — small in absolute terms.
- A_lift validated in-sample (same distribution).

## 17. Open questions

- **Multi-position steering**: apply the eval direction across the full
  prompt sequence rather than as a single residual-stream addition.
- **Contrastive training on behavior-class**: train the classifier to
  predict downstream behavior rather than condition label.
- **Larger pair samples (n=100+)** to tighten A_lift CI and study the
  predict-TIE bucket.
- **Replication on different model families**, especially less heavily-
  tuned models, to test which findings are Gemma-3-RLHF-specific.

## Figures

All figures are written to `results/figures/writeup/` unless noted.

| § | figure | file |
|---|---|---|
| TL;DR | Forest plot of all major findings (8-D + 4-D rows) | `forest_plot_findings_with_4d.png` |
| 3 | PC↔V/A scatter (PC1 vs V, PC2 vs A) | `pc_axes_vs_va.png` |
| 3 | All-pairs A_lift circumplex (with k-means cluster overlay) | `alift_atlas_kmeans.png` |
| 3 | A_lift confusion matrix (predict-WIN/LOSS/TIE × actual) | `alift_confusion_matrix.png` |
| 3 | Embedding isometry: PCA vs UMAP vs diffusion-maps | `embedding_isometry.png` |
| 3 | PCA loadings: PC alignment with V/A and top emotions per PC | `pc_loadings.png` |
| 4 | A_lift vs V_lift as predictors of pullback advantage | `vlift_predictor.png` |
| 4 | A_lift at n=96 (extension): margin + \|A_lift\|→off_gap | `alift_n100_extension.png` |
| 4 | Per-pair n=40 chord results (8-D and 4-D side-by-side) | `pair_n40_results.png`, `pair_n40_results_4d.png` |
| 4 | Per-waypoint V/A trajectory (4 anchor pairs) | `waypoint_trajectories_va.png` |
| 4 | Running-stats convergence (effect-size vs n) | `running_stats_trajectory.png` |
| 5 | Composition stratified per-pair scatter (raw vs NM) | `composition_n20_raw_vs_nm.png`, `composition_stratified_scatter.png` |
| 6 | Refusal/engagement bars | `refusal_engagement_bars.png` |
| 7 | Dimension ablation curve (4–32 dims) | `dim_ablation_writeup.png` |
| 7 | Denser dimension ablation (filling 6/10/12/14/24) | `dim_ablation_denser.png` |
| 7 | β ablation curve | `beta_ablation_writeup.png` |
| 7 | Adaptive vs fixed KDE bandwidth (PCA-8, n=800) | `adaptive_kde_geodesic.png` |
| 7 | Silverman vs clustered_NN behavioral comparison (n=40) | `silverman_vs_clustered_nn.png` |
| 7 | Pullback vs geodesic scatter | `pullback_vs_geodesic_scatter_writeup.png` |
| 7 | Path deviation distribution | `path_deviation_writeup.png` |
| 7 | Cumulative Bhattacharyya per pair | `bhattacharyya_per_pair.png` |
| 7 | Teleportation index per pair | `teleportation_index_per_pair.png` |
| 8 | Behavioral metrics across d∈{4,6,8} (n=40 each) | `dim_behavioral_compare.png` |
| 8 | 4-D vs 8-D per-pair outcome scatter | `outcome_4d_vs_8d_scatter.png` |
| 8 | 4-D vs 8-D LINEAR positive control (n=40) | `linear_4d_vs_8d_positive_control.png` |
| 8 | 2-D vs 8-D LINEAR control (d=2 falsification, n=40) | `linear_2d_vs_8d.png` |
| 8 | Subspace sweep figures | `subspace_sweep_writeup.png` |
| 9 | TV vs matched segmented-constant control, K=8 (n=40) | `tv8_vs_cv8_n40.png` |
| 9 | TV vs matched segmented-constant control, K=16 (n=40) | `tv16_vs_cv16_n40.png` |
| 10 | Geometric-vs-behavioral dissociation single panel | `geom_vs_behavioral_dissociation.png` |
| 11 | Eval-direction top/bottom emotions | `eval_direction_top_bottom_emotions.png` |
| 11 | Steering dose-response | `eval_steering_dose_response.png` |
| 11 | Framing cosine heatmap | `framing_cosine_heatmap.png` |
