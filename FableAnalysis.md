# Project Assessment — manifold-emotions

*Written 2026-06-09 by Claude (Fable 5), after reviewing writeup.md, results/outline.md,
results/follow-up-plan.md, all day journals (results/day1–6.md), the results/ inventory,
and the code structure.*

## TL;DR

The science is in better shape than the writeup currently sells — the geometric–behavioral
dissociation and the compensation-for-underfitting reframe are a genuinely strong
negative-space result. The two things to absolutely do before publishing: scale the
time-varying experiment past n=12 (with a smoothed-TV variant to break the discontinuity
confound) and resolve the M_y coverage question. Everything else is polish and hygiene.

---

## 1. How to characterize the outcomes

A coherent, publishable story — stronger than a typical null-replication because the
dissociations are clean. Three tiers:

### Tier 1 — robust findings that should headline

- **The geometric–behavioral dissociation.** The curved metric is geometrically real
  (isometry edge +0.049 at 8-D over 14,535 pairs, peaking at +0.085 at 6-D), but the
  geometric peak does *not* translate behaviorally — d=4 and d=6 give the same ~+0.018
  behavioral on-manifold gain despite a 35% difference in geometric edge. This is the
  cleanest expression of the project's thesis and is currently buried in §6.5.
- **The compensation reframe.** The 4-D geodesic edge (p=0.002) recovers only ~22% of
  what 4-D linear loses vs 8-D linear (p<0.0001). "The curved metric helps when you've
  under-fit the manifold" is the honest, memorable version of the claim, and the positive
  control establishing it is the project's strongest statistical result.
- **The time-varying flip.** Goodfire's central distinguishing claim — trajectory-following
  during generation — produces the opposite of the predicted effect (pullback 8/12 wins
  constant-vector → 1/12 wins time-varying). Most newsworthy single finding, but also the
  most fragile (see §2.1).
- **Interpretability ≠ steering.** The unified pattern across eval-awareness, refusal, and
  contrastive-vs-differential probes is the most transferable lesson and deserves the
  abstract placement the outline's framing notes already call for.

### Tier 2 — real but conditional

- A_lift as a one-sided positive-effect detector (n=96, r=+0.276, p=0.007;
  arousal-specific per the V_lift control).
- Pullback's on-manifold edge (p=0.051 at 8-D, p=0.023 at 6-D).
- Composition pathology surviving only as a structure-dependent claim at matched norms.
- Silverman bandwidth helping the geodesic only (p=0.001, geodesic off-M_y E).

### Tier 3 — frame as scoping, not findings

Refusal-gate decoupling, anxious-is-hard-for-Gemma, the σ-sweep non-generalization
(the outline's "cautionary tale" appendix idea is right).

### Framing notes

- `results/outline.md` §0 ("behaviorally inert") is stale — superseded by the 4-D/6-D
  geodesic significance results.
- writeup.md's TL;DR still cites the n=40 A_lift numbers (r=+0.384) even though the body
  says the n=96 numbers should replace them in the headline. **Fix: TL;DR should cite
  r=+0.276, n=96.**
- Suggested one-line thesis: *curvature routes "where you end up on the manifold," not
  "which V/A point you arrive at."*
- Position the work as adaptation/refinement/characterization-of-limits (per the outline's
  own framing notes), generous to both source papers.

---

## 2. Where the current results are vulnerable

In order of how much they threaten the story:

### 2.1 The time-varying result is n=12

The claim "one of the project's stronger negative findings" rests on 12 pairs, one lost to
a multi-word-label parser bug, with a design that confounds *time-variation* with
*segment-boundary discontinuity* (the writeup's own interpretation #1). Two cheap fixes
before publishing:

1. Fix the parser and extend to the full n=40 with the batched judge — cheap now, and the
   effect size (+0.128) is large enough that n=40 should be decisive.
2. Add a **smoothed-TV condition** (cross-fade between waypoint vectors instead of hard
   switches).
   - If smoothed-TV recovers pullback's advantage → "discontinuities hurt"
     (interpretation #1; per-token might actually work).
   - If it doesn't → "constant-vector already integrates the trajectory"
     (interpretation #2; Goodfire's temporal claim genuinely dead on this domain).

Currently the two interpretations can't be distinguished, and "per-token is unlikely to
recover" is speculation presented with more confidence than n=12 supports.

### 2.2 Mixed sequential/batched judging contaminates two key results

- Silverman comparison: n=4 sequential + n=36 batched.
- n=96 A_lift extension: 40 sequential + 56 batched.

Per-pair noise (~0.01–0.03) is quantified and the paired design absorbs it, but the
cleanest fix is to re-judge the sequential subsets through the batched pipeline so every
comparison is single-pipeline. At 50% batch cost this is the cheapest credibility
purchase available.

### 2.3 Multiple comparisons

Dozens of Wilcoxon tests; several headline numbers sit at p=0.02–0.05. The p=0.002
geodesic result and p<0.0001 positive control survive any correction; pullback's
p=0.051/0.023 results don't obviously. Recommended: a short statistical-honesty paragraph
designating primary vs exploratory endpoints rather than post-hoc correction — fits the
candid tone already established. Also: report the TV reversals as two-sided p or the
reversed one-sided p (~0.002–0.008) rather than "p=0.998 against the hypothesis."

### 2.4 A_lift extension sampling

The 60 new pairs were stratified by A_lift quintile, so the n=96 correlation is computed
over a non-random sample — stratification on the predictor inflates predictor variance
and can bias r either way. One sentence of acknowledgment, or compute r with
original-distribution weighting.

### 2.5 Journal findings missing from the writeup

- **happy→sad alone drives a +0.404 arousal-tracking win** (Day 3) — important
  "is the pullback story really one pair?" caveat.
- **Eval-framing specificity** (Day 4): explicit → "performance enthusiasm" vs subtle →
  "quiet competence", cos range 0.38–0.96 across framings — strengthens §7.
- **PC3–8 orthogonal semantic structure** (Day 6) — cognition/empathy/anger axes.

### 2.6 Is M_y complete?

Day 3 records the judge run dying at 28/171 emotions on credit exhaustion; the A_lift
atlas covers 10,584 pairs (~146 emotions), not 14,535 (171). If ~25 emotions still lack
behavior centroids, that silently shapes off-M_y E (fewer centroids → inflated distances)
and the atlas. Either finish judging the remainder with the batched pipeline (cheap now)
or state actual M_y coverage explicitly — the writeup currently reads as if all 171 are
in M_y.

**RESOLVED (2026-06-10, by direct inspection of `data/manifold_y.npz`):** M_y contains
all 171 emotions with 50 stories each (one, "scared", has 49). The interrupted Day 3 run
was evidently completed later. The atlas's 10,584 pairs are exactly 14,535 minus the
3,951 pairs whose V/A separation is < 1.0, which `analyze_alift_all_pairs.py` skips as
degenerate — not missing emotions. Off-M_y E was therefore computed against the full
171-centroid M_y all along. Writeup needs only one sentence: state full coverage and the
degenerate-pair filter.

---

## 3. What comes next — recommended order

1. **Harden, then ship Writeup 1.** Fix the TL;DR n=96 numbers, fold in the missing
   journal findings (§2.5), add the endpoints paragraph (§2.3), resolve M_y coverage
   (§2.6), and promote the dissociation + compensation reframe higher in the narrative.
   Figures are in excellent shape — all 32 referenced figures exist (plus 9 unreferenced;
   note `silverman_vs_clustered_nn.png` is cited in §6.5 text but missing from the
   figures table).
2. **TV at n=40 + smoothed-TV variant** (§2.1). The one place where modest compute spend
   materially changes what can be claimed; the TV flip is the finding most likely to draw
   scrutiny.
3. **A d=2 behavioral run as a falsification test of the compensation reframe.** The
   hypothesis makes a quantitative prediction: at d=2 linear under-fits even worse than
   at d=4, so the geodesic edge should be *larger*. Geometric machinery and n=40 harness
   already exist; one overnight chain converts the reframe from interpretation to tested
   prediction.
4. **Second model** (follow-up plan + writeup §12 both flag it). For the
   "interpretability ≠ steering" claim especially, a base or less-RLHF'd model is the
   highest-value generalization test. Writeup-2-scale work; don't block Writeup 1 on it.
5. **Diffusion maps: priority dropped.** The follow-up plan (written before the
   dissociation result) rated it highest-information. But diffusion-2 already beats PCA-2
   geometrically *and* geometric edges have been shown not to translate behaviorally —
   so expected information from diffusion-map *steering* is lower than the plan assumed.

---

## 4. Repo and code changes

- **Commit the untracked work.** ~40 experiment scripts, the batched judge module, and
  five feature docs that the writeup's reproducibility depends on sit untracked on top of
  a single initial commit. Per project conventions: `feat/` branches (or a few by area —
  e.g. `feat/batched-judge`, `feat/dim-ablation-experiments`) and PRs. The shell chains
  and orchestrators should be committed too — they're the actual provenance of cited
  results.
- **Refactor dimension-variant duplication before the robustness study.**
  `run_pullback_experiment_{4d,6d,8d_silverman}.py` differ by ~60 lines of constants;
  setup scripts and chain shells are 90%+ copies; `bootstrap_ci()` is defined four times
  across analysis scripts. Parameterize by `--dim`/`--bandwidth` flags and move
  `bootstrap_ci` into `src/manifold_emotions/`. This matters now because the follow-up
  plan multiplies variants (12-D, 16-D, diffusion, GMM) — copy-paste becomes unmanageable
  exactly when paired comparisons demand identical logic.
- **`judge_text_batched.py` is genuinely new code** (batch API, chunking, polling,
  custom_id mapping, resumable cache — not a duplicate of `judge_text.py`) and is
  load-bearing for results. Commit it and ideally add a test alongside the existing 10
  test files.
- **Rotate the Anthropic API key in `.env`.** Gitignored (good), but plaintext on disk
  and surfaced into tool transcripts during this review — cheap to rotate.
- **Docs:** OVERVIEW.md and feature docs are current. The one stale artifact is
  `results/outline.md`, which disagrees with writeup.md in several places (§0 framing,
  A_lift numbers, figure list) — update it or add a header marking it superseded so
  future drafting doesn't pull the wrong numbers.
