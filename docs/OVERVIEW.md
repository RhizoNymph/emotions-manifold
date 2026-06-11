```yaml
Overview:
  description: |
    Research codebase combining Anthropic's emotion-vectors paper
    (171 emotion concepts as diff-in-means activation directions) with
    Goodfire's manifold-steering paper (geodesic-aware activation steering).
    Goal: fit an activation manifold M_h and behavior manifold M_y over
    emotion concepts; test isometry M_h ↔ M_y; compare manifold-aware
    steering (geodesic, pullback) against linear steering as baseline.
  subsystems:
    capture: vLLM forks (Gemma 3) emit residual-stream activations to disk via a
             registered capture consumer. Used for emotion-vector extraction.
    vectors: Diff-in-means construction over per-emotion stories produces
             a (171, hidden) tensor of activation directions.
    manifold_h: PCA-projected centroids + density-aware KDE Riemannian
                metric (G_E) over the activation subspace.
    behavior_y: Anthropic-style Claude judge rates each story for
                (valence, arousal) on [-5, +5], yielding M_y centroids.
    geodesic: JAX-based shortest-path solver under G_E.
    pullback: kernel-weighted barycenter inverse — given an M_y target
              point y*, return an M_h subspace point as Σ w_i(y*) h_i
              with w_i ∝ Gaussian on ||y_i − y*||/σ.
    steering: applies path-of-vectors as residual-stream additive
              perturbations during vLLM generation; judges continuations.
    dashboard: Plotly Dash 3D viewer for paths in M_h subspace and M_y.
  data_flow: |
    vLLM (captures) → emotion_vectors.npz → manifold_h.npz
                                          ↘
                                            geodesics_cache.npz
                                          ↗
    Anthropic API ratings → manifold_y.npz
    {manifold_h, manifold_y} → pullback paths → steering trajectories
                            → vLLM continuations → judge ratings
                            → off-M_y E + M_y-line distance metrics
Features Index:
  emotion_vectors:
    description: Diff-in-means activation directions over 171 emotions
    entry_points: [scripts/compute_vectors.py]
    depends_on: [capture]
    doc: docs/features/emotion_vectors.md
  manifold_h:
    description: PCA + KDE-bandwidth Riemannian metric over M_h centroids
    entry_points: [scripts/fit_manifold.py]
    depends_on: [emotion_vectors]
    doc: docs/features/manifold_h.md
  behavior_manifold:
    description: Claude-judge valence/arousal ratings of per-emotion stories
    entry_points: [scripts/build_behavior_manifold.py]
    depends_on: []
    doc: docs/features/behavior_manifold.md
  pullback:
    description: Kernel-barycenter inverse of an M_y straight line into M_h subspace; supports constant or adaptive K-NN-distance σ
    entry_points: [scripts/experiments/run_chord.py, scripts/run_pullback_sigma_sweep.py, scripts/plotting/dashboard.py]
    depends_on: [manifold_h, behavior_manifold]
    doc: docs/features/pullback.md
  steering_experiment:
    description: Compare pullback vs M_h geodesic vs M_h linear under generation+judge metrics (off-M_y E, M_y-line distance)
    entry_points: [scripts/experiments/run_chord.py, scripts/orchestration/run_chain.py]
    depends_on: [pullback, manifold_h, behavior_manifold]
    doc: docs/features/steering_experiment.md
  isometry:
    description: Pairwise-distance correlation between M_h subspace and M_y over common labels
    entry_points: [scripts/analysis/check_isometry.py]
    depends_on: [manifold_h, behavior_manifold]
    doc: docs/features/isometry.md
  manifold_alternatives:
    description: Alternative embeddings (UMAP, diffusion maps) and adaptive-bandwidth KDE for the activation manifold; geometric and geodesic isometry comparisons
    entry_points: [scripts/analysis/embedding_isometry.py, scripts/analysis/manifold_alternatives.py]
    depends_on: [emotion_vectors, behavior_manifold]
    doc: docs/features/manifold_alternatives.md
  dimension_ablation:
    description: G_E geodesic isometry edge as a function of PCA subspace dimensionality (4/8/16/32 + denser 6/10/12/14/24)
    entry_points: [scripts/run_dimension_ablation.py (archive branch), scripts/run_denser_dim_sweep.py (archive branch)]
    depends_on: [manifold_h, behavior_manifold]
    doc: docs/features/dimension_ablation.md
  d6_behavioral:
    description: Behavioral re-run of n=40 chord at d=6 (where the denser dim sweep showed peak G_E edge +0.085), parallel to d=4 and d=8 setups
    entry_points: [scripts/fit_manifold.py --dim 6 --geodesics, scripts/experiments/run_chord.py --config experiments/chord_6d.yaml, scripts/orchestration/run_chain.py --config experiments/chord_6d.yaml, scripts/analysis/analyze_chord.py --results-dir results/pullback_6d]
    depends_on: [manifold_h, behavior_manifold]
    doc: docs/features/d6_behavioral.md
  time_varying_steering:
    description: Stepping through waypoints during generation (K=8 segments × 12 tokens) via vLLM /v1/completions with manual Gemma chat template — Goodfire's central distinguishing claim
    entry_points: [scripts/run_time_varying_steering.py (archive branch), scripts/analysis/analyze_time_varying.py]
    depends_on: [manifold_h, behavior_manifold]
    doc: docs/features/time_varying_steering.md
  positive_controls:
    description: 4-D linear vs 8-D linear positive control disentangling "curved metric helps" from "geodesic compensates for dim loss"
    entry_points: [scripts/analysis/analyze_4d_linear_vs_8d_linear.py]
    depends_on: []
    doc: docs/features/positive_controls.md
  dashboard:
    description: 3D plotly Dash viewer for M_h paths and M_y projections; interactive σ slider for pullback
    entry_points: [scripts/plotting/dashboard.py]
    depends_on: [pullback, manifold_h, behavior_manifold]
    doc: docs/features/dashboard.md
```

Also see [`docs/roadmap.md`](roadmap.md) for the experimental goals and Goodfire/Anthropic source pointers.

Daily journals at `results/day*.md` track day-by-day findings and pending experiments.

## Code layout

```
src/manifold_emotions/        package: all shared logic
  analysis/stats.py           bootstrap CIs, paired-gap reports
  experiments/chord.py        chord experiment (ChordRunConfig, run_chord_pair)
  experiments/chain.py        multi-host chain runner with resume
  manifold/geodesic_cache.py  all-pairs geodesic cache build/save/load
experiments/                  per-variant YAML configs (chord_{8d,4d,6d,8d_silverman}.yaml)
experiments/pairs/            tracked pair lists (alift_n40.json)
scripts/                      pipeline stages (generate/extract/compute/fit/build/precompute)
scripts/experiments/          run_chord.py — single chord entry point, --config selects variant
scripts/orchestration/        run_chain.py (multi-host, resume), run_chain_batched_judge.py (phase 2)
scripts/analysis/             analyze_* — parameterized by --results-dir where variant-generic
scripts/plotting/             plot_*, dashboard.py, make_writeup_figures.py
```

Historical one-off scripts (per-dimension copies, day-specific run logs,
shell chains) are preserved verbatim on the `archive/disorganized-scripts`
branch; entries above marked "(archive branch)" live only there.
