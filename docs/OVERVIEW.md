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
    entry_points: [scripts/run_pullback_experiment.py, scripts/run_pullback_sigma_sweep.py, scripts/dashboard.py]
    depends_on: [manifold_h, behavior_manifold]
    doc: docs/features/pullback.md
  steering_experiment:
    description: Compare pullback vs M_h geodesic vs M_h linear under generation+judge metrics (off-M_y E, M_y-line distance)
    entry_points: [scripts/run_pullback_experiment.py, scripts/run_steering_experiment.py]
    depends_on: [pullback, manifold_h, behavior_manifold]
    doc: docs/features/steering_experiment.md
  isometry:
    description: Pairwise-distance correlation between M_h subspace and M_y over common labels
    entry_points: [scripts/check_isometry.py]
    depends_on: [manifold_h, behavior_manifold]
    doc: docs/features/isometry.md
  dashboard:
    description: 3D plotly Dash viewer for M_h paths and M_y projections; interactive σ slider for pullback
    entry_points: [scripts/dashboard.py]
    depends_on: [pullback, manifold_h, behavior_manifold]
    doc: docs/features/dashboard.md
```

Also see [`docs/roadmap.md`](roadmap.md) for the experimental goals and Goodfire/Anthropic source pointers.

Daily journals at `results/day*.md` track day-by-day findings and pending experiments.
