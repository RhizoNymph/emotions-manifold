# Pullback construction

Kernel-weighted barycenter inverse of an M_y straight line back into the
M_h subspace. The conceptual goal: given two emotion centroids and a
straight line between them in the 2-D valence/arousal behavior space
(M_y), construct an activation-space (M_h) trajectory that *would*
produce that behavior trajectory if used as a steering path.

This implements Goodfire's pullback / inverse construction (their §3.3)
on the Anthropic emotion corpus.

## Scope

- Compute the pullback path in subspace + full residual-stream
  coordinates.
- Compute the two baseline paths (M_h geodesic, M_h linear).
- Compute per-waypoint subspace distances pullback↔geodesic,
  pullback↔linear, geodesic↔linear.
- Support both constant σ and adaptive per-waypoint σ (K-NN in M_y).

## Non-scope

- Trajectory generation and judging (lives in `steering/experiment.py`
  and `steering/pullback_experiment.py`).
- Geodesic fitting (lives in `manifold/geodesic.py`).

## Construction

For each waypoint k along the M_y chord at parameter t ∈ [0, 1]:

1. Compute the M_y waypoint y*_k = (1−t) · y_start + t · y_end.
2. Compute the per-anchor weight on every M_y centroid y_i:
   `w_i(y*_k) = softmax(-||y_i − y*_k||² / (2 σ(y*_k)²))`
3. Lift to M_h subspace: `h*_k = Σ_i w_i · h_i_subspace`.

Endpoints are snapped to the exact start/end centroids so all three
paths share endpoints (lengths comparable under G_E).

## σ (kernel bandwidth) spec

Two modes, selected by the `sigma` argument:

- **Constant σ** (float, or `None` → median NN distance in M_y).
  All waypoints share the same σ. Suitable when M_y density is roughly
  uniform along the chord.
- **Adaptive σ** (string `"knn:K"`): at each waypoint y*_k, σ(y*_k) =
  distance from y*_k to its K-th nearest M_y centroid. Sparse regions
  get wide kernels; dense regions get narrow ones.

Why adaptive matters at 171-scale: the 4-pair pullback comparison
showed that a single global σ cannot work for all pairs at once:
- σ ≈ median NN (0.077) is right for dense pairs (excited↔weary,
  depressed↔energized) but too narrow for terrified→serene (the chord
  passes through a sparse part of M_y).
- σ = 1.0 (the 9-point sweep optimum on a single dense pair) rescues
  terrified→serene but collapses the dense pairs to global-mean
  behavior.

K-NN-distance σ from each y* itself adapts the kernel width to local
M_y density along the chord. See `results/day3.md` for the 4-pair
numbers.

## Files

- `src/manifold_emotions/manifold/pullback.py` — `construct_pullback_path`,
  `compute_pullback`, `_resolve_sigma`, `PullbackResult` dataclass.
- `src/manifold_emotions/steering/pullback_experiment.py` — wraps the
  geometric pullback with generation + judging and computes
  off-M_y E and M_y-line distance per trajectory.
- `scripts/experiments/run_chord.py` — CLI for a single pair or
  default 4-pair sweep; accepts `--sigma <float>` or `--sigma knn:K`.
- `scripts/experiments/run_pullback_sigma_sweep.py` — sweeps a list of σ values on
  a single pair, reusing the same kernel-barycenter machinery.
- `scripts/plotting/dashboard.py` — interactive σ slider on the 3D viewer; only
  constant σ for now.
- `tests/test_pullback.py` — unit tests for kernel weights, median NN,
  K-NN distance, endpoint snapping, hull membership, and adaptive σ
  asymmetry.

## Key exports

```python
from manifold_emotions.manifold.pullback import (
    PullbackResult,         # dataclass: all three paths + distances + σ trace
    SigmaSpec,              # float | str | None
    construct_pullback_path, # → (my_path, pullback_sub, σ_mean, σ_per_waypoint, σ_spec)
    compute_pullback,       # → PullbackResult
)
```

## Invariants

- Pullback endpoints are snapped to `manifold.centroids_subspace[start_idx]`
  and `[end_idx]` so all three paths share endpoints.
- σ values are positive floats. Adaptive σ is clipped to ensure K ∈ [1, N].
- `sigma_per_waypoint.shape == (num_waypoints,)`.
- The constant-σ behavior is unchanged from the pre-adaptive version
  (regression tests cover this — see `test_pullback_path_starts_and_ends_at_centroids`).
- For an aligned circumplex (synthetic ring in both spaces),
  pullback↔geodesic distance < pullback↔linear distance under default σ
  (`test_pullback_on_ring_lands_near_geodesic_not_linear`).

## Provenance for experimental outputs

- σ=0.077 (171-default median NN) 4-pair: `results/pullback/<pair>.json`
- σ=1.0 4-pair: `results/pullback/<pair>_sigma1000.json`
- σ=knn:K 4-pair: `results/pullback/<pair>_knn{K}.json`
- 30-scale Day 2 baseline: `data/30emotions/pullback/<pair>.json`
- Per-pair σ-sweep tables: `results/pullback_sigma/<pair>_sigma{xxxx}.json`
- Each JSON now includes `sigma_spec` and `sigma_per_waypoint` so we
  can reconstruct the kernel used at every waypoint after the fact.
