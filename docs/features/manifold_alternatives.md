# Manifold alternatives

Alternative embeddings (UMAP, diffusion maps) and adaptive-bandwidth
KDE constructions for the activation manifold, compared against the
production PCA-8 + clustered-NN-bandwidth setup on two axes:

1. **Geometric isometry**: chord-distance Pearson correlation with V/A
   distance across all 171×170/2 = 14,535 emotion pairs.
2. **Geodesic isometry**: G_E geodesic length Pearson correlation with
   V/A distance across an 800-pair sample. Tests whether a different
   density estimator changes the curved-metric edge.

## Scope

- Compute alternative embeddings of the 5376-D residual-stream
  centroids: PCA-{2,4,8,16,32}, UMAP-{2,4,8} × {n_neighbors=15,30},
  diffusion-maps-{2,4,8} (Coifman-Lafon, alpha=1.0).
- Compute geometric isometry per embedding.
- Implement adaptive (variable) bandwidth KDE: per-centroid bandwidth
  h_i = c · d_k(c_i) where d_k is the k-th nearest centroid distance.
- Compute G_E geodesic isometry under fixed (Silverman), fixed
  (clustered-NN, production default), adaptive (k=5), and adaptive
  (k=10) bandwidth.

## Non-scope

- Behavioral validation. The adaptive-bandwidth KDE is fit and
  geodesic-tested only on existing centroid data; running vLLM
  steering experiments under the alternative metric is left as a
  follow-up.

## Files

- `scripts/analysis/embedding_isometry.py` — fast geometric-isometry-only
  comparison; writes `results/manifold_alternatives/embedding_isometry.{json,png}`.
- `scripts/analysis/manifold_alternatives.py` — full comparison including
  geodesic isometry under adaptive vs fixed KDE bandwidth; writes
  `results/manifold_alternatives/adaptive_kde_geodesic.{json,png}`.

## Key results

- PCA-2 has the strongest *geometric* isometry to V/A (+0.845),
  beating PCA-8 (+0.710) and raw 5376-D activations (+0.655).
- Diffusion-maps-2 narrowly beats PCA-2 (+0.868 vs +0.845).
- UMAP underperforms PCA at every setting tested.
- Adaptive-bandwidth KDE does not change the geodesic isometry edge
  meaningfully vs the production fixed-bandwidth setup.

## Invariants

- Same 171-centroid common-label set as `check_isometry.py`.
- Same 800-pair sample (SEED=5678) as `run_dimension_ablation.py`
  so geodesic comparisons are matched.
- Adaptive KDE uses unnormalized kernel sum for energy (same
  semantics as `GaussianKDE` — see `manifold/density.py` docs).
