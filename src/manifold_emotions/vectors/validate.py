"""Phase 3 validation: does the extracted emotion-vector space recapitulate
human-like emotional structure?

Per Anthropic's protocol, the gate checks are:
1. PCA over the (num_emotions, hidden_size) vector matrix yields a
   leading axis that correlates strongly with LLM-judged valence
   (Anthropic reports r=0.81 against PAD norms in Sonnet 4.5; we
   accept anything above ~0.5 as evidence the geometry is meaningful).
2. A secondary axis correlates with arousal (Anthropic reports r=0.66).
3. Together these two axes recover the affective circumplex layout.

We compute these correlations and return a structured report. Logit
lens and max-activating examples are separate validators (TODO).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from .judge import EmotionRating
from .pca import PCAResult, align_axis_signs, fit_pca

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AxisReport:
    """Correlation between one PCA axis and one judged dimension."""

    axis_index: int
    dimension_name: str  # "valence" or "arousal"
    pearson_r: float
    explained_variance_ratio: float


@dataclass(frozen=True, slots=True)
class ValidationReport:
    pca: PCAResult
    # Mapping from dimension ("valence", "arousal") to the best-matching axis.
    best_axis: dict[str, AxisReport]
    # All axis × dimension correlations for inspection.
    all_correlations: dict[str, list[float]]
    # The numeric scores in the same order as the row labels (for downstream use).
    valence: np.ndarray
    arousal: np.ndarray
    # The emotion labels in matched order with the input vectors.
    labels: tuple[str, ...]


def _best_axis(
    pca: PCAResult,
    scores: np.ndarray,
    dimension_name: str,
    candidate_axes: range,
) -> AxisReport:
    """Pick the axis with the strongest |Pearson r| against scores."""
    best: AxisReport | None = None
    for axis in candidate_axes:
        corr = float(np.corrcoef(pca.projections[:, axis], scores)[0, 1])
        if not np.isfinite(corr):
            continue
        if best is None or abs(corr) > abs(best.pearson_r):
            best = AxisReport(
                axis_index=axis,
                dimension_name=dimension_name,
                pearson_r=corr,
                explained_variance_ratio=float(pca.explained_variance_ratio[axis]),
            )
    if best is None:
        raise ValueError(f"no finite correlation found for {dimension_name}")
    return best


def validate_emotion_vectors(
    vectors: np.ndarray,
    labels: tuple[str, ...],
    ratings: dict[str, EmotionRating],
    *,
    candidate_axis_count: int = 5,
) -> ValidationReport:
    """Run PCA on the emotion vectors and report which axes match valence/arousal.

    ``vectors[i]`` corresponds to ``labels[i]``. ``ratings`` must contain
    every label as a key. ``candidate_axis_count`` limits which axes
    are considered for the valence/arousal best-match (we look only at
    the top few since they carry most of the variance).
    """
    missing = [label for label in labels if label not in ratings]
    if missing:
        raise ValueError(
            f"missing judge ratings for {len(missing)} labels (first 5: {missing[:5]})"
        )

    valence = np.array([ratings[label].valence for label in labels], dtype=np.float64)
    arousal = np.array([ratings[label].arousal for label in labels], dtype=np.float64)

    n_components = min(candidate_axis_count, vectors.shape[0], vectors.shape[1])
    pca = fit_pca(vectors, n_components=n_components)
    # Align signs so PC1 correlates positively with valence by convention
    # — purely cosmetic, makes plots readable.
    pca = align_axis_signs(pca, valence, axis_indices=(0,))
    pca = align_axis_signs(pca, arousal, axis_indices=(1, 2))

    candidate_axes = range(n_components)
    best_valence = _best_axis(pca, valence, "valence", candidate_axes)
    best_arousal_axes = range(n_components)
    # If valence already grabbed the best axis, arousal can't reuse it.
    best_arousal = _best_axis(
        pca,
        arousal,
        "arousal",
        (a for a in best_arousal_axes if a != best_valence.axis_index),
    )

    all_corr_valence = [
        float(np.corrcoef(pca.projections[:, a], valence)[0, 1])
        for a in range(n_components)
    ]
    all_corr_arousal = [
        float(np.corrcoef(pca.projections[:, a], arousal)[0, 1])
        for a in range(n_components)
    ]

    log.info(
        "validate.pca",
        n_components=n_components,
        explained_variance_ratio=[
            round(r, 3) for r in pca.explained_variance_ratio.tolist()
        ],
        best_valence_axis=best_valence.axis_index,
        best_valence_r=round(best_valence.pearson_r, 3),
        best_arousal_axis=best_arousal.axis_index,
        best_arousal_r=round(best_arousal.pearson_r, 3),
    )

    return ValidationReport(
        pca=pca,
        best_axis={"valence": best_valence, "arousal": best_arousal},
        all_correlations={"valence": all_corr_valence, "arousal": all_corr_arousal},
        valence=valence,
        arousal=arousal,
        labels=labels,
    )
