"""PCA + axis-alignment tests on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from manifold_emotions.vectors.pca import align_axis_signs, fit_pca


def test_pca_recovers_axes_from_anisotropic_data() -> None:
    """A 2-D blob stretched along x should give PC1 ≈ x-axis."""
    rng = np.random.default_rng(seed=0)
    n = 200
    # Stretched 10× along x, 1× along y, embedded in 5-D.
    x = rng.normal(0, 10.0, size=(n, 1))
    y = rng.normal(0, 1.0, size=(n, 1))
    other = rng.normal(0, 0.1, size=(n, 3))
    data = np.hstack([x, y, other])

    pca = fit_pca(data, n_components=3)
    # PC1 should be aligned with the first axis (up to sign).
    assert abs(pca.components[0, 0]) > 0.95
    # And PC1 should explain most of the variance.
    assert pca.explained_variance_ratio[0] > 0.95


def test_pca_projections_recover_signal() -> None:
    """If we put a strong linear signal along PC1, its projections should track it."""
    rng = np.random.default_rng(seed=1)
    n = 50
    signal = np.linspace(-1, 1, n)
    direction = np.array([1.0, 0.0, 0.0])
    noise = rng.normal(0, 0.01, size=(n, 3))
    data = signal[:, None] * direction[None, :] * 10.0 + noise

    pca = fit_pca(data, n_components=3)
    corr = abs(np.corrcoef(pca.projections[:, 0], signal)[0, 1])
    assert corr > 0.99


def test_align_axis_signs_flips_when_correlation_negative() -> None:
    rng = np.random.default_rng(seed=2)
    n = 30
    signal = np.linspace(-1, 1, n)
    # Build data where PC1 will be anti-correlated with signal by sign convention.
    data = -signal[:, None] * np.array([[1.0, 0.0]]) * 5.0 + rng.normal(
        0, 0.05, size=(n, 2)
    )
    pca = fit_pca(data, n_components=2)
    initial_corr = np.corrcoef(pca.projections[:, 0], signal)[0, 1]

    aligned = align_axis_signs(pca, signal, axis_indices=(0,))
    aligned_corr = np.corrcoef(aligned.projections[:, 0], signal)[0, 1]

    # The flip should make the correlation non-negative.
    assert aligned_corr >= 0
    # And in fact close to |initial_corr|.
    assert aligned_corr == pytest.approx(abs(initial_corr), abs=1e-9)


def test_pca_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        fit_pca(np.zeros(5), n_components=2)
    with pytest.raises(ValueError):
        fit_pca(np.zeros((3, 5)), n_components=10)
