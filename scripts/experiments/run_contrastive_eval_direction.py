"""Contrastive eval direction — alternative to differential probe.

The differential probe (run_eval_awareness_probe.py) computed
Δh = h_eval - h_natural averaged across pairs. That captures the
average activation *shift* but weights every coordinate equally
regardless of whether it actually discriminates eval-noticing
internal state from baseline.

This script instead trains a linear classifier on captured activations
at the last prompt token, with:
  - positive class: the 4 eval framings (explicit, roleplay, subtle, meta)
  - negative class: natural + neutral_tag

The classifier's weight vector gives the direction along which the
two classes are most separable — automatically downweighting noise
dimensions where the two distributions overlap.

Outputs:
  - direction vector saved as .npy (full residual + subspace projection)
  - comparison to the differential direction (cosine, top emotions)
  - classifier accuracy on held-out fold (sanity check)

Then run_eval_awareness_steering.py can be pointed at this new
direction to test whether contrastive geometry gives a cleaner
behavioral steering signal than differential.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold

sys.path.insert(0, str(Path(__file__).parent))
from run_eval_awareness_probe import (  # noqa: E402
    _load_with_fallback,
    TASKS,
)
from manifold_emotions.extraction.capture import _slug  # noqa: E402

log = structlog.get_logger(__name__)

POS_FRAMINGS = ("eval_explicit", "eval_roleplay", "eval_subtle", "eval_meta")
NEG_FRAMINGS = ("natural", "neutral_tag")


def _load_framing(config, framing: str) -> list[tuple[int, np.ndarray]]:
    """Load all available captures for one framing as (task_idx, last_pos_act) pairs."""
    capture_root = config.capture.root
    layer = config.model.target_layer
    hook = config.model.hook_point
    pairs = []
    for task_idx in range(len(TASKS)):
        tag = f"eval_probe_{framing}"
        req_id = f"{tag}_probe_{task_idx:04d}"
        path = capture_root / _slug(tag) / _slug(req_id) / f"{layer}_{hook}.bin"
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            acts = _load_with_fallback(path)
        except Exception as e:
            log.warn("contrastive.load_failure", path=str(path), error=str(e))
            continue
        # Last prompt token
        pairs.append((task_idx, acts[-1].astype(np.float32)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results/eval_awareness_contrastive")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--C", type=float, default=1.0,
                        help="L2 regularization inverse strength")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all captures, labeled
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    framing_parts: list[str] = []
    for framing in POS_FRAMINGS + NEG_FRAMINGS:
        pairs = _load_framing(config, framing)
        if not pairs:
            log.warn("contrastive.no_captures", framing=framing)
            continue
        X_f = np.stack([a for _, a in pairs], axis=0)
        y_f = np.ones(X_f.shape[0]) if framing in POS_FRAMINGS else np.zeros(X_f.shape[0])
        X_parts.append(X_f)
        y_parts.append(y_f)
        framing_parts.extend([framing] * X_f.shape[0])
        log.info("contrastive.loaded", framing=framing, n=X_f.shape[0])

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    print()
    print(f"==== Contrastive eval direction ====")
    print(f"  total samples: {X.shape[0]}  hidden_size: {X.shape[1]}")
    print(f"  positive (eval framings): {int(y.sum())}")
    print(f"  negative (natural + neutral_tag): {int((1 - y).sum())}")

    # K-fold cross-validation: report mean accuracy as sanity
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_accs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(
            C=args.C, class_weight="balanced", max_iter=2000, solver="lbfgs",
        )
        clf.fit(X[train_idx], y[train_idx])
        acc = clf.score(X[test_idx], y[test_idx])
        fold_accs.append(acc)
        log.info("contrastive.fold", fold=fold_idx, acc=acc)
    print(f"  classifier accuracy ({args.n_folds}-fold CV): "
          f"{np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")

    # Now fit on the FULL data for the final direction
    clf_full = LogisticRegression(
        C=args.C, class_weight="balanced", max_iter=2000, solver="lbfgs",
    )
    clf_full.fit(X, y)
    direction_full = clf_full.coef_[0].astype(np.float32)
    full_train_acc = clf_full.score(X, y)
    print(f"  full-data fit train accuracy: {full_train_acc:.3f}")
    print(f"  ||direction (full)||: {np.linalg.norm(direction_full):.3f}")

    # Project into subspace
    manifold = FittedManifold.load(config.paths.manifold_h)
    direction_sub = manifold.project(direction_full[None, :])[0]
    print(f"  ||direction (subspace)||: {np.linalg.norm(direction_sub):.3f}")

    # Compare to differential direction (eval_vs_neutral_mean from v2)
    diff_full = np.load("results/eval_awareness_v2/full_eval_vs_neutral_mean.npy")
    diff_sub = np.load("results/eval_awareness_v2/sub_eval_vs_neutral_mean.npy")

    def cos(a, b):
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))

    cos_full = cos(direction_full, diff_full)
    cos_sub = cos(direction_sub, diff_sub)
    print()
    print(f"==== Contrastive vs differential ====")
    print(f"  cos(contrastive, differential) full:     {cos_full:+.3f}")
    print(f"  cos(contrastive, differential) subspace: {cos_sub:+.3f}")
    print(f"  ||diff (full)||:     {np.linalg.norm(diff_full):.3f}")
    print(f"  ||diff (subspace)||: {np.linalg.norm(diff_sub):.3f}")

    # Top emotions aligned with contrastive direction
    emotion_labels = list(manifold.labels)
    centroids_sub = manifold.centroids_subspace.astype(np.float32)

    def cosines(v):
        norms = np.linalg.norm(centroids_sub, axis=1) * np.linalg.norm(v)
        return (centroids_sub @ v) / np.where(norms > 0, norms, 1.0)

    cos_em = cosines(direction_sub)
    order = np.argsort(cos_em)[::-1]
    top_emotions = [(emotion_labels[i], float(cos_em[i])) for i in order[:args.top_k]]
    bottom_emotions = [(emotion_labels[i], float(cos_em[i]))
                       for i in np.argsort(cos_em)[:args.top_k]]

    print()
    print(f"==== TOP {args.top_k} emotions aligned with contrastive direction ====")
    for em, c in top_emotions:
        print(f"  {c:+.3f}  {em}")
    print()
    print(f"==== BOTTOM {args.top_k} ====")
    for em, c in bottom_emotions:
        print(f"  {c:+.3f}  {em}")

    # Save artifacts
    np.save(out_dir / "full_contrastive_eval.npy", direction_full)
    np.save(out_dir / "sub_contrastive_eval.npy", direction_sub)
    summary = {
        "C": args.C,
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "cv_accuracy_mean": float(np.mean(fold_accs)),
        "cv_accuracy_std": float(np.std(fold_accs)),
        "full_train_accuracy": float(full_train_acc),
        "norm_full": float(np.linalg.norm(direction_full)),
        "norm_subspace": float(np.linalg.norm(direction_sub)),
        "cos_vs_differential_full": cos_full,
        "cos_vs_differential_subspace": cos_sub,
        "top_emotions": top_emotions,
        "bottom_emotions": bottom_emotions,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"saved {out_dir/'_summary.json'}")


if __name__ == "__main__":
    main()
