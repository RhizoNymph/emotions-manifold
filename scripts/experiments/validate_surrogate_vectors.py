"""GPU validation of the offline surrogate optimizer's promised vectors (Idea D).

surrogate_optimizer.py dumped, per top-headroom pair, the steering vectors it PREDICTS
will reach each V/A target waypoint much closer than linear. This script actually
generates from those vectors on the vLLM fork, judges the output, and answers the two
questions the offline screen can't:

  1. Real headroom: does the optimized vector reach the target closer than linear *in
     reality* (not just in surrogate prediction)?
  2. Surrogate fidelity off the linear region: does the achieved distance match what the
     surrogate predicted (opt_pred)? If reality is much worse, the optimizer exploited
     surrogate error and a closed loop needs the judge in the loop, not a fixed surrogate.

For each dumped pair it generates the optimized trajectory AND re-generates linear (so the
judge is matched), then reports per-pair mean distance to target for each, plus the
surrogate's predicted distance. Batched judging.

    uv run python scripts/experiments/validate_surrogate_vectors.py --host http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.judge_text_batched import judge_texts_batched
from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.experiments.chord import NEUTRAL_PROMPTS
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import compute_pullback
from manifold_emotions.steering.experiment import _aggregate_waypoint_behavior, _text_id
from manifold_emotions.steering.trajectory import generate_along_path


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vectors-dir", type=Path, default=Path("results/surrogate_optimizer/validation_vectors"))
    ap.add_argument("--manifold", type=Path, default=Path("data/manifold_h.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/surrogate_optimizer/validation_results.json"))
    ap.add_argument("--host", default=None, help="override vLLM base_url")
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--steering-scale", type=float, default=8.0)
    args = ap.parse_args()

    config = load_config()
    if args.host:
        config = dataclasses.replace(config, vllm_server=dataclasses.replace(config.vllm_server, base_url=args.host))
    manifold = FittedManifold.load(args.manifold)
    behavior = BehaviorManifold.load(config.paths.manifold_y)
    prompts = list(NEUTRAL_PROMPTS[: args.num_prompts])

    files = sorted(args.vectors_dir.glob("opt_*.npz"))
    if not files:
        raise SystemExit(f"no opt_*.npz in {args.vectors_dir} (run surrogate_optimizer.py first)")

    results = []
    for f in files:
        npz = np.load(f, allow_pickle=True)
        s, e = (str(x) for x in npz["pair"])
        opt_full = npz["opt_full"].astype(np.float32)  # (K, hidden), already x scale
        targets = npz["targets"]  # (K, 2)
        opt_pred = npz["opt_pred"]  # (K, 2) surrogate-predicted V/A
        k = opt_full.shape[0]

        geom = compute_pullback(manifold, behavior, s, e, num_waypoints=k, sigma=None)
        lin_steer = np.asarray(geom.linear_full, dtype=np.float32) * args.steering_scale

        opt_cont = await generate_along_path(config, opt_full, prompts, concurrency=args.concurrency)
        lin_cont = await generate_along_path(config, lin_steer, prompts, concurrency=args.concurrency)

        passages = [(f"opt_{s}_{e}_{_text_id(c)}", c.text) for c in opt_cont]
        passages += [(f"lin_{s}_{e}_{_text_id(c)}", c.text) for c in lin_cont]
        ratings = await judge_texts_batched(config, passages, cache_path=None)

        def idx(prefix: str) -> dict[str, tuple[float, float]]:
            p = f"{prefix}_{s}_{e}_"
            return {tid[len(p):]: (r.valence, r.arousal) for tid, r in ratings.items() if tid.startswith(p)}

        opt_mean, _ = _aggregate_waypoint_behavior(opt_cont, idx("opt"), k)
        lin_mean, _ = _aggregate_waypoint_behavior(lin_cont, idx("lin"), k)

        def mean_dist(beh: np.ndarray) -> float:
            m = np.all(np.isfinite(beh), axis=1)
            return float(np.linalg.norm(beh[m] - targets[m], axis=1).mean())

        opt_d, lin_d = mean_dist(opt_mean), mean_dist(lin_mean)
        pred_d = float(np.linalg.norm(opt_pred - targets, axis=1).mean())
        results.append({
            "pair": f"{s}->{e}",
            "optimized_actual_dist": opt_d,
            "linear_actual_dist": lin_d,
            "real_headroom": lin_d - opt_d,
            "surrogate_predicted_dist": pred_d,
            "surrogate_optimism": opt_d - pred_d,  # >0 => reality worse than promised
        })
        print(f"  {s}->{e}: optimized {opt_d:.3f}  linear {lin_d:.3f}  "
              f"real_headroom {lin_d-opt_d:+.3f}  (surrogate promised {pred_d:.3f})")

    rh = np.array([r["real_headroom"] for r in results])
    opt = np.array([r["surrogate_optimism"] for r in results])
    print(f"\n=== validation (n={len(results)} pairs) ===")
    print(f"  mean real headroom (linear - optimized): {rh.mean():+.3f}")
    print(f"  pairs where optimized beat linear:        {(rh > 0).sum()}/{len(rh)}")
    print(f"  mean surrogate optimism (actual - promised): {opt.mean():+.3f}  (high => surrogate exploited)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_pairs": len(results),
        "mean_real_headroom": float(rh.mean()),
        "pairs_optimized_beat_linear": int((rh > 0).sum()),
        "mean_surrogate_optimism": float(opt.mean()),
        "per_pair": results,
    }, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
