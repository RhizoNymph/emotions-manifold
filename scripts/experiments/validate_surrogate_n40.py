"""Un-selected (n=40) GPU validation of the surrogate optimizer's promised vectors.

``validate_surrogate_vectors.py`` validated only the 5 TOP-PREDICTED-HEADROOM pairs,
so its +1.45 mean real headroom is a best-case, selection-biased estimate. This script
validates ALL 40 chord pairs the optimizer dumped, giving an un-selected population
estimate, and adds a coherence judgement (same A/B/C judge as the composition
experiments) over both optimized and linear completions.

Two phases, split so the orchestrator can run generation and judging as separate
commands (generation is vLLM, judging is the Anthropic API):

  gen    Generate the optimized AND matched-linear trajectories for every pair and
         SAVE the completions to disk (the n=12 TV run lost its completions — never
         again). Pairs are sharded round-robin across ``--hosts`` (a single 3090 would
         take too long); each host runs on the shared event loop with its own base_url,
         mirroring scripts/orchestration/run_chain.py. Resumable: pairs whose
         completions file already exists are skipped (``--force`` re-runs).

  judge  Load the saved completions, judge V/A (batched Anthropic API) and coherence
         (same judge as run_composition_experiment.py), matched opt-vs-linear in the
         SAME batch per pair — exactly the 5-pair design. Writes per-pair real headroom,
         surrogate optimism, and coherence (opt vs linear) to validation_results_n40.json.

    # (a) generation across hosts, from this worktree
    uv run python scripts/experiments/validate_surrogate_n40.py gen \
        --hosts http://localhost:8000/v1,http://node1:8000/v1,http://node2:8000/v1

    # (b) V/A + coherence judging (Anthropic API)
    uv run python scripts/experiments/validate_surrogate_n40.py judge

Every network call is skipped under ``--dry-run`` (builds requests/payloads only).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import structlog

from manifold_emotions.behavior.judge_text_batched import judge_texts_batched
from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import Config, load_config
from manifold_emotions.experiments.chain import split_pairs
from manifold_emotions.experiments.chord import NEUTRAL_PROMPTS
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import compute_pullback
from manifold_emotions.steering.experiment import _aggregate_waypoint_behavior, _text_id
from manifold_emotions.steering.trajectory import (
    SteeredContinuation,
    _build_payload,
    generate_along_path,
)

# The coherence judge lives in the composition experiment script (A/B/C ->
# coherent/mixed/absent). Reuse it verbatim, exactly as run_tone_experiment.py does.
sys.path.insert(0, str(Path(__file__).parent))
from run_composition_experiment import judge_coherence  # noqa: E402

log = structlog.get_logger(__name__)

DEFAULT_VECTORS = Path("results/surrogate_optimizer/validation_vectors_n40")
DEFAULT_OUT = Path("results/surrogate_optimizer/n40")


def _slug(s: str, e: str) -> str:
    """Filesystem-safe pair identifier (labels may contain spaces, e.g. 'at ease')."""
    return f"{s}__{e}".replace(" ", "-")


def _load_pair_vectors(path: Path) -> dict:
    npz = np.load(path, allow_pickle=True)
    s, e = (str(x) for x in npz["pair"])
    return {
        "s": s,
        "e": e,
        "opt_full": npz["opt_full"].astype(np.float32),  # (K, hidden), already * scale
        "targets": np.asarray(npz["targets"], dtype=np.float64),  # (K, 2)
        "opt_pred": np.asarray(npz["opt_pred"], dtype=np.float64),  # (K, 2)
    }


def _serialize(conts: list[SteeredContinuation]) -> list[dict]:
    return [
        {
            "waypoint_index": c.waypoint_index,
            "prompt_index": c.prompt_index,
            "text": c.text,
            "finish_reason": c.finish_reason,
        }
        for c in conts
    ]


def _deserialize(rows: list[dict]) -> list[SteeredContinuation]:
    return [
        SteeredContinuation(
            waypoint_index=r["waypoint_index"],
            prompt_index=r["prompt_index"],
            text=r["text"],
            finish_reason=r.get("finish_reason"),
        )
        for r in rows
    ]


# --------------------------------------------------------------------------- gen


async def _gen_pair(
    config: Config,
    vec: dict,
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    prompts: list[str],
    steering_scale: float,
    concurrency: int,
    completions_dir: Path,
    dry_run: bool,
) -> None:
    s, e = vec["s"], vec["e"]
    opt_full = vec["opt_full"]
    k = opt_full.shape[0]
    geom = compute_pullback(manifold, behavior, s, e, num_waypoints=k, sigma=None)
    lin_steer = np.asarray(geom.linear_full, dtype=np.float32) * steering_scale

    if dry_run:
        # Build one representative payload for each arm to exercise the CPU path.
        _build_payload(config, prompts[0], opt_full[0], max_tokens=128)
        _build_payload(config, prompts[0], lin_steer[0], max_tokens=128)
        log.info(
            "surrogate_n40.gen.dry_run_pair",
            pair=f"{s}->{e}", host=config.vllm_server.base_url,
            k=k, num_prompts=len(prompts), generations=2 * k * len(prompts),
        )
        return

    opt_cont = await generate_along_path(config, opt_full, prompts, concurrency=concurrency)
    lin_cont = await generate_along_path(config, lin_steer, prompts, concurrency=concurrency)

    completions_dir.mkdir(parents=True, exist_ok=True)
    out = completions_dir / f"{_slug(s, e)}.json"
    out.write_text(json.dumps({
        "pair": [s, e],
        "k": k,
        "num_prompts": len(prompts),
        "steering_scale": steering_scale,
        "opt": _serialize(opt_cont),
        "lin": _serialize(lin_cont),
    }, indent=2))
    log.info("surrogate_n40.gen.pair_done", pair=f"{s}->{e}",
             host=config.vllm_server.base_url, opt=len(opt_cont), lin=len(lin_cont),
             saved=str(out))


async def run_gen(args: argparse.Namespace) -> None:
    config = load_config()
    manifold = FittedManifold.load(args.manifold)
    behavior = BehaviorManifold.load(config.paths.manifold_y)
    prompts = list(NEUTRAL_PROMPTS[: args.num_prompts])

    files = sorted(args.vectors_dir.glob("opt_*.npz"))
    if not files:
        raise SystemExit(
            f"no opt_*.npz in {args.vectors_dir} "
            "(run surrogate_optimizer.py --n-validate -1 first)")
    vecs = [_load_pair_vectors(f) for f in files]

    completions_dir = args.out_dir / "completions"
    if not args.force:
        pending = [v for v in vecs
                   if not (completions_dir / f"{_slug(v['s'], v['e'])}.json").exists()]
    else:
        pending = list(vecs)
    log.info("surrogate_n40.gen.start", total=len(vecs), pending=len(pending),
             skipped=len(vecs) - len(pending), hosts=args.hosts, dry_run=args.dry_run)

    hosts = args.hosts.split(",") if args.hosts else [config.vllm_server.base_url]
    shares = split_pairs(pending, len(hosts))

    async def worker(host: str, share: list[dict]) -> None:
        cfg = dataclasses.replace(
            config, vllm_server=dataclasses.replace(config.vllm_server, base_url=host))
        for vec in share:
            await _gen_pair(cfg, vec, manifold, behavior, prompts,
                            args.steering_scale, args.concurrency,
                            completions_dir, args.dry_run)

    async with asyncio.TaskGroup() as tg:
        for host, share in zip(hosts, shares, strict=True):
            tg.create_task(worker(host, share))

    if args.dry_run:
        total = sum(2 * v["opt_full"].shape[0] * len(prompts) for v in pending)
        k = vecs[0]["opt_full"].shape[0]
        print(f"[dry-run] would generate {total} completions across {len(hosts)} host(s) "
              f"for {len(pending)} pairs ({len(prompts)} prompts, K={k})")
    else:
        print(f"gen done: {len(pending)} pairs -> {completions_dir}/")


# ------------------------------------------------------------------------- judge


def _index_ratings(ratings: dict, prefix: str, s: str, e: str) -> dict[str, tuple[float, float]]:
    p = f"{prefix}_{s}_{e}_"
    return {tid[len(p):]: (r.valence, r.arousal) for tid, r in ratings.items() if tid.startswith(p)}


def _mean_dist(beh: np.ndarray, targets: np.ndarray) -> float:
    m = np.all(np.isfinite(beh), axis=1)
    return float(np.linalg.norm(beh[m] - targets[m], axis=1).mean())


def _coherent_frac(labels: list[str]) -> float:
    if not labels:
        return float("nan")
    return sum(1 for x in labels if x == "coherent") / len(labels)


async def run_judge(args: argparse.Namespace) -> None:
    config = load_config()
    completions_dir = args.out_dir / "completions"
    files = sorted(args.vectors_dir.glob("opt_*.npz"))
    if not files:
        raise SystemExit(f"no opt_*.npz in {args.vectors_dir}")

    va_dir = args.out_dir / "va_ratings"
    coh_dir = args.out_dir / "coherence_ratings"
    va_dir.mkdir(parents=True, exist_ok=True)
    coh_dir.mkdir(parents=True, exist_ok=True)

    # Pairs are judged concurrently: each pair is two Batches-API jobs (V/A,
    # then coherence), and batch queue latency dominates wall-clock, so a
    # sequential loop over 40 pairs costs ~a day where a concurrent one costs
    # ~one batch turnaround. Per-pair caches keep this resumable either way.
    sem = asyncio.Semaphore(args.judge_concurrency)

    async def _judge_pair(f: Path) -> dict | None:
        vec = _load_pair_vectors(f)
        s, e, targets, opt_pred = vec["s"], vec["e"], vec["targets"], vec["opt_pred"]
        k = vec["opt_full"].shape[0]
        cpath = completions_dir / f"{_slug(s, e)}.json"
        if not cpath.exists():
            log.warning("surrogate_n40.judge.no_completions", pair=f"{s}->{e}", path=str(cpath))
            return None
        blob = json.loads(cpath.read_text())
        opt_cont = _deserialize(blob["opt"])
        lin_cont = _deserialize(blob["lin"])

        passages = [(f"opt_{s}_{e}_{_text_id(c)}", c.text) for c in opt_cont]
        passages += [(f"lin_{s}_{e}_{_text_id(c)}", c.text) for c in lin_cont]

        if args.dry_run:
            log.info("surrogate_n40.judge.dry_run_pair", pair=f"{s}->{e}", passages=len(passages))
            return None

        async with sem:
            # V/A: matched opt+linear in the SAME batch, exactly as the 5-pair version.
            va = await judge_texts_batched(config, passages, cache_path=va_dir / f"{_slug(s, e)}.json")
            # Coherence: same A/B/C judge as the composition experiments, same passages.
            coh = await judge_coherence(config, passages, cache_path=coh_dir / f"{_slug(s, e)}.json")

        opt_mean, _ = _aggregate_waypoint_behavior(opt_cont, _index_ratings(va, "opt", s, e), k)
        lin_mean, _ = _aggregate_waypoint_behavior(lin_cont, _index_ratings(va, "lin", s, e), k)
        opt_d, lin_d = _mean_dist(opt_mean, targets), _mean_dist(lin_mean, targets)
        pred_d = float(np.linalg.norm(opt_pred - targets, axis=1).mean())

        opt_coh = [coh[tid].label for tid, _ in passages
                   if tid.startswith(f"opt_{s}_{e}_") and tid in coh]
        lin_coh = [coh[tid].label for tid, _ in passages
                   if tid.startswith(f"lin_{s}_{e}_") and tid in coh]
        opt_cf, lin_cf = _coherent_frac(opt_coh), _coherent_frac(lin_coh)

        row = {
            "pair": f"{s}->{e}",
            "optimized_actual_dist": opt_d,
            "linear_actual_dist": lin_d,
            "real_headroom": lin_d - opt_d,
            "surrogate_predicted_dist": pred_d,
            "surrogate_optimism": opt_d - pred_d,  # >0 => reality worse than promised
            "opt_coherent_frac": opt_cf,
            "lin_coherent_frac": lin_cf,
            "coherence_gap": opt_cf - lin_cf,  # >0 => optimized more coherent than linear
            "n_opt": len(opt_coh),
            "n_lin": len(lin_coh),
        }
        print(f"  {s}->{e}: opt {opt_d:.3f}  lin {lin_d:.3f}  headroom {lin_d-opt_d:+.3f}  "
              f"(promised {pred_d:.3f})  coh opt {opt_cf:.2f} lin {lin_cf:.2f}")
        return row

    gathered = await asyncio.gather(*(_judge_pair(f) for f in files))
    results = [r for r in gathered if r is not None]

    if args.dry_run:
        print(f"[dry-run] judge would score {len(files)} pairs "
              f"(opt+lin V/A + coherence per pair)")
        return

    rh = np.array([r["real_headroom"] for r in results])
    opt = np.array([r["surrogate_optimism"] for r in results])
    cg = np.array([r["coherence_gap"] for r in results])
    print(f"\n=== n=40 validation (n={len(results)} pairs) ===")
    print(f"  mean real headroom (linear - optimized): {rh.mean():+.3f}")
    print(f"  pairs where optimized beat linear:        {(rh > 0).sum()}/{len(rh)}")
    print(f"  mean surrogate optimism (actual - promised): {opt.mean():+.3f}")
    print(f"  mean coherence gap (opt - lin):           {np.nanmean(cg):+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_pairs": len(results),
        "mean_real_headroom": float(rh.mean()),
        "pairs_optimized_beat_linear": int((rh > 0).sum()),
        "mean_surrogate_optimism": float(opt.mean()),
        "mean_coherence_gap": float(np.nanmean(cg)),
        "per_pair": results,
    }, indent=2))
    print(f"\nsaved {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vectors-dir", type=Path, default=DEFAULT_VECTORS)
    ap.add_argument("--manifold", type=Path, default=Path("data/manifold_h.npz"))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="dir for saved completions and rating caches")
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--steering-scale", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="build requests/payloads without sending any network call")

    sub = ap.add_subparsers(dest="phase", required=True)

    g = sub.add_parser("gen", help="generate + save opt/linear completions across hosts")
    g.add_argument("--hosts", default=None,
                   help="comma-separated vLLM base URLs; pairs are sharded round-robin")
    g.add_argument("--concurrency", type=int, default=16)
    g.add_argument("--force", action="store_true",
                   help="re-generate pairs even if completions exist")
    g.set_defaults(func=run_gen)

    j = sub.add_parser("judge", help="judge V/A + coherence over saved completions")
    j.add_argument("--out", type=Path,
                   default=Path("results/surrogate_optimizer/validation_results_n40.json"))
    j.add_argument("--judge-concurrency", type=int, default=12,
                   help="pairs judged concurrently (each pair = 2 batch jobs)")
    j.set_defaults(func=run_judge)

    args = ap.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
