"""Coherence-judge the bijective-spline chord completions (day 10 follow-up).

The judged n=40 bijective run showed the spline beats linear at target-tracking
(+0.085/+0.091 M_y-line margin); the un-selected inverse-steering validation
showed target gains there come with a small coherence cost (-0.032
coherent-fraction). This script asks the symmetric question: does the spline's
target win cost coherence too?

Reuses the composition-experiment A/B/C coherence judge verbatim (same prompt,
same scale as every other coherence number in the project) over the saved
completions in data/pullback_spline_bijective_8d/, and reports per-pair
coherent-fraction gaps (method - linear) with bootstrap CIs and two-sided
Wilcoxon, matching the n=40 validation's analysis.

Run from the repo root (needs data/, results/, .env):
    uv run python scripts/analysis/judge_bijective_coherence.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from run_composition_experiment import judge_coherence  # noqa: E402

from manifold_emotions.analysis.stats import bootstrap_ci
from manifold_emotions.config import load_config

COMPLETIONS_DIR = Path("data/pullback_spline_bijective_8d")
CACHE_DIR = Path("results/pullback_spline_bijective_8d_coherence")
OUT_PATH = Path("results/spline_analysis_bijective_8d/coherence_summary.json")
METHODS = ("spline_induced", "spline_density", "linear")
PAIR_CONCURRENCY = 4  # x config.judge.concurrency in-flight requests


def _coherent_frac(labels: list[str]) -> float:
    if not labels:
        return float("nan")
    return sum(1 for x in labels if x == "coherent") / len(labels)


async def main() -> None:
    config = load_config()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(COMPLETIONS_DIR.glob("completions_*.json"))
    if not files:
        raise SystemExit(f"no completions under {COMPLETIONS_DIR}")

    sem = asyncio.Semaphore(PAIR_CONCURRENCY)

    async def judge_pair(path: Path) -> dict:
        entries = json.loads(path.read_text())
        slug = path.stem.removeprefix("completions_")
        passages = [(e["text_id"], e["text"]) for e in entries]
        async with sem:
            ratings = await judge_coherence(
                config, passages, cache_path=CACHE_DIR / f"{slug}.json")
        by_method: dict[str, list[str]] = {m: [] for m in METHODS}
        for e in entries:
            if e["method"] in by_method and e["text_id"] in ratings:
                by_method[e["method"]].append(ratings[e["text_id"]].label)
        row = {"pair": slug}
        for m in METHODS:
            row[f"{m}_coherent_frac"] = _coherent_frac(by_method[m])
            row[f"{m}_n"] = len(by_method[m])
        print(f"  {slug}: " + "  ".join(
            f"{m}={row[f'{m}_coherent_frac']:.3f}" for m in METHODS))
        return row

    rows = await asyncio.gather(*(judge_pair(f) for f in files))
    rows = sorted(rows, key=lambda r: r["pair"])

    summary: dict = {"n_pairs": len(rows), "per_pair": rows}
    lin = np.array([r["linear_coherent_frac"] for r in rows])
    print(f"\n=== bijective-spline coherence (n={len(rows)} pairs) ===")
    print(f"  linear coherent-frac mean: {np.nanmean(lin):.3f}")
    for m in ("spline_induced", "spline_density"):
        v = np.array([r[f"{m}_coherent_frac"] for r in rows])
        gap = v - lin
        ok = np.isfinite(gap)
        lo, hi = bootstrap_ci(gap[ok], np.mean)
        w = stats.wilcoxon(gap[ok])
        summary[m] = {
            "coherent_frac_mean": float(np.nanmean(v)),
            "gap_vs_linear_mean": float(gap[ok].mean()),
            "gap_ci": [float(lo), float(hi)],
            "wilcoxon_p_two_sided": float(w.pvalue),
        }
        print(f"  {m}: coherent-frac {np.nanmean(v):.3f}  "
              f"gap vs linear {gap[ok].mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"p={w.pvalue:.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
