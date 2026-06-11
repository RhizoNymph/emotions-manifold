"""Sample 60 new emotion pairs for the n=100+ extension, stratified by
A_lift quintile, excluding the existing 40 pairs from the n=40 chord set.

Writes:
- results/alift_n100_extension/sampled_pairs.json
- results/alift_n100_extension/sampled_pairs.txt  (one "label1 label2" per line)

Used by alift_n100_extension_chain.sh.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold


SEED = 13579  # different from the original A_lift sample to avoid overlap
N_PER_QUINTILE = 12  # 5 quintiles × 12 = 60 new pairs
OUT_DIR = Path("results/alift_n100_extension")

# 40 existing pairs from alift_4d_chain.sh (parsed manually)
EXISTING_PAIRS_RAW = """
happy sad
excited weary
depressed energized
terrified serene
hope unhappy
contemptuous hope
contemptuous playful
obstinate proud
disgusted pleased
indignant loving
proud puzzled
self-conscious thankful
smug unhappy
jubilant mad
energized terrified
anxious droopy
indifferent stressed
cheerful lazy
depressed vengeful
calm ecstatic
amused ashamed
grumpy hopeful
proud sympathetic
brooding proud
brooding pleased
afraid loving
loving restless
ecstatic puzzled
distressed self-confident
hope sensitive
at_ease obstinate
kind vulnerable
at_ease envious
at_ease disdainful
inspired irate
happy jealous
droopy uneasy
content excited
bored unnerved
overwhelmed sluggish
"""


def _normalize(label: str) -> str:
    return label.strip().replace("_", " ")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    beh = BehaviorManifold.load(Path("data/manifold_y.npz"))
    labels = list(beh.labels)

    existing: set[frozenset[str]] = set()
    for line in EXISTING_PAIRS_RAW.strip().splitlines():
        a, b = line.strip().split(maxsplit=1)
        # The chain script uses space-separated, where multi-word labels
        # are encoded with underscores. e.g. "at_ease obstinate".
        a = _normalize(a)
        b = _normalize(b)
        existing.add(frozenset([a, b]))
    print(f"existing pairs: {len(existing)}")

    arr = np.load("results/alift_all_pairs/all_pairs.npz")
    a_lift = arr["a_lift"]
    pair_i = arr["i"]
    pair_j = arr["j"]
    n_total = len(a_lift)
    print(f"total atlas pairs: {n_total}")

    # Exclude existing pairs
    candidate_mask = np.ones(n_total, dtype=bool)
    for k in range(n_total):
        a = labels[pair_i[k]]
        b = labels[pair_j[k]]
        if frozenset([a, b]) in existing:
            candidate_mask[k] = False
    n_avail = int(candidate_mask.sum())
    print(f"after excluding existing: {n_avail}")

    candidate_idx = np.where(candidate_mask)[0]
    a_lift_avail = a_lift[candidate_idx]

    # Stratified sampling by A_lift quintile
    quintile_edges = np.quantile(a_lift_avail, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"quintile edges: {quintile_edges.round(3)}")
    rng = np.random.default_rng(SEED)
    sampled = []
    for q in range(5):
        lo, hi = quintile_edges[q], quintile_edges[q + 1]
        if q == 4:
            bucket = candidate_idx[(a_lift_avail >= lo) & (a_lift_avail <= hi)]
        else:
            bucket = candidate_idx[(a_lift_avail >= lo) & (a_lift_avail < hi)]
        if len(bucket) < N_PER_QUINTILE:
            print(f"  quintile {q+1}: only {len(bucket)} candidates, taking all")
            chosen = bucket
        else:
            chosen = rng.choice(bucket, size=N_PER_QUINTILE, replace=False)
        for k in chosen:
            i, j = int(pair_i[k]), int(pair_j[k])
            sampled.append({
                "a_lift_quintile": q + 1,
                "label_i": labels[i],
                "label_j": labels[j],
                "a_lift": float(a_lift[k]),
                "atlas_idx": int(k),
            })
        print(f"  quintile {q+1} [{lo:+.3f}, {hi:+.3f}]: sampled {len(chosen)} pairs")

    out_json = {
        "seed": SEED,
        "n_sampled": len(sampled),
        "quintile_edges": quintile_edges.tolist(),
        "pairs": sampled,
    }
    (OUT_DIR / "sampled_pairs.json").write_text(json.dumps(out_json, indent=2))

    lines = []
    for s in sampled:
        a = s["label_i"].replace(" ", "_")
        b = s["label_j"].replace(" ", "_")
        lines.append(f"{a} {b}")
    (OUT_DIR / "sampled_pairs.txt").write_text("\n".join(lines) + "\n")

    print(f"\nsaved {OUT_DIR/'sampled_pairs.json'}")
    print(f"saved {OUT_DIR/'sampled_pairs.txt'} ({len(lines)} pairs)")
    print(f"\nfirst 5 sampled pairs:")
    for s in sampled[:5]:
        print(f"  q{s['a_lift_quintile']}  {s['label_i']:>20s} → {s['label_j']:<20s}  A_lift={s['a_lift']:+.3f}")


if __name__ == "__main__":
    main()
