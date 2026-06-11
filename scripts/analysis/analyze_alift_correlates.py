"""Cross-correlate A_lift with pre-computed pair_alignment metrics.

Asks: does A_lift correlate with any geometric pair-property that
we already have for all 14,535 pairs? If yes, A_lift is potentially
predictable from cheaper geometric measurements; if no, it captures
information not in the existing alignment metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config


def main():
    cfg = load_config()
    beh = BehaviorManifold.load(cfg.paths.manifold_y)
    labels = list(beh.labels)
    label_idx = {l: i for i, l in enumerate(labels)}

    # Load A_lift data
    arr = np.load("results/alift_all_pairs/all_pairs.npz")
    a_lift = arr["a_lift"]
    v_lift = arr["v_lift"]
    chord_len = arr["chord_len"]
    mid_V = arr["midpoint_V"]
    mid_A = arr["midpoint_A"]
    i_idx = arr["i"]
    j_idx = arr["j"]

    # Build pair index (i,j) -> A_lift row index
    pair_to_row = {}
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        pair_to_row[(int(ii), int(jj))] = k

    # Load pair_alignment.json
    align = json.load(open("results/pair_alignment.json"))
    print(f"Loaded {len(align)} pair_alignment records")

    # Build aligned arrays for each metric
    metrics_to_check = [
        "h_distance", "v_diff", "a_diff",
        "participation_ratio", "top_pc_fraction",
        "near_chord_centroid_count",
        "ge_geodesic_length", "ge_linear_length", "ge_length_gap",
        "max_chord_deflection",
    ]
    metric_arrays = {m: [] for m in metrics_to_check}
    aligned_a_lift = []
    aligned_v_lift = []
    aligned_chord = []
    aligned_mid_V = []
    aligned_mid_A = []
    matched = 0
    for rec in align:
        if rec["start"] not in label_idx or rec["end"] not in label_idx:
            continue
        i = label_idx[rec["start"]]
        j = label_idx[rec["end"]]
        if i > j:
            i, j = j, i
        if (i, j) not in pair_to_row:
            continue
        k = pair_to_row[(i, j)]
        for m in metrics_to_check:
            if rec.get(m) is None:
                metric_arrays[m].append(np.nan)
            else:
                metric_arrays[m].append(float(rec[m]))
        aligned_a_lift.append(a_lift[k])
        aligned_v_lift.append(v_lift[k])
        aligned_chord.append(chord_len[k])
        aligned_mid_V.append(mid_V[k])
        aligned_mid_A.append(mid_A[k])
        matched += 1
    print(f"  Matched {matched} pairs across both sources")

    aligned_a_lift = np.array(aligned_a_lift)
    aligned_v_lift = np.array(aligned_v_lift)
    aligned_chord = np.array(aligned_chord)
    aligned_mid_V = np.array(aligned_mid_V)
    aligned_mid_A = np.array(aligned_mid_A)

    print()
    print("==== Correlations: A_lift vs pair_alignment metrics ====")
    print(f"  {'metric':>30s}  {'Pearson r':>10s}  {'Spearman r':>10s}  {'N':>6s}")
    results = {}
    for m, vals in metric_arrays.items():
        vals = np.array(vals)
        finite = np.isfinite(vals) & np.isfinite(aligned_a_lift)
        if finite.sum() < 50:
            print(f"  {m:>30s}  -- insufficient data --")
            continue
        p_r, p_p = stats.pearsonr(vals[finite], aligned_a_lift[finite])
        s_r, s_p = stats.spearmanr(vals[finite], aligned_a_lift[finite])
        print(f"  {m:>30s}  {p_r:>+10.3f}  {s_r:>+10.3f}  {finite.sum():>6d}")
        results[m] = {
            "pearson_r": float(p_r), "pearson_p": float(p_p),
            "spearman_r": float(s_r), "spearman_p": float(s_p),
            "n": int(finite.sum()),
        }

    # Also: A_lift vs simple V/A features
    print()
    print("==== Correlations: A_lift vs midpoint V/A and chord len ====")
    print(f"  {'feature':>30s}  {'Pearson r':>10s}  {'Spearman r':>10s}")
    simple_features = {
        "midpoint_V": aligned_mid_V,
        "midpoint_A": aligned_mid_A,
        "chord_length": aligned_chord,
        "abs(midpoint_V - 3.5)": np.abs(aligned_mid_V - 3.5),
        "abs(midpoint_A - 4.5)": np.abs(aligned_mid_A - 4.5),
    }
    for name, vals in simple_features.items():
        finite = np.isfinite(vals) & np.isfinite(aligned_a_lift)
        if finite.sum() < 50:
            continue
        p_r, _ = stats.pearsonr(vals[finite], aligned_a_lift[finite])
        s_r, _ = stats.spearmanr(vals[finite], aligned_a_lift[finite])
        print(f"  {name:>30s}  {p_r:>+10.3f}  {s_r:>+10.3f}")
        results[name] = {"pearson_r": float(p_r), "spearman_r": float(s_r)}

    out_dir = Path("results/alift_all_pairs")
    (out_dir / "correlates.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_dir/'correlates.json'}")


if __name__ == "__main__":
    main()
