"""Append 171-scale pullback results to day3.md, with 30-vs-171 comparison.

Reads the newly-overwritten ``results/pullback/<pair>.json`` files and
the snapshotted 30-emotion versions at ``data/30emotions/pullback/``,
diffs the key metrics, and appends a comparison section to day3.md.

Run with:
    uv run python scripts/append_pullback_to_day3.py
"""

from __future__ import annotations

import json
from pathlib import Path

DAY3 = Path("results/day3.md")
PAIRS: tuple[tuple[str, str], ...] = (
    ("excited", "weary"),
    ("depressed", "energized"),
    ("happy", "sad"),
    ("terrified", "serene"),
)


def _flatten(row: dict) -> dict[str, float]:
    g = row["geometry"]
    t = row["trajectories"]
    return {
        "sigma": row.get("sigma", float("nan")),
        "pullback_to_geodesic": g.get("mean_dist_pullback_to_geodesic"),
        "pullback_to_linear": g.get("mean_dist_pullback_to_linear"),
        "ge_pullback": g.get("pullback_length"),
        "ge_geodesic": g.get("geodesic_length"),
        "ge_linear": g.get("linear_length"),
        "pull_off_my": t["pullback"]["off_manifold_energy"],
        "pull_my_line": t["pullback"]["my_geodesic_distance"],
        "geo_off_my": t["geodesic"]["off_manifold_energy"],
        "geo_my_line": t["geodesic"]["my_geodesic_distance"],
        "lin_off_my": t["linear"]["off_manifold_energy"],
        "lin_my_line": t["linear"]["my_geodesic_distance"],
    }


def main() -> None:
    sections: list[str] = ["## 171-scale pullback results", ""]

    summary_rows: list[str] = [
        "### 30 vs 171 comparison (key metrics)",
        "",
        "Each row reports the *change* going from 30-emotion to 171-emotion:",
        "  Δ pull_my_line = pullback M_y-line distance at 171 minus at 30",
        "  Δ margin = (linear M_y-line − pullback M_y-line) at 171 minus at 30",
        "  positive margin means pullback wins more at 171; negative means it wins less.",
        "",
        "| pair | σ@30 | σ@171 | Δ pull_my_line | Δ margin (pullback wins more if positive) | bidirectional shape (still closer to geodesic?) |",
        "|---|---:|---:|---:|---:|:---:|",
    ]

    for start, end in PAIRS:
        new_path = Path(f"results/pullback/{start}_{end}.json")
        old_path = Path(f"data/30emotions/pullback/{start}_{end}.json")
        if not new_path.exists():
            sections += [f"### {start} → {end}", "", "(missing 171-scale result)", ""]
            continue
        if not old_path.exists():
            sections += [f"### {start} → {end}", "", "(missing 30-scale backup)", ""]
            continue

        new = _flatten(json.loads(new_path.read_text()))
        old = _flatten(json.loads(old_path.read_text()))

        delta_my_line = new["pull_my_line"] - old["pull_my_line"]
        old_margin = old["lin_my_line"] - old["pull_my_line"]
        new_margin = new["lin_my_line"] - new["pull_my_line"]
        delta_margin = new_margin - old_margin

        closer_to_geo = (new["pullback_to_geodesic"] < new["pullback_to_linear"])

        summary_rows.append(
            f"| {start}→{end} | {old['sigma']:.3f} | {new['sigma']:.3f} | "
            f"{delta_my_line:+.3f} | {delta_margin:+.3f} | "
            f"{'✓' if closer_to_geo else '✗'} |"
        )

        sections += [
            f"### {start} → {end}",
            "",
            f"σ at 30: {old['sigma']:.3f}, σ at 171: {new['sigma']:.3f}",
            f"(σ should shrink at 171 because the median NN distance shrinks)",
            "",
            f"| metric | 30-emotion | 171-emotion | Δ |",
            "|---|---:|---:|---:|",
            f"| pullback ↔ geodesic | {old['pullback_to_geodesic']:.3f} | {new['pullback_to_geodesic']:.3f} | {new['pullback_to_geodesic'] - old['pullback_to_geodesic']:+.3f} |",
            f"| pullback ↔ linear   | {old['pullback_to_linear']:.3f} | {new['pullback_to_linear']:.3f} | {new['pullback_to_linear'] - old['pullback_to_linear']:+.3f} |",
            f"| G_E pullback length | {old['ge_pullback']:.2f} | {new['ge_pullback']:.2f} | {new['ge_pullback'] - old['ge_pullback']:+.2f} |",
            f"| G_E geodesic length | {old['ge_geodesic']:.2f} | {new['ge_geodesic']:.2f} | {new['ge_geodesic'] - old['ge_geodesic']:+.2f} |",
            f"| G_E linear length   | {old['ge_linear']:.2f} | {new['ge_linear']:.2f} | {new['ge_linear'] - old['ge_linear']:+.2f} |",
            f"| pullback off-M_y E  | {old['pull_off_my']:.3f} | {new['pull_off_my']:.3f} | {new['pull_off_my'] - old['pull_off_my']:+.3f} |",
            f"| pullback M_y-line   | {old['pull_my_line']:.3f} | {new['pull_my_line']:.3f} | {new['pull_my_line'] - old['pull_my_line']:+.3f} |",
            f"| linear M_y-line     | {old['lin_my_line']:.3f} | {new['lin_my_line']:.3f} | {new['lin_my_line'] - old['lin_my_line']:+.3f} |",
            f"| geodesic M_y-line   | {old['geo_my_line']:.3f} | {new['geo_my_line']:.3f} | {new['geo_my_line'] - old['geo_my_line']:+.3f} |",
            "",
            f"Margin (linear − pullback) on M_y-line: {old_margin:+.3f} at 30 → "
            f"{new_margin:+.3f} at 171 (change: {delta_margin:+.3f})",
            "",
        ]

    body = "\n".join(summary_rows + ["", "---", ""] + sections)

    text = DAY3.read_text() if DAY3.exists() else ""
    # Replace the placeholder if present, else append.
    placeholder = "(pending — appended by scripts/append_pullback_to_day3.py)"
    if placeholder in text:
        text = text.replace(placeholder, body)
    else:
        text += "\n\n" + body + "\n"
    DAY3.write_text(text)
    print(f"appended pullback comparison to {DAY3}")


if __name__ == "__main__":
    main()
