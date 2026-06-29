from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs_images_pooled"
INPUT_CSV = OUTPUT_DIR / "pcp2_normlog_cereb_purkinje_spots_per_cell_all_cells.csv"
OUTPUT_DATAPOINTS_CSV = OUTPUT_DIR / "pcp2_cereb_slice_replicate_region_averages.csv"
OUTPUT_STATS_CSV = OUTPUT_DIR / "pcp2_cereb_slice_replicate_region_averages_ttests.csv"
OUTPUT_PNG = OUTPUT_DIR / "pcp2_cereb_slice_replicate_region_averages_barplot.png"

CONDITION_ORDER = ["b2", "b3_5nM", "b3_10nM", "b3_20nM"]
CONDITION_COLORS = {
    "b2": "#4C78A8",
    "b3_5nM": "#E45756",
    "b3_10nM": "#F58518",
    "b3_20nM": "#54A24B",
}


def build_slice_replicate_df(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["condition", "sample", "concentration", "biological_slice_id", "rep"], as_index=False)
        .agg(
            mean_pcp2_spots_per_cell=("pcp2_spot_count", "mean"),
            n_cells=("pcp2_spot_count", "size"),
            n_fovs=("slice_id", lambda s: int(pd.Series(s.astype(str)).nunique())),
            fov_ids=("slice_id", lambda s: ",".join(sorted(pd.Series(s.astype(str)).unique()))),
        )
        .sort_values(
            by=["condition", "biological_slice_id", "rep"],
            key=lambda s: s.map({c: i for i, c in enumerate(CONDITION_ORDER)}) if s.name == "condition" else s,
        )
        .reset_index(drop=True)
    )
    grouped["slice_replicate_label"] = grouped.apply(
        lambda row: f"slice{row['biological_slice_id']} rep{row['rep']}",
        axis=1,
    )
    grouped["analysis_definition"] = (
        "average PCP2 spots per Purkinje cell, averaged across all cells from cerebellar FOVs "
        "within each biological slice and replicate"
    )
    return grouped


def pairwise_ttests(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for i, cond_a in enumerate(CONDITION_ORDER):
        vals_a = df.loc[df["condition"] == cond_a, "mean_pcp2_spots_per_cell"].to_numpy(dtype=float)
        for cond_b in CONDITION_ORDER[i + 1 :]:
            vals_b = df.loc[df["condition"] == cond_b, "mean_pcp2_spots_per_cell"].to_numpy(dtype=float)
            test = stats.ttest_ind(vals_a, vals_b, equal_var=True, nan_policy="omit")
            p_value = float(test.pvalue)
            rows.append(
                {
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "n_a": int(len(vals_a)),
                    "n_b": int(len(vals_b)),
                    "mean_a": float(np.mean(vals_a)) if len(vals_a) else np.nan,
                    "mean_b": float(np.mean(vals_b)) if len(vals_b) else np.nan,
                    "student_ttest_pvalue": p_value,
                    "significance_label": (
                        "****"
                        if p_value < 1e-4
                        else "***"
                        if p_value < 1e-3
                        else "**"
                        if p_value < 1e-2
                        else "*"
                        if p_value < 5e-2
                        else "ns"
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_region_averages(df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.3))
    x = np.arange(len(CONDITION_ORDER))
    means = []
    rng = np.random.default_rng(73)
    for cond in CONDITION_ORDER:
        vals = df.loc[df["condition"] == cond, "mean_pcp2_spots_per_cell"].to_numpy(dtype=float)
        means.append(float(np.mean(vals)) if len(vals) else np.nan)
    ax.bar(x, means, color=[CONDITION_COLORS[c] for c in CONDITION_ORDER], width=0.62, alpha=0.82)

    shuffled_df = df.sample(frac=1.0, random_state=73).reset_index(drop=True)
    for i, cond in enumerate(CONDITION_ORDER):
        sub = shuffled_df.loc[shuffled_df["condition"] == cond]
        vals = sub["mean_pcp2_spots_per_cell"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.1, 0.1, size=len(sub))
        ax.scatter(np.full(len(sub), x[i]) + jitter, vals, color="black", s=28, alpha=0.75, zorder=3)
        if np.isfinite(means[i]):
            ax.text(x[i], means[i] + max(0.3, 0.03 * max(means)), f"{means[i]:.2f}", ha="center", va="bottom", fontsize=9)

    ymax = max(float(df["mean_pcp2_spots_per_cell"].max()), 1.0)
    axis_top = ymax * 1.55
    line_h = max(axis_top * 0.015, 0.3)
    top_margin = max(axis_top * 0.03, 0.8)
    available = max(axis_top - ymax - top_margin - 2 * line_h, line_h * len(stats_df))
    step = max(available / max(len(stats_df), 1), line_h * 0.8)
    total_span = step * max(len(stats_df) - 1, 0) + line_h
    base_y = max(ymax + top_margin, axis_top - total_span - top_margin)
    for idx, row in enumerate(stats_df.itertuples(index=False)):
        i = CONDITION_ORDER.index(row.condition_a)
        j = CONDITION_ORDER.index(row.condition_b)
        y = base_y + idx * step
        ax.plot([x[i], x[i], x[j], x[j]], [y, y + line_h, y + line_h, y], color="black", linewidth=1.0)
        ax.text((x[i] + x[j]) / 2.0, y + line_h + max(0.08, 0.01 * ymax), row.significance_label, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x, CONDITION_ORDER)
    ax.set_ylabel("Average PCP2 spots per Purkinje cell")
    ax.set_title("Cerebellar PCP2 Spots per Slice/Replicate")
    ax.set_ylim(0, axis_top)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    region_df = build_slice_replicate_df(df)
    stats_df = pairwise_ttests(region_df)
    region_df.to_csv(OUTPUT_DATAPOINTS_CSV, index=False)
    stats_df.to_csv(OUTPUT_STATS_CSV, index=False)
    plot_region_averages(region_df, stats_df)


if __name__ == "__main__":
    main()
