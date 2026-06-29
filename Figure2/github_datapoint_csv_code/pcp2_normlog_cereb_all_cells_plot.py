from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import images_pooled_barseq_analysis as pooled
import pcp2_slice_replicate_ratio_normlog_plot as normlog
import replicate_barseq_analysis as base


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs_images_pooled"
CONDITION_ORDER = ["b2", "b3_5nM", "b3_10nM", "b3_20nM"]
CONDITION_COLORS = {
    "b2": "#4C78A8",
    "b3_5nM": "#E45756",
    "b3_10nM": "#F58518",
    "b3_20nM": "#54A24B",
}


def collect_per_cell_df() -> pd.DataFrame:
    manifest, _ = pooled.parse_manifest()
    scale_df = normlog.build_cereb_normalization_scales(manifest)
    scale_map = {
        (row.condition, row.sample, row.concentration, str(row.biological_slice_id), str(row.rep)): float(row.normalization_max_q99_99)
        for row in scale_df.itertuples(index=False)
    }

    keys = sorted(
        {key for key, channel in manifest if key.region == "cereb" and channel == "pcp2"},
        key=lambda k: (CONDITION_ORDER.index(k.condition), int(k.slice_id), int(k.rep)),
    )
    rows: list[dict] = []
    for key in keys:
        scale_hi = scale_map[normlog.slice_group_key(key)]
        dapi = pooled.load_image(manifest, key, "dapi")
        pcp2_raw = pooled.load_image_raw(manifest, key, "pcp2")
        seg_spots, pcp2_bs = normlog.blob_log_spots_normalized(
            pcp2_raw,
            scale_hi=scale_hi,
            brightness_min=normlog.CEREB_BRIGHTNESS_MIN_NORM,
            overlap=normlog.CEREB_LOG_OVERLAP,
        )
        count_spots, _ = normlog.barseq_style_spots_raw16_count_cereb(pcp2_raw)
        labels, mask, cell_df = pooled.segment_pcp2_purkinje_fast(
            pcp2_bs=pcp2_bs,
            dapi=dapi,
            spots=seg_spots,
            sample=key.sample,
        )
        cell_df = normlog.filter_accepted_purkinje_cells(cell_df)
        if cell_df.empty or labels.max() <= 0:
            continue

        radius_multiplier = normlog.purkinje_territory_radius_multiplier_for_key(key)
        _, cell_counts = normlog.assign_spots_to_expanded_purkinje_territories(
            labels.shape,
            cell_df,
            count_spots,
            radius_multiplier=radius_multiplier,
        )

        cell_df = cell_df.copy()
        cell_df["pcp2_spot_count"] = cell_df["cell_id"].map(lambda x: int(cell_counts[int(x)]))
        for row in cell_df.itertuples(index=False):
            rows.append(
                {
                    "condition": key.condition,
                    "sample": key.sample,
                    "concentration": key.concentration,
                    "slice_id": key.slice_id,
                    "rep": key.rep,
                    "image_uid": pooled.image_uid(key),
                    "biological_slice_id": pooled.biological_slice_id(key.slice_id),
                    "cell_id": int(row.cell_id),
                    "area_px": float(row.area_px),
                    "disk_radius_px": float(row.disk_radius_px),
                    "pcp2_spot_count": int(row.pcp2_spot_count),
                    "pcp2_spot_calling_method": (
                        "BARseq-style raw16 bandpass + peak_local_max count caller for cerebellum "
                        "plus normalized blob_log seed caller and condition-specific expanded Purkinje territories"
                    ),
                    "cereb_seed_threshold_norm": float(normlog.CEREB_BRIGHTNESS_MIN_NORM),
                    "cereb_count_threshold_raw16": float(normlog.CEREB_COUNT_BRIGHTNESS_MIN_RAW16),
                    "purkinje_territory_radius_multiplier": float(radius_multiplier),
                    "pcp2_spot_radius_range_px": "4-6 diameter",
                }
            )
    return pd.DataFrame(rows)


def overall_anova_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    row: dict[str, float | int | str] = {"test": "one_way_anova"}
    for cond in CONDITION_ORDER:
        vals = df.loc[df["condition"] == cond, "pcp2_spot_count"].to_numpy(dtype=float)
        grouped.append(vals)
        row[f"n_{cond}"] = int(len(vals))
        row[f"mean_{cond}"] = float(np.mean(vals)) if len(vals) else np.nan
    result = stats.f_oneway(*grouped)
    p_value = float(result.pvalue)
    if p_value < 1e-4:
        label = "****"
    elif p_value < 1e-3:
        label = "***"
    elif p_value < 1e-2:
        label = "**"
    elif p_value < 5e-2:
        label = "*"
    else:
        label = "ns"
    row["f_statistic"] = float(result.statistic)
    row["anova_pvalue"] = p_value
    row["significance_label"] = label
    return pd.DataFrame([row])


def plot_all_cells(df: pd.DataFrame, stats_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    x = np.arange(len(CONDITION_ORDER))
    means = []
    rng = np.random.default_rng(17)
    for cond in CONDITION_ORDER:
        vals = df.loc[df["condition"] == cond, "pcp2_spot_count"].to_numpy(dtype=float)
        means.append(float(np.mean(vals)) if len(vals) else np.nan)

    ax.bar(x, means, color=[CONDITION_COLORS[c] for c in CONDITION_ORDER], width=0.62, alpha=0.82)
    shuffled_df = df.sample(frac=1.0, random_state=17).reset_index(drop=True)
    for i, cond in enumerate(CONDITION_ORDER):
        vals = shuffled_df.loc[shuffled_df["condition"] == cond, "pcp2_spot_count"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(np.full(len(vals), x[i]) + jitter, vals, color="black", s=10, alpha=0.35, zorder=3)
        ax.text(x[i], means[i] + max(0.5, 0.03 * max(means)), f"{means[i]:.2f}", ha="center", va="bottom", fontsize=9)

    ymax = max(float(df["pcp2_spot_count"].max()), 1.0)
    base_y = ymax * 1.05
    line_h = max(ymax * 0.02, 0.3)
    row = stats_df.iloc[0]
    ax.plot([x[0], x[0], x[-1], x[-1]], [base_y, base_y + line_h, base_y + line_h, base_y], color="black", linewidth=1.0)
    ax.text(
        (x[0] + x[-1]) / 2.0,
        base_y + line_h + max(0.1, 0.01 * ymax),
        f"ANOVA {row['significance_label']} (p={row['anova_pvalue']:.2e})",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_xticks(x, CONDITION_ORDER)
    ax.set_ylabel("PCP2 spots per Purkinje cell")
    ax.set_title("PCP2 Spots Per Purkinje Cell Across All Cells")
    ax.set_ylim(0, base_y + 4 * line_h)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    per_cell_df = collect_per_cell_df()
    stats_df = overall_anova_stats(per_cell_df)
    per_cell_df.to_csv(OUTPUT_DIR / "pcp2_normlog_cereb_purkinje_spots_per_cell_all_cells.csv", index=False)
    stats_df.to_csv(OUTPUT_DIR / "pcp2_normlog_cereb_purkinje_spots_per_cell_all_cells_anova_stats.csv", index=False)
    stats_df.to_csv(OUTPUT_DIR / "pcp2_normlog_cereb_purkinje_spots_per_cell_all_cells_pairwise_stats.csv", index=False)
    plot_all_cells(
        per_cell_df,
        stats_df,
        OUTPUT_DIR / "pcp2_normlog_cereb_purkinje_spots_per_cell_all_cells_barplot.png",
    )


if __name__ == "__main__":
    main()
