from __future__ import annotations

from pathlib import Path
import zlib

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import images_pooled_barseq_analysis as pooled
import pcp2_slice_replicate_ratio_normlog_plot as normlog
from pcp2_gaussian_condition_plots import plot_condition_bars, CONDITION_COLORS, CONDITION_ORDER


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs_images_pooled"
NORMLOG_RATIO_CSV = OUTPUT_DIR / "pcp2_slice_replicate_normlog_ratio_datapoints.csv"
NORMLOG_CEREB_FOV_CSV = OUTPUT_DIR / "pcp2_slice_replicate_normlog_cereb_per_fov.csv"
NORMLOG_BRAINSTEM_FOV_CSV = OUTPUT_DIR / "pcp2_slice_replicate_normlog_brainstem_per_fov.csv"


def overall_anova_stats(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grouped = []
    row: dict[str, float | int | str] = {"test": "one_way_anova"}
    for cond in CONDITION_ORDER:
        vals = (
            df.loc[df["condition"] == cond, value_col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
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


def plot_condition_bars_with_anova(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    out_png: Path,
    y_limit_max: float | None = None,
) -> pd.DataFrame:
    stats_df = overall_anova_stats(df, value_col)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    x = np.arange(len(CONDITION_ORDER))
    means = []
    grouped_vals: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(11)
    for cond in CONDITION_ORDER:
        vals = (
            df.loc[df["condition"] == cond, value_col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
        grouped_vals[cond] = vals
        means.append(float(vals.mean()) if len(vals) else np.nan)
    ax.bar(x, means, color=[CONDITION_COLORS[c] for c in CONDITION_ORDER], width=0.62, alpha=0.82)
    finite_means = [m for m in means if np.isfinite(m)]
    mean_ref = max(finite_means) if finite_means else 1.0
    for i, cond in enumerate(CONDITION_ORDER):
        vals = grouped_vals[cond]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), x[i]) + jitter, vals, color="black", s=22, alpha=0.6, zorder=3)
        if np.isfinite(means[i]):
            ax.text(x[i], means[i] + max(0.03, 0.04 * mean_ref), f"{means[i]:.2f}", ha="center", va="bottom", fontsize=9)
    ymax = max(np.nanmax(df[value_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)), 1.0)
    axis_top = float(y_limit_max) if y_limit_max is not None else ymax * 1.18
    line_h = max(axis_top * 0.015, 0.02 if axis_top <= 2 else 0.3)
    top_margin = max(axis_top * 0.03, 0.03 if axis_top <= 2 else 0.8)
    base_y = max(ymax + top_margin, axis_top - 3 * line_h - top_margin)
    row = stats_df.iloc[0]
    ax.plot([x[0], x[0], x[-1], x[-1]], [base_y, base_y + line_h, base_y + line_h, base_y], color="black", linewidth=1.0)
    ax.text(
        (x[0] + x[-1]) / 2.0,
        base_y + line_h + max(0.01, 0.02 * ymax),
        f"ANOVA {row['significance_label']} (p={row['anova_pvalue']:.2e})",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_ylim(0, axis_top)
    ax.set_xticks(x, CONDITION_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    return stats_df


def build_brainstem_same_size_region_df(cereb_fov_df: pd.DataFrame) -> pd.DataFrame:
    manifest, _ = pooled.parse_manifest()
    scale_df = normlog.build_cereb_normalization_scales(manifest)
    scale_map = {
        (row.condition, row.sample, row.concentration, str(row.biological_slice_id), str(row.rep)): float(row.normalization_max_q99_99)
        for row in scale_df.itertuples(index=False)
    }

    cereb_group = (
        cereb_fov_df.groupby(
            ["condition", "sample", "concentration", "biological_slice_id", "rep"],
            as_index=False,
        )
        .agg(
            avg_purkinje_cell_area_px=("avg_purkinje_cell_area_px", "mean"),
            avg_purkinje_cell_count=("n_purkinje_cells", "mean"),
            cereb_fov_ids=("fov_id", lambda x: ",".join(sorted({str(v) for v in x}))),
            source_slice_ids=("slice_id", lambda x: ",".join(sorted({str(v) for v in x}))),
        )
        .copy()
    )

    rows: list[dict] = []
    for row in cereb_group.itertuples(index=False):
        region_radius = float(np.sqrt(float(row.avg_purkinje_cell_area_px) / np.pi))
        sample_count = max(20, int(round(float(row.avg_purkinje_cell_count))))
        brain_keys = sorted(
            [
                key
                for key, channel in manifest
                if channel == "pcp2"
                and key.region == "brainstem"
                and key.condition == row.condition
                and key.sample == row.sample
                and key.concentration == row.concentration
                and pooled.biological_slice_id(key.slice_id) == str(row.biological_slice_id)
                and str(key.rep) == str(row.rep)
            ],
            key=lambda k: int(k.slice_id),
        )
        sampled_values: list[float] = []
        brainstem_fov_ids: list[str] = []
        brainstem_source_slice_ids: list[str] = []
        for key in brain_keys:
            raw = pooled.load_image_raw(manifest, key, "pcp2")
            scale_hi = scale_map[normlog.slice_group_key(key)]
            coords, _ = normlog.blob_log_spots_normalized(
                raw,
                scale_hi=scale_hi,
                brightness_min=normlog.BRAINSTEM_BRIGHTNESS_MIN_NORM,
                overlap=normlog.BRAINSTEM_LOG_OVERLAP,
            )
            coords = coords.astype(np.float32, copy=False)
            h, w = raw.shape
            margin = int(np.ceil(region_radius))
            if h <= 2 * margin or w <= 2 * margin:
                continue
            seed_label = f"{row.condition}_{row.biological_slice_id}_{row.rep}_{key.slice_id}"
            rng = np.random.default_rng(zlib.crc32(seed_label.encode("utf-8")))
            centers_r = rng.integers(margin, h - margin, size=sample_count)
            centers_c = rng.integers(margin, w - margin, size=sample_count)
            for cr, cc in zip(centers_r, centers_c):
                if len(coords):
                    d2 = (coords[:, 0] - float(cr)) ** 2 + (coords[:, 1] - float(cc)) ** 2
                    sampled_values.append(float(np.sum(d2 <= region_radius ** 2)))
                else:
                    sampled_values.append(0.0)
            brainstem_fov_ids.append(str(key.slice_id)[1])
            brainstem_source_slice_ids.append(str(key.slice_id))

        rows.append(
            {
                "condition": row.condition,
                "sample": row.sample,
                "concentration": row.concentration,
                "biological_slice_id": str(row.biological_slice_id),
                "rep": str(row.rep),
                "slice_replicate_label": f"slice{row.biological_slice_id} rep{row.rep}",
                "avg_purkinje_cell_area_px": float(row.avg_purkinje_cell_area_px),
                "avg_purkinje_cell_count": float(row.avg_purkinje_cell_count),
                "brainstem_fov_ids": ",".join(brainstem_fov_ids),
                "brainstem_source_slice_ids": ",".join(brainstem_source_slice_ids),
                "brainstem_region_radius_px": region_radius,
                "brainstem_regions_sampled_per_fov": sample_count,
                "value": float(np.mean(sampled_values)) if sampled_values else np.nan,
                "pcp2_spot_calling_method": (
                    "brainstem spots from normalized LoG caller; value is average PCP2 spots in sampled "
                    "brainstem regions matched to average Purkinje cell size"
                ),
                "brainstem_spot_threshold_min_intensity": float(normlog.BRAINSTEM_BRIGHTNESS_MIN_NORM),
                "pcp2_spot_radius_range_px": "4-6 diameter",
                "pcp2_normalization_method": (
                    "per biological slice+rep: min=0, max=q99.99 from cerebellum PCP2 images, "
                    "applied to cerebellum and brainstem PCP2 before LoG spot calling"
                ),
                "brainstem_denominator_definition": "sampled same-size brainstem regions matched to average Purkinje cell size",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ratio_df = pd.read_csv(NORMLOG_RATIO_CSV)
    cereb_fov_df = pd.read_csv(NORMLOG_CEREB_FOV_CSV)
    brainstem_fov_df = pd.read_csv(NORMLOG_BRAINSTEM_FOV_CSV)

    cereb_plot_df = ratio_df[
        [
            "condition",
            "sample",
            "concentration",
            "biological_slice_id",
            "rep",
            "slice_replicate_label",
            "avg_purkinje_spots_per_cell",
            "pcp2_spot_calling_method",
            "cereb_spot_threshold_min_intensity",
            "pcp2_spot_radius_range_px",
            "pcp2_normalization_method",
        ]
    ].copy()
    cereb_plot_df = cereb_plot_df.rename(columns={"avg_purkinje_spots_per_cell": "value"})
    cereb_stats = plot_condition_bars(
        cereb_plot_df,
        value_col="value",
        ylabel="Avg PCP2 spots per Purkinje cell",
        title="PCP2 Spots in Cerebellar Purkinje Cells (Normalized LoG)",
        out_png=OUTPUT_DIR / "pcp2_normlog_cereb_purkinje_spots_per_cell_barplot.png",
        y_limit_max=120.0,
    )
    cereb_plot_df.to_csv(OUTPUT_DIR / "pcp2_normlog_cereb_purkinje_spots_per_cell_datapoints.csv", index=False)
    cereb_stats.to_csv(OUTPUT_DIR / "pcp2_normlog_cereb_purkinje_spots_per_cell_pairwise_stats.csv", index=False)

    cereb_fov_df = cereb_fov_df.copy()
    cereb_fov_df["cereb_total_spots_per_purkinje_cell"] = (
        cereb_fov_df["cereb_spot_count_total"] / cereb_fov_df["n_purkinje_cells"]
    )
    cereb_group = (
        cereb_fov_df.groupby(
            ["condition", "sample", "concentration", "biological_slice_id", "rep"],
            as_index=False,
        )
        .agg(
            n_cereb_fovs=("fov_id", "nunique"),
            cereb_fov_ids=("fov_id", lambda x: ",".join(sorted({str(v) for v in x}))),
            source_slice_ids=("slice_id", lambda x: ",".join(sorted({str(v) for v in x}))),
            avg_cereb_total_spots_per_purkinje_cell=("cereb_total_spots_per_purkinje_cell", "mean"),
            avg_purkinje_cell_count=("n_purkinje_cells", "mean"),
            pcp2_spot_calling_method=("pcp2_spot_calling_method", "first"),
            cereb_spot_threshold_min_intensity=("pcp2_spot_threshold_min_intensity", "first"),
            pcp2_spot_radius_range_px=("pcp2_spot_radius_range_px", "first"),
            normalization_min=("normalization_min", "first"),
            normalization_max_q99_99=("normalization_max_q99_99", "mean"),
        )
        .copy()
    )
    brainstem_group = (
        brainstem_fov_df.groupby(
            ["condition", "sample", "concentration", "biological_slice_id", "rep"],
            as_index=False,
        )
        .agg(
            n_brainstem_fovs=("fov_id", "nunique"),
            brainstem_fov_ids=("fov_id", lambda x: ",".join(sorted({str(v) for v in x}))),
            brainstem_source_slice_ids=("slice_id", lambda x: ",".join(sorted({str(v) for v in x}))),
            avg_brainstem_spots_per_image=("brainstem_spot_count", "mean"),
            brainstem_spot_threshold_min_intensity=("pcp2_spot_threshold_min_intensity", "first"),
        )
        .copy()
    )
    ratio_plot_df = cereb_group.merge(
        brainstem_group,
        on=["condition", "sample", "concentration", "biological_slice_id", "rep"],
        how="left",
    )
    ratio_plot_df["slice_replicate_label"] = ratio_plot_df.apply(
        lambda row: f"slice{row['biological_slice_id']} rep{row['rep']}",
        axis=1,
    )
    ratio_plot_df["pcp2_normalization_method"] = (
        "per biological slice+rep: min=0, max=q99.99 from cerebellum PCP2 images, "
        "applied to cerebellum and brainstem PCP2 before LoG spot calling"
    )
    ratio_plot_df["value"] = (
        ratio_plot_df["avg_cereb_total_spots_per_purkinje_cell"]
        / ratio_plot_df["avg_brainstem_spots_per_image"]
    )
    ratio_stats = plot_condition_bars(
        ratio_plot_df,
        value_col="value",
        ylabel="Avg(total cereb FOV spots / Purkinje cells) / avg brainstem image spots",
        title="PCP2 Cerebellum FOV Spots-per-Cell Over Brainstem Image Spots (Normalized LoG)",
        out_png=OUTPUT_DIR / "pcp2_normlog_cereb_fovspots_per_cell_over_brainstem_image_spots_barplot.png",
    )
    ratio_plot_df.to_csv(
        OUTPUT_DIR / "pcp2_normlog_cereb_fovspots_per_cell_over_brainstem_image_spots_datapoints.csv",
        index=False,
    )
    ratio_stats.to_csv(
        OUTPUT_DIR / "pcp2_normlog_cereb_fovspots_per_cell_over_brainstem_image_spots_pairwise_stats.csv",
        index=False,
    )

    brainstem_plot_df = build_brainstem_same_size_region_df(cereb_fov_df)
    brainstem_stats = plot_condition_bars(
        brainstem_plot_df,
        value_col="value",
        ylabel="Avg PCP2 spots per brainstem cell-sized region",
        title="PCP2 Spots in Brainstem Cell-Sized Regions (Normalized LoG)",
        out_png=OUTPUT_DIR / "pcp2_normlog_brainstem_spots_per_cell_barplot.png",
        y_limit_max=1.0,
    )
    brainstem_plot_df.to_csv(
        OUTPUT_DIR / "pcp2_normlog_brainstem_spots_per_cell_datapoints.csv",
        index=False,
    )
    brainstem_stats.to_csv(
        OUTPUT_DIR / "pcp2_normlog_brainstem_spots_per_cell_pairwise_stats.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
