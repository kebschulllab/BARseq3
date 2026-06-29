from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import images_pooled_barseq_analysis as pooled
import malat1_per_cell_colored_plot as colored_plot


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs_images_pooled"
PER_CELL_CSV = OUTPUT_DIR / "malat1_pooled_per_cell.csv"
PER_IMAGE_CSV = OUTPUT_DIR / "malat1_pooled_per_image.csv"
SUMMARY_CSV = OUTPUT_DIR / "malat1_pooled_condition_summary.csv"
STATS_CSV = OUTPUT_DIR / "malat1_pooled_signal_over_noise_pairwise_stats.csv"
VALUE_COL = "malat1_signal_over_noise_ratio"
MAX_ALLOWED_SIGNAL_OVER_NOISE = 15.0


def pairwise_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for i, cond_a in enumerate(pooled.CONDITION_ORDER):
        vals_a = df.loc[df["condition"] == cond_a, VALUE_COL].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        for cond_b in pooled.CONDITION_ORDER[i + 1 :]:
            vals_b = df.loc[df["condition"] == cond_b, VALUE_COL].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            test = stats.ttest_ind(vals_a, vals_b, equal_var=False, nan_policy="omit")
            p_value = float(test.pvalue)
            rows.append(
                {
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "n_a": int(len(vals_a)),
                    "n_b": int(len(vals_b)),
                    "mean_a": float(np.mean(vals_a)) if len(vals_a) else np.nan,
                    "mean_b": float(np.mean(vals_b)) if len(vals_b) else np.nan,
                    "welch_ttest_pvalue": p_value,
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


def main() -> None:
    per_cell_df = pd.read_csv(PER_CELL_CSV)
    denom = per_cell_df["malat1_intensity_outside_expanded_dapi_inside_expanded_nissl_mean"].replace(0, np.nan)
    per_cell_df[VALUE_COL] = per_cell_df["malat1_intensity_inside_expanded_dapi_mean"] / denom
    per_cell_df = per_cell_df.replace([np.inf, -np.inf], np.nan)
    per_cell_df = per_cell_df.loc[per_cell_df[VALUE_COL].isna() | (per_cell_df[VALUE_COL] <= MAX_ALLOWED_SIGNAL_OVER_NOISE)].copy()

    per_image_df = pd.read_csv(PER_IMAGE_CSV)
    per_image_rows: list[dict] = []
    for image_uid, sub in per_cell_df.groupby("image_uid", sort=False):
        row = per_image_df.loc[per_image_df["image_uid"] == image_uid].iloc[0].to_dict()
        vals = sub[VALUE_COL].dropna()
        row["mean_ratio"] = float(vals.mean()) if len(vals) else np.nan
        row["median_ratio"] = float(vals.median()) if len(vals) else np.nan
        per_image_rows.append(row)
    per_image_df = pd.DataFrame(per_image_rows)

    summary_rows: list[dict] = []
    for cond in pooled.CONDITION_ORDER:
        sub = per_cell_df.loc[per_cell_df["condition"] == cond, VALUE_COL].dropna()
        img_sub = per_image_df.loc[per_image_df["condition"] == cond]
        summary_rows.append(
            {
                "condition": cond,
                "n_images": int(len(img_sub)),
                "n_cells": int((per_cell_df["condition"] == cond).sum()),
                "pooled_mean_ratio": float(sub.mean()) if len(sub) else np.nan,
                "pooled_median_ratio": float(sub.median()) if len(sub) else np.nan,
                "mean_image_mean_ratio": float(img_sub["mean_ratio"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(img_sub) else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    stats_df = pairwise_stats(per_cell_df)

    per_cell_df.to_csv(PER_CELL_CSV, index=False)
    per_image_df.to_csv(PER_IMAGE_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    stats_df.to_csv(STATS_CSV, index=False)

    pooled.plot_condition_bar_with_points(
        summary_df.rename(columns={"pooled_mean_ratio": VALUE_COL}),
        per_cell_df,
        value_col=VALUE_COL,
        ylabel="MALAT1 signal / noise",
        title="MALAT1 Per-Cell Signal-to-Noise Ratio",
        filename="malat1_pooled_ratio_barplot.png",
        add_pairwise_ttests=True,
        show_group_means_at_bottom=False,
    )

    colored_plot.main()


if __name__ == "__main__":
    main()
