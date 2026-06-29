from __future__ import annotations

import os
import re
import sys
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import stats
from skimage import draw

import replicate_barseq_analysis as base


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "images"
OUTPUT_DIR = ROOT / "analysis_outputs_images_pooled"
OUTPUT_DIR.mkdir(exist_ok=True)
CELLPOSE_V4_VENDOR_DIR = ROOT / ".vendor_cellpose_v4_np1"
CELLPOSE_V4_MODEL_DIR = ROOT / ".cellpose_models"

FILE_RE = re.compile(
    r"^(?P<channel>dapi|nissl|malat1|pcp2)_(?P<sample>b2|b3)_"
    r"(?:(?P<region_a>cereb|brainstem)(?:_(?P<conc_a>[^_]+))?|(?P<conc_b>[^_]+)_(?P<region_b>cereb|brainstem))"
    r"_slice(?P<slice>\d+)rep(?P<rep>\d+)\.(?:tif|tiff)$",
    re.IGNORECASE,
)

CONDITION_ORDER = ["b2", "b3_5nM", "b3_10nM", "b3_20nM"]
CONDITION_COLORS = {
    "b2": "#4C78A8",
    "b3_5nM": "#E45756",
    "b3_10nM": "#F58518",
    "b3_20nM": "#54A24B",
}
MALAT1_MANUAL_OVERRIDES = {
    # This image is visibly offset relative to DAPI/Nissl; using a lower
    # source crop for MALAT1 gives a better cross-channel match than a large
    # in-crop translation.
    "b3_20nM_slice12rep2_cereb": {
        "source_row_offset_px": 10,
        "source_col_offset_px": 0,
        "shift_row_px": 0.0,
    },
}
PCP2_SPOT_THRESHOLD = 0.0
PCP2_SPOT_MIN_RADIUS_PX = 2.0
PCP2_SPOT_MAX_RADIUS_PX = 4.0
PCP2_SPOT_MIN_DISTANCE = 4
PCP2_SPOT_SMOOTH_SIGMA = 0.45
PCP2_SPOT_BANDPASS_LOW_SIGMA = 1.4
PCP2_SPOT_BANDPASS_HIGH_SIGMA = 2.8
PCP2_SPOT_COUNT_THRESHOLD = 0.0
PCP2_SPOT_COUNT_MIN_DISTANCE = 3
PCP2_SPOT_COUNT_SMOOTH_SIGMA = 0.45
PCP2_SPOTMASK_DENSITY_RADIUS_PX = 16
PCP2_SPOTMASK_DENSITY_SMOOTH_SIGMA = 2.2
PCP2_SPOTMASK_MIN_DISTANCE_PX = 24
PCP2_SPOTMASK_MIN_AREA_PX = 700
PCP2_PURKINJE_MIN_EQ_DIAMETER_PX = 36.0
PCP2_PURKINJE_MAX_EQ_DIAMETER_PX = 95.0
PCP2_PURKINJE_MAX_AXIS_RATIO = 2.0
PCP2_PURKINJE_MIN_CIRCULARITY = 0.22
PCP2_PURKINJE_MIN_RADIUS_PX = 18.0
PCP2_PURKINJE_MAX_RADIUS_PX = 32.0
PCP2_PURKINJE_RADIUS_PAD_PX = 4.0
PCP2_PURKINJE_MIN_DISK_FREE_FRACTION = 0.45
PCP2_BRAINSTEM_RANDOM_REGION_MULTIPLIER = 2


def parse_manifest() -> tuple[dict[tuple[base.ImageKey, str], Path], pd.DataFrame]:
    manifest: dict[tuple[base.ImageKey, str], Path] = {}
    rows: list[dict] = []
    for path in sorted(INPUT_DIR.iterdir()):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        gd = match.groupdict()
        region = gd["region_a"] or gd["region_b"]
        concentration = gd["conc_a"] or gd["conc_b"] or "base"
        key = base.ImageKey(
            sample=gd["sample"],
            region=region,
            concentration=concentration,
            slice_id=gd["slice"],
            rep=gd["rep"],
        )
        manifest[(key, gd["channel"].lower())] = path
        rows.append(
            {
                "path": str(path),
                "channel": gd["channel"].lower(),
                "sample": key.sample,
                "region": key.region,
                "concentration": key.concentration,
                "slice_id": key.slice_id,
                "rep": key.rep,
                "condition": key.condition,
            }
        )
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(OUTPUT_DIR / "images_manifest.csv", index=False)
    return manifest, manifest_df


def load_image(manifest: dict[tuple[base.ImageKey, str], Path], key: base.ImageKey, channel: str) -> np.ndarray:
    return base.read_image(manifest[(key, channel)])


def load_image_raw(manifest: dict[tuple[base.ImageKey, str], Path], key: base.ImageKey, channel: str) -> np.ndarray:
    return tifffile.imread(manifest[(key, channel)])


def image_uid(key: base.ImageKey) -> str:
    return f"{key.condition}_slice{key.slice_id}rep{key.rep}_{key.region}"


def biological_slice_id(slice_id: str | int) -> str:
    return str(slice_id)[0]


def plot_condition_bar_with_points(
    summary_df: pd.DataFrame,
    image_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    filename: str,
    add_pairwise_ttests: bool = False,
    show_group_counts: bool = False,
    show_top_bar_values: bool = True,
    show_group_means_at_bottom: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    x = np.arange(len(CONDITION_ORDER))
    means = [float(summary_df.loc[summary_df["condition"] == cond, value_col].iloc[0]) if (summary_df["condition"] == cond).any() else np.nan for cond in CONDITION_ORDER]
    ax.bar(x, means, color=[CONDITION_COLORS[c] for c in CONDITION_ORDER], width=0.62, alpha=0.82)
    rng = np.random.default_rng(7)
    finite_means = [m for m in means if np.isfinite(m)]
    mean_ref = max(finite_means) if finite_means else 1.0
    condition_values: dict[str, np.ndarray] = {}
    for i, cond in enumerate(CONDITION_ORDER):
        sub = image_df.loc[image_df["condition"] == cond, value_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        condition_values[cond] = sub
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        ax.scatter(np.full(len(sub), x[i]) + jitter, sub, color="black", s=18, alpha=0.6, zorder=3)
        if show_top_bar_values and np.isfinite(means[i]):
            ax.text(x[i], means[i] + max(0.03, 0.04 * mean_ref), f"{means[i]:.2f}", ha="center", va="bottom")
    ax.set_xticks(x, CONDITION_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    summary_vals = summary_df[value_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy() if value_col in summary_df.columns else np.array([])
    image_vals = image_df[value_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy() if value_col in image_df.columns else np.array([])
    combined = np.concatenate([summary_vals, image_vals]) if len(summary_vals) or len(image_vals) else np.array([1.0])
    ymax = float(np.nanmax(combined)) if len(combined) else 1.0
    upper_ylim = max(1.0, float(ymax) * 1.18)

    if add_pairwise_ttests:
        comparisons: list[tuple[int, int, float, str]] = []
        for i, cond_a in enumerate(CONDITION_ORDER):
            for j in range(i + 1, len(CONDITION_ORDER)):
                cond_b = CONDITION_ORDER[j]
                vals_a = condition_values.get(cond_a, np.array([]))
                vals_b = condition_values.get(cond_b, np.array([]))
                if len(vals_a) < 2 or len(vals_b) < 2:
                    continue
                p_value = float(stats.ttest_ind(vals_a, vals_b, equal_var=False, nan_policy="omit").pvalue)
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
                comparisons.append((i, j, p_value, label))

        if comparisons:
            base_height = float(ymax) + max(0.08, 0.06 * max(float(ymax), 1.0))
            step = max(0.08, 0.08 * max(float(ymax), 1.0))
            line_height = max(0.02, 0.02 * max(float(ymax), 1.0))
            for idx, (i, j, _p_value, label) in enumerate(comparisons):
                y = base_height + idx * step
                ax.plot([x[i], x[i], x[j], x[j]], [y, y + line_height, y + line_height, y], color="black", linewidth=1.0)
                ax.text((x[i] + x[j]) / 2.0, y + line_height + max(0.01, 0.015 * max(float(ymax), 1.0)), label, ha="center", va="bottom", fontsize=9)
            upper_ylim = max(upper_ylim, base_height + len(comparisons) * step + 2.5 * line_height)

    ax.set_ylim(0, upper_ylim)

    if show_group_counts:
        count_y = 0.02 * upper_ylim
        for i, cond in enumerate(CONDITION_ORDER):
            n_points = len(condition_values.get(cond, np.array([])))
            ax.text(x[i], count_y, f"n={n_points}", ha="center", va="bottom", fontsize=10)

    if show_group_means_at_bottom:
        mean_y = 0.02 * upper_ylim
        for i, cond in enumerate(CONDITION_ORDER):
            if np.isfinite(means[i]):
                ax.text(x[i], mean_y, f"{means[i]:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def plot_condition_region_density(
    pooled_df: pd.DataFrame,
    image_df: pd.DataFrame,
    filename: str,
) -> None:
    groups = []
    labels = []
    colors = []
    for cond in CONDITION_ORDER:
        for region, shade in (("cereb", "#222222"), ("brainstem", "#888888")):
            groups.append((cond, region))
            labels.append(f"{cond}\n{region}")
            colors.append(CONDITION_COLORS[cond] if region == "cereb" else shade)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(groups))
    means = []
    for cond, region in groups:
        sub = pooled_df[(pooled_df["condition"] == cond) & (pooled_df["region"] == region)]
        means.append(float(sub["spots_per_100k_px"].iloc[0]) if len(sub) else np.nan)
    ax.bar(x, means, color=colors, width=0.68, alpha=0.82)
    rng = np.random.default_rng(9)
    finite_means = [m for m in means if np.isfinite(m)]
    mean_ref = max(finite_means) if finite_means else 1.0
    for i, (cond, region) in enumerate(groups):
        sub = image_df[(image_df["condition"] == cond) & (image_df["region"] == region)]["spots_per_100k_px"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        ax.scatter(np.full(len(sub), x[i]) + jitter, sub, color="black", s=18, alpha=0.55, zorder=3)
        if np.isfinite(means[i]):
            ax.text(x[i], means[i] + max(0.05, 0.03 * mean_ref), f"{means[i]:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Spots per 100k px")
    ax.set_title("PCP2 Pooled Spot Density Across All Slices and Replicates")
    ymax = image_df["spots_per_100k_px"].replace([np.inf, -np.inf], np.nan).dropna().max()
    if not np.isfinite(ymax):
        ymax = 1.0
    ax.set_ylim(0, max(1.0, float(ymax) * 1.18))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def plot_condition_region_point_barplot(
    point_df: pd.DataFrame,
    value_col: str,
    filename: str,
    ylabel: str,
    title: str,
) -> None:
    groups = []
    labels = []
    colors = []
    for cond in CONDITION_ORDER:
        for region, shade in (("cereb", "#222222"), ("brainstem", "#888888")):
            groups.append((cond, region))
            labels.append(f"{cond}\n{region}")
            colors.append(CONDITION_COLORS[cond] if region == "cereb" else shade)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(groups))
    means = []
    for cond, region in groups:
        sub = point_df[(point_df["condition"] == cond) & (point_df["region"] == region)][value_col]
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        means.append(float(sub.mean()) if len(sub) else np.nan)
    ax.bar(x, means, color=colors, width=0.68, alpha=0.82)
    rng = np.random.default_rng(11)
    finite_means = [m for m in means if np.isfinite(m)]
    mean_ref = max(finite_means) if finite_means else 1.0
    for i, (cond, region) in enumerate(groups):
        sub = point_df[(point_df["condition"] == cond) & (point_df["region"] == region)][value_col]
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        ax.scatter(np.full(len(sub), x[i]) + jitter, sub, color="black", s=14, alpha=0.45, zorder=3)
        if np.isfinite(means[i]):
            ax.text(x[i], means[i] + max(0.05, 0.03 * mean_ref), f"{means[i]:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ymax = point_df[value_col].replace([np.inf, -np.inf], np.nan).dropna().max()
    if not np.isfinite(ymax):
        ymax = 1.0
    ax.set_ylim(0, max(1.0, float(ymax) * 1.18))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def call_pcp2_spots_16bit(image_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_float = image_raw.astype(np.float32, copy=False)
    image_bs = base.remove_background(image_float, sigma=20)
    bandpass = base.filters.gaussian(image_bs, sigma=PCP2_SPOT_BANDPASS_LOW_SIGMA, preserve_range=True) - base.filters.gaussian(
        image_bs, sigma=PCP2_SPOT_BANDPASS_HIGH_SIGMA, preserve_range=True
    )
    smooth = base.filters.gaussian(np.clip(bandpass, 0, None), sigma=PCP2_SPOT_SMOOTH_SIGMA, preserve_range=True)
    coords = base.feature.peak_local_max(
        smooth,
        min_distance=PCP2_SPOT_MIN_DISTANCE,
        threshold_abs=0.0,
        exclude_border=False,
    )
    if len(coords):
        coords = np.unique(coords.astype(np.int32, copy=False), axis=0)
    return coords, image_bs


def call_pcp2_spots_16bit_sensitive(image_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_float = image_raw.astype(np.float32, copy=False)
    image_bs = base.remove_background(image_float, sigma=20)
    bandpass = base.filters.gaussian(image_bs, sigma=PCP2_SPOT_BANDPASS_LOW_SIGMA, preserve_range=True) - base.filters.gaussian(
        image_bs, sigma=PCP2_SPOT_BANDPASS_HIGH_SIGMA, preserve_range=True
    )
    smooth = base.filters.gaussian(np.clip(bandpass, 0, None), sigma=PCP2_SPOT_COUNT_SMOOTH_SIGMA, preserve_range=True)
    coords = base.feature.peak_local_max(
        smooth,
        min_distance=PCP2_SPOT_COUNT_MIN_DISTANCE,
        threshold_abs=0.0,
        exclude_border=False,
    )
    if len(coords):
        coords = np.unique(coords.astype(np.int32, copy=False), axis=0)
    return coords, image_bs


def average_brainstem_spots_for_matched_areas(
    brainstem_spots: np.ndarray,
    image_shape: tuple[int, int],
    area_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(area_values) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    spot_map = np.zeros(image_shape, dtype=np.int32)
    for r, c in brainstem_spots:
        spot_map[int(r), int(c)] += 1
    integral = spot_map.cumsum(axis=0).cumsum(axis=1)

    def window_mean_count(side: int) -> float:
        h, w = image_shape
        if side <= 0:
            return np.nan
        side = min(side, h, w)
        sums = integral[side - 1 :, side - 1 :].astype(np.float64)
        if side > 1:
            sums = sums.copy()
            sums[1:, :] -= integral[:-side, side - 1 :]
            sums[:, 1:] -= integral[side - 1 :, :-side]
            sums[1:, 1:] += integral[:-side, :-side]
        return float(sums.mean()) if sums.size else np.nan

    sides = np.maximum(1, np.rint(np.sqrt(area_values)).astype(np.int32))
    unique_sides = np.unique(sides)
    side_to_mean = {int(side): window_mean_count(int(side)) for side in unique_sides}
    means = np.array([side_to_mean[int(side)] for side in sides], dtype=np.float32)
    return means, sides


def average_brainstem_intensity_for_matched_areas(
    brainstem_image_bs: np.ndarray,
    area_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(area_values) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    integral = brainstem_image_bs.astype(np.float64).cumsum(axis=0).cumsum(axis=1)

    def window_mean_intensity_sum(side: int) -> float:
        h, w = brainstem_image_bs.shape
        if side <= 0:
            return np.nan
        side = min(side, h, w)
        sums = integral[side - 1 :, side - 1 :].astype(np.float64)
        if side > 1:
            sums = sums.copy()
            sums[1:, :] -= integral[:-side, side - 1 :]
            sums[:, 1:] -= integral[side - 1 :, :-side]
            sums[1:, 1:] += integral[:-side, :-side]
        return float(sums.mean()) if sums.size else np.nan

    sides = np.maximum(1, np.rint(np.sqrt(area_values)).astype(np.int32))
    unique_sides = np.unique(sides)
    side_to_mean = {int(side): window_mean_intensity_sum(int(side)) for side in unique_sides}
    means = np.array([side_to_mean[int(side)] for side in sides], dtype=np.float32)
    return means, sides


def save_pcp2_pooled_qc(
    key: base.ImageKey,
    cereb_pcp2_bs: np.ndarray,
    cereb_mask: np.ndarray,
    cereb_spots: np.ndarray,
    brainstem_pcp2_bs: np.ndarray,
    brainstem_mask: np.ndarray,
    brainstem_spots: np.ndarray,
    n_purkinje_cells: int,
    n_brainstem_cells: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    axes[0].imshow(base.fixed_range_rescale(cereb_pcp2_bs, 150, 1000), cmap="gray")
    if len(cereb_spots):
        axes[0].plot(cereb_spots[:, 1], cereb_spots[:, 0], ".", color="red", markersize=2.2)
    if np.any(cereb_mask):
        axes[0].contour(cereb_mask, colors="yellow", linewidths=0.9, linestyles="dotted")
    axes[0].set_title(f"{key.condition} cereb spots + mask ({n_purkinje_cells} cells)")
    axes[0].axis("off")

    axes[1].imshow(cereb_mask.astype(np.float32), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Cereb mask")
    axes[1].axis("off")

    axes[2].imshow(base.fixed_range_rescale(brainstem_pcp2_bs, 150, 1000), cmap="gray")
    if len(brainstem_spots):
        axes[2].plot(brainstem_spots[:, 1], brainstem_spots[:, 0], ".", color="red", markersize=2.2)
    if np.any(brainstem_mask):
        axes[2].contour(brainstem_mask, colors="yellow", linewidths=0.9, linestyles="dotted")
    axes[2].set_title(f"{key.condition} brainstem spots + random regions ({n_brainstem_cells} regions)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"pcp2_pooled_qc_{key.condition}_slice{key.slice_id}rep{key.rep}.png", dpi=200)
    plt.close()


def save_malat1_pooled_panel(
    key: base.ImageKey,
    dapi: np.ndarray,
    nissl: np.ndarray,
    malat1: np.ndarray,
    nuclei_labels: np.ndarray,
    cell_labels: np.ndarray,
) -> None:
    expanded_nuclei_mask = base.segmentation.expand_labels(nuclei_labels, distance=2) > 0
    expanded_cell_mask = base.segmentation.expand_labels(cell_labels, distance=256) > 0
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))
    panels = [
        (malat1, f"{key.condition} slice{key.slice_id} MALAT1", "magma", base.MALAT1_INTENSITY_RANGE),
        (dapi, "DAPI signal", "Blues", None),
        (nissl, "Nissl signal", "Greens", None),
        (expanded_cell_mask.astype(np.float32), "Expanded Nissl mask", "gray", (0.0, 1.0)),
    ]
    for ax, (image, title, cmap, fixed_range) in zip(axes, panels):
        if fixed_range is None:
            display = base.percentile_rescale(image, 1, 99.8)
        else:
            display = base.fixed_range_rescale(image, fixed_range[0], fixed_range[1])
        ax.imshow(display, cmap=cmap)
        ax.contour(expanded_nuclei_mask, levels=[0.5], colors="cyan", linewidths=0.9, linestyles="dotted")
        ax.contour(expanded_cell_mask, levels=[0.5], colors="yellow", linewidths=0.9, linestyles="dotted")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"malat1_pooled_panel_{key.condition}_slice{key.slice_id}rep{key.rep}.png", dpi=200)
    plt.close()


def analyze_malat1_pooled(manifest: dict[tuple[base.ImageKey, str], Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_cell_tables: list[pd.DataFrame] = []
    per_image_rows: list[dict] = []
    keys = sorted(
        {key for key, channel in manifest if key.region == "cereb" and channel == "malat1"},
        key=lambda k: (CONDITION_ORDER.index(k.condition), int(k.slice_id), int(k.rep)),
    )
    value_col = "malat1_intensity_per_unit_area_outside_expanded_dapi_inside_expanded_nissl_over_inside_expanded_dapi_ratio"
    for idx, key in enumerate(keys, start=1):
        print(f"[MALAT1] {idx}/{len(keys)} {image_uid(key)}", flush=True)
        dapi = load_image(manifest, key, "dapi")
        nissl = load_image(manifest, key, "nissl")
        malat1 = load_image(manifest, key, "malat1")
        roi = base.select_sparse_roi(dapi)
        dapi_roi = base.crop(dapi, roi)
        nissl_roi = base.crop(nissl, roi)
        override = MALAT1_MANUAL_OVERRIDES.get(image_uid(key))
        if override is None:
            malat1_roi = base.crop(malat1, roi)
        else:
            source_roi = (
                roi[0] + int(override.get("source_row_offset_px", 0)),
                roi[1] + int(override.get("source_col_offset_px", 0)),
                roi[2],
            )
            malat1_roi = base.crop(malat1, source_roi)
        transform = base.estimate_malat1_transform(dapi_roi, nissl_roi, malat1_roi)
        if override is not None:
            angle = float(override.get("angle_deg", transform["angle"]))
            shift_row = float(override.get("shift_row_px", transform["shift_rc"][0]))
            shift_col = float(override.get("shift_col_px", transform["shift_rc"][1]))
            transform = {
                "angle": angle,
                "shift_rc": (shift_row, shift_col),
                "score": float(transform["score"]),
            }
        malat1_aligned = base.rigid_transform(
            malat1_roi,
            angle=float(transform["angle"]),
            shift_rc=tuple(transform["shift_rc"]),
        )
        nuclei_labels = base.segment_nuclei(dapi_roi)
        cell_labels = base.segment_cells_cellpose(nissl_roi, dapi_roi, nuclei_labels)
        per_cell = base.compute_per_cell_malat1_ratio(key.condition, key, malat1_aligned, nuclei_labels, cell_labels)
        save_malat1_pooled_panel(key, dapi_roi, nissl_roi, malat1_aligned, nuclei_labels, cell_labels)
        per_cell["image_uid"] = image_uid(key)
        per_cell["slice_id"] = key.slice_id
        per_cell["rep"] = key.rep
        per_cell_tables.append(per_cell)
        vals = per_cell[value_col].replace([np.inf, -np.inf], np.nan).dropna()
        per_image_rows.append(
            {
                "condition": key.condition,
                "sample": key.sample,
                "concentration": key.concentration,
                "slice_id": key.slice_id,
                "rep": key.rep,
                "image_uid": image_uid(key),
                "n_cells": int(len(per_cell)),
                "mean_ratio": float(vals.mean()) if len(vals) else np.nan,
                "median_ratio": float(vals.median()) if len(vals) else np.nan,
                "roi_row_start": roi[0],
                "roi_col_start": roi[1],
                "roi_size_px": roi[2],
                "malat1_rotation_deg": float(transform["angle"]),
                "malat1_shift_row_px": float(transform["shift_rc"][0]),
                "malat1_shift_col_px": float(transform["shift_rc"][1]),
                "malat1_alignment_score": float(transform["score"]),
            }
        )
    per_cell_df = pd.concat(per_cell_tables, ignore_index=True)
    per_image_df = pd.DataFrame(per_image_rows)
    summary_rows = []
    for cond in CONDITION_ORDER:
        sub = per_cell_df[per_cell_df["condition"] == cond][value_col].replace([np.inf, -np.inf], np.nan).dropna()
        img_sub = per_image_df[per_image_df["condition"] == cond]
        summary_rows.append(
            {
                "condition": cond,
                "n_images": int(len(img_sub)),
                "n_cells": int(len(sub)),
                "pooled_mean_ratio": float(sub.mean()) if len(sub) else np.nan,
                "pooled_median_ratio": float(sub.median()) if len(sub) else np.nan,
                "mean_image_mean_ratio": float(img_sub["mean_ratio"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(img_sub) else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    per_cell_df.to_csv(OUTPUT_DIR / "malat1_pooled_per_cell.csv", index=False)
    per_image_df.to_csv(OUTPUT_DIR / "malat1_pooled_per_image.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "malat1_pooled_condition_summary.csv", index=False)
    plot_condition_bar_with_points(
        summary_df.rename(columns={"pooled_mean_ratio": value_col}),
        per_cell_df,
        value_col=value_col,
        ylabel="Mean outside expanded DAPI / mean inside expanded DAPI",
        title="MALAT1 Pooled Across All Slices and Replicates",
        filename="malat1_pooled_ratio_barplot.png",
        add_pairwise_ttests=True,
    )
    return per_cell_df, per_image_df, summary_df


def segment_pcp2_purkinje_fast(
    pcp2_bs: np.ndarray,
    dapi: np.ndarray,
    spots: np.ndarray,
    sample: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if len(spots) == 0:
        empty = np.zeros_like(pcp2_bs, dtype=np.int32)
        return empty, empty > 0, pd.DataFrame()

    spot_seed = np.zeros_like(pcp2_bs, dtype=np.float32)
    for r, c in spots:
        spot_seed[int(r), int(c)] += 1.0

    local_counts = base.ndi.convolve(
        spot_seed,
        base.morphology.disk(PCP2_SPOTMASK_DENSITY_RADIUS_PX).astype(np.float32),
        mode="constant",
        cval=0.0,
    )
    density = base.filters.gaussian(
        local_counts,
        sigma=PCP2_SPOTMASK_DENSITY_SMOOTH_SIGMA,
        preserve_range=True,
    )
    support_threshold = 3.0 if sample == "b2" else 4.0
    support = density >= support_threshold
    support = base.morphology.binary_closing(support, base.morphology.disk(4))
    support = base.morphology.binary_opening(support, base.morphology.disk(1))
    support = base.morphology.remove_small_objects(support, 250)
    support = base.morphology.remove_small_holes(support, 200)
    if not np.any(support):
        empty = np.zeros_like(pcp2_bs, dtype=np.int32)
        return empty, empty > 0, pd.DataFrame()

    maxima = base.feature.peak_local_max(
        density,
        min_distance=PCP2_SPOTMASK_MIN_DISTANCE_PX,
        threshold_abs=3.5 if sample == "b2" else 5.0,
        labels=support,
        exclude_border=False,
    )
    markers = np.zeros_like(pcp2_bs, dtype=np.int32)
    if len(maxima):
        markers[tuple(maxima.T)] = np.arange(1, len(maxima) + 1)
        markers = base.ndi.label(markers > 0)[0]
    labels = base.segmentation.watershed(-density, markers, mask=support) if markers.max() > 0 else base.measure.label(support)

    candidate_rows: list[dict] = []
    for prop in base.measure.regionprops(labels, intensity_image=pcp2_bs):
        perimeter = float(getattr(prop, "perimeter_crofton", 0.0) or 0.0)
        axis_ratio = float(prop.axis_major_length / max(prop.axis_minor_length, 1e-6))
        circularity = float((4.0 * np.pi * prop.area / (perimeter * perimeter)) if perimeter > 0 else 0.0)
        eq_diameter = float(prop.equivalent_diameter_area)
        label_mask = labels == int(prop.label)
        n_spots = int(spot_seed[label_mask].sum()) if np.any(label_mask) else 0
        min_spots = 3 if sample == "b2" else 4
        if prop.area < PCP2_SPOTMASK_MIN_AREA_PX:
            continue
        if eq_diameter < PCP2_PURKINJE_MIN_EQ_DIAMETER_PX or eq_diameter > PCP2_PURKINJE_MAX_EQ_DIAMETER_PX:
            continue
        if axis_ratio > PCP2_PURKINJE_MAX_AXIS_RATIO:
            continue
        if circularity < PCP2_PURKINJE_MIN_CIRCULARITY:
            continue
        if n_spots < min_spots:
            continue
        spot_coords = spots[label_mask[spots[:, 0], spots[:, 1]]] if len(spots) else np.empty((0, 2), dtype=np.int32)
        if len(spot_coords):
            center_row = float(np.mean(spot_coords[:, 0]))
            center_col = float(np.mean(spot_coords[:, 1]))
            distances = np.sqrt((spot_coords[:, 0] - center_row) ** 2 + (spot_coords[:, 1] - center_col) ** 2)
            raw_radius = float(np.percentile(distances, 85) + PCP2_PURKINJE_RADIUS_PAD_PX)
        else:
            center_row = float(prop.centroid[0])
            center_col = float(prop.centroid[1])
            raw_radius = float(eq_diameter * 0.5)
        disk_radius = float(np.clip(raw_radius, PCP2_PURKINJE_MIN_RADIUS_PX, PCP2_PURKINJE_MAX_RADIUS_PX))
        candidate_rows.append(
            {
                "center_row": center_row,
                "center_col": center_col,
                "disk_radius_px": disk_radius,
                "seed_spot_count": n_spots,
            }
        )

    if not candidate_rows:
        empty = np.zeros_like(pcp2_bs, dtype=np.int32)
        return empty, empty > 0, pd.DataFrame()

    final_labels = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    rows: list[dict] = []
    for candidate in sorted(candidate_rows, key=lambda row: (row["seed_spot_count"], row["disk_radius_px"]), reverse=True):
        rr, cc = draw.disk(
            (candidate["center_row"], candidate["center_col"]),
            candidate["disk_radius_px"],
            shape=pcp2_bs.shape,
        )
        if len(rr) == 0:
            continue
        free_mask = final_labels[rr, cc] == 0
        if free_mask.mean() < PCP2_PURKINJE_MIN_DISK_FREE_FRACTION:
            continue
        rr = rr[free_mask]
        cc = cc[free_mask]
        if len(rr) == 0:
            continue
        final_labels[rr, cc] = next_id
        pixel_area = int(len(rr))
        eq_diameter = float(np.sqrt((4.0 * pixel_area) / np.pi))
        local_mask = final_labels == next_id
        rows.append(
            {
                "cell_id": next_id,
                "area_px": pixel_area,
                "centroid_row": float(np.mean(rr)),
                "centroid_col": float(np.mean(cc)),
                "equivalent_diameter_px": eq_diameter,
                "axis_ratio": 1.0,
                "circularity": 1.0,
                "mean_pcp2_intensity": float(np.mean(pcp2_bs[rr, cc])),
                "pcp2_spot_seed_count": int(candidate["seed_spot_count"]),
                "disk_radius_px": float(candidate["disk_radius_px"]),
            }
        )
        next_id += 1

    final_mask = final_labels > 0
    if not rows:
        empty = np.zeros_like(pcp2_bs, dtype=np.int32)
        return empty, empty > 0, pd.DataFrame()
    return final_labels.astype(np.int32), final_mask, pd.DataFrame(rows)


def segment_pcp2_brainstem_cells_fast(
    pcp2_bs: np.ndarray,
    spots: np.ndarray,
    radius_hint_px: float | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if len(spots) == 0:
        empty = np.zeros_like(pcp2_bs, dtype=np.int32)
        return empty, empty > 0, pd.DataFrame()

    spot_seed = np.zeros_like(pcp2_bs, dtype=np.float32)
    for r, c in spots:
        spot_seed[int(r), int(c)] += 1.0

    local_counts = base.ndi.convolve(
        spot_seed,
        base.morphology.disk(max(10, PCP2_SPOTMASK_DENSITY_RADIUS_PX - 2)).astype(np.float32),
        mode="constant",
        cval=0.0,
    )
    density = base.filters.gaussian(
        local_counts,
        sigma=max(1.6, PCP2_SPOTMASK_DENSITY_SMOOTH_SIGMA - 0.4),
        preserve_range=True,
    )
    support = density >= 1.8
    support = base.morphology.binary_closing(support, base.morphology.disk(3))
    support = base.morphology.binary_opening(support, base.morphology.disk(1))
    support = base.morphology.remove_small_objects(support, 60)
    support = base.morphology.remove_small_holes(support, 60)
    if not np.any(support):
        empty = np.zeros_like(pcp2_bs, dtype=np.int32)
        return empty, empty > 0, pd.DataFrame()

    brainstem_min_distance = max(18, int(round((radius_hint_px or 24.0) * 0.7)))
    maxima = base.feature.peak_local_max(
        density,
        min_distance=brainstem_min_distance,
        threshold_abs=2.0,
        labels=support,
        exclude_border=False,
    )
    markers = np.zeros_like(pcp2_bs, dtype=np.int32)
    if len(maxima):
        markers[tuple(maxima.T)] = np.arange(1, len(maxima) + 1)
        markers = base.ndi.label(markers > 0)[0]
    labels = base.segmentation.watershed(-density, markers, mask=support) if markers.max() > 0 else base.measure.label(support)

    if radius_hint_px is None or not np.isfinite(radius_hint_px):
        radius_min = PCP2_PURKINJE_MIN_RADIUS_PX
        radius_max = PCP2_PURKINJE_MAX_RADIUS_PX
    else:
        radius_min = max(PCP2_PURKINJE_MIN_RADIUS_PX, float(radius_hint_px) * 0.75)
        radius_max = min(PCP2_PURKINJE_MAX_RADIUS_PX, float(radius_hint_px) * 1.25)
        radius_min = min(radius_min, radius_max)

    candidate_rows: list[dict] = []
    for prop in base.measure.regionprops(labels, intensity_image=pcp2_bs):
        label_mask = labels == int(prop.label)
        n_spots = int(spot_seed[label_mask].sum()) if np.any(label_mask) else 0
        if n_spots < 2:
            continue
        spot_coords = spots[label_mask[spots[:, 0], spots[:, 1]]] if len(spots) else np.empty((0, 2), dtype=np.int32)
        if len(spot_coords):
            center_row = float(np.mean(spot_coords[:, 0]))
            center_col = float(np.mean(spot_coords[:, 1]))
            distances = np.sqrt((spot_coords[:, 0] - center_row) ** 2 + (spot_coords[:, 1] - center_col) ** 2)
            raw_radius = float(np.percentile(distances, 85) + PCP2_PURKINJE_RADIUS_PAD_PX)
        else:
            center_row = float(prop.centroid[0])
            center_col = float(prop.centroid[1])
            raw_radius = float(np.sqrt(prop.area / np.pi))
        disk_radius = float(np.clip(raw_radius, radius_min, radius_max))
        candidate_rows.append(
            {
                "center_row": center_row,
                "center_col": center_col,
                "disk_radius_px": disk_radius,
                "seed_spot_count": n_spots,
            }
        )

    if not candidate_rows:
        empty = np.zeros_like(pcp2_bs, dtype=np.int32)
        return empty, empty > 0, pd.DataFrame()

    final_labels = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    rows: list[dict] = []
    for candidate in sorted(candidate_rows, key=lambda row: (row["seed_spot_count"], row["disk_radius_px"]), reverse=True):
        rr, cc = draw.disk(
            (candidate["center_row"], candidate["center_col"]),
            candidate["disk_radius_px"],
            shape=pcp2_bs.shape,
        )
        if len(rr) == 0:
            continue
        free_mask = final_labels[rr, cc] == 0
        if free_mask.mean() < PCP2_PURKINJE_MIN_DISK_FREE_FRACTION:
            continue
        rr = rr[free_mask]
        cc = cc[free_mask]
        if len(rr) == 0:
            continue
        final_labels[rr, cc] = next_id
        pixel_area = int(len(rr))
        eq_diameter = float(np.sqrt((4.0 * pixel_area) / np.pi))
        rows.append(
            {
                "cell_id": next_id,
                "area_px": pixel_area,
                "centroid_row": float(np.mean(rr)),
                "centroid_col": float(np.mean(cc)),
                "equivalent_diameter_px": eq_diameter,
                "axis_ratio": 1.0,
                "circularity": 1.0,
                "mean_pcp2_intensity": float(np.mean(pcp2_bs[rr, cc])),
                "pcp2_spot_seed_count": int(candidate["seed_spot_count"]),
                "disk_radius_px": float(candidate["disk_radius_px"]),
            }
        )
        next_id += 1

    final_mask = final_labels > 0
    if not rows:
        empty = np.zeros_like(pcp2_bs, dtype=np.int32)
        return empty, empty > 0, pd.DataFrame()
    return final_labels.astype(np.int32), final_mask, pd.DataFrame(rows)


def sample_random_brainstem_regions(
    image_shape: tuple[int, int],
    radii_px: np.ndarray,
    seed_key: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if len(radii_px) == 0:
        empty = np.zeros(image_shape, dtype=np.int32)
        return empty, empty > 0, pd.DataFrame()

    rng = np.random.default_rng(zlib.crc32(seed_key.encode("utf-8")) & 0xFFFFFFFF)
    h, w = image_shape
    occupancy = np.zeros(image_shape, dtype=bool)
    labels = np.zeros(image_shape, dtype=np.int32)
    rows: list[dict] = []
    order = np.argsort(-radii_px)
    next_id = 1

    for idx in order:
        radius = float(np.clip(radii_px[idx], PCP2_PURKINJE_MIN_RADIUS_PX, PCP2_PURKINJE_MAX_RADIUS_PX))
        row_low = int(np.ceil(radius))
        row_high = int(np.floor(h - radius))
        col_low = int(np.ceil(radius))
        col_high = int(np.floor(w - radius))
        if row_high <= row_low or col_high <= col_low:
            continue
        best: tuple[np.ndarray, np.ndarray, float] | None = None
        for _ in range(300):
            center_row = rng.integers(row_low, row_high + 1)
            center_col = rng.integers(col_low, col_high + 1)
            rr, cc = draw.disk((center_row, center_col), radius, shape=image_shape)
            if len(rr) == 0:
                continue
            overlap_fraction = float(occupancy[rr, cc].mean())
            if best is None or overlap_fraction < best[2]:
                best = (rr, cc, overlap_fraction)
            if overlap_fraction <= 0.05:
                break
        if best is None:
            continue
        rr, cc, _ = best
        new_mask = ~occupancy[rr, cc]
        rr = rr[new_mask]
        cc = cc[new_mask]
        if len(rr) == 0:
            continue
        labels[rr, cc] = next_id
        occupancy[rr, cc] = True
        pixel_area = int(len(rr))
        rows.append(
            {
                "cell_id": next_id,
                "area_px": pixel_area,
                "centroid_row": float(np.mean(rr)),
                "centroid_col": float(np.mean(cc)),
                "equivalent_diameter_px": float(np.sqrt((4.0 * pixel_area) / np.pi)),
                "axis_ratio": 1.0,
                "circularity": 1.0,
                "mean_pcp2_intensity": np.nan,
                "pcp2_spot_seed_count": np.nan,
                "disk_radius_px": radius,
            }
        )
        next_id += 1

    mask = labels > 0
    if not rows:
        empty = np.zeros(image_shape, dtype=np.int32)
        return empty, empty > 0, pd.DataFrame()
    return labels.astype(np.int32), mask, pd.DataFrame(rows)


def analyze_pcp2_pooled(manifest: dict[tuple[base.ImageKey, str], Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_image_rows: list[dict] = []
    per_cell_tables: list[pd.DataFrame] = []
    brainstem_cell_tables: list[pd.DataFrame] = []
    point_plot_rows: list[dict] = []
    point_plot_intensity_rows: list[dict] = []
    pair_rows: list[dict] = []
    radii_by_group: dict[tuple[str, str, str, str, str], list[np.ndarray]] = {}
    processed_brainstem_uids: set[str] = set()
    keys = sorted(
        {key for key, channel in manifest if key.region == "cereb" and channel == "pcp2"},
        key=lambda k: (CONDITION_ORDER.index(k.condition), int(k.slice_id), int(k.rep)),
    )
    for idx, cereb_key in enumerate(keys, start=1):
        print(f"[PCP2] {idx}/{len(keys)} {image_uid(cereb_key)}", flush=True)
        brainstem_key = base.ImageKey(cereb_key.sample, "brainstem", cereb_key.concentration, cereb_key.slice_id, cereb_key.rep)
        dapi_cereb = load_image(manifest, cereb_key, "dapi")
        pcp2_cereb = load_image(manifest, cereb_key, "pcp2")
        pcp2_cereb_raw = load_image_raw(manifest, cereb_key, "pcp2")
        pcp2_brainstem = load_image(manifest, brainstem_key, "pcp2")
        pcp2_brainstem_raw = load_image_raw(manifest, brainstem_key, "pcp2")
        cereb_seg_spots, cereb_pcp2_bs = call_pcp2_spots_16bit(pcp2_cereb_raw)
        brainstem_spots, brainstem_pcp2_bs = call_pcp2_spots_16bit_sensitive(pcp2_brainstem_raw)
        cereb_spots, _ = call_pcp2_spots_16bit_sensitive(pcp2_cereb_raw)
        purkinje_labels, cereb_mask, per_cell_df = segment_pcp2_purkinje_fast(
            cereb_pcp2_bs,
            dapi_cereb,
            cereb_seg_spots,
            cereb_key.sample,
        )
        cereb_spots = base.filter_spots(cereb_spots, cereb_mask)
        brainstem_labels = np.zeros_like(brainstem_pcp2_bs, dtype=np.int32)
        brainstem_mask = np.zeros_like(brainstem_pcp2_bs, dtype=bool)
        brainstem_cell_df = pd.DataFrame()
        if not per_cell_df.empty and "disk_radius_px" in per_cell_df.columns:
            repeated_radii = np.repeat(
                per_cell_df["disk_radius_px"].to_numpy(dtype=np.float32),
                PCP2_BRAINSTEM_RANDOM_REGION_MULTIPLIER,
            )
            brainstem_labels, brainstem_mask, brainstem_cell_df = sample_random_brainstem_regions(
                brainstem_pcp2_bs.shape,
                repeated_radii,
                seed_key=image_uid(brainstem_key),
            )
        save_pcp2_pooled_qc(
            key=cereb_key,
            cereb_pcp2_bs=cereb_pcp2_bs,
            cereb_mask=cereb_mask,
            cereb_spots=cereb_spots,
            brainstem_pcp2_bs=brainstem_pcp2_bs,
            brainstem_mask=brainstem_mask,
            brainstem_spots=brainstem_spots,
            n_purkinje_cells=int(per_cell_df["cell_id"].nunique()) if not per_cell_df.empty else 0,
            n_brainstem_cells=int(brainstem_cell_df["cell_id"].nunique()) if not brainstem_cell_df.empty else 0,
        )
        avg_brainstem_spots_per_cell = np.nan
        avg_brainstem_intensity_per_cell = np.nan
        processed_brainstem_uids.add(image_uid(brainstem_key))
        if not brainstem_cell_df.empty:
            brainstem_cell_spot_counts = np.zeros(int(brainstem_labels.max()) + 1, dtype=np.int32)
            for r, c in brainstem_spots:
                label_id = int(brainstem_labels[int(r), int(c)])
                if label_id > 0:
                    brainstem_cell_spot_counts[label_id] += 1
            brainstem_cell_df = brainstem_cell_df.copy()
            brainstem_cell_df["condition"] = cereb_key.condition
            brainstem_cell_df["sample"] = brainstem_key.sample
            brainstem_cell_df["concentration"] = brainstem_key.concentration
            brainstem_cell_df["slice_id"] = brainstem_key.slice_id
            brainstem_cell_df["rep"] = brainstem_key.rep
            brainstem_cell_df["image_uid"] = image_uid(brainstem_key)
            brainstem_cell_df["region"] = "brainstem"
            brainstem_cell_df["brainstem_region_definition"] = "random matched-size regions using Purkinje-cell radii from paired cerebellar FOV"
            brainstem_cell_df["pcp2_spot_count"] = brainstem_cell_df["cell_id"].map(lambda x: int(brainstem_cell_spot_counts[int(x)]))
            brainstem_cell_df["mean_pcp2_intensity"] = brainstem_cell_df["cell_id"].map(
                lambda x: float(brainstem_pcp2_bs[brainstem_labels == int(x)].mean()) if np.any(brainstem_labels == int(x)) else np.nan
            )
            brainstem_cell_df["pcp2_intensity_sum"] = brainstem_cell_df["cell_id"].map(
                lambda x: float(brainstem_pcp2_bs[brainstem_labels == int(x)].sum()) if np.any(brainstem_labels == int(x)) else np.nan
            )
            avg_brainstem_spots_per_cell = float(brainstem_cell_df["pcp2_spot_count"].mean())
            avg_brainstem_intensity_per_cell = float(brainstem_cell_df["pcp2_intensity_sum"].mean())
            brainstem_cell_tables.append(brainstem_cell_df)
            for _, row in brainstem_cell_df.iterrows():
                point_plot_rows.append(
                    {
                        "condition": cereb_key.condition,
                        "region": "brainstem",
                        "image_uid": image_uid(brainstem_key),
                        "cell_id": int(row["cell_id"]),
                        "value": float(row["pcp2_spot_count"]),
                    }
                )
                point_plot_intensity_rows.append(
                    {
                        "condition": cereb_key.condition,
                        "region": "brainstem",
                        "image_uid": image_uid(brainstem_key),
                        "cell_id": int(row["cell_id"]),
                        "value": float(row["pcp2_intensity_sum"]),
                    }
                )
        if not per_cell_df.empty:
            group_key = (
                cereb_key.condition,
                cereb_key.sample,
                cereb_key.concentration,
                biological_slice_id(cereb_key.slice_id),
                cereb_key.rep,
            )
            radii_by_group.setdefault(group_key, []).append(
                per_cell_df["disk_radius_px"].to_numpy(dtype=np.float32)
            )
            cell_spot_counts = np.zeros(int(purkinje_labels.max()) + 1, dtype=np.int32)
            for r, c in cereb_spots:
                label_id = int(purkinje_labels[int(r), int(c)])
                if label_id > 0:
                    cell_spot_counts[label_id] += 1
            per_cell_df = per_cell_df.copy()
            per_cell_df["condition"] = cereb_key.condition
            per_cell_df["sample"] = cereb_key.sample
            per_cell_df["concentration"] = cereb_key.concentration
            per_cell_df["slice_id"] = cereb_key.slice_id
            per_cell_df["rep"] = cereb_key.rep
            per_cell_df["image_uid"] = image_uid(cereb_key)
            per_cell_df["region"] = "cereb"
            per_cell_df["pcp2_spot_count"] = per_cell_df["cell_id"].map(lambda x: int(cell_spot_counts[int(x)]))
            per_cell_df["pcp2_intensity_sum"] = per_cell_df["area_px"] * per_cell_df["mean_pcp2_intensity"]
            per_cell_df["brainstem_region_definition"] = "random matched-size regions using Purkinje-cell radii from paired brainstem FOV"
            per_cell_df["brainstem_average_spots_per_segmented_cell"] = avg_brainstem_spots_per_cell
            per_cell_df["brainstem_average_spots_in_matched_area"] = avg_brainstem_spots_per_cell
            per_cell_df["brainstem_average_intensity_per_segmented_cell"] = avg_brainstem_intensity_per_cell
            per_cell_df["brainstem_average_intensity_in_matched_area"] = avg_brainstem_intensity_per_cell
            per_cell_df["brainstem_n_segmented_cells"] = int(brainstem_cell_df["cell_id"].nunique()) if not brainstem_cell_df.empty else 0
            per_cell_tables.append(per_cell_df)
            for _, row in per_cell_df.iterrows():
                point_plot_rows.append(
                    {
                        "condition": cereb_key.condition,
                        "region": "cereb",
                        "image_uid": image_uid(cereb_key),
                        "cell_id": int(row["cell_id"]),
                        "value": float(row["pcp2_spot_count"]),
                    }
                )
                point_plot_intensity_rows.append(
                    {
                        "condition": cereb_key.condition,
                        "region": "cereb",
                        "image_uid": image_uid(cereb_key),
                        "cell_id": int(row["cell_id"]),
                        "value": float(row["pcp2_intensity_sum"]),
                    }
                )
                point_plot_intensity_rows.append(
                    {
                        "condition": cereb_key.condition,
                        "region": "brainstem",
                        "image_uid": image_uid(cereb_key),
                        "cell_id": int(row["cell_id"]),
                        "value": float(row["brainstem_average_intensity_per_segmented_cell"]),
                    }
                )
        cereb_area = int(cereb_mask.sum())
        brain_area = int(np.prod(pcp2_brainstem.shape))
        cereb_avg_spots_per_cell = float(per_cell_df["pcp2_spot_count"].mean()) if not per_cell_df.empty else np.nan
        cereb_avg_intensity_per_cell = float(per_cell_df["pcp2_intensity_sum"].mean()) if not per_cell_df.empty else np.nan
        n_cereb_cells = int(per_cell_df["cell_id"].nunique()) if not per_cell_df.empty else 0
        n_brainstem_cells = int(brainstem_cell_df["cell_id"].nunique()) if not brainstem_cell_df.empty else 0
        per_image_rows.extend(
            [
                {
                    "condition": cereb_key.condition,
                    "sample": cereb_key.sample,
                    "concentration": cereb_key.concentration,
                    "slice_id": cereb_key.slice_id,
                    "rep": cereb_key.rep,
                    "image_uid": image_uid(cereb_key),
                    "region": "cereb",
                    "area_px": cereb_area,
                    "spot_count": int(len(cereb_spots)),
                    "spots_per_100k_px": float(len(cereb_spots) / cereb_area * 100000) if cereb_area else np.nan,
                    "n_segmented_purkinje_cells": n_cereb_cells,
                    "n_segmented_cells": n_cereb_cells,
                    "avg_spots_per_segmented_cell": cereb_avg_spots_per_cell,
                    "avg_intensity_per_segmented_cell": cereb_avg_intensity_per_cell,
                },
                {
                    "condition": cereb_key.condition,
                    "sample": cereb_key.sample,
                    "concentration": cereb_key.concentration,
                    "slice_id": cereb_key.slice_id,
                    "rep": cereb_key.rep,
                    "image_uid": image_uid(brainstem_key),
                    "region": "brainstem",
                    "area_px": brain_area,
                    "spot_count": int(len(brainstem_spots)),
                    "spots_per_100k_px": float(len(brainstem_spots) / brain_area * 100000) if brain_area else np.nan,
                    "n_segmented_purkinje_cells": np.nan,
                    "n_segmented_cells": n_brainstem_cells,
                    "avg_spots_per_segmented_cell": avg_brainstem_spots_per_cell,
                    "avg_intensity_per_segmented_cell": avg_brainstem_intensity_per_cell,
                    "brainstem_region_definition": "random matched-size regions using Purkinje-cell radii from paired cerebellar FOV",
                },
            ]
        )
        cereb_density = float(len(cereb_spots) / cereb_area * 100000) if cereb_area else np.nan
        brain_density = float(len(brainstem_spots) / brain_area * 100000) if brain_area else np.nan
        pair_rows.append(
            {
                "condition": cereb_key.condition,
                "sample": cereb_key.sample,
                "concentration": cereb_key.concentration,
                "slice_id": cereb_key.slice_id,
                "rep": cereb_key.rep,
                "pair_uid": f"{cereb_key.condition}_slice{cereb_key.slice_id}rep{cereb_key.rep}",
                "purkinje_spots_per_100k_px": cereb_density,
                "brainstem_spots_per_100k_px": brain_density,
                "purkinje_enrichment_vs_brainstem": cereb_density / brain_density if np.isfinite(brain_density) and brain_density > 0 else np.nan,
                "avg_purkinje_spots_per_cell": cereb_avg_spots_per_cell,
                "avg_brainstem_spots_per_cell": avg_brainstem_spots_per_cell,
                "purkinje_to_brainstem_spot_ratio": (
                    cereb_avg_spots_per_cell / avg_brainstem_spots_per_cell
                    if np.isfinite(avg_brainstem_spots_per_cell) and avg_brainstem_spots_per_cell > 0
                    else np.nan
                ),
            }
        )
    extra_brainstem_keys = sorted(
        {key for key, channel in manifest if key.region == "brainstem" and channel == "pcp2"},
        key=lambda k: (CONDITION_ORDER.index(k.condition), int(k.slice_id), int(k.rep)),
    )
    for brainstem_key in extra_brainstem_keys:
        if image_uid(brainstem_key) in processed_brainstem_uids:
            continue
        group_key = (
            brainstem_key.condition,
            brainstem_key.sample,
            brainstem_key.concentration,
            biological_slice_id(brainstem_key.slice_id),
            brainstem_key.rep,
        )
        radius_chunks = radii_by_group.get(group_key, [])
        pcp2_brainstem = load_image(manifest, brainstem_key, "pcp2")
        pcp2_brainstem_raw = load_image_raw(manifest, brainstem_key, "pcp2")
        brainstem_spots, brainstem_pcp2_bs = call_pcp2_spots_16bit_sensitive(pcp2_brainstem_raw)
        brainstem_labels = np.zeros_like(brainstem_pcp2_bs, dtype=np.int32)
        brainstem_cell_df = pd.DataFrame()
        avg_brainstem_spots_per_cell = np.nan
        avg_brainstem_intensity_per_cell = np.nan
        if radius_chunks:
            repeated_radii = np.repeat(
                np.concatenate(radius_chunks).astype(np.float32),
                PCP2_BRAINSTEM_RANDOM_REGION_MULTIPLIER,
            )
            brainstem_labels, _, brainstem_cell_df = sample_random_brainstem_regions(
                brainstem_pcp2_bs.shape,
                repeated_radii,
                seed_key=image_uid(brainstem_key),
            )
        if not brainstem_cell_df.empty:
            brainstem_cell_spot_counts = np.zeros(int(brainstem_labels.max()) + 1, dtype=np.int32)
            for r, c in brainstem_spots:
                label_id = int(brainstem_labels[int(r), int(c)])
                if label_id > 0:
                    brainstem_cell_spot_counts[label_id] += 1
            brainstem_cell_df = brainstem_cell_df.copy()
            brainstem_cell_df["condition"] = brainstem_key.condition
            brainstem_cell_df["sample"] = brainstem_key.sample
            brainstem_cell_df["concentration"] = brainstem_key.concentration
            brainstem_cell_df["slice_id"] = brainstem_key.slice_id
            brainstem_cell_df["rep"] = brainstem_key.rep
            brainstem_cell_df["image_uid"] = image_uid(brainstem_key)
            brainstem_cell_df["region"] = "brainstem"
            brainstem_cell_df["brainstem_region_definition"] = "random matched-size regions using Purkinje-cell radii from all cerebellar FOVs in the same biological slice and replicate"
            brainstem_cell_df["pcp2_spot_count"] = brainstem_cell_df["cell_id"].map(lambda x: int(brainstem_cell_spot_counts[int(x)]))
            brainstem_cell_df["mean_pcp2_intensity"] = brainstem_cell_df["cell_id"].map(
                lambda x: float(brainstem_pcp2_bs[brainstem_labels == int(x)].mean()) if np.any(brainstem_labels == int(x)) else np.nan
            )
            brainstem_cell_df["pcp2_intensity_sum"] = brainstem_cell_df["cell_id"].map(
                lambda x: float(brainstem_pcp2_bs[brainstem_labels == int(x)].sum()) if np.any(brainstem_labels == int(x)) else np.nan
            )
            avg_brainstem_spots_per_cell = float(brainstem_cell_df["pcp2_spot_count"].mean())
            avg_brainstem_intensity_per_cell = float(brainstem_cell_df["pcp2_intensity_sum"].mean())
            brainstem_cell_tables.append(brainstem_cell_df)
            for _, row in brainstem_cell_df.iterrows():
                point_plot_rows.append(
                    {
                        "condition": brainstem_key.condition,
                        "region": "brainstem",
                        "image_uid": image_uid(brainstem_key),
                        "cell_id": int(row["cell_id"]),
                        "value": float(row["pcp2_spot_count"]),
                    }
                )
                point_plot_intensity_rows.append(
                    {
                        "condition": brainstem_key.condition,
                        "region": "brainstem",
                        "image_uid": image_uid(brainstem_key),
                        "cell_id": int(row["cell_id"]),
                        "value": float(row["pcp2_intensity_sum"]),
                    }
                )
        brain_area = int(np.prod(pcp2_brainstem.shape))
        n_brainstem_cells = int(brainstem_cell_df["cell_id"].nunique()) if not brainstem_cell_df.empty else 0
        per_image_rows.append(
            {
                "condition": brainstem_key.condition,
                "sample": brainstem_key.sample,
                "concentration": brainstem_key.concentration,
                "slice_id": brainstem_key.slice_id,
                "rep": brainstem_key.rep,
                "image_uid": image_uid(brainstem_key),
                "region": "brainstem",
                "area_px": brain_area,
                "spot_count": int(len(brainstem_spots)),
                "spots_per_100k_px": float(len(brainstem_spots) / brain_area * 100000) if brain_area else np.nan,
                "n_segmented_purkinje_cells": np.nan,
                "n_segmented_cells": n_brainstem_cells,
                "avg_spots_per_segmented_cell": avg_brainstem_spots_per_cell,
                "avg_intensity_per_segmented_cell": avg_brainstem_intensity_per_cell,
                "brainstem_region_definition": "random matched-size regions using Purkinje-cell radii from all cerebellar FOVs in the same biological slice and replicate",
            }
        )
    per_image_df = pd.DataFrame(per_image_rows)
    pair_df = pd.DataFrame(pair_rows)
    per_cell_out_df = pd.concat(per_cell_tables, ignore_index=True) if per_cell_tables else pd.DataFrame()
    brainstem_cell_out_df = pd.concat(brainstem_cell_tables, ignore_index=True) if brainstem_cell_tables else pd.DataFrame()
    point_plot_df = pd.DataFrame(point_plot_rows)
    point_plot_intensity_df = pd.DataFrame(point_plot_intensity_rows)
    pooled_rows = []
    enrichment_rows = []
    for cond in CONDITION_ORDER:
        cereb = per_image_df[(per_image_df["condition"] == cond) & (per_image_df["region"] == "cereb")]
        brain = per_image_df[(per_image_df["condition"] == cond) & (per_image_df["region"] == "brainstem")]
        cereb_area = int(cereb["area_px"].sum())
        brain_area = int(brain["area_px"].sum())
        cereb_spots = int(cereb["spot_count"].sum())
        brain_spots = int(brain["spot_count"].sum())
        cereb_density = cereb_spots / cereb_area * 100000 if cereb_area else np.nan
        brain_density = brain_spots / brain_area * 100000 if brain_area else np.nan
        pooled_rows.extend(
            [
                {"condition": cond, "region": "cereb", "total_area_px": cereb_area, "total_spots": cereb_spots, "spots_per_100k_px": cereb_density},
                {"condition": cond, "region": "brainstem", "total_area_px": brain_area, "total_spots": brain_spots, "spots_per_100k_px": brain_density},
            ]
        )
        enrichment_rows.append(
            {
                "condition": cond,
                "n_cereb_images": int(len(cereb)),
                "n_brainstem_images": int(len(brain)),
                "total_purkinje_spots": cereb_spots,
                "total_brainstem_spots": brain_spots,
                "pooled_purkinje_spots_per_100k_px": cereb_density,
                "pooled_brainstem_spots_per_100k_px": brain_density,
                "pooled_purkinje_enrichment_vs_brainstem": cereb_density / brain_density if np.isfinite(brain_density) and brain_density > 0 else np.nan,
                "pooled_brainstem_to_purkinje_ratio": brain_density / cereb_density if np.isfinite(cereb_density) and cereb_density > 0 else np.nan,
            }
        )
    pooled_df = pd.DataFrame(pooled_rows)
    enrichment_df = pd.DataFrame(enrichment_rows)
    per_image_df.to_csv(OUTPUT_DIR / "pcp2_pooled_per_image.csv", index=False)
    pair_df.to_csv(OUTPUT_DIR / "pcp2_pooled_per_pair.csv", index=False)
    pooled_df.to_csv(OUTPUT_DIR / "pcp2_pooled_condition_region_summary.csv", index=False)
    enrichment_df.to_csv(OUTPUT_DIR / "pcp2_pooled_condition_enrichment_summary.csv", index=False)
    if not per_cell_out_df.empty:
        per_cell_out_df.to_csv(OUTPUT_DIR / "pcp2_pooled_segmented_cells.csv", index=False)
    if not brainstem_cell_out_df.empty:
        brainstem_cell_out_df.to_csv(OUTPUT_DIR / "pcp2_pooled_brainstem_segmented_cells.csv", index=False)
    if not point_plot_df.empty:
        point_plot_df.to_csv(OUTPUT_DIR / "pcp2_pooled_per_cell_vs_matched_brainstem.csv", index=False)
        plot_condition_region_point_barplot(
            point_plot_df,
            value_col="value",
            filename="pcp2_pooled_density_by_region_barplot.png",
            ylabel="Spots per Purkinje cell or random matched brainstem region",
            title="PCP2 Per-Cell Cerebellum vs Random Matched Brainstem Regions",
        )
    else:
        plot_condition_region_density(pooled_df, per_image_df, "pcp2_pooled_density_by_region_barplot.png")
    if not point_plot_intensity_df.empty:
        point_plot_intensity_df.to_csv(OUTPUT_DIR / "pcp2_pooled_per_cell_intensity_vs_matched_brainstem.csv", index=False)
        plot_condition_region_point_barplot(
            point_plot_intensity_df,
            value_col="value",
            filename="pcp2_pooled_intensity_by_region_barplot.png",
            ylabel="PCP2 intensity per Purkinje cell or random matched brainstem region",
            title="PCP2 Per-Cell Cerebellum vs Random Matched Brainstem Intensity",
        )
    plot_condition_bar_with_points(
        enrichment_df.rename(columns={"pooled_purkinje_enrichment_vs_brainstem": "value"}),
        pair_df.rename(columns={"purkinje_enrichment_vs_brainstem": "value"}),
        value_col="value",
        ylabel="Pooled Purkinje spot density / pooled brainstem spot density",
        title="PCP2 Pooled Enrichment Across All Slices and Replicates",
        filename="pcp2_pooled_enrichment_barplot.png",
    )
    return per_image_df, pooled_df, enrichment_df


def main() -> None:
    manifest, manifest_df = parse_manifest()
    analyze_malat1_pooled(manifest)
    analyze_pcp2_pooled(manifest)
    summary = {
        "n_files": int(len(manifest_df)),
        "conditions": sorted(manifest_df["condition"].unique().tolist()),
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "run_summary.csv", index=False)


if __name__ == "__main__":
    main()
