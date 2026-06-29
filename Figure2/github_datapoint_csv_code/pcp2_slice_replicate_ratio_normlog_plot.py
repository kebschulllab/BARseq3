from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.feature import blob_log

import images_pooled_barseq_analysis as pooled
import pcp2_slice_replicate_ratio_plot as base_ratio
import replicate_barseq_analysis as base


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs_images_pooled"

LOG_MIN_SIGMA = 1.4
LOG_MAX_SIGMA = 2.2
LOG_NUM_SIGMA = 6
LOG_OVERLAP = 0.5
LOG_RESPONSE_THRESHOLD = 0.012
LOG_SMOOTH_SIGMA = 0.45
NORM_PERCENTILE = 99.99
CEREB_BRIGHTNESS_MIN_NORM = 0.18
BRAINSTEM_BRIGHTNESS_MIN_NORM = 0.0
CEREB_LOG_OVERLAP = 0.35
BRAINSTEM_LOG_OVERLAP = 0.5
CEREB_COUNT_BRIGHTNESS_MIN_NORM = 0.22
CEREB_COUNT_LOG_OVERLAP = 0.25
CEREB_COUNT_LOG_RESPONSE_THRESHOLD = 0.016
CEREB_COUNT_BRIGHTNESS_MIN_RAW16 = 300.0
CEREB_COUNT_LOG_RESPONSE_THRESHOLD_RAW16 = 12.0
PURKINJE_TERRITORY_RADIUS_MULTIPLIER_B2 = 2.0
PURKINJE_TERRITORY_RADIUS_MULTIPLIER_B3 = 4.0
PURKINJE_MIN_ACCEPTED_AREA_PX = 400.0
PURKINJE_MIN_ACCEPTED_CIRCULARITY = 0.8
PURKINJE_MAX_ACCEPTED_AXIS_RATIO = 1.5


def slice_group_key(key: base.ImageKey) -> tuple[str, str, str, str, str]:
    return (
        key.condition,
        key.sample,
        key.concentration,
        pooled.biological_slice_id(key.slice_id),
        str(key.rep),
    )


def build_cereb_normalization_scales(
    manifest: dict[tuple[base.ImageKey, str], Path]
) -> pd.DataFrame:
    groups: dict[tuple[str, str, str, str, str], list[base.ImageKey]] = {}
    for key, channel in manifest:
        if key.region != "cereb" or channel != "pcp2":
            continue
        groups.setdefault(slice_group_key(key), []).append(key)

    rows: list[dict] = []
    for group_key, keys in sorted(groups.items()):
        pixel_blocks = []
        for key in sorted(keys, key=lambda k: int(k.slice_id)):
            raw = pooled.load_image_raw(manifest, key, "pcp2")
            pixel_blocks.append(raw.reshape(-1))
        all_pixels = np.concatenate(pixel_blocks) if pixel_blocks else np.array([], dtype=np.float32)
        q_hi = float(np.percentile(all_pixels, NORM_PERCENTILE)) if all_pixels.size else 1.0
        if not np.isfinite(q_hi) or q_hi <= 0:
            q_hi = 1.0
        rows.append(
            {
                "condition": group_key[0],
                "sample": group_key[1],
                "concentration": group_key[2],
                "biological_slice_id": group_key[3],
                "rep": group_key[4],
                "normalization_min": 0.0,
                "normalization_max_q99_99": q_hi,
                "source_region": "cereb",
                "source_n_images": len(keys),
                "source_slice_ids": ",".join(sorted({str(k.slice_id) for k in keys})),
            }
        )
    return pd.DataFrame(rows)


def normalize_from_slice_scale(image_raw: np.ndarray, scale_hi: float) -> np.ndarray:
    image_float = image_raw.astype(np.float32, copy=False)
    if not np.isfinite(scale_hi) or scale_hi <= 0:
        return np.zeros_like(image_float, dtype=np.float32)
    return np.clip(image_float / float(scale_hi), 0.0, 1.0).astype(np.float32, copy=False)


def blob_log_spots_normalized(
    image_raw: np.ndarray,
    scale_hi: float,
    brightness_min: float,
    overlap: float,
    response_threshold: float = LOG_RESPONSE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    image_norm = normalize_from_slice_scale(image_raw, scale_hi)
    image_bs = base.remove_background(image_norm, sigma=20)
    smooth = base.filters.gaussian(np.clip(image_bs, 0, None), sigma=LOG_SMOOTH_SIGMA, preserve_range=True)
    blobs = blob_log(
        smooth,
        min_sigma=LOG_MIN_SIGMA,
        max_sigma=LOG_MAX_SIGMA,
        num_sigma=LOG_NUM_SIGMA,
        threshold=response_threshold,
        overlap=overlap,
        exclude_border=False,
    )
    coords = np.rint(blobs[:, :2]).astype(np.int32) if len(blobs) else np.empty((0, 2), dtype=np.int32)
    if len(coords):
        h, w = smooth.shape
        keep = (coords[:, 0] >= 0) & (coords[:, 0] < h) & (coords[:, 1] >= 0) & (coords[:, 1] < w)
        coords = coords[keep]
        if len(coords):
            keep = image_bs[coords[:, 0], coords[:, 1]] >= brightness_min
            coords = coords[keep]
            coords = np.unique(coords, axis=0)
    return coords, image_bs


def blob_log_spots_raw16_count_cereb(
    image_raw: np.ndarray,
    brightness_min: float = CEREB_COUNT_BRIGHTNESS_MIN_RAW16,
    overlap: float = CEREB_COUNT_LOG_OVERLAP,
    response_threshold: float = CEREB_COUNT_LOG_RESPONSE_THRESHOLD_RAW16,
) -> tuple[np.ndarray, np.ndarray]:
    image_float = image_raw.astype(np.float32, copy=False)
    image_bs = base.remove_background(image_float, sigma=20)
    smooth = base.filters.gaussian(np.clip(image_bs, 0, None), sigma=LOG_SMOOTH_SIGMA, preserve_range=True)
    blobs = blob_log(
        smooth,
        min_sigma=LOG_MIN_SIGMA,
        max_sigma=LOG_MAX_SIGMA,
        num_sigma=LOG_NUM_SIGMA,
        threshold=response_threshold,
        overlap=overlap,
        exclude_border=False,
    )
    coords = np.rint(blobs[:, :2]).astype(np.int32) if len(blobs) else np.empty((0, 2), dtype=np.int32)
    if len(coords):
        h, w = smooth.shape
        keep = (coords[:, 0] >= 0) & (coords[:, 0] < h) & (coords[:, 1] >= 0) & (coords[:, 1] < w)
        coords = coords[keep]
        if len(coords):
            keep = image_bs[coords[:, 0], coords[:, 1]] >= brightness_min
            coords = coords[keep]
            coords = np.unique(coords, axis=0)
    return coords, image_bs


def barseq_style_spots_raw16_count_cereb(image_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords, image_bs = pooled.call_pcp2_spots_16bit(image_raw)
    if len(coords):
        keep = image_bs[coords[:, 0], coords[:, 1]] >= CEREB_COUNT_BRIGHTNESS_MIN_RAW16
        coords = coords[keep]
    return coords, image_bs


def purkinje_territory_radius_multiplier_for_key(key: base.ImageKey) -> float:
    return PURKINJE_TERRITORY_RADIUS_MULTIPLIER_B2 if key.sample == "b2" else PURKINJE_TERRITORY_RADIUS_MULTIPLIER_B3


def filter_accepted_purkinje_cells(cell_df: pd.DataFrame) -> pd.DataFrame:
    if cell_df.empty:
        return cell_df.copy()
    keep = (
        (cell_df["area_px"] >= PURKINJE_MIN_ACCEPTED_AREA_PX)
        & (cell_df["circularity"] >= PURKINJE_MIN_ACCEPTED_CIRCULARITY)
        & (cell_df["axis_ratio"] <= PURKINJE_MAX_ACCEPTED_AXIS_RATIO)
    )
    return cell_df.loc[keep].copy()


def assign_spots_to_expanded_purkinje_territories(
    image_shape: tuple[int, int],
    cell_df: pd.DataFrame,
    count_spots: np.ndarray,
    radius_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    if cell_df.empty:
        return np.zeros(image_shape, dtype=np.int32), np.zeros(1, dtype=np.int32)

    territory_labels = np.zeros(image_shape, dtype=np.int32)
    dist_best = np.full(image_shape, np.inf, dtype=np.float32)
    for row in cell_df.itertuples(index=False):
        rr, cc = pooled.draw.disk(
            (float(row.centroid_row), float(row.centroid_col)),
            float(row.disk_radius_px) * radius_multiplier,
            shape=image_shape,
        )
        if len(rr) == 0:
            continue
        dist = ((rr - float(row.centroid_row)) ** 2 + (cc - float(row.centroid_col)) ** 2).astype(np.float32)
        update = dist < dist_best[rr, cc]
        if np.any(update):
            territory_labels[rr[update], cc[update]] = int(row.cell_id)
            dist_best[rr[update], cc[update]] = dist[update]

    cell_counts = np.zeros(int(cell_df["cell_id"].max()) + 1, dtype=np.int32)
    for r, c in count_spots:
        label_id = int(territory_labels[int(r), int(c)])
        if label_id > 0:
            cell_counts[label_id] += 1
    return territory_labels, cell_counts


def collect_cereb_per_fov(
    manifest: dict[tuple[base.ImageKey, str], Path],
    scale_map: dict[tuple[str, str, str, str, str], float],
) -> pd.DataFrame:
    keys = sorted(
        {key for key, channel in manifest if key.region == "cereb" and channel == "pcp2"},
        key=lambda k: (base_ratio.CONDITION_ORDER.index(k.condition), int(k.slice_id), int(k.rep)),
    )
    rows: list[dict] = []
    for key in keys:
        scale_hi = scale_map[slice_group_key(key)]
        dapi = pooled.load_image(manifest, key, "dapi")
        pcp2_raw = pooled.load_image_raw(manifest, key, "pcp2")
        seg_spots, pcp2_bs = blob_log_spots_normalized(
            pcp2_raw,
            scale_hi=scale_hi,
            brightness_min=CEREB_BRIGHTNESS_MIN_NORM,
            overlap=CEREB_LOG_OVERLAP,
        )
        count_spots, _ = barseq_style_spots_raw16_count_cereb(pcp2_raw)
        labels, mask, cell_df = pooled.segment_pcp2_purkinje_fast(
            pcp2_bs=pcp2_bs,
            dapi=dapi,
            spots=seg_spots,
            sample=key.sample,
        )
        cell_df = filter_accepted_purkinje_cells(cell_df)
        count_spots = base.filter_spots(count_spots, mask)
        if cell_df.empty or labels.max() <= 0:
            avg_spots = np.nan
            avg_area = np.nan
            n_cells = 0
        else:
            _, cell_counts = assign_spots_to_expanded_purkinje_territories(
                labels.shape,
                cell_df,
                count_spots,
                radius_multiplier=purkinje_territory_radius_multiplier_for_key(key),
            )
            cell_df = cell_df.copy()
            cell_df["pcp2_spot_count"] = cell_df["cell_id"].map(lambda x: int(cell_counts[int(x)]))
            avg_spots = float(cell_df["pcp2_spot_count"].mean()) if len(cell_df) else np.nan
            avg_area = float(cell_df["area_px"].mean()) if len(cell_df) else np.nan
            n_cells = int(cell_df["cell_id"].nunique())

        rows.append(
            {
                "condition": key.condition,
                "sample": key.sample,
                "concentration": key.concentration,
                "slice_id": key.slice_id,
                "rep": key.rep,
                "image_uid": pooled.image_uid(key),
                "biological_slice_id": pooled.biological_slice_id(key.slice_id),
                "fov_id": str(key.slice_id)[1],
                "n_purkinje_cells": n_cells,
                "avg_purkinje_cell_area_px": avg_area,
                "avg_purkinje_spots_per_cell": avg_spots,
                "cereb_spot_count_total": int(len(count_spots)),
                "pcp2_spot_threshold_min_intensity": 0.0,
                "pcp2_spot_radius_range_px": "4-6 diameter",
                "pcp2_spot_calling_method": "BARseq-style raw16 bandpass + peak_local_max count caller for cerebellum plus normalized blob_log seed caller and condition-specific expanded Purkinje territories",
                "normalization_min": 0.0,
                "normalization_max_q99_99": scale_hi,
            }
        )
    return pd.DataFrame(rows)


def collect_brainstem_per_fov(
    manifest: dict[tuple[base.ImageKey, str], Path],
    scale_map: dict[tuple[str, str, str, str, str], float],
) -> pd.DataFrame:
    keys = sorted(
        {key for key, channel in manifest if key.region == "brainstem" and channel == "pcp2"},
        key=lambda k: (base_ratio.CONDITION_ORDER.index(k.condition), int(k.slice_id), int(k.rep)),
    )
    rows: list[dict] = []
    for key in keys:
        scale_hi = scale_map[slice_group_key(key)]
        pcp2_raw = pooled.load_image_raw(manifest, key, "pcp2")
        count_spots, _ = blob_log_spots_normalized(
            pcp2_raw,
            scale_hi=scale_hi,
            brightness_min=BRAINSTEM_BRIGHTNESS_MIN_NORM,
            overlap=BRAINSTEM_LOG_OVERLAP,
        )
        rows.append(
            {
                "condition": key.condition,
                "sample": key.sample,
                "concentration": key.concentration,
                "slice_id": key.slice_id,
                "rep": key.rep,
                "image_uid": pooled.image_uid(key),
                "biological_slice_id": pooled.biological_slice_id(key.slice_id),
                "fov_id": str(key.slice_id)[1],
                "brainstem_area_px": int(pcp2_raw.shape[0] * pcp2_raw.shape[1]),
                "brainstem_spot_count": int(len(count_spots)),
                "pcp2_spot_threshold_min_intensity": BRAINSTEM_BRIGHTNESS_MIN_NORM,
                "pcp2_spot_radius_range_px": "4-6 diameter",
                "pcp2_spot_calling_method": "blob_log on per-slice q99.99 cerebellum-normalized PCP2",
                "normalization_min": 0.0,
                "normalization_max_q99_99": scale_hi,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    manifest, _ = pooled.parse_manifest()
    scale_df = build_cereb_normalization_scales(manifest)
    scale_map = {
        (row.condition, row.sample, row.concentration, str(row.biological_slice_id), str(row.rep)): float(row.normalization_max_q99_99)
        for row in scale_df.itertuples(index=False)
    }

    cereb_per_fov = collect_cereb_per_fov(manifest, scale_map)
    brainstem_per_fov = collect_brainstem_per_fov(manifest, scale_map)
    slice_rep_df = base_ratio.build_slice_replicate_table(cereb_per_fov, brainstem_per_fov)
    slice_rep_df["cereb_spot_threshold_min_intensity"] = 0.0
    slice_rep_df["brainstem_spot_threshold_min_intensity"] = BRAINSTEM_BRIGHTNESS_MIN_NORM
    slice_rep_df["pcp2_spot_radius_range_px"] = "4-6 diameter"
    slice_rep_df["pcp2_normalization_method"] = (
        "per biological slice+rep: min=0, max=q99.99 from cerebellum PCP2 images, "
        "applied to cerebellum and brainstem PCP2 before LoG spot calling"
    )
    slice_rep_df["pcp2_spot_calling_method"] = (
        f"blob_log on per-slice cerebellum-normalized PCP2, min_sigma={LOG_MIN_SIGMA}, "
        f"max_sigma={LOG_MAX_SIGMA}, num_sigma={LOG_NUM_SIGMA}, seed_response_threshold={LOG_RESPONSE_THRESHOLD}, "
        f"cereb_seed_brightness>={CEREB_BRIGHTNESS_MIN_NORM}, cereb_seed_overlap={CEREB_LOG_OVERLAP}, "
        f"cereb_count=BARseq-style raw16 bandpass + peak_local_max caller with raw cutoff {CEREB_COUNT_BRIGHTNESS_MIN_RAW16:g}, "
        f"spots assigned to expanded Purkinje territories x{PURKINJE_TERRITORY_RADIUS_MULTIPLIER_B2:g} for b2 and x{PURKINJE_TERRITORY_RADIUS_MULTIPLIER_B3:g} for b3, "
        f"brainstem_brightness>={BRAINSTEM_BRIGHTNESS_MIN_NORM}, brainstem_overlap={BRAINSTEM_LOG_OVERLAP}, q={NORM_PERCENTILE}"
    )
    stats_df = base_ratio.pairwise_stats(slice_rep_df)

    scale_df.to_csv(OUTPUT_DIR / "pcp2_slice_replicate_normlog_normalization_scales.csv", index=False)
    cereb_per_fov.to_csv(OUTPUT_DIR / "pcp2_slice_replicate_normlog_cereb_per_fov.csv", index=False)
    brainstem_per_fov.to_csv(OUTPUT_DIR / "pcp2_slice_replicate_normlog_brainstem_per_fov.csv", index=False)
    slice_rep_df.to_csv(OUTPUT_DIR / "pcp2_slice_replicate_normlog_ratio_datapoints.csv", index=False)
    stats_df.to_csv(OUTPUT_DIR / "pcp2_slice_replicate_normlog_ratio_pairwise_stats.csv", index=False)
    base_ratio.plot_ratio_bars(slice_rep_df, stats_df, OUTPUT_DIR / "pcp2_slice_replicate_normlog_ratio_barplots.png")


if __name__ == "__main__":
    main()
