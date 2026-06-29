from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import exposure, feature, filters, measure, morphology, segmentation, transform
from skimage.registration import phase_cross_correlation


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "analysis_tifffiles_replicates"
OUTPUT_DIR = ROOT / "analysis_outputs_replicates"
OUTPUT_DIR.mkdir(exist_ok=True)
ROI_MASK_DIR = ROOT / "analysis_tifffiles_replicates_roi_masks"
ROI_MASK_DIR.mkdir(exist_ok=True)

FILE_RE = re.compile(
    r"^(?P<channel>dapi|nissl|malat1|pcp2)_(?P<sample>b2|b3)_(?P<region>cereb|brainstem)"
    r"(?:_(?P<conc>[^_]+))?_slice(?P<slice>\d+)rep(?P<rep>\d+)\.tif$"
)

MALAT1_INTENSITY_RANGE = (150.0, 2500.0)
MALAT1_NUCLEUS_EXPANSION_PX = 6
PCP2_INTENSITY_RANGE = (150.0, 1000.0)
PCP2_SPOT_THRESHOLD = 200.0
PCP2_MIN_DISTANCE = 2
PCP2_SPOT_SMOOTH_SIGMA = 0.6
PCP2_SPOT_CANDIDATE_THRESHOLD_FRACTION = 0.55
PURKINJE_SEED_THRESHOLD_B2 = 150.0
PURKINJE_SEED_THRESHOLD_B3 = 220.0
PURKINJE_TARGET_DIAMETER_PX = 75.0
PURKINJE_MIN_EQ_DIAMETER_PX = 50.0
PURKINJE_MAX_EQ_DIAMETER_PX = 100.0
PURKINJE_MIN_AREA_PX = 400
PURKINJE_MERGE_DILATION_PX = 2
PURKINJE_MERGE_CLOSE_PX = 4
PURKINJE_MIN_SOLIDITY = 0.45
PURKINJE_WATERSHED_MIN_DISTANCE = 22
PURKINJE_SUPPORT_THRESHOLD_FRACTION = 0.55
PURKINJE_NUCLEUS_CAPTURE_RADIUS_PX = 40
PURKINJE_NUCLEUS_EXPANSION_PX = 30
PURKINJE_LOCAL_SIGNAL_THRESHOLD_FRACTION = 0.45
PURKINJE_LINE_GROUP_RADIUS_PX = 90
PURKINJE_LINE_MIN_COMPONENT_SIZE = 3
TILE_HEIGHT = 200
TILE_WIDTH = 200
TILE_AREA_PX = TILE_HEIGHT * TILE_WIDTH

_CELLPOSE_MODEL = None


@dataclass(frozen=True)
class ImageKey:
    sample: str
    region: str
    concentration: str
    slice_id: str
    rep: str

    @property
    def condition(self) -> str:
        if self.sample == "b2":
            return "b2"
        return f"{self.sample}_{self.concentration}"

    @property
    def slug(self) -> str:
        return f"{self.condition}_slice{self.slice_id}rep{self.rep}_{self.region}"


def get_cellpose_model():
    global _CELLPOSE_MODEL
    if _CELLPOSE_MODEL is None:
        try:
            from numba.core.dispatcher import Dispatcher

            if not hasattr(Dispatcher, "_codex_cache_patch"):
                Dispatcher.enable_caching = lambda self: None
                Dispatcher._codex_cache_patch = True
        except Exception:
            pass
        from cellpose import models

        _CELLPOSE_MODEL = models.Cellpose(gpu=False, model_type="cyto")
    return _CELLPOSE_MODEL


def read_image(path: Path) -> np.ndarray:
    import tifffile

    return tifffile.imread(path).astype(np.float32)


def percentile_rescale(image: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    lo, hi = np.percentile(image, [low, high])
    if hi <= lo:
        hi = lo + 1
    return exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0, 1))


def fixed_range_rescale(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((image - lo) / (hi - lo), 0, 1)


def remove_background(image: np.ndarray, sigma: float) -> np.ndarray:
    background = filters.gaussian(image, sigma=sigma, preserve_range=True)
    return np.clip(image - background, 0, None)


def get_purkinje_seed_threshold(sample: str, concentration: str) -> float:
    if sample == "b2":
        return PURKINJE_SEED_THRESHOLD_B2
    return PURKINJE_SEED_THRESHOLD_B3


def segment_nuclei(dapi: np.ndarray) -> np.ndarray:
    dapi_bs = remove_background(dapi, sigma=18)
    smooth = filters.gaussian(dapi_bs, sigma=1.4, preserve_range=True)
    high = float(filters.threshold_otsu(smooth))
    low = max(high * 0.4, float(np.percentile(smooth[smooth > 0], 45)) if np.any(smooth > 0) else 0.0)
    binary = filters.apply_hysteresis_threshold(smooth, low, high)
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.binary_opening(binary, morphology.disk(1))
    binary = morphology.remove_small_objects(binary, 60)
    binary = morphology.remove_small_holes(binary, 80)
    distance = ndi.distance_transform_edt(binary)
    maxima = feature.peak_local_max(distance, min_distance=10, labels=binary, exclude_border=False)
    markers = np.zeros(binary.shape, dtype=np.int32)
    if len(maxima):
        markers[tuple(maxima.T)] = np.arange(1, len(maxima) + 1)
    else:
        seed_mask = smooth > high
        seed_mask = morphology.remove_small_objects(seed_mask, 20)
        markers = measure.label(seed_mask).astype(np.int32)
        return segmentation.watershed(-distance, markers, mask=binary)
    markers = ndi.label(markers > 0)[0]
    return segmentation.watershed(-distance, markers, mask=binary)


def rigid_transform(image: np.ndarray, angle: float, shift_rc: tuple[float, float], order: int = 1) -> np.ndarray:
    rotated = transform.rotate(image, angle=angle, resize=False, preserve_range=True, mode="constant", cval=0)
    shifted = ndi.shift(rotated, shift=shift_rc, order=order, mode="constant", cval=0)
    return shifted.astype(np.float32)


def parse_manifest() -> tuple[dict[tuple[ImageKey, str], Path], pd.DataFrame]:
    manifest: dict[tuple[ImageKey, str], Path] = {}
    rows: list[dict] = []
    for path in sorted(INPUT_DIR.iterdir()):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        gd = match.groupdict()
        key = ImageKey(
            sample=gd["sample"],
            region=gd["region"],
            concentration=gd["conc"] or "base",
            slice_id=gd["slice"],
            rep=gd["rep"],
        )
        manifest[(key, gd["channel"])] = path
        rows.append(
            {
                "path": str(path),
                "channel": gd["channel"],
                "sample": key.sample,
                "region": key.region,
                "concentration": key.concentration,
                "slice_id": key.slice_id,
                "rep": key.rep,
                "condition": key.condition,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "replicate_manifest.csv", index=False)
    return manifest, df


def load_image(manifest: dict[tuple[ImageKey, str], Path], key: ImageKey, channel: str) -> np.ndarray:
    path = manifest[(key, channel)]
    return read_image(path)


def select_sparse_roi(dapi: np.ndarray, roi_size: int = 384, stride: int = 128) -> tuple[int, int, int]:
    nuclei_labels = segment_nuclei(dapi)
    props = measure.regionprops(nuclei_labels)
    centroids = np.array([p.centroid for p in props], dtype=np.float32) if props else np.empty((0, 2), dtype=np.float32)
    best = None
    fallback = None
    h, w = dapi.shape
    for r0 in range(0, h - roi_size + 1, stride):
        for c0 in range(0, w - roi_size + 1, stride):
            if len(centroids):
                inside = (
                    (centroids[:, 0] >= r0)
                    & (centroids[:, 0] < r0 + roi_size)
                    & (centroids[:, 1] >= c0)
                    & (centroids[:, 1] < c0 + roi_size)
                )
                count = int(inside.sum())
            else:
                count = 0
            local_mask = nuclei_labels[r0 : r0 + roi_size, c0 : c0 + roi_size] > 0
            occupancy = float(local_mask.mean())
            center_penalty = abs((r0 + roi_size / 2) - h / 2) / h + abs((c0 + roi_size / 2) - w / 2) / w
            fallback_score = count + 400.0 * max(0.0, 0.03 - occupancy) + 4.0 * center_penalty
            if fallback is None or fallback_score < fallback[0]:
                fallback = (fallback_score, r0, c0, count)
            if count < 16 or count > 44 or occupancy < 0.035:
                continue
            density_penalty = abs(count - 28)
            occupancy_penalty = 80.0 * max(0.0, occupancy - 0.07)
            score = density_penalty + occupancy_penalty + 3.0 * center_penalty
            if best is None or score < best[0]:
                best = (score, r0, c0, count)
    chosen = best if best is not None else fallback
    if chosen is None:
        return ((h - roi_size) // 2, (w - roi_size) // 2, roi_size)
    return (int(chosen[1]), int(chosen[2]), roi_size)


def crop(image: np.ndarray, roi: tuple[int, int, int]) -> np.ndarray:
    r0, c0, size = roi
    return image[r0 : r0 + size, c0 : c0 + size]


def unit_scale(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    lo = float(np.percentile(image, 5))
    hi = float(np.percentile(image, 99.5))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0, 1).astype(np.float32)


def rough_nissl_cell_mask(nissl: np.ndarray) -> np.ndarray:
    nissl_bs = remove_background(nissl, sigma=18)
    smooth = filters.gaussian(nissl_bs, sigma=1.6, preserve_range=True)
    threshold = np.percentile(smooth, 70)
    mask = smooth > threshold
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = morphology.binary_opening(mask, morphology.disk(1))
    mask = morphology.remove_small_objects(mask, 100)
    mask = morphology.remove_small_holes(mask, 100)
    return mask


def build_malat1_alignment_target(dapi_roi: np.ndarray, nissl_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nuclei_labels = segment_nuclei(dapi_roi)
    nuclei_mask = nuclei_labels > 0
    expanded_nuclei = morphology.binary_dilation(nuclei_mask, morphology.disk(MALAT1_NUCLEUS_EXPANSION_PX))
    cell_mask = rough_nissl_cell_mask(nissl_roi)
    combined_cell_mask = cell_mask | expanded_nuclei
    dapi_feat = unit_scale(filters.gaussian(remove_background(dapi_roi, sigma=14), sigma=1.2, preserve_range=True))
    nissl_body = unit_scale(filters.gaussian(remove_background(nissl_roi, sigma=18), sigma=1.6, preserve_range=True))
    nissl_edge = unit_scale(filters.sobel(nissl_body))
    target = (1.6 * dapi_feat + 0.9 * nissl_body + 1.2 * nissl_edge).astype(np.float32)
    target *= morphology.binary_dilation(combined_cell_mask, morphology.disk(10)).astype(np.float32)
    return target, expanded_nuclei, combined_cell_mask


def score_malat1_alignment(
    aligned_malat_feat: np.ndarray,
    target_feat: np.ndarray,
    nuclei_mask: np.ndarray,
    cell_mask: np.ndarray,
) -> float:
    tissue_mask = morphology.binary_dilation(cell_mask, morphology.disk(10))
    overlap = tissue_mask & (aligned_malat_feat > 0)
    if overlap.sum() < 1500:
        return -np.inf

    a = aligned_malat_feat[overlap]
    b = target_feat[overlap]
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    corr = float((a * b).mean())

    cell_mean = float(aligned_malat_feat[cell_mask].mean()) if cell_mask.any() else 0.0
    outside_mask = tissue_mask & (~cell_mask)
    outside_mean = float(aligned_malat_feat[outside_mask].mean()) if outside_mask.any() else 0.0
    nucleus_mean = float(aligned_malat_feat[nuclei_mask].mean()) if nuclei_mask.any() else 0.0
    cell_enrichment = np.log1p(cell_mean) - np.log1p(outside_mean)
    nucleus_enrichment = np.log1p(nucleus_mean) - np.log1p(outside_mean)
    return float(corr + 0.35 * cell_enrichment + 0.15 * nucleus_enrichment)


def estimate_malat1_transform(dapi_roi: np.ndarray, nissl_roi: np.ndarray, malat1_roi: np.ndarray) -> dict[str, float | tuple[float, float]]:
    target_feat, nuclei_mask, cell_mask = build_malat1_alignment_target(dapi_roi, nissl_roi)
    max_shift_px = 45.0
    scale = 0.5
    target_small = transform.rescale(target_feat, scale, preserve_range=True, anti_aliasing=True).astype(np.float32)
    nuclei_small = transform.rescale(nuclei_mask.astype(np.float32), scale, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    cell_small = transform.rescale(cell_mask.astype(np.float32), scale, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    malat_feat = unit_scale(filters.gaussian(remove_background(malat1_roi, sigma=14), sigma=1.0, preserve_range=True))
    malat_small = transform.rescale(malat_feat, scale, preserve_range=True, anti_aliasing=True).astype(np.float32)

    best = None
    coarse_angles = np.arange(-4.0, 4.01, 0.5)
    for angle in coarse_angles:
        rotated = transform.rotate(malat_small, angle=angle, resize=False, preserve_range=True, mode="constant", cval=0)
        shift, _, _ = phase_cross_correlation(target_small, rotated, upsample_factor=20)
        if np.max(np.abs(shift / scale)) > max_shift_px:
            continue
        aligned = ndi.shift(rotated, shift=shift, order=1, mode="constant", cval=0)
        score = score_malat1_alignment(aligned, target_small, nuclei_small, cell_small)
        score -= 0.0005 * float((shift[0] / scale) ** 2 + (shift[1] / scale) ** 2)
        if best is None or score > best[0]:
            best = (score, float(angle), float(shift[0] / scale), float(shift[1] / scale))

    if best is None:
        return {"angle": 0.0, "shift_rc": (0.0, 0.0), "score": np.nan}

    fine_angles = np.arange(best[1] - 0.6, best[1] + 0.61, 0.1)
    refined = None
    for angle in fine_angles:
        rotated = transform.rotate(malat_feat, angle=angle, resize=False, preserve_range=True, mode="constant", cval=0)
        shift, _, _ = phase_cross_correlation(target_feat, rotated, upsample_factor=25)
        if np.max(np.abs(shift)) > max_shift_px:
            continue
        aligned = ndi.shift(rotated, shift=shift, order=1, mode="constant", cval=0)
        score = score_malat1_alignment(aligned, target_feat, nuclei_mask, cell_mask)
        score -= 0.0005 * float(shift[0] ** 2 + shift[1] ** 2)
        if refined is None or score > refined[0]:
            refined = (score, float(angle), float(shift[0]), float(shift[1]))

    chosen = refined if refined is not None else best

    # Rescue weak fits with a local biologically plausible grid search.
    if (
        chosen is None
        or chosen[0] < 0.15
        or max(abs(chosen[2]), abs(chosen[3])) > 30.0
    ):
        grid_best = None
        angle_center = float(chosen[1]) if chosen is not None else 0.0
        row_center = float(chosen[2]) if chosen is not None else 0.0
        col_center = float(chosen[3]) if chosen is not None else 0.0
        for angle in np.arange(angle_center - 0.8, angle_center + 0.81, 0.2):
            rotated = transform.rotate(malat_feat, angle=angle, resize=False, preserve_range=True, mode="constant", cval=0)
            for dr in np.arange(row_center - 12.0, row_center + 12.1, 4.0):
                for dc in np.arange(col_center - 12.0, col_center + 12.1, 4.0):
                    if max(abs(dr), abs(dc)) > max_shift_px:
                        continue
                    aligned = ndi.shift(rotated, shift=(dr, dc), order=1, mode="constant", cval=0)
                    score = score_malat1_alignment(aligned, target_feat, nuclei_mask, cell_mask)
                    score -= 0.0005 * float(dr ** 2 + dc ** 2)
                    if grid_best is None or score > grid_best[0]:
                        grid_best = (score, float(angle), float(dr), float(dc))
        if grid_best is not None and (chosen is None or grid_best[0] > chosen[0]):
            chosen = grid_best

    if chosen is None:
        return {"angle": 0.0, "shift_rc": (0.0, 0.0), "score": np.nan}

    # Final refinement: choose the transform that most strongly enriches MALAT1
    # inside the segmented nuclei and, secondarily, the surrounding cell territory.
    malat_score = filters.gaussian(remove_background(malat1_roi, sigma=10), sigma=0.8, preserve_range=True).astype(np.float32)
    nuclei_refine = morphology.binary_dilation(nuclei_mask, morphology.disk(2))
    tissue_refine = morphology.binary_dilation(cell_mask, morphology.disk(8))
    outside_refine = tissue_refine & (~cell_mask)

    def direct_alignment_score(angle: float, dr: float, dc: float) -> float:
        aligned = rigid_transform(malat_score, angle=angle, shift_rc=(dr, dc), order=1)
        nuc_mean = float(aligned[nuclei_refine].mean()) if nuclei_refine.any() else 0.0
        cell_mean = float(aligned[cell_mask].mean()) if cell_mask.any() else 0.0
        if outside_refine.any():
            outside_mean = float(aligned[outside_refine].mean())
        else:
            outside_mean = float(aligned[~tissue_refine].mean()) if (~tissue_refine).any() else 0.0
        tissue_vals = aligned[tissue_refine] if tissue_refine.any() else aligned.ravel()
        top_thresh = float(np.percentile(tissue_vals, 99.2)) if tissue_vals.size else 0.0
        top_mask = aligned >= top_thresh
        top_in_nuc = float((top_mask & nuclei_refine).sum()) / max(float(top_mask.sum()), 1.0)
        nucleus_enrichment = np.log1p(nuc_mean) - np.log1p(outside_mean)
        cell_enrichment = np.log1p(cell_mean) - np.log1p(outside_mean)
        shift_penalty = 0.0006 * float(dr ** 2 + dc ** 2)
        angle_penalty = 0.012 * abs(angle)
        return float(1.7 * nucleus_enrichment + 0.45 * cell_enrichment + 0.7 * top_in_nuc - shift_penalty - angle_penalty)

    centers = [(float(chosen[1]), float(chosen[2]), float(chosen[3]))]
    if chosen[0] < 0.35:
        centers.append((0.0, 0.0, 0.0))
    direct_best = None
    for angle_center, row_center, col_center in centers:
        for angle in np.arange(angle_center - 1.0, angle_center + 1.01, 0.2):
            for dr in np.arange(row_center - 8.0, row_center + 8.1, 2.0):
                for dc in np.arange(col_center - 8.0, col_center + 8.1, 2.0):
                    if max(abs(dr), abs(dc)) > max_shift_px:
                        continue
                    score = direct_alignment_score(float(angle), float(dr), float(dc))
                    if direct_best is None or score > direct_best[0]:
                        direct_best = (score, float(angle), float(dr), float(dc))

    if direct_best is not None:
        return {"angle": direct_best[1], "shift_rc": (direct_best[2], direct_best[3]), "score": direct_best[0]}
    return {"angle": chosen[1], "shift_rc": (chosen[2], chosen[3]), "score": chosen[0]}


def segment_cells_cellpose(nissl: np.ndarray, dapi_aligned: np.ndarray, nuclei_labels: np.ndarray) -> np.ndarray:
    # Build a broad soma support mask so lighter Nissl cells are retained in
    # the MALAT1 per-cell analysis instead of disappearing when only the
    # brightest cell bodies are segmented.
    nissl_bs = remove_background(nissl, sigma=18)
    nissl_smooth = filters.gaussian(nissl_bs, sigma=1.8, preserve_range=True)
    positive = nissl_smooth[nissl_smooth > 0]
    if positive.size:
        low = float(np.percentile(positive, 34))
        high = float(np.percentile(positive, 58))
        body_mask = filters.apply_hysteresis_threshold(nissl_smooth, low, high)
    else:
        body_mask = np.zeros_like(nissl_smooth, dtype=bool)
    nucleus_band = segmentation.expand_labels(nuclei_labels, distance=16) > 0
    light_body = nucleus_band & (nissl_smooth > (float(np.percentile(positive, 24)) if positive.size else 0.0))
    body_mask |= light_body
    body_mask |= morphology.binary_dilation(nuclei_labels > 0, morphology.disk(10))
    body_mask = morphology.binary_closing(body_mask, morphology.disk(4))
    body_mask = morphology.binary_opening(body_mask, morphology.disk(1))
    body_mask = morphology.remove_small_objects(body_mask, 120)
    body_mask = morphology.remove_small_holes(body_mask, 180)

    if not np.any(body_mask):
        return np.zeros_like(nuclei_labels, dtype=np.int32)

    guidance = (
        1.0 * percentile_rescale(nissl_smooth, 1, 99.8)
        + 0.45 * percentile_rescale(filters.gaussian(remove_background(dapi_aligned, sigma=14), sigma=1.2, preserve_range=True), 1, 99.8)
    ).astype(np.float32)
    watershed_labels = segmentation.watershed(-guidance, nuclei_labels, mask=body_mask)

    final_labels = np.zeros_like(nuclei_labels, dtype=np.int32)
    for nucleus_id in sorted(int(x) for x in np.unique(nuclei_labels) if x > 0):
        cell_mask = watershed_labels == nucleus_id
        if not np.any(cell_mask):
            continue
        if int(cell_mask.sum()) < 80:
            rescue = (segmentation.expand_labels((nuclei_labels == nucleus_id).astype(np.int32), distance=14) > 0) & body_mask
            cell_mask |= rescue
        if int(cell_mask.sum()) < 80:
            continue
        final_labels[cell_mask & (final_labels == 0)] = nucleus_id
    return final_labels


def segment_purkinje(nissl: np.ndarray) -> np.ndarray:
    nissl_bs = remove_background(nissl, sigma=24)
    smooth = filters.gaussian(nissl_bs, sigma=2.0, preserve_range=True)
    candidate_soma = smooth > np.percentile(smooth, 97)
    candidate_soma = morphology.binary_opening(candidate_soma, morphology.disk(1))
    candidate_soma = morphology.binary_closing(candidate_soma, morphology.disk(2))
    candidate_soma = morphology.remove_small_objects(candidate_soma, 40)
    low_nissl = smooth < np.percentile(smooth, 35)
    low_nissl = morphology.binary_closing(low_nissl, morphology.disk(10))
    low_nissl = morphology.remove_small_objects(low_nissl, 1000)
    dist_from_low_nissl = ndi.distance_transform_edt(~low_nissl)
    purkinje = candidate_soma & (dist_from_low_nissl >= 5) & (dist_from_low_nissl <= 60)
    purkinje = morphology.remove_small_objects(purkinje, 40)
    purkinje = morphology.binary_dilation(purkinje, morphology.disk(6))
    return purkinje


def segment_purkinje_cells_from_pcp2(
    pcp2: np.ndarray, dapi: np.ndarray, sample: str, concentration: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    pcp2_bs = remove_background(pcp2, sigma=20)
    seed_threshold = get_purkinje_seed_threshold(sample, concentration)
    support_threshold = max(seed_threshold * PURKINJE_SUPPORT_THRESHOLD_FRACTION, 80.0)
    local_signal_threshold = max(seed_threshold * PURKINJE_LOCAL_SIGNAL_THRESHOLD_FRACTION, 60.0)
    seed_mask = pcp2_bs > seed_threshold
    support_mask = pcp2_bs > support_threshold
    support_mask = morphology.binary_dilation(support_mask, morphology.disk(PURKINJE_MERGE_DILATION_PX))
    support_mask = morphology.binary_closing(support_mask, morphology.disk(PURKINJE_MERGE_CLOSE_PX))
    support_mask = morphology.binary_opening(support_mask, morphology.disk(1))
    support_mask = morphology.remove_small_objects(support_mask, 120)
    support_mask = morphology.remove_small_holes(support_mask, 120)
    smooth_broad = filters.gaussian(pcp2_bs, sigma=4.0, preserve_range=True)
    local_signal_mask = smooth_broad > local_signal_threshold
    nuclei_labels = segment_nuclei(dapi)
    candidate_markers = np.zeros_like(nuclei_labels, dtype=np.int32)
    next_id = 1
    for prop in measure.regionprops(nuclei_labels):
        rr, cc = int(round(prop.centroid[0])), int(round(prop.centroid[1]))
        r0 = max(0, rr - PURKINJE_NUCLEUS_CAPTURE_RADIUS_PX)
        r1 = min(pcp2.shape[0], rr + PURKINJE_NUCLEUS_CAPTURE_RADIUS_PX + 1)
        c0 = max(0, cc - PURKINJE_NUCLEUS_CAPTURE_RADIUS_PX)
        c1 = min(pcp2.shape[1], cc + PURKINJE_NUCLEUS_CAPTURE_RADIUS_PX + 1)
        local_peak = float(pcp2_bs[r0:r1, c0:c1].max())
        local_support = bool(support_mask[r0:r1, c0:c1].any())
        local_signal = bool(local_signal_mask[r0:r1, c0:c1].any())
        if local_peak < support_threshold:
            continue
        if not (local_support or local_signal):
            continue
        candidate_markers[nuclei_labels == prop.label] = next_id
        next_id += 1

    if next_id == 1:
        return np.zeros_like(nuclei_labels, dtype=np.int32), np.zeros_like(seed_mask, dtype=bool), pd.DataFrame(), pcp2_bs

    expanded_candidate_labels = segmentation.expand_labels(candidate_markers, distance=PURKINJE_NUCLEUS_EXPANSION_PX)
    merged = support_mask | ((expanded_candidate_labels > 0) & local_signal_mask)
    merged = morphology.binary_closing(merged, morphology.disk(6))
    merged = morphology.binary_opening(merged, morphology.disk(1))
    merged = morphology.remove_small_objects(merged, PURKINJE_MIN_AREA_PX)
    merged = morphology.remove_small_holes(merged, PURKINJE_MIN_AREA_PX)

    labels = segmentation.watershed(-ndi.distance_transform_edt(merged), candidate_markers, mask=merged)

    keep_mask = np.zeros_like(merged, dtype=bool)
    for prop in measure.regionprops(labels, intensity_image=pcp2):
        area = int(prop.area)
        eq_diameter = float(prop.equivalent_diameter_area)
        solidity = float(prop.solidity)
        if area < PURKINJE_MIN_AREA_PX:
            continue
        if eq_diameter < PURKINJE_MIN_EQ_DIAMETER_PX or eq_diameter > PURKINJE_MAX_EQ_DIAMETER_PX:
            continue
        if solidity < PURKINJE_MIN_SOLIDITY:
            continue
        keep_mask[labels == prop.label] = True

    filtered_labels = measure.label(keep_mask)
    if filtered_labels.max() > 0:
        centroid_mask = np.zeros_like(keep_mask, dtype=bool)
        cell_centers: list[tuple[int, int, int]] = []
        for prop in measure.regionprops(filtered_labels):
            rr = int(round(prop.centroid[0]))
            cc = int(round(prop.centroid[1]))
            centroid_mask[rr, cc] = True
            cell_centers.append((int(prop.label), rr, cc))
        centroid_groups = morphology.binary_dilation(centroid_mask, morphology.disk(PURKINJE_LINE_GROUP_RADIUS_PX))
        centroid_groups = morphology.binary_closing(centroid_groups, morphology.disk(8))
        centroid_group_labels = measure.label(centroid_groups)
        component_counts: dict[int, int] = {}
        label_to_component: dict[int, int] = {}
        for label_id, rr, cc in cell_centers:
            component_id = int(centroid_group_labels[rr, cc])
            label_to_component[label_id] = component_id
            component_counts[component_id] = component_counts.get(component_id, 0) + 1
        grouped_mask = np.zeros_like(keep_mask, dtype=bool)
        for label_id, component_id in label_to_component.items():
            if component_id > 0 and component_counts.get(component_id, 0) >= PURKINJE_LINE_MIN_COMPONENT_SIZE:
                grouped_mask[filtered_labels == label_id] = True
        filtered_labels = measure.label(grouped_mask)

    rows: list[dict] = []
    for prop in measure.regionprops(filtered_labels, intensity_image=pcp2):
        rows.append(
            {
                "cell_id": int(prop.label),
                "area_px": int(prop.area),
                "centroid_row": float(prop.centroid[0]),
                "centroid_col": float(prop.centroid[1]),
                "major_axis_length_px": float(prop.axis_major_length),
                "minor_axis_length_px": float(prop.axis_minor_length),
                "equivalent_diameter_px": float(prop.equivalent_diameter_area),
                "solidity": float(prop.solidity),
                "mean_pcp2_intensity": float(prop.mean_intensity),
            }
        )
    return filtered_labels.astype(np.int32), keep_mask, pd.DataFrame(rows), pcp2_bs


def compute_per_cell_malat1_ratio(condition: str, key: ImageKey, malat1: np.ndarray, nuclei_labels: np.ndarray, cell_labels: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    expanded_nuclei_labels = segmentation.expand_labels(nuclei_labels, distance=MALAT1_NUCLEUS_EXPANSION_PX)
    expanded_cell_labels = segmentation.expand_labels(cell_labels, distance=256)
    for label_id in range(1, int(cell_labels.max()) + 1):
        nucleus_mask = expanded_nuclei_labels == label_id
        cell_mask = expanded_cell_labels == label_id
        if not nucleus_mask.any() or not cell_mask.any():
            continue
        outside_mask = cell_mask & (~nucleus_mask)
        inside_mean = float(malat1[nucleus_mask].mean()) if nucleus_mask.any() else np.nan
        outside_mean = float(malat1[outside_mask].mean()) if outside_mask.any() else np.nan
        ratio = outside_mean / inside_mean if inside_mean and not np.isnan(inside_mean) else np.nan
        rows.append(
            {
                "condition": condition,
                "sample": key.sample,
                "concentration": key.concentration,
                "slice_id": key.slice_id,
                "rep": key.rep,
                "region": key.region,
                "cell_id": label_id,
                "expanded_dapi_area_px": int(nucleus_mask.sum()),
                "expanded_nissl_until_neighbor_area_px": int(cell_mask.sum()),
                "outside_expanded_dapi_inside_expanded_nissl_area_px": int(outside_mask.sum()),
                "malat1_intensity_inside_expanded_dapi_mean": inside_mean,
                "malat1_intensity_outside_expanded_dapi_inside_expanded_nissl_mean": outside_mean,
                "malat1_intensity_per_unit_area_outside_expanded_dapi_inside_expanded_nissl_over_inside_expanded_dapi_ratio": ratio,
            }
        )
    return pd.DataFrame(rows)


def save_malat_panel(
    condition: str,
    key: ImageKey,
    dapi: np.ndarray,
    nissl: np.ndarray,
    malat1: np.ndarray,
    nuclei_labels: np.ndarray,
    cell_labels: np.ndarray,
) -> None:
    expanded_nuclei_mask = segmentation.expand_labels(nuclei_labels, distance=MALAT1_NUCLEUS_EXPANSION_PX) > 0
    expanded_cell_mask = segmentation.expand_labels(cell_labels, distance=256) > 0
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))
    panels = [
        (malat1, f"{condition} slice{key.slice_id} MALAT1", "magma", MALAT1_INTENSITY_RANGE),
        (dapi, "DAPI signal", "Blues", None),
        (nissl, "Nissl signal", "Greens", None),
        (expanded_cell_mask.astype(np.float32), "Expanded Nissl mask", "gray", (0.0, 1.0)),
    ]
    for ax, (image, title, cmap, fixed_range) in zip(axes, panels):
        if fixed_range is None:
            display = percentile_rescale(image, 1, 99.8)
        else:
            display = fixed_range_rescale(image, fixed_range[0], fixed_range[1])
        ax.imshow(display, cmap=cmap)
        ax.contour(expanded_nuclei_mask, levels=[0.5], colors="cyan", linewidths=0.9, linestyles="dotted")
        ax.contour(expanded_cell_mask, levels=[0.5], colors="yellow", linewidths=0.9, linestyles="dotted")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"malat1_panel_{condition}_slice{key.slice_id}rep{key.rep}.png", dpi=200)
    plt.close()


def save_malat_barplot(df: pd.DataFrame) -> None:
    value_col = "malat1_intensity_per_unit_area_outside_expanded_dapi_inside_expanded_nissl_over_inside_expanded_dapi_ratio"
    order = ["b2", "b3_10nM", "b3_20nM"]
    colors = {"b2": "#4C78A8", "b3_10nM": "#F58518", "b3_20nM": "#54A24B"}
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = np.arange(len(order))
    means = [df[df["condition"] == cond][value_col].replace([np.inf, -np.inf], np.nan).dropna().mean() for cond in order]
    ax.bar(x, means, color=[colors[c] for c in order], width=0.62, alpha=0.75)
    rng = np.random.default_rng(0)
    finite_means = [m for m in means if np.isfinite(m)]
    mean_ref = max(finite_means) if finite_means else 1.0
    for i, cond in enumerate(order):
        sub = df[df["condition"] == cond][value_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        ax.scatter(np.full(len(sub), x[i]) + jitter, sub, color="black", s=10, alpha=0.45, zorder=3)
        if len(sub) and np.isfinite(means[i]):
            ax.text(x[i], means[i] + max(0.03, 0.04 * mean_ref), f"{means[i]:.2f}", ha="center", va="bottom")
    ax.set_xticks(x, order)
    ax.set_ylabel("Mean outside expanded DAPI / mean inside expanded DAPI")
    ax.set_title("MALAT1 Per-Cell Intensity-Per-Area Ratio Across Replicates")
    ymax = df[value_col].replace([np.inf, -np.inf], np.nan).dropna().max()
    ax.set_ylim(0, max(1.0, float(ymax) * 1.15))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "malat1_replicates_per_cell_ratio_barplot.png", dpi=200)
    plt.close()


def save_notice_figure(filename: str, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def roi_mask_path_for_key(manifest: dict[tuple[ImageKey, str], Path], key: ImageKey) -> Path:
    return ROI_MASK_DIR / f"{manifest[(key, 'pcp2')].stem}__roi_mask.tif"


def load_roi_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = read_image(path)
    if mask.shape != expected_shape:
        raise ValueError(f"ROI mask shape mismatch for {path.name}: expected {expected_shape}, got {mask.shape}")
    return mask > 0


def build_pcp2_roi_manifest(manifest: dict[tuple[ImageKey, str], Path]) -> pd.DataFrame:
    rows: list[dict] = []
    keys = sorted({key for key, channel in manifest if channel == "pcp2"}, key=lambda k: (k.condition, k.slice_id, k.rep, k.region))
    for key in keys:
        image_path = manifest[(key, "pcp2")]
        mask_path = roi_mask_path_for_key(manifest, key)
        rows.append(
            {
                "condition": key.condition,
                "sample": key.sample,
                "concentration": key.concentration,
                "region": key.region,
                "slice_id": key.slice_id,
                "rep": key.rep,
                "source_image": str(image_path),
                "roi_mask_path": str(mask_path),
                "roi_role": "purkinje_layer" if key.region == "cereb" else "brainstem_reference",
                "mask_exists": mask_path.exists(),
            }
        )
    return pd.DataFrame(rows)


def call_pcp2_spots(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_bs = remove_background(image, sigma=20)
    smooth = filters.gaussian(image_bs, sigma=PCP2_SPOT_SMOOTH_SIGMA, preserve_range=True)
    candidate_threshold = max(PCP2_SPOT_THRESHOLD * PCP2_SPOT_CANDIDATE_THRESHOLD_FRACTION, 60.0)
    coords = feature.peak_local_max(
        smooth,
        min_distance=PCP2_MIN_DISTANCE,
        threshold_abs=candidate_threshold,
        exclude_border=False,
    )
    if len(coords):
        coords = coords[image_bs[coords[:, 0], coords[:, 1]] >= PCP2_SPOT_THRESHOLD]
    return coords, image_bs


def filter_spots(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return coords
    keep = [mask[int(r), int(c)] for r, c in coords]
    return coords[np.asarray(keep, dtype=bool)]


def tile_spot_counts(coords: np.ndarray, image_shape: tuple[int, int], region_mask: np.ndarray | None = None) -> pd.DataFrame:
    n_rows = image_shape[0] // TILE_HEIGHT
    n_cols = image_shape[1] // TILE_WIDTH
    tile_counts = np.zeros((n_rows, n_cols), dtype=np.int32)
    tile_mask_fraction = np.ones((n_rows, n_cols), dtype=np.float32)
    if region_mask is not None:
        for tile_r in range(n_rows):
            for tile_c in range(n_cols):
                r0 = tile_r * TILE_HEIGHT
                r1 = r0 + TILE_HEIGHT
                c0 = tile_c * TILE_WIDTH
                c1 = c0 + TILE_WIDTH
                tile_mask_fraction[tile_r, tile_c] = float(region_mask[r0:r1, c0:c1].mean())
    for r, c in coords:
        tile_r = int(r) // TILE_HEIGHT
        tile_c = int(c) // TILE_WIDTH
        if tile_r < n_rows and tile_c < n_cols and tile_mask_fraction[tile_r, tile_c] >= 0.99:
            tile_counts[tile_r, tile_c] += 1
    rows = []
    for tile_r in range(n_rows):
        for tile_c in range(n_cols):
            if tile_mask_fraction[tile_r, tile_c] < 0.99:
                continue
            rows.append(
                {
                    "tile_row": tile_r,
                    "tile_col": tile_c,
                    "tile_height_px": TILE_HEIGHT,
                    "tile_width_px": TILE_WIDTH,
                    "tile_area_px2": TILE_AREA_PX,
                    "tile_mask_fraction": float(tile_mask_fraction[tile_r, tile_c]),
                    "spots_per_200x200_px_area": int(tile_counts[tile_r, tile_c]),
                }
            )
    return pd.DataFrame(rows)


def centered_window_spot_counts(coords: np.ndarray, image_shape: tuple[int, int], per_cell_df: pd.DataFrame) -> pd.DataFrame:
    half_h = TILE_HEIGHT // 2
    half_w = TILE_WIDTH // 2
    rows = []
    if per_cell_df.empty:
        return pd.DataFrame(rows)
    for _, row in per_cell_df.iterrows():
        center_r = int(round(float(row["centroid_row"])))
        center_c = int(round(float(row["centroid_col"])))
        r0 = center_r - half_h
        c0 = center_c - half_w
        r1 = r0 + TILE_HEIGHT
        c1 = c0 + TILE_WIDTH
        if r0 < 0 or c0 < 0 or r1 > image_shape[0] or c1 > image_shape[1]:
            continue
        in_window = 0
        for r, c in coords:
            if r0 <= int(r) < r1 and c0 <= int(c) < c1:
                in_window += 1
        rows.append(
            {
                "tile_row": int(r0 // TILE_HEIGHT),
                "tile_col": int(c0 // TILE_WIDTH),
                "tile_height_px": TILE_HEIGHT,
                "tile_width_px": TILE_WIDTH,
                "tile_area_px2": TILE_AREA_PX,
                "tile_mask_fraction": 1.0,
                "spots_per_200x200_px_area": int(in_window),
                "cell_id": int(row["cell_id"]),
                "window_center_row": center_r,
                "window_center_col": center_c,
            }
        )
    return pd.DataFrame(rows)


def save_pair_alignment_qc(pair_id: str, b2_cereb_dapi: np.ndarray, b3_cereb_aligned: np.ndarray, cereb_overlap_mask: np.ndarray, b2_brainstem: np.ndarray, b3_brainstem_aligned: np.ndarray, brainstem_overlap_mask: np.ndarray) -> None:
    cereb_rgb = np.dstack([percentile_rescale(b3_cereb_aligned, 1, 99.8), np.zeros_like(b2_cereb_dapi), percentile_rescale(b2_cereb_dapi, 1, 99.8)])
    brain_rgb = np.dstack([percentile_rescale(b3_brainstem_aligned, 1, 99.8), np.zeros_like(b2_brainstem), percentile_rescale(b2_brainstem, 1, 99.8)])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cereb_rgb)
    axes[0].contour(cereb_overlap_mask, colors="yellow", linewidths=0.8)
    axes[0].set_title(f"{pair_id} cereb overlap")
    axes[1].imshow(brain_rgb)
    axes[1].contour(brainstem_overlap_mask, colors="yellow", linewidths=0.8)
    axes[1].set_title(f"{pair_id} brainstem overlap")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"pcp2_alignment_{pair_id}.png", dpi=200)
    plt.close()


def save_pcp2_qc(
    condition: str,
    slice_id: str,
    rep: str,
    cereb_pcp2_bs: np.ndarray,
    cereb_mask: np.ndarray,
    cereb_spots: np.ndarray,
    brainstem_pcp2_bs: np.ndarray,
    brainstem_spots: np.ndarray,
    n_purkinje_cells: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    panels = [
        (axes[0], cereb_pcp2_bs, cereb_mask, cereb_spots, f"{condition} cereb PCP2-segmented Purkinje cells ({n_purkinje_cells})"),
        (axes[1], brainstem_pcp2_bs, None, brainstem_spots, f"{condition} brainstem PCP2 spots"),
    ]
    for ax, image, mask, spots, title in panels:
        ax.imshow(fixed_range_rescale(image, *PCP2_INTENSITY_RANGE), cmap="gray")
        if len(spots):
            ax.plot(spots[:, 1], spots[:, 0], ".", color="red", markersize=2.3)
        if mask is not None and np.any(mask):
            ax.contour(mask, colors="yellow", linewidths=0.8, linestyles="dotted")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"pcp2_qc_{condition}_slice{slice_id}rep{rep}.png", dpi=200)
    plt.close()


def analyze_malat1_replicates(manifest: dict[tuple[ImageKey, str], Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_cell_tables: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    keys = sorted({key for key, channel in manifest if key.region == "cereb" and channel == "malat1"}, key=lambda k: (k.condition, k.slice_id, k.rep))
    for key in keys:
        dapi = load_image(manifest, key, "dapi")
        nissl = load_image(manifest, key, "nissl")
        malat1 = load_image(manifest, key, "malat1")
        roi = select_sparse_roi(dapi)
        dapi_roi = crop(dapi, roi)
        nissl_roi = crop(nissl, roi)
        malat1_roi = crop(malat1, roi)
        malat1_transform = estimate_malat1_transform(dapi_roi, nissl_roi, malat1_roi)
        malat1_aligned = rigid_transform(malat1_roi, angle=float(malat1_transform["angle"]), shift_rc=tuple(malat1_transform["shift_rc"]))
        nuclei_labels = segment_nuclei(dapi_roi)
        cell_labels = segment_cells_cellpose(nissl_roi, dapi_roi, nuclei_labels)
        per_cell = compute_per_cell_malat1_ratio(key.condition, key, malat1_aligned, nuclei_labels, cell_labels)
        per_cell_tables.append(per_cell)
        summary_rows.append(
            {
                "condition": key.condition,
                "sample": key.sample,
                "concentration": key.concentration,
                "slice_id": key.slice_id,
                "rep": key.rep,
                "roi_row_start": roi[0],
                "roi_col_start": roi[1],
                "roi_size_px": roi[2],
                "malat1_rotation_deg": float(malat1_transform["angle"]),
                "malat1_shift_row_px": float(malat1_transform["shift_rc"][0]),
                "malat1_shift_col_px": float(malat1_transform["shift_rc"][1]),
                "malat1_alignment_score": float(malat1_transform["score"]),
                "n_cells": int(len(per_cell)),
                "mean_ratio": float(per_cell["malat1_intensity_per_unit_area_outside_expanded_dapi_inside_expanded_nissl_over_inside_expanded_dapi_ratio"].replace([np.inf, -np.inf], np.nan).dropna().mean()),
            }
        )
        save_malat_panel(key.condition, key, dapi_roi, nissl_roi, malat1_aligned, nuclei_labels, cell_labels)
    per_cell_df = pd.concat(per_cell_tables, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    per_cell_df.to_csv(OUTPUT_DIR / "malat1_replicates_per_cell.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "malat1_replicates_per_image_summary.csv", index=False)
    save_malat_barplot(per_cell_df)
    return per_cell_df, summary_df


def analyze_pcp2_replicates(manifest: dict[tuple[ImageKey, str], Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    purkinje_columns = [
        "condition",
        "sample",
        "concentration",
        "slice_id",
        "rep",
        "n_segmented_purkinje_cells",
        "purkinje_layer_area_px",
        "purkinje_layer_spots",
        "purkinje_layer_spots_per_100k_px",
        "brainstem_area_px",
        "brainstem_spots",
        "brainstem_spots_per_100k_px",
        "brainstem_spots_expected_in_purkinje_layer_area",
        "purkinje_layer_enrichment_vs_brainstem",
    ]
    fov_columns = [
        "condition",
        "sample",
        "concentration",
        "slice_id",
        "rep",
        "n_segmented_purkinje_cells",
        "purkinje_layer_area_px",
        "purkinje_layer_spots",
        "purkinje_layer_spots_per_100k_px",
        "brainstem_area_px",
        "brainstem_spots",
        "brainstem_spots_per_100k_px",
        "brainstem_to_purkinje_layer_spots_per_area_ratio",
    ]
    tile_columns = [
        "tile_row",
        "tile_col",
        "tile_height_px",
        "tile_width_px",
        "tile_area_px2",
        "tile_mask_fraction",
        "spots_per_200x200_px_area",
        "cell_id",
        "window_center_row",
        "window_center_col",
        "condition",
        "sample",
        "concentration",
        "slice_id",
        "rep",
        "region",
        "group",
    ]

    pair_rows: list[dict] = []
    fov_rows: list[dict] = []
    tile_tables: list[pd.DataFrame] = []
    per_cell_tables: list[pd.DataFrame] = []
    keys = sorted({key for key, channel in manifest if key.region == "cereb" and channel == "pcp2"}, key=lambda k: (k.condition, k.slice_id, k.rep))
    for cereb_key in keys:
        brainstem_key = ImageKey(cereb_key.sample, "brainstem", cereb_key.concentration, cereb_key.slice_id, cereb_key.rep)
        dapi_cereb = load_image(manifest, cereb_key, "dapi")
        pcp2_cereb = load_image(manifest, cereb_key, "pcp2")
        pcp2_brainstem = load_image(manifest, brainstem_key, "pcp2")
        purkinje_labels, cereb_mask, per_cell_df, pcp2_cereb_bs = segment_purkinje_cells_from_pcp2(
            pcp2_cereb, dapi_cereb, cereb_key.sample, cereb_key.concentration
        )
        cereb_spots, pcp2_cereb_bs = call_pcp2_spots(pcp2_cereb)
        brainstem_spots, pcp2_brainstem_bs = call_pcp2_spots(pcp2_brainstem)
        cereb_spots = filter_spots(cereb_spots, cereb_mask)
        n_purkinje_cells = int(per_cell_df["cell_id"].nunique()) if not per_cell_df.empty else 0

        if not per_cell_df.empty:
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
            per_cell_df["pcp2_spot_count"] = per_cell_df["cell_id"].map(lambda x: int(cell_spot_counts[int(x)]))
            per_cell_df["pcp2_spots_per_100k_px"] = per_cell_df["pcp2_spot_count"] / per_cell_df["area_px"] * 100000
            per_cell_tables.append(per_cell_df)

        purkinje_area_px = int(cereb_mask.sum())
        brainstem_area_px = int(np.prod(pcp2_brainstem.shape))
        purkinje_density = len(cereb_spots) / purkinje_area_px * 100000 if purkinje_area_px else np.nan
        brainstem_density = len(brainstem_spots) / brainstem_area_px * 100000 if brainstem_area_px else np.nan
        pair_rows.append(
            {
                "condition": cereb_key.condition,
                "sample": cereb_key.sample,
                "concentration": cereb_key.concentration,
                "slice_id": cereb_key.slice_id,
                "rep": cereb_key.rep,
                "n_segmented_purkinje_cells": n_purkinje_cells,
                "purkinje_layer_area_px": purkinje_area_px,
                "purkinje_layer_spots": int(len(cereb_spots)),
                "purkinje_layer_spots_per_100k_px": purkinje_density,
                "brainstem_area_px": brainstem_area_px,
                "brainstem_spots": int(len(brainstem_spots)),
                "brainstem_spots_per_100k_px": brainstem_density,
                "brainstem_spots_expected_in_purkinje_layer_area": float(len(brainstem_spots) * (purkinje_area_px / brainstem_area_px)) if brainstem_area_px else np.nan,
                "purkinje_layer_enrichment_vs_brainstem": purkinje_density / brainstem_density if brainstem_density else np.nan,
            }
        )
        fov_rows.append(
            {
                "condition": cereb_key.condition,
                "sample": cereb_key.sample,
                "concentration": cereb_key.concentration,
                "slice_id": cereb_key.slice_id,
                "rep": cereb_key.rep,
                "n_segmented_purkinje_cells": n_purkinje_cells,
                "purkinje_layer_area_px": purkinje_area_px,
                "purkinje_layer_spots": int(len(cereb_spots)),
                "purkinje_layer_spots_per_100k_px": purkinje_density,
                "brainstem_area_px": brainstem_area_px,
                "brainstem_spots": int(len(brainstem_spots)),
                "brainstem_spots_per_100k_px": brainstem_density,
                "brainstem_to_purkinje_layer_spots_per_area_ratio": brainstem_density / purkinje_density if purkinje_density else np.nan,
            }
        )

        tile_cereb = centered_window_spot_counts(cereb_spots, pcp2_cereb.shape, per_cell_df)
        tile_cereb["condition"] = cereb_key.condition
        tile_cereb["sample"] = cereb_key.sample
        tile_cereb["concentration"] = cereb_key.concentration
        tile_cereb["slice_id"] = cereb_key.slice_id
        tile_cereb["rep"] = cereb_key.rep
        tile_cereb["region"] = "cereb"
        tile_cereb["group"] = f"{cereb_key.condition}_cereb"
        tile_tables.append(tile_cereb)

        tile_brain = tile_spot_counts(brainstem_spots, pcp2_brainstem.shape, None)
        tile_brain["condition"] = cereb_key.condition
        tile_brain["sample"] = cereb_key.sample
        tile_brain["concentration"] = cereb_key.concentration
        tile_brain["slice_id"] = cereb_key.slice_id
        tile_brain["rep"] = cereb_key.rep
        tile_brain["region"] = "brainstem"
        tile_brain["group"] = f"{cereb_key.condition}_brainstem"
        tile_tables.append(tile_brain)

        save_pcp2_qc(
            cereb_key.condition,
            cereb_key.slice_id,
            cereb_key.rep,
            pcp2_cereb_bs,
            cereb_mask,
            cereb_spots,
            pcp2_brainstem_bs,
            brainstem_spots,
            n_purkinje_cells,
        )

    purkinje_df = pd.DataFrame(pair_rows, columns=purkinje_columns)
    fov_df = pd.DataFrame(fov_rows, columns=fov_columns)
    tile_df = pd.concat(tile_tables, ignore_index=True) if tile_tables else pd.DataFrame(columns=tile_columns)
    per_cell_out_df = pd.concat(per_cell_tables, ignore_index=True) if per_cell_tables else pd.DataFrame()
    method_text = (
        f"peak_local_max on background-subtracted PCP2 smoothed with sigma={PCP2_SPOT_SMOOTH_SIGMA:g}, "
        f"candidate threshold={PCP2_SPOT_THRESHOLD * PCP2_SPOT_CANDIDATE_THRESHOLD_FRACTION:g}, "
        f"min_distance={PCP2_MIN_DISTANCE}, and final intensity filter >= {PCP2_SPOT_THRESHOLD:g}; "
        f"Purkinje cells segmented from cerebellar PCP2 after "
        f"background-subtracted thresholding at b2={PURKINJE_SEED_THRESHOLD_B2:g}, "
        f"b3_10nM={PURKINJE_SEED_THRESHOLD_B3:g}, "
        f"b3_20nM={PURKINJE_SEED_THRESHOLD_B3:g}, with DAPI nuclei as candidate seeds, "
        f"blob merging, and watershed splitting "
        f"(diameter gate {PURKINJE_MIN_EQ_DIAMETER_PX:g}-{PURKINJE_MAX_EQ_DIAMETER_PX:g}px)"
    )
    for df in (purkinje_df, fov_df, tile_df, per_cell_out_df):
        if not df.empty:
            df["spot_calling_method"] = method_text
    pd.DataFrame(
        [
            {
                "note": "Inter-FOV alignment disabled for replicate PCP2 analysis. Purkinje cells are segmented directly from cerebellar PCP2 images; brainstem uses whole-FOV spot density."
            }
        ]
    ).to_csv(OUTPUT_DIR / "pcp2_alignment_transforms.csv", index=False)
    purkinje_df.to_csv(OUTPUT_DIR / "pcp2_replicates_purkinje_vs_brainstem.csv", index=False)
    fov_df.to_csv(OUTPUT_DIR / "pcp2_replicates_brainstem_to_cereb_fov_spot_density.csv", index=False)
    tile_df.to_csv(OUTPUT_DIR / "pcp2_replicates_spots_per_200x200_px_area_tiles.csv", index=False)
    per_cell_out_df.to_csv(OUTPUT_DIR / "pcp2_replicates_segmented_purkinje_cells.csv", index=False)
    save_summary_barplot(
        purkinje_df,
        value_col="purkinje_layer_enrichment_vs_brainstem",
        group_col="condition",
        order=["b2", "b3_10nM", "b3_20nM"],
        colors={"b2": "#4C78A8", "b3_10nM": "#F58518", "b3_20nM": "#54A24B"},
        ylabel="Segmented Purkinje-cell spot density / brainstem spot density",
        title="PCP2 Purkinje Enrichment Across Replicates",
        filename="pcp2_replicates_purkinje_vs_brainstem_barplot.png",
    )
    save_summary_barplot(
        fov_df,
        value_col="brainstem_to_purkinje_layer_spots_per_area_ratio",
        group_col="condition",
        order=["b2", "b3_10nM", "b3_20nM"],
        colors={"b2": "#4C78A8", "b3_10nM": "#F58518", "b3_20nM": "#54A24B"},
        ylabel="Brainstem spots per area / segmented Purkinje-cell spots per area",
        title="PCP2 Brainstem/Purkinje Ratio Across Replicates",
        filename="pcp2_replicates_brainstem_to_cereb_ratio_barplot.png",
    )
    save_tile_group_barplot(tile_df)
    pd.DataFrame(
        [
            {
                "note": "Legacy ROI-mask step removed. Purkinje cells are now segmented automatically from the cerebellar PCP2 images."
            }
        ]
    ).to_csv(OUTPUT_DIR / "pcp2_replicates_missing_roi_masks.csv", index=False)
    pd.DataFrame(
        [
            {
                "note": "Legacy ROI-mask manifest retained as a note only. PCP2 replicate analysis no longer depends on user ROI masks."
            }
        ]
    ).to_csv(OUTPUT_DIR / "pcp2_roi_mask_manifest.csv", index=False)
    return purkinje_df, fov_df, tile_df, per_cell_out_df


def save_summary_barplot(df: pd.DataFrame, value_col: str, group_col: str, order: list[str], colors: dict[str, str], ylabel: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = np.arange(len(order))
    means = [df[df[group_col] == cond][value_col].replace([np.inf, -np.inf], np.nan).dropna().mean() for cond in order]
    ax.bar(x, means, color=[colors[c] for c in order], width=0.62, alpha=0.8)
    rng = np.random.default_rng(1)
    finite_means = [m for m in means if np.isfinite(m)]
    mean_ref = max(finite_means) if finite_means else 1.0
    for i, cond in enumerate(order):
        sub = df[df[group_col] == cond][value_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        ax.scatter(np.full(len(sub), x[i]) + jitter, sub, color="black", s=18, alpha=0.55, zorder=3)
        if len(sub) and np.isfinite(means[i]):
            ax.text(x[i], means[i] + max(0.03, 0.05 * mean_ref), f"{means[i]:.2f}", ha="center", va="bottom")
    ax.set_xticks(x, order)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ymax = df[value_col].replace([np.inf, -np.inf], np.nan).dropna().max()
    ax.set_ylim(0, max(1.0, float(ymax) * 1.15))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def save_tile_group_barplot(tile_df: pd.DataFrame) -> None:
    order = [
        "b2_cereb",
        "b2_brainstem",
        "b3_10nM_cereb",
        "b3_10nM_brainstem",
        "b3_20nM_cereb",
        "b3_20nM_brainstem",
    ]
    labels = ["b2 cereb", "b2 brainstem", "b3 10nM cereb", "b3 10nM brainstem", "b3 20nM cereb", "b3 20nM brainstem"]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B", "#B279A2"]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(order))
    means = [tile_df[tile_df["group"] == group]["spots_per_200x200_px_area"].mean() for group in order]
    ax.bar(x, means, color=colors, width=0.65, alpha=0.82)
    rng = np.random.default_rng(2)
    finite_means = [m for m in means if np.isfinite(m)]
    mean_ref = max(finite_means) if finite_means else 1.0
    for i, group in enumerate(order):
        sub = tile_df[tile_df["group"] == group]["spots_per_200x200_px_area"].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        ax.scatter(np.full(len(sub), x[i]) + jitter, sub, color="black", s=5, alpha=0.12, zorder=3, linewidths=0)
        if len(sub) and np.isfinite(means[i]):
            ax.text(x[i], means[i] + max(0.02, 0.04 * mean_ref), f"{means[i]:.2f}", ha="center", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Spots per 200 x 200 px area")
    ax.set_title("PCP2 Spots Per 200 x 200 px Area Across Replicates")
    ymax = tile_df["spots_per_200x200_px_area"].max()
    ax.set_ylim(0, max(1.0, float(ymax) * 1.15))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pcp2_replicates_spots_per_200x200_px_area_barplot.png", dpi=200)
    plt.close()


def main():
    manifest, manifest_df = parse_manifest()
    analyze_malat1_replicates(manifest)
    analyze_pcp2_replicates(manifest)
    summary = {
        "n_files": int(len(manifest_df)),
        "conditions": sorted(manifest_df["condition"].unique().tolist()),
        "slices": sorted(manifest_df["slice_id"].astype(str).unique().tolist()),
        "reps": sorted(manifest_df["rep"].astype(str).unique().tolist()),
    }
    pd.Series(summary).to_json(OUTPUT_DIR / "replicate_analysis_summary.json", indent=2)


if __name__ == "__main__":
    main()
