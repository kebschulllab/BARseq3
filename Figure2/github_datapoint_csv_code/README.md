# Datapoint CSV Code Only

This folder now contains only the code needed for:

- MALAT1 per-cell ratio datapoints
- PCP2 mean amplicons per cell in Purkinje-cell FOVs
- PCP2 mean amplicons per matched brainstem regions

## MALAT1

- `images_pooled_barseq_analysis.py`
  - Builds the pooled MALAT1 per-cell and per-image tables from the raw images.
- `replicate_barseq_analysis.py`
  - Shared segmentation, alignment, and per-cell MALAT1 measurement functions used by the pooled script.
- `refresh_malat1_signal_over_noise.py`
  - Converts the MALAT1 per-cell outputs into the final signal/noise-style per-cell ratio table.

Relevant datapoint CSV:
- `malat1_signal_over_noise_datapoints.csv`

## PCP2 mean amplicons per cell in Purkinje-cell FOVs

- `pcp2_slice_replicate_ratio_normlog_plot.py`
  - Contains the accepted PCP2 spot-calling logic used for cerebellar quantification.
- `pcp2_normlog_cereb_all_cells_plot.py`
  - Builds the current all-cells cerebellar PCP2 table from the raw images.
- `pcp2_cereb_slice_replicate_ttests.py`
  - Converts the cerebellar all-cells table into one mean-amplcons-per-cell datapoint per biological slice + replicate.

Relevant datapoint CSV:
- `pcp2_cereb_region_average_spots_datapoints.csv`

## PCP2 mean amplicons per brainstem regions

- `pcp2_normlog_condition_plots.py`
  - Builds the slice + replicate brainstem matched-region averages from the raw images.
