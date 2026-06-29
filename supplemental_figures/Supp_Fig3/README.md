# Datapoint CSV Code Only

This folder now contains only the code needed for:

- Malat1 per-cell ratio datapoints


## Malat1

- `images_pooled_barseq_analysis.py`
  - Builds the pooled Malat1 per-cell and per-image tables from the raw images.
- `replicate_barseq_analysis.py`
  - Shared segmentation, alignment, and per-cell Malat1 measurement functions used by the pooled script.
- `refresh_malat1_signal_over_noise.py`
  - Converts the Malat1 per-cell outputs into the final signal/noise-style per-cell ratio table.


