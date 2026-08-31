# 3D Underwater Acoustic Field Prediction — A Demo

This repository is a self-contained GitHub demonstration of a conditional CNN for 3-D transmission-loss (TL) field prediction.

## Inputs and output

- Environmental vector: `[52]` = longitude, latitude, and a 50-level SSP.
- Background TL field: `[4, 36, 250]`.
- Predicted TL field: `[4, 36, 250]`.

The CNN architecture and the default model scale are kept unchanged from the original code: `base_ch=128`, `cond_dim=128`, and `num_blocks=8`.

## EOF-K geo-steady field

Bathymetry is not available in this public example, so K-means uses normalized EOF coefficients of the training SSP only.

1. For each of the 121 source grid points, average its SSP over all training year-month combinations to form `T [121,50]`.
2. Apply centered SVD/EOF and retain the minimum `K` for cumulative explained variance >= 99%.
3. Z-score the retained EOF scores and apply K-means to obtain `M` acoustic subregions.
4. For each subregion, average all 3-D TL fields belonging to its source grid points across all training years and months.
5. Validation and test samples use the already-fitted EOF-K labels and geo-steady fields; they never participate in their construction.

For an arbitrary source coordinate, `eofk.location_to_cluster()` performs great-circle nearest-grid mapping before reading the corresponding subregion label.

## Synthetic data

No `shareddata/` folder is required. The logical dataset contains exactly:

```text
13 years x 12 months x 121 grid points = 18,876 samples
```

Source positions form an 11 x 11 grid spanning 133°W–132°W and 36°N–37°N. SSP and TL fields are generated deterministically from the sample index, so the same sample is reproducible without storing large `.npy` files.

The synthetic TL contains broad physics-inspired effects including spreading, range-dependent absorption, SSP-related depth structure, azimuth anisotropy, refractive bands, interference-like structure, shadow-like regions, seasonality, and interannual variability. It is intended only for running and testing the deep-learning pipeline, not as a substitute for an acoustic propagation solver.

## Files

```text
train.py             training / validation / testing entry point
config.py            data geometry and year split
synthetic_data.py    synthetic SSP and TL generator
eofk.py              EOF, K-means, nearest-neighbor mapping, geo-steady fields
dataset.py           PyTorch Dataset and background-field assignment
model.py             CNN architecture
losses.py            original mixed loss
train_utils.py       training-only normalization and utility functions
my_functions.py      original helper functions
requirements.txt     dependencies
```

## Run

```bash
pip install -r requirements.txt
python train.py
```

The dataset uses a chronological split: years 1–11 are the training set, year 12 is the validation set, and year 13 is the test set. EOF decomposition, K-means clustering, geo-steady TL fields, and all normalization statistics are fitted exclusively from years 1–11. Years 12 and 13 are never used to construct the background fields or preprocessing statistics.

EOF-K metadata and geo-steady fields are saved under `baseline_cnn_outputs/` together with the trained model and normalization parameters.
