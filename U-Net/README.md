# U-Net for 3D Underwater Acoustic Field Prediction

This repository provides a self-contained U-Net example for predicting 3-D underwater acoustic transmission-loss (TL) fields. The model takes a 52-dimensional environmental vector and a 3-D geo-steady background field as inputs and predicts a TL field with shape `[4, 36, 250]`.

## Data setup

No `shareddata/` directory or large external dataset is required. The code constructs 18,876 logical samples at runtime:

```text
13 years × 12 months × 121 source locations = 18,876 samples
```

The source locations form an 11 × 11 regular grid spanning 133°W–132°W and 36°N–37°N.

Each environmental vector contains:

- the first 2 elements: longitude and latitude;
- the remaining 50 elements: a synthetic sound speed profile (SSP).

The synthetic SSP includes vertical structure together with spatial, seasonal, and interannual variability. The synthetic TL fields include geometric spreading, range-dependent absorption, depth structure, azimuthal variability, interference-like fluctuations, shadow-zone-like structure, and environment-dependent variations. These data are intended for testing the training pipeline and code interfaces; they are not a substitute for results from a dedicated acoustic propagation solver.

## Data split

A chronological extrapolation split is used:

- Years 1–11: training set, 15,972 samples;
- Year 12: validation set, 1,452 samples;
- Year 13: test set, 1,452 samples.

EOF decomposition, K-means clustering, geo-steady fields, environmental-input normalization, and TL normalization are fitted exclusively from the training set. The validation and test sets do not contribute to these statistics.

## EOF-K geo-steady field

For each source grid point, the SSP is first averaged over all training years and all 12 months, producing a `121 × 50` long-term SSP matrix. EOF decomposition is then performed using SVD, and the minimum number of components whose cumulative explained variance reaches 99% is retained.

Bathymetry is not available in this dataset, so K-means uses only the retained EOF coefficients. The EOF coefficients are Z-score normalized before clustering.

After the acoustic subregions are defined, all 3-D TL fields from the training years, all months, and all source grid points within each subregion are averaged to obtain a time-invariant geo-steady field. Each sample uses the geo-steady field associated with the EOF-K subregion of its source location.

The repository also includes great-circle nearest-neighbor mapping for assigning an arbitrary source location to the closest point on the 11 × 11 source grid.

## U-Net

The default U-Net configuration is:

```text
x1_dim   = 52
in_ch    = 4
base_ch  = 32
cond_dim = 256
dropout  = 0.1
```

The environmental vector is encoded by an MLP and injected additively at the U-Net bottleneck. The network output shape is `[4, 36, 250]`.

The default model contains 10,373,636 parameters.

The loss combines weighted L1 loss and gradient loss. Along the range dimension, the first 125 bins use weight 1 and the remaining 125 bins use weight 2.

## Project files

```text
train.py           training, validation, and testing entry point
config.py          dataset geometry, year split, and default training parameters
synthetic_data.py  synthetic SSP and 3-D TL generation
eofk.py            EOF, K-means, nearest-neighbor mapping, and geo-steady field calculation
dataset.py         PyTorch Dataset
model.py           U-Net model
losses.py          MixedLoss
train_utils.py     normalization, random seeding, and model-complexity utilities
requirements.txt   Python dependencies
```

## Running the code

Install dependencies:

```bash
pip install -r requirements.txt
```

Train with the default configuration:

```bash
python train.py
```

For a quick pipeline check, reduce the number of epochs:

```bash
python train.py --epochs 1
```

To change the number of EOF-K subregions:

```bash
python train.py --clusters 5
```

## Output files

Training outputs are saved to `outputs_unet_eofk/` by default, including:

```text
best.pt
last.pt
training_history.npy

grid_lonlat.npy
grid_cluster_labels.npy
long_term_ssp.npy

eof_mean.npy
eof_modes.npy
eof_explained_ratio.npy
eof_score_mean.npy
eof_score_std.npy
kmeans_centers.npy

geo_steady_fields.npy
geo_steady_cluster_counts.npy

x1_mean.npy
x1_std.npy
t_mean.npy
t_std.npy
```

`grid_cluster_labels.npy` can be used to visualize the EOF-K spatial partition, while `geo_steady_fields.npy` stores the long-term steady TL background field for each acoustic subregion.
