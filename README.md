# CubeCast

CubeCast is a compact Python library for modeling and forecasting tensor
streams with shape `(time, location, feature)`. It combines nonlinear latent
state dynamics, ICA-based seasonal coordinates, automatic location grouping,
and online regime selection behind a small API.

This repository provides the official Python implementation of *Non-Linear
Mining of Social Activities in Tensor Streams* by Kawabata et al. (KDD 2020).

## Features

- Forecasts multi-location, multi-feature tensor streams.
- Represents the smoothed shared trend in a truncated SVD basis.
- Represents phase-averaged seasonal residuals in an SVD-reduced subspace,
  then applies FastICA within that subspace.
- Fits diagonal quadratic state dynamics with lmfit's
  Levenberg-Marquardt optimizer.
- Discovers location groups through recursive, MDL-guided bisection.
- Reuses known regimes or creates a new regime when it reduces description
  cost.
- Supports rolling-window updates and multi-step forecasting.
- Exposes additive trend and seasonal decompositions.
- Handles missing values with per-series mean imputation.

## Installation

CubeCast requires Python 3.9 or later.

```bash
cd cubecast
pip install .
```

Install the optional Matplotlib integration with:

```bash
pip install ".[plot]"
```

The core runtime dependencies are NumPy, scikit-learn, and lmfit.

## Quick start

```python
import numpy as np

from cubecast import CubeCast


# Weekly search volumes: time x countries x keywords
observations = np.load("search_volume.npy")

model = CubeCast(
    period=52,
    window_size=104,
    max_components=4,
)
model.fit(observations)

# Forecast the next year.
forecast = model.predict(52)

# Incorporate a new quarter and forecast again.
model.update(new_observations)
forecast = model.predict(52)
```

The returned forecast has shape `(steps, location, feature)`.

## Streaming updates

Call `fit()` once on an initial history. As each observation arrives, keep the
time axis and pass it to `update()`, then call `predict()` for a new forecast:

```python
model.fit(initial_history)  # (time, location, feature)

for observation in data_stream:  # each item is (location, feature)
    model.update(observation[None, :, :])
    forecast = model.predict(12)
```

`update()` also accepts a batch with shape `(batch_time, location, feature)`.
Pass only newly observed data, not previous forecasts. With `window_size` set,
each update refits a candidate model on the retained rolling window instead of
performing a constant-time parameter update. For seasonal models, the initial
history and `window_size` must contain at least `2 * period` time points.

## Input data

CubeCast expects a floating-point array with three dimensions:

```text
(time, location, feature)
```

For example, a Google Trends tensor may use weeks as time points, countries as
locations, and search terms as features.

- At least four time points are required.
- A seasonal model requires at least two complete periods.
- Set `period=None` to disable seasonality.
- Location and feature dimensions must remain fixed across `update()` calls.
- Each rolling window is standardized independently for every
  location-feature series.
- `NaN` values are replaced by the mean of their location-feature series.
- Infinite values are rejected.

## Trend and seasonality pipeline

SVD and FastICA do not produce two competing decompositions in this library.
SVD is used in two separate places, and FastICA is used only in the seasonal
pipeline.

Neither method defines trend or seasonality by itself. The moving average
defines the trend signal, and subtracting that signal plus grouping residuals
by phase defines the seasonal profile. SVD and FastICA only provide compact
coordinates for those already constructed signals.

### Trend coordinates

The model first averages the normalized tensor over locations and smooths the
result with a moving average of length `period`. A truncated SVD basis is fitted
to this smoothed `(time, feature)` matrix. Projecting the smoothed signal into
that basis produces the latent trend coordinates `z`.

FastICA is not involved in the trend representation.

### Seasonal coordinates

The seasonal path is:

```text
shared signal
  -> subtract smoothed trend
  -> average residuals by phase within the period
  -> select and project onto a low-rank SVD subspace
  -> fit FastICA in that reduced subspace
  -> obtain ICA seasonal coordinates
```

The explicit SVD projection serves two purposes: its BIC/MDL score selects the
dimension supplied as `n_components` to FastICA, and it removes numerically
empty directions before FastICA whitening. FastICA then rotates the retained
subspace toward approximately statistically independent coordinates. It does
not choose the number of components itself, and statistical independence is an
optimization objective rather than a guarantee.

FastICA is fitted to the SVD-reduced phase profile, not to the raw tensor. This
pipeline is therefore not equivalent to applying FastICA directly to the
observations, nor does it treat an SVD component as a seasonal component.

For each time point, the current detrended signal is projected through the same
SVD basis and transformed by the fitted FastICA model. The implementation then
expresses these ICA scores as a phase template multiplied by a time-varying
intensity. This product is the value exposed as `seasonal_latent`.

In compact notation, the observation model is:

```text
reconstruction = baseline + W @ z + U @ q
q = seasonal_intensity * seasonal_phase_template
```

Here, `W` and `U` are specific to the active location group. Therefore,
`seasonal_latent` is a shared ICA-coordinate state, while the returned
`seasonal` array is its contribution after projection to one location and
feature.

## Inspecting the learned structure

```python
print(model.groups_)
# ((0, 1), (2, 3))

print(model.n_regimes_)
print(model.active_regime_)
print(model.trend_components_)
print(model.seasonal_components_)
print(model.transition_optimizer_)
# lmfit.leastsq
```

`trend_components_` is the selected SVD trend dimension.
`seasonal_components_` is the SVD/BIC-selected seasonal subspace dimension that
is passed to FastICA as `n_components`.

`groups_` contains the automatically discovered location clusters for the
active regime. A group split is retained only when its reduction in residual
cost pays for the additional model parameters under the BIC/MDL approximation.

## Decomposition

The additive decomposition for one location and feature is available without
plotting dependencies:

```python
components = model.decompose(location=0, feature=0)

observed = components["observed"]
reconstructed = components["reconstruction"]
trend = components["trend"]
seasonal = components["seasonal"]
trend_latent = components["trend_latent"]
seasonal_latent = components["seasonal_latent"]
```

The decomposition satisfies:

```text
reconstruction = baseline + trend + seasonal
```

The latent seasonal state is the regularized
`intensity * seasonal_phase_template` representation of the FastICA scores.
The group-specific observation matrix maps this shared ICA-coordinate state
into the seasonal contribution for a particular location and feature.

## Visualization

```python
figure, axes = model.plot_decomposition(
    location=0,
    feature=0,
    path="decomposition.png",
)
```

The saved figure uses separate panels and vertical scales for:

1. observed and reconstructed values;
2. trend contribution;
3. seasonal contribution;
4. latent trend components; and
5. latent seasonal components.

Run the complete synthetic example with:

```bash
python examples/plot_decomposition.py
```

The example contains two distinct seasonal sources and saves its result to
`examples/decomposition.png`.

## API

### `CubeCast(period=None, window_size=None, max_components=4)`

- `period`: seasonal period, or `None` for a nonseasonal model.
- `window_size`: number of recent time points retained during updates. `None`
  retains the full stream.
- `max_components`: upper bound for automatically selected trend and seasonal
  dimensions.

### Methods

- `fit(tensor)`: initialize the latent space and first regime.
- `update(tensor)`: append observations and select or create a regime.
- `predict(steps)`: produce a multi-step tensor forecast.
- `fit_predict(tensor, steps)`: fit and forecast in one call.
- `decompose(location, feature)`: return additive and latent components.
- `plot_decomposition(location, feature, path=None)`: plot and optionally save
  a decomposition.

## Implementation notes

The external SVD projection is distinct from FastICA's internal whitening. It
defines the seasonal subspace and prevents rank-deficient input from reaching
FastICA. FastICA uses a fixed random seed for reproducibility and finds a
rotation only within that retained subspace.

State dynamics are initialized without quadratic terms and then optimized with
lmfit. If a window is too short to identify all LM parameters, the
implementation retains the linear least-squares transition.

Observation matrices remain ordinary least-squares problems because they are
linear once the latent states and location assignments are fixed. Latent bases
are fixed after `fit()` so stored regimes remain comparable. Seasonal profiles
are refreshed from the active rolling window.

State forecasts are clipped to six training standard deviations to prevent
unstable quadratic extrapolation.

## Development

Run the test suite from this directory:

```bash
python -m unittest discover -s tests -v
```

Build and validate the distribution:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

## Reference

Koki Kawabata, Yasuko Matsubara, Takato Honda, and Yasushi Sakurai. 2020.
"Non-Linear Mining of Social Activities in Tensor Streams." In *Proceedings of
the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*,
2093-2102. https://doi.org/10.1145/3394486.3403260

```bibtex
@inproceedings{kawabata2020cubecast,
  author = {Kawabata, Koki and Matsubara, Yasuko and Honda, Takato and
            Sakurai, Yasushi},
  title = {Non-Linear Mining of Social Activities in Tensor Streams},
  booktitle = {Proceedings of the 26th ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining},
  year = {2020},
  pages = {2093--2102},
  doi = {10.1145/3394486.3403260}
}
```

## License

MIT
