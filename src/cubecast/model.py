"""A small, practical implementation of the CubeCast model."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from lmfit import Parameters, minimize
from sklearn.decomposition import FastICA


@dataclass(frozen=True)
class _Group:
    locations: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class _Regime:
    groups: Tuple[_Group, ...]
    assignments: np.ndarray
    bias: np.ndarray
    linear: np.ndarray
    quadratic: np.ndarray
    state_center: np.ndarray
    state_scale: np.ndarray
    optimizer: str
    data_code: float
    sample_count: int


class CubeCast:
    """Forecast a ``(time, location, feature)`` tensor stream.

    Parameters
    ----------
    period:
        Seasonal period, such as ``52`` for weekly data with yearly
        seasonality. Use ``None`` to disable the seasonal component.
    window_size:
        Number of recent observations retained by :meth:`update`. ``None``
        retains the complete stream.
    max_components:
        Small upper bound for automatically selected latent dimensions.
    """

    def __init__(
        self,
        period: Optional[int] = None,
        window_size: Optional[int] = None,
        max_components: int = 4,
    ) -> None:
        if period is not None and period < 1:
            raise ValueError("period must be positive or None")
        if window_size is not None and window_size < 4:
            raise ValueError("window_size must be at least 4 or None")
        if max_components < 1:
            raise ValueError("max_components must be positive")

        self.period = period
        self.window_size = window_size
        self.max_components = max_components
        self._fitted = False

    def fit(self, tensor: np.ndarray) -> "CubeCast":
        """Fit the first regime and initialize the shared latent space."""
        values = self._validate_tensor(tensor)

        self._time_seen = values.shape[0]
        self._history = self._trim(values)
        self._check_training_length(self._history.shape[0])
        self._n_locations, self._n_features = values.shape[1:]

        normalized = self._normalize_history()
        phases = self._phases()
        self._initialize_latent_space(normalized, phases)
        states, trends, seasonal = self._coordinates(normalized, phases)

        regime = self._fit_regime(normalized, states, trends, seasonal)
        self._regimes = [regime]
        self._active = 0
        self._last_state = states[-1].copy()
        self._cache_decomposition(normalized, trends, seasonal)
        self._fitted = True
        self._publish_state()
        return self

    def update(self, tensor: np.ndarray) -> "CubeCast":
        """Append observations, detect a regime, and update model state."""
        self._require_fitted()
        values = self._validate_tensor(tensor)
        if values.shape[1:] != (self._n_locations, self._n_features):
            raise ValueError("location and feature dimensions cannot change")

        self._time_seen += values.shape[0]
        self._history = self._trim(np.concatenate((self._history, values)))
        self._check_training_length(self._history.shape[0])

        normalized = self._normalize_history()
        phases = self._phases()
        states, trends, seasonal = self._coordinates(
            normalized, phases, update_season=True
        )
        candidate = self._fit_regime(normalized, states, trends, seasonal)

        existing_codes = [
            self._data_code(regime, normalized, states, trends, seasonal)[0]
            for regime in self._regimes
        ]
        best = int(np.argmin(existing_codes))
        new_code = candidate.data_code + self._model_code(candidate)

        if new_code < existing_codes[best]:
            self._regimes.append(candidate)
            self._active = len(self._regimes) - 1
        else:
            self._active = best

        self._last_state = states[-1].copy()
        self._cache_decomposition(normalized, trends, seasonal)
        self._publish_state()
        return self

    def predict(self, steps: int) -> np.ndarray:
        """Forecast ``steps`` future tensors."""
        self._require_fitted()
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("steps must be a positive integer")

        regime = self._regimes[self._active]
        state = self._last_state.copy()
        result = np.empty((steps, self._n_locations, self._n_features))

        for offset in range(steps):
            state = self._advance(regime, state)
            phase = (self._time_seen + offset) % (self.period or 1)
            trend = state[: self._trend_rank]
            intensity = state[self._trend_rank :]
            seasonal = intensity * self._season_table[phase]
            design = np.concatenate((trend, seasonal, [1.0]))

            for group in regime.groups:
                prediction = design @ group.coefficients
                result[offset, group.locations] = prediction

        return result * self.scale_ + self.mean_

    def fit_predict(self, tensor: np.ndarray, steps: int) -> np.ndarray:
        """Fit the model and immediately forecast future values."""
        return self.fit(tensor).predict(steps)

    def decompose(self, location: int = 0, feature: int = 0) -> dict:
        """Return observed, reconstructed, trend, and seasonal values."""
        self._require_fitted()
        if not 0 <= location < self._n_locations:
            raise IndexError("location is out of range")
        if not 0 <= feature < self._n_features:
            raise IndexError("feature is out of range")

        regime = self._regimes[self._active]
        group = next(item for item in regime.groups if location in item.locations)
        coefficients = group.coefficients[:, feature]
        split = self._trend_rank
        end = split + self._season_rank

        scale = self.scale_[location, feature]
        mean = self.mean_[location, feature]
        baseline = np.full(self._history.shape[0], mean + scale * coefficients[-1])
        trend = scale * (self._last_trends @ coefficients[:split])
        seasonal = scale * (self._last_seasonal @ coefficients[split:end])
        observed = mean + scale * self._last_normalized[:, location, feature]
        start = self._time_seen - self._history.shape[0]

        return {
            "time": start + np.arange(self._history.shape[0]),
            "observed": observed.copy(),
            "baseline": baseline,
            "trend": trend,
            "seasonal": seasonal,
            "reconstruction": baseline + trend + seasonal,
            "trend_latent": self._last_trends.copy(),
            "seasonal_latent": self._last_seasonal.copy(),
        }

    def plot_decomposition(
        self,
        location: int = 0,
        feature: int = 0,
        path=None,
    ):
        """Plot a decomposition and optionally save it to ``path``."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError(
                "plotting requires: pip install 'cubecast[plot]'"
            ) from error

        values = self.decompose(location, feature)
        time = values["time"]
        figure, axes = plt.subplots(
            5, 1, figsize=(10, 10), sharex=True, constrained_layout=True
        )

        axes[0].plot(time, values["observed"], label="observed", alpha=0.65)
        axes[0].plot(time, values["reconstruction"], label="reconstructed")
        axes[0].set_title(f"Location {location}, feature {feature}")
        axes[0].set_ylabel("value")
        axes[0].legend()

        axes[1].plot(time, values["trend"], color="C0")
        axes[1].axhline(0.0, color="0.6", linewidth=0.8)
        axes[1].set_ylabel("trend")

        axes[2].plot(time, values["seasonal"], color="C1")
        axes[2].axhline(0.0, color="0.6", linewidth=0.8)
        axes[2].set_ylabel("seasonal")

        for index, component in enumerate(values["trend_latent"].T, start=1):
            axes[3].plot(time, component, label=f"z{index}")
        axes[3].set_ylabel("trend latent")
        axes[3].legend(ncol=2)

        for index, component in enumerate(values["seasonal_latent"].T, start=1):
            axes[4].plot(time, component, "--", label=f"s{index}")
        axes[4].set_ylabel("seasonal latent")
        axes[4].set_xlabel("time")
        if values["seasonal_latent"].shape[1]:
            axes[4].legend(ncol=2)
        else:
            axes[4].text(
                0.5,
                0.5,
                "No seasonal components",
                ha="center",
                va="center",
                transform=axes[4].transAxes,
            )

        if path is not None:
            from pathlib import Path

            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=160, bbox_inches="tight")
        return figure, axes

    def _initialize_latent_space(
        self, values: np.ndarray, phases: np.ndarray
    ) -> None:
        shared = values.mean(axis=1)
        trend = self._smooth(shared)
        self._trend_basis = self._select_basis(trend, allow_zero=False)
        self._trend_rank = self._trend_basis.shape[0]

        if self.period is None:
            self._season_ica = None
            self._season_projection = np.empty((0, self._n_features))
            self._season_table = np.empty((1, 0))
            self._season_rank = 0
            return

        residual = shared - trend
        profile = self._phase_means(residual, phases)
        self._season_projection = self._select_basis(profile, allow_zero=True)
        self._season_rank = self._season_projection.shape[0]
        if self._season_rank == 0:
            self._season_ica = None
            self._season_table = np.empty((self.period, 0))
            return

        self._season_ica = FastICA(
            n_components=self._season_rank,
            whiten="unit-variance",
            random_state=0,
        )
        reduced_profile = profile @ self._season_projection.T
        self._season_table = self._season_ica.fit_transform(reduced_profile)

    def _coordinates(
        self,
        values: np.ndarray,
        phases: np.ndarray,
        update_season: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        shared = values.mean(axis=1)
        smooth = self._smooth(shared)
        trends = smooth @ self._trend_basis.T

        if self._season_rank == 0:
            seasonal = np.empty((values.shape[0], 0))
            states = trends
            return states, trends, seasonal

        reduced = (shared - smooth) @ self._season_projection.T
        scores = self._season_ica.transform(reduced)
        if update_season:
            current = self._phase_means(scores, phases)
            present = np.bincount(phases, minlength=self.period) > 0
            self._season_table[present] = current[present]

        template = self._season_table[phases]
        ridge = 1e-8 + 1e-6 * np.mean(template * template, axis=0)
        intensity = scores * template / (template * template + ridge)
        intensity = np.clip(intensity, -5.0, 5.0)
        seasonal = intensity * template
        states = np.concatenate((trends, intensity), axis=1)
        return states, trends, seasonal

    def _fit_regime(
        self,
        values: np.ndarray,
        states: np.ndarray,
        trends: np.ndarray,
        seasonal: np.ndarray,
    ) -> _Regime:
        bias, linear, quadratic, center, scale, optimizer = self._fit_transition(
            states
        )
        groups, assignments = self._fit_groups(values, trends, seasonal)
        provisional = _Regime(
            groups=groups,
            assignments=assignments,
            bias=bias,
            linear=linear,
            quadratic=quadratic,
            state_center=center,
            state_scale=scale,
            optimizer=optimizer,
            data_code=0.0,
            sample_count=0,
        )
        code, count = self._data_code(
            provisional, values, states, trends, seasonal
        )
        return _Regime(
            groups=groups,
            assignments=assignments,
            bias=bias,
            linear=linear,
            quadratic=quadratic,
            state_center=center,
            state_scale=scale,
            optimizer=optimizer,
            data_code=code,
            sample_count=count,
        )

    def _fit_transition(
        self, states: np.ndarray
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
    ]:
        current, following = states[:-1], states[1:]
        dimension = states.shape[1]
        bias = np.empty(dimension)
        linear = np.empty((dimension, dimension))
        quadratic = np.zeros(dimension)

        for target in range(dimension):
            design = np.column_stack((np.ones(current.shape[0]), current))
            coefficients = np.linalg.lstsq(
                design, following[:, target], rcond=None
            )[0]
            bias[target] = coefficients[0]
            linear[target] = coefficients[1:]

        optimizer = "numpy.linalg.lstsq"
        parameter_count = dimension * (dimension + 2)
        if following.size >= parameter_count:
            parameters = Parameters()
            for target in range(dimension):
                parameters.add(f"bias_{target}", value=bias[target])
                parameters.add(f"quadratic_{target}", value=0.0)
                for source in range(dimension):
                    parameters.add(
                        f"linear_{target}_{source}",
                        value=linear[target, source],
                    )

            result = minimize(
                self._transition_residual,
                parameters,
                args=(current, following),
                method="leastsq",
            )
            fitted = np.array([value.value for value in result.params.values()])
            if result.success and np.all(np.isfinite(fitted)):
                for target in range(dimension):
                    bias[target] = result.params[f"bias_{target}"].value
                    quadratic[target] = result.params[
                        f"quadratic_{target}"
                    ].value
                    for source in range(dimension):
                        linear[target, source] = result.params[
                            f"linear_{target}_{source}"
                        ].value
                optimizer = "lmfit.leastsq"

        center = states.mean(axis=0)
        scale = states.std(axis=0)
        scale[scale < 1e-6] = 1.0
        return bias, linear, quadratic, center, scale, optimizer

    @staticmethod
    def _transition_residual(
        parameters: Parameters,
        current: np.ndarray,
        following: np.ndarray,
    ) -> np.ndarray:
        predicted = np.empty_like(following)
        dimension = following.shape[1]
        for target in range(dimension):
            predicted[:, target] = parameters[f"bias_{target}"].value
            predicted[:, target] += (
                parameters[f"quadratic_{target}"].value
                * current[:, target] ** 2
            )
            for source in range(dimension):
                predicted[:, target] += (
                    parameters[f"linear_{target}_{source}"].value
                    * current[:, source]
                )
        return (predicted - following).ravel()

    def _fit_groups(
        self, values: np.ndarray, trends: np.ndarray, seasonal: np.ndarray
    ) -> Tuple[Tuple[_Group, ...], np.ndarray]:
        location_features = values.transpose(1, 0, 2).reshape(
            self._n_locations, -1
        )
        pending = [np.arange(self._n_locations)]
        accepted: List[np.ndarray] = []

        while pending:
            locations = pending.pop()
            split = self._split_locations(location_features, locations)
            if split is not None and self._split_improves(
                values, trends, seasonal, locations, split
            ):
                pending.extend(split)
            else:
                accepted.append(locations)

        accepted.sort(key=lambda item: int(item.min()))
        groups = tuple(
            self._fit_group(values, trends, seasonal, locations)
            for locations in accepted
        )
        assignments = np.empty(self._n_locations, dtype=int)
        for group_index, group in enumerate(groups):
            assignments[group.locations] = group_index
        return groups, assignments

    def _fit_group(
        self,
        values: np.ndarray,
        trends: np.ndarray,
        seasonal: np.ndarray,
        locations: np.ndarray,
    ) -> _Group:
        locations = np.sort(locations)
        design = np.column_stack((trends, seasonal, np.ones(values.shape[0])))
        target = values[:, locations].mean(axis=1)
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        return _Group(locations.copy(), coefficients)

    def _split_locations(
        self, features: np.ndarray, locations: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if locations.size < 2:
            return None
        centered = features[locations] - features[locations].mean(axis=0)
        if np.linalg.norm(centered) < 1e-10:
            return None
        scores = np.linalg.svd(centered, full_matrices=False)[0][:, 0]
        order = np.argsort(scores)
        midpoint = locations.size // 2
        return locations[order[:midpoint]], locations[order[midpoint:]]

    def _split_improves(
        self,
        values: np.ndarray,
        trends: np.ndarray,
        seasonal: np.ndarray,
        parent: np.ndarray,
        children: Sequence[np.ndarray],
    ) -> bool:
        parent_rss = self._group_rss(
            self._fit_group(values, trends, seasonal, parent),
            values,
            trends,
            seasonal,
        )
        child_rss = sum(
            self._group_rss(
                self._fit_group(values, trends, seasonal, child),
                values,
                trends,
                seasonal,
            )
            for child in children
        )
        samples = values.shape[0] * parent.size * self._n_features
        parameters = (
            self._trend_rank + self._season_rank + 1
        ) * self._n_features
        parent_bic = self._gaussian_code(parent_rss, samples)
        parent_bic += parameters * np.log(samples)
        child_bic = self._gaussian_code(child_rss, samples)
        child_bic += len(children) * parameters * np.log(samples)
        return child_bic < parent_bic

    def _group_rss(
        self,
        group: _Group,
        values: np.ndarray,
        trends: np.ndarray,
        seasonal: np.ndarray,
    ) -> float:
        design = np.column_stack((trends, seasonal, np.ones(values.shape[0])))
        residual = values[:, group.locations] - (design @ group.coefficients)[:, None]
        return float(np.sum(residual * residual))

    def _data_code(
        self,
        regime: _Regime,
        values: np.ndarray,
        states: np.ndarray,
        trends: np.ndarray,
        seasonal: np.ndarray,
    ) -> Tuple[float, int]:
        observation_rss = sum(
            self._group_rss(group, values, trends, seasonal)
            for group in regime.groups
        )
        predicted = self._transition(regime, states[:-1])
        state_residual = (states[1:] - predicted) / regime.state_scale
        dynamic_rss = float(np.sum(state_residual * state_residual))
        count = values.size + state_residual.size
        return self._gaussian_code(observation_rss + dynamic_rss, count), count

    def _model_code(self, regime: _Regime) -> float:
        state_dimension = self._trend_rank + self._season_rank
        transition_parameters = state_dimension * (state_dimension + 2)
        group_parameters = len(regime.groups) * (
            self._trend_rank + self._season_rank + 1
        ) * self._n_features
        return (transition_parameters + group_parameters) * np.log(
            max(regime.sample_count, 2)
        )

    def _transition(self, regime: _Regime, states: np.ndarray) -> np.ndarray:
        return (
            regime.bias
            + states @ regime.linear.T
            + states * states * regime.quadratic
        )

    def _advance(self, regime: _Regime, state: np.ndarray) -> np.ndarray:
        following = self._transition(regime, state[None])[0]
        lower = regime.state_center - 6.0 * regime.state_scale
        upper = regime.state_center + 6.0 * regime.state_scale
        return np.clip(following, lower, upper)

    def _select_basis(self, matrix: np.ndarray, allow_zero: bool) -> np.ndarray:
        _, singular_values, vectors = np.linalg.svd(matrix, full_matrices=False)
        tolerance = (
            max(matrix.shape) * singular_values[0] * np.finfo(float).eps
            if singular_values.size
            else 0.0
        )
        numerical_rank = int(np.sum(singular_values > tolerance))
        minimum_rank = 0 if allow_zero else 1
        maximum = min(
            self.max_components,
            max(numerical_rank, minimum_rank),
            singular_values.size,
        )
        first = 0 if allow_zero else 1
        samples = matrix.size
        energy_floor = max(np.sum(singular_values**2) * 1e-10, 1e-12)
        scores = []

        for rank in range(first, maximum + 1):
            rss = max(np.sum(singular_values[rank:] ** 2), energy_floor)
            parameters = rank * (sum(matrix.shape) - rank)
            bic = self._gaussian_code(rss, samples)
            bic += parameters * np.log(max(samples, 2))
            scores.append((bic, rank))

        rank = min(scores)[1]
        return vectors[:rank]

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        if self.period is None:
            return values
        left = (self.period - 1) // 2
        right = self.period // 2
        padded = np.pad(values, ((left, right), (0, 0)), mode="edge")
        totals = np.vstack((np.zeros((1, values.shape[1])), padded.cumsum(axis=0)))
        return (totals[self.period :] - totals[: -self.period]) / self.period

    def _phase_means(self, values: np.ndarray, phases: np.ndarray) -> np.ndarray:
        result = np.zeros((self.period, values.shape[1]))
        for phase in range(self.period):
            selected = values[phases == phase]
            if selected.size:
                result[phase] = selected.mean(axis=0)
        return result

    def _normalize_history(self) -> np.ndarray:
        if np.any(np.all(np.isnan(self._history), axis=0)):
            raise ValueError("each location-feature series needs a finite value")
        self.mean_ = np.nanmean(self._history, axis=0)
        filled = np.where(np.isnan(self._history), self.mean_, self._history)
        self.scale_ = filled.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return (filled - self.mean_) / self.scale_

    def _phases(self) -> np.ndarray:
        start = self._time_seen - self._history.shape[0]
        return (start + np.arange(self._history.shape[0])) % (self.period or 1)

    def _trim(self, values: np.ndarray) -> np.ndarray:
        if self.window_size is None:
            return values.copy()
        return values[-self.window_size :].copy()

    def _check_training_length(self, length: int) -> None:
        if length < 4:
            raise ValueError("at least four time points are required")
        if self.period is not None and length < 2 * self.period:
            raise ValueError("seasonal fitting requires at least two periods")

    @staticmethod
    def _validate_tensor(tensor: np.ndarray) -> np.ndarray:
        values = np.asarray(tensor, dtype=float)
        if values.ndim != 3:
            raise ValueError("tensor must have shape (time, location, feature)")
        if 0 in values.shape:
            raise ValueError("tensor dimensions cannot be empty")
        if np.any(np.isinf(values)):
            raise ValueError("tensor cannot contain infinite values")
        return values

    @staticmethod
    def _gaussian_code(rss: float, samples: int) -> float:
        return samples * np.log(max(rss / max(samples, 1), 1e-12))

    def _publish_state(self) -> None:
        regime = self._regimes[self._active]
        self.n_regimes_ = len(self._regimes)
        self.active_regime_ = self._active
        self.groups_ = tuple(
            tuple(int(location) for location in group.locations)
            for group in regime.groups
        )
        self.trend_components_ = self._trend_rank
        self.seasonal_components_ = self._season_rank
        self.transition_optimizer_ = regime.optimizer

    def _cache_decomposition(
        self,
        normalized: np.ndarray,
        trends: np.ndarray,
        seasonal: np.ndarray,
    ) -> None:
        self._last_normalized = normalized.copy()
        self._last_trends = trends.copy()
        self._last_seasonal = seasonal.copy()

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit must be called before this operation")
