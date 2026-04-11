"""Synthetic dataset generation utilities for filter benchmarking.

The generators in this module are built around the observation model

    y_obs = f(x, state) + noise

and are intended for benchmarking Kalman filters, particle filters, and more
robust alternatives under known signal and noise assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
import math
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


SignalGenerator = Callable[..., "SignalResult"]
NoiseGenerator = Callable[..., "NoiseResult"]

SIGNAL_MODELS: Dict[str, SignalGenerator] = {}
NOISE_MODELS: Dict[str, NoiseGenerator] = {}


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a single signal or noise model."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeSchedule:
    """Sequence of models to apply across x with explicit switch times.

    The first regime is active from the beginning of the series until the first
    switch time. Each additional switch time activates the next regime.

    Example:
        RegimeSchedule(
            regimes=(
                ModelSpec("linear", {"slope": 0.5}),
                ModelSpec("sinusoidal", {"amplitude": 2.0, "frequency": 0.03}),
                ModelSpec("random_walk", {"step_scale": 0.4}),
            ),
            switch_times=(100, 250),
        )
    """

    regimes: Sequence[Union[str, ModelSpec, Mapping[str, Any]]]
    switch_times: Sequence[float]
    carry_forward_state: bool = False


@dataclass(frozen=True)
class InputSpec:
    """Configuration for generating x values."""

    kind: str = "grid"
    params: Dict[str, Any] = field(default_factory=dict)
    sort_values: bool = True


@dataclass(frozen=True)
class DatasetConfig:
    """High-level dataset generation configuration."""

    n_samples: int = 500
    input_spec: InputSpec = field(default_factory=InputSpec)
    signal: Union[str, ModelSpec, Mapping[str, Any], RegimeSchedule] = field(
        default_factory=lambda: ModelSpec("linear", {"slope": 1.0})
    )
    noise: Optional[Union[str, ModelSpec, Mapping[str, Any], RegimeSchedule]] = field(
        default_factory=lambda: ModelSpec("gaussian", {"scale": 0.25})
    )
    noise_mode: str = "additive"
    mixed_multiplicative_weight: float = 1.0
    x: Optional[Sequence[float]] = None
    seed: Optional[int] = None
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalResult:
    values: np.ndarray
    latent_state: Optional[np.ndarray] = None
    extra_columns: Dict[str, np.ndarray] = field(default_factory=dict)
    terminal_state: Dict[str, float] = field(default_factory=dict)


@dataclass
class NoiseResult:
    values: np.ndarray
    extra_columns: Dict[str, np.ndarray] = field(default_factory=dict)
    terminal_state: Dict[str, float] = field(default_factory=dict)


@dataclass
class GeneratedDataset:
    """Dataset plus generation metadata."""

    data: pd.DataFrame
    config: DatasetConfig
    metadata: Dict[str, Any]

    def summary(self) -> pd.Series:
        return dataset_summary(self)


@dataclass(frozen=True)
class StandardSuiteCase:
    """One dataset specification in the standard filter benchmark suite."""

    name: str
    signal_name: str
    noise_name: str
    config: DatasetConfig


def signal_model(name: str) -> Callable[[SignalGenerator], SignalGenerator]:
    def decorator(func: SignalGenerator) -> SignalGenerator:
        SIGNAL_MODELS[name] = func
        return func

    return decorator


def noise_model(name: str) -> Callable[[NoiseGenerator], NoiseGenerator]:
    def decorator(func: NoiseGenerator) -> NoiseGenerator:
        NOISE_MODELS[name] = func
        return func

    return decorator


def model(name: str, **params: Any) -> ModelSpec:
    return ModelSpec(name=name, params=dict(params))


def regime_schedule(
    regimes: Sequence[Union[str, ModelSpec, Mapping[str, Any]]],
    switch_times: Sequence[float],
    *,
    carry_forward_state: bool = False,
) -> RegimeSchedule:
    return RegimeSchedule(
        regimes=tuple(regimes),
        switch_times=tuple(switch_times),
        carry_forward_state=carry_forward_state,
    )


def available_signal_models() -> list[str]:
    return sorted(SIGNAL_MODELS)


def available_noise_models() -> list[str]:
    return sorted(NOISE_MODELS)


def list_presets() -> list[str]:
    return sorted(PRESET_BUILDERS)


def generate_dataset(
    config: Optional[DatasetConfig] = None,
    *,
    n_samples: int = 500,
    input_spec: Optional[InputSpec] = None,
    signal: Union[str, ModelSpec, Mapping[str, Any], RegimeSchedule] = "linear",
    noise: Optional[Union[str, ModelSpec, Mapping[str, Any], RegimeSchedule]] = "gaussian",
    noise_mode: str = "additive",
    mixed_multiplicative_weight: float = 1.0,
    x: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    name: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> GeneratedDataset:
    """Generate a synthetic dataset.

    If `config` is supplied it is used as-is. Otherwise a DatasetConfig is
    constructed from the keyword arguments.
    """

    if config is None:
        config = DatasetConfig(
            n_samples=n_samples,
            input_spec=input_spec or InputSpec(),
            signal=signal,
            noise=noise,
            noise_mode=noise_mode,
            mixed_multiplicative_weight=mixed_multiplicative_weight,
            x=x,
            seed=seed,
            name=name,
            metadata=metadata or {},
        )

    rng = np.random.default_rng(config.seed)
    x_values = _resolve_inputs(config, rng)

    signal_result, signal_regime_index = _evaluate_signal_spec(config.signal, x_values, rng)
    y_true = signal_result.values

    if config.noise is None:
        noise_result = NoiseResult(values=np.zeros_like(y_true))
        noise_regime_index = np.zeros(len(y_true), dtype=int)
    else:
        noise_result, noise_regime_index = _evaluate_noise_spec(config.noise, x_values, y_true, rng)

    realized_noise = _apply_noise(
        y_true,
        noise_result.values,
        mode=config.noise_mode,
        mixed_multiplicative_weight=config.mixed_multiplicative_weight,
    )
    y_obs = y_true + realized_noise

    frame = pd.DataFrame(
        {
            "x": x_values,
            "y_true": y_true,
            "noise_draw": noise_result.values,
            "noise": realized_noise,
            "y_obs": y_obs,
            "signal_regime": signal_regime_index.astype(int),
            "noise_regime": noise_regime_index.astype(int),
        }
    )

    if signal_result.latent_state is not None:
        frame["latent_state"] = signal_result.latent_state

    for column_name, values in signal_result.extra_columns.items():
        frame[column_name] = values

    for column_name, values in noise_result.extra_columns.items():
        frame[column_name] = values

    dataset_name = config.name or _default_dataset_name(config)
    metadata_dict = {
        "name": dataset_name,
        "noise_mode": config.noise_mode,
        "seed": config.seed,
        "signal": _serialize_spec(config.signal),
        "noise": _serialize_spec(config.noise),
        "input_spec": asdict(config.input_spec),
        "n_samples": len(frame),
    }
    metadata_dict.update(config.metadata)

    return GeneratedDataset(data=frame, config=config, metadata=metadata_dict)


def generate_preset(name: str, *, n_samples: int = 500, seed: Optional[int] = None) -> GeneratedDataset:
    return generate_dataset(get_preset_config(name, n_samples=n_samples, seed=seed))


def get_preset_config(name: str, *, n_samples: int = 500, seed: Optional[int] = None) -> DatasetConfig:
    try:
        builder = PRESET_BUILDERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preset '{name}'. Available presets: {list_presets()}") from exc
    return builder(n_samples=n_samples, seed=seed)


def generate_benchmark_suite(
    preset_names: Optional[Sequence[str]] = None,
    *,
    n_samples: int = 500,
    base_seed: int = 0,
) -> Dict[str, GeneratedDataset]:
    selected = list(preset_names or list_presets())
    return {
        preset_name: generate_preset(preset_name, n_samples=n_samples, seed=base_seed + index)
        for index, preset_name in enumerate(selected)
    }


def get_standard_suite_cases(
    *,
    n_samples: int = 500,
    base_seed: Optional[int] = 0,
) -> list[StandardSuiteCase]:
    """Build the standard grid benchmark suite.

    The suite is the Cartesian product of:
        true values = linear, logistic, exponential, logarithmic, mean reverting curve
        noise = normal, Laplace, uniform, Cauchy, gamma, Poisson
    """

    signal_specs = _standard_suite_signal_specs()
    noise_specs = _standard_suite_noise_specs()

    cases: list[StandardSuiteCase] = []
    case_index = 0
    for signal_name, signal_spec in signal_specs.items():
        for noise_name, noise_spec in noise_specs.items():
            case_seed = None if base_seed is None else base_seed + case_index
            case_name = f"{signal_name}__{noise_name}"
            config = DatasetConfig(
                n_samples=n_samples,
                input_spec=InputSpec(kind="grid", params={"start": 0.0, "stop": 10.0}),
                signal=signal_spec,
                noise=noise_spec,
                noise_mode="additive",
                seed=case_seed,
                name=case_name,
                metadata={
                    "suite": "standard_grid",
                    "signal_name": signal_name,
                    "noise_name": noise_name,
                },
            )
            cases.append(
                StandardSuiteCase(
                    name=case_name,
                    signal_name=signal_name,
                    noise_name=noise_name,
                    config=config,
                )
            )
            case_index += 1

    return cases


def generate_standard_suite(
    *,
    n_samples: int = 500,
    base_seed: Optional[int] = 0,
) -> Dict[str, GeneratedDataset]:
    """Generate all datasets in the standard grid benchmark suite."""

    return {
        case.name: generate_dataset(case.config)
        for case in get_standard_suite_cases(n_samples=n_samples, base_seed=base_seed)
    }


def test_filter_on_standard_suite(
    filter_func: Callable[..., Any],
    *,
    n_samples: int = 500,
    base_seed: Optional[int] = 0,
    target_column: str = "y_true",
    estimate_key: Optional[str] = None,
    estimate_column: int = 0,
    include_failures: bool = True,
) -> pd.DataFrame:
    """Run a filter over the standard suite and return a score summary.

    The filter can use any of these common signatures:
        filter_func(y_obs)
        filter_func(y_obs, x)
        filter_func(dataset=dataset, y_obs=y_obs, x=x, y_true=y_true)
        filter_func(dataframe)

    The filter should return an array-like estimate with one estimate per
    observation. If it returns a mapping or dataframe, set `estimate_key` to the
    output column/key to score.
    """

    rows: list[Dict[str, Any]] = []

    for case in get_standard_suite_cases(n_samples=n_samples, base_seed=base_seed):
        dataset = generate_dataset(case.config)
        row: Dict[str, Any] = {
            "case": case.name,
            "signal": case.signal_name,
            "noise": case.noise_name,
            "n_samples": len(dataset.data),
            "seed": dataset.config.seed,
        }

        try:
            filter_output = _call_filter_func(filter_func, dataset)
            estimates = _coerce_filter_estimates(
                filter_output,
                n_samples=len(dataset.data),
                estimate_key=estimate_key,
                estimate_column=estimate_column,
            )
            row.update(_score_estimates(dataset, estimates, target_column=target_column))
            row["status"] = "ok"
            row["error"] = ""
        except Exception as exc:
            if not include_failures:
                raise
            row.update(
                {
                    "rmse": np.nan,
                    "mae": np.nan,
                    "bias": np.nan,
                    "error_std": np.nan,
                    "max_abs_error": np.nan,
                    "corr": np.nan,
                    "status": "failed",
                    "error": str(exc),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def dataset_summary(dataset: GeneratedDataset) -> pd.Series:
    frame = dataset.data
    noise = frame["noise"].to_numpy()
    signal_values = frame["y_true"].to_numpy()
    signal_std = float(np.std(signal_values, ddof=0))
    noise_std = float(np.std(noise, ddof=0))
    snr = np.inf if noise_std == 0 else signal_std / noise_std

    return pd.Series(
        {
            "rows": int(len(frame)),
            "x_min": float(np.min(frame["x"])),
            "x_max": float(np.max(frame["x"])),
            "signal_mean": float(np.mean(signal_values)),
            "signal_std": signal_std,
            "noise_mean": float(np.mean(noise)),
            "noise_std": noise_std,
            "observed_mean": float(np.mean(frame["y_obs"])),
            "observed_std": float(np.std(frame["y_obs"], ddof=0)),
            "snr_like": float(snr),
        },
        name=dataset.metadata.get("name", "dataset"),
    )


def plot_dataset(
    dataset: GeneratedDataset,
    *,
    include_noise: bool = False,
    title: Optional[str] = None,
):
    """Quick plot helper. Raises ImportError if matplotlib is unavailable."""

    import matplotlib.pyplot as plt

    frame = dataset.data
    fig, axes = plt.subplots(2 if include_noise else 1, 1, sharex=True, figsize=(11, 6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].plot(frame["x"], frame["y_true"], label="True signal", linewidth=2)
    axes[0].scatter(frame["x"], frame["y_obs"], label="Observed", s=12, alpha=0.6)
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="best")
    axes[0].set_title(title or dataset.metadata.get("name", "Synthetic dataset"))

    if include_noise:
        axes[1].plot(frame["x"], frame["noise"], label="Realized noise")
        axes[1].set_ylabel("Noise")
        axes[1].legend(loc="best")

    axes[-1].set_xlabel("x")
    fig.tight_layout()
    return fig, axes


def _resolve_inputs(config: DatasetConfig, rng: np.random.Generator) -> np.ndarray:
    if config.x is not None:
        x_values = np.asarray(config.x, dtype=float)
        if x_values.ndim != 1:
            raise ValueError("Custom x must be one-dimensional.")
        return x_values
    return _sample_inputs(config.n_samples, config.input_spec, rng)


def _sample_inputs(
    n_samples: int,
    input_spec: InputSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    kind = input_spec.kind.lower()
    params = dict(input_spec.params)

    if kind == "grid":
        start = params.get("start", 0.0)
        stop = params.get("stop", float(n_samples - 1))
        endpoint = params.get("endpoint", True)
        x_values = np.linspace(start, stop, n_samples, endpoint=endpoint)
    elif kind == "uniform":
        low = params.get("low", 0.0)
        high = params.get("high", 1.0)
        x_values = rng.uniform(low, high, n_samples)
    elif kind == "normal":
        mean = params.get("mean", 0.0)
        scale = params.get("scale", 1.0)
        x_values = rng.normal(mean, scale, n_samples)
    elif kind == "mixture_gaussian":
        means = np.asarray(params.get("means", (-2.0, 2.0)), dtype=float)
        scales = np.asarray(params.get("scales", (0.6, 0.6)), dtype=float)
        weights = np.asarray(params.get("weights", np.ones(len(means))), dtype=float)
        if not (len(means) == len(scales) == len(weights)):
            raise ValueError("means, scales, and weights must have the same length.")
        weights = weights / np.sum(weights)
        components = rng.choice(len(means), size=n_samples, p=weights)
        x_values = rng.normal(means[components], scales[components])
    elif kind == "irregular_time":
        start = params.get("start", 0.0)
        mean_step = params.get("mean_step", 1.0)
        if mean_step <= 0:
            raise ValueError("mean_step must be positive.")
        increments = rng.exponential(mean_step, size=max(n_samples - 1, 1))
        x_values = np.empty(n_samples, dtype=float)
        x_values[0] = start
        if n_samples > 1:
            x_values[1:] = start + np.cumsum(increments)
    else:
        raise ValueError(
            f"Unknown input kind '{input_spec.kind}'. "
            "Available kinds: grid, uniform, normal, mixture_gaussian, irregular_time."
        )

    if input_spec.sort_values:
        return np.sort(x_values)
    return x_values


def _evaluate_signal_spec(
    spec: Union[str, ModelSpec, Mapping[str, Any], RegimeSchedule],
    x_values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[SignalResult, np.ndarray]:
    schedule = _coerce_regime_schedule(spec)
    if schedule is None:
        result = _run_signal_model(_coerce_model_spec(spec), x_values, rng, initial_state=None)
        return result, np.zeros(len(x_values), dtype=int)
    return _evaluate_signal_schedule(schedule, x_values, rng)


def _evaluate_noise_spec(
    spec: Union[str, ModelSpec, Mapping[str, Any], RegimeSchedule],
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
) -> tuple[NoiseResult, np.ndarray]:
    schedule = _coerce_regime_schedule(spec)
    if schedule is None:
        result = _run_noise_model(_coerce_model_spec(spec), x_values, y_true, rng, initial_state=None)
        return result, np.zeros(len(x_values), dtype=int)
    return _evaluate_noise_schedule(schedule, x_values, y_true, rng)


def _evaluate_signal_schedule(
    schedule: RegimeSchedule,
    x_values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[SignalResult, np.ndarray]:
    regimes = [_coerce_model_spec(item) for item in schedule.regimes]
    boundaries = _regime_boundaries(x_values, schedule.switch_times, len(regimes))

    values = np.empty(len(x_values), dtype=float)
    regime_index = np.empty(len(x_values), dtype=int)
    latent_state: Optional[np.ndarray] = None
    extra_columns: Dict[str, np.ndarray] = {}
    previous_terminal_state: Optional[Dict[str, float]] = None

    for idx, spec in enumerate(regimes):
        start, stop = boundaries[idx], boundaries[idx + 1]
        initial_state = previous_terminal_state if schedule.carry_forward_state else None
        result = _run_signal_model(spec, x_values[start:stop], rng, initial_state=initial_state)

        values[start:stop] = result.values
        regime_index[start:stop] = idx

        if result.latent_state is not None:
            if latent_state is None:
                latent_state = np.full(len(x_values), np.nan, dtype=float)
            latent_state[start:stop] = result.latent_state

        for column_name, segment_values in result.extra_columns.items():
            if column_name not in extra_columns:
                extra_columns[column_name] = np.full(len(x_values), np.nan, dtype=float)
            extra_columns[column_name][start:stop] = segment_values

        previous_terminal_state = result.terminal_state

    terminal_state = previous_terminal_state or {}
    return (
        SignalResult(
            values=values,
            latent_state=latent_state,
            extra_columns=extra_columns,
            terminal_state=terminal_state,
        ),
        regime_index,
    )


def _evaluate_noise_schedule(
    schedule: RegimeSchedule,
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
) -> tuple[NoiseResult, np.ndarray]:
    regimes = [_coerce_model_spec(item) for item in schedule.regimes]
    boundaries = _regime_boundaries(x_values, schedule.switch_times, len(regimes))

    values = np.empty(len(x_values), dtype=float)
    regime_index = np.empty(len(x_values), dtype=int)
    extra_columns: Dict[str, np.ndarray] = {}
    previous_terminal_state: Optional[Dict[str, float]] = None

    for idx, spec in enumerate(regimes):
        start, stop = boundaries[idx], boundaries[idx + 1]
        initial_state = previous_terminal_state if schedule.carry_forward_state else None
        result = _run_noise_model(
            spec,
            x_values[start:stop],
            y_true[start:stop],
            rng,
            initial_state=initial_state,
        )

        values[start:stop] = result.values
        regime_index[start:stop] = idx

        for column_name, segment_values in result.extra_columns.items():
            if column_name not in extra_columns:
                extra_columns[column_name] = np.full(len(x_values), np.nan, dtype=float)
            extra_columns[column_name][start:stop] = segment_values

        previous_terminal_state = result.terminal_state

    terminal_state = previous_terminal_state or {}
    return NoiseResult(values=values, extra_columns=extra_columns, terminal_state=terminal_state), regime_index


def _run_signal_model(
    spec: ModelSpec,
    x_values: np.ndarray,
    rng: np.random.Generator,
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    try:
        generator = SIGNAL_MODELS[spec.name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown signal model '{spec.name}'. Available models: {available_signal_models()}"
        ) from exc
    return generator(x_values, rng, spec.params, initial_state=initial_state)


def _run_noise_model(
    spec: ModelSpec,
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    try:
        generator = NOISE_MODELS[spec.name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown noise model '{spec.name}'. Available models: {available_noise_models()}"
        ) from exc
    return generator(x_values, y_true, rng, spec.params, initial_state=initial_state)


def _coerce_model_spec(spec: Union[str, ModelSpec, Mapping[str, Any]]) -> ModelSpec:
    if isinstance(spec, ModelSpec):
        return spec
    if isinstance(spec, str):
        return ModelSpec(name=spec, params={})
    if isinstance(spec, Mapping):
        if "name" not in spec:
            raise ValueError("Model mappings must include a 'name' field.")
        name = str(spec["name"])
        params = dict(spec.get("params", {}))
        for key, value in spec.items():
            if key not in {"name", "params"}:
                params[key] = value
        return ModelSpec(name=name, params=params)
    raise TypeError(f"Unsupported model specification: {type(spec)!r}")


def _coerce_regime_schedule(
    spec: Union[str, ModelSpec, Mapping[str, Any], RegimeSchedule],
) -> Optional[RegimeSchedule]:
    if isinstance(spec, RegimeSchedule):
        return _validate_regime_schedule(spec)
    if isinstance(spec, Mapping) and "regimes" in spec:
        schedule = RegimeSchedule(
            regimes=tuple(spec["regimes"]),
            switch_times=tuple(spec.get("switch_times", ())),
            carry_forward_state=bool(spec.get("carry_forward_state", False)),
        )
        return _validate_regime_schedule(schedule)
    return None


def _validate_regime_schedule(schedule: RegimeSchedule) -> RegimeSchedule:
    if len(schedule.regimes) == 0:
        raise ValueError("Regime schedules must contain at least one regime.")
    if len(schedule.regimes) != len(schedule.switch_times) + 1:
        raise ValueError("Regime schedules require len(regimes) == len(switch_times) + 1.")
    switch_times = np.asarray(schedule.switch_times, dtype=float)
    if len(switch_times) > 1 and np.any(np.diff(switch_times) <= 0):
        raise ValueError("switch_times must be strictly increasing.")
    return schedule


def _regime_boundaries(
    x_values: np.ndarray,
    switch_times: Sequence[float],
    regime_count: int,
) -> list[int]:
    if len(x_values) == 0:
        raise ValueError("x_values must not be empty.")
    if not np.all(np.diff(x_values) >= 0):
        raise ValueError("Regime switching requires x to be sorted in ascending order.")

    boundaries = [0]
    for switch_time in switch_times:
        boundary = int(np.searchsorted(x_values, switch_time, side="left"))
        boundaries.append(boundary)
    boundaries.append(len(x_values))

    if len(boundaries) != regime_count + 1:
        raise ValueError("Boundary count does not match the regime count.")

    for start, stop in zip(boundaries, boundaries[1:]):
        if stop <= start:
            raise ValueError(
                "Each regime must cover at least one sample. Adjust switch_times or the x domain."
            )

    return boundaries


def _apply_noise(
    y_true: np.ndarray,
    noise_draw: np.ndarray,
    *,
    mode: str,
    mixed_multiplicative_weight: float,
) -> np.ndarray:
    normalized_mode = mode.lower()
    if normalized_mode == "additive":
        return noise_draw
    if normalized_mode == "multiplicative":
        return y_true * noise_draw
    if normalized_mode == "mixed":
        return noise_draw + mixed_multiplicative_weight * y_true * noise_draw
    raise ValueError("noise_mode must be 'additive', 'multiplicative', or 'mixed'.")


def _standard_suite_signal_specs() -> Dict[str, ModelSpec]:
    return {
        "linear": model("linear", intercept=1.0, slope=0.6),
        "logistic": model("logistic", lower=0.0, upper=5.0, midpoint=5.0, growth=1.1),
        "exponential": model("exponential", amplitude=1.0, rate=0.22, offset=0.0),
        "logarithmic": model("logarithmic", scale=2.0, offset=0.0, shift=1.0),
        "mean_reverting_curve": model("mean_reverting_curve", mean=1.0, initial_value=5.0, rate=0.55),
    }


def _standard_suite_noise_specs() -> Dict[str, ModelSpec]:
    return {
        "normal": model("gaussian", loc=0.0, scale=0.5),
        "laplace": model("laplace", loc=0.0, scale=0.35),
        "uniform": model("uniform", low=-0.75, high=0.75),
        "cauchy": model("cauchy", loc=0.0, scale=0.2),
        "gamma": model("gamma", shape=2.0, scale=0.35, center=True),
        "poisson": model("poisson", rate=1.0, center=True),
    }


def _call_filter_func(filter_func: Callable[..., Any], dataset: GeneratedDataset) -> Any:
    frame = dataset.data
    y_obs = frame["y_obs"].to_numpy()
    x_values = frame["x"].to_numpy()
    y_true = frame["y_true"].to_numpy()

    aliases: Dict[str, Any] = {
        "dataset": dataset,
        "generated_dataset": dataset,
        "data": frame,
        "df": frame,
        "dataframe": frame,
        "frame": frame,
        "x": x_values,
        "t": x_values,
        "time": x_values,
        "times": x_values,
        "y": y_obs,
        "z": y_obs,
        "y_obs": y_obs,
        "observed": y_obs,
        "observation": y_obs,
        "observations": y_obs,
        "measurements": y_obs,
        "y_true": y_true,
        "truth": y_true,
        "target": y_true,
    }

    try:
        signature = inspect.signature(filter_func)
    except (TypeError, ValueError):
        return filter_func(y_obs)

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return filter_func(**aliases)

    positional_args = []
    keyword_args: Dict[str, Any] = {}
    matched_any = False

    for parameter in signature.parameters.values():
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if parameter.name not in aliases:
            continue

        matched_any = True
        value = aliases[parameter.name]
        if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            positional_args.append(value)
        else:
            keyword_args[parameter.name] = value

    if matched_any:
        return filter_func(*positional_args, **keyword_args)

    positional_params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if len(positional_params) >= 2:
        return filter_func(y_obs, x_values)
    return filter_func(y_obs)


def _coerce_filter_estimates(
    filter_output: Any,
    *,
    n_samples: int,
    estimate_key: Optional[str],
    estimate_column: int,
) -> np.ndarray:
    if isinstance(filter_output, Mapping):
        if estimate_key is None:
            candidate_keys = ("estimate", "estimates", "y_hat", "filtered", "prediction", "predictions")
            estimate_key = next((key for key in candidate_keys if key in filter_output), None)
        if estimate_key is None or estimate_key not in filter_output:
            raise ValueError("Filter returned a mapping; provide estimate_key or use a standard estimate key.")
        raw_estimates = filter_output[estimate_key]
    elif isinstance(filter_output, pd.DataFrame):
        if estimate_key is None:
            candidate_keys = ("estimate", "estimates", "y_hat", "filtered", "prediction", "predictions")
            estimate_key = next((key for key in candidate_keys if key in filter_output.columns), None)
        if estimate_key is None:
            if filter_output.shape[1] == 1:
                raw_estimates = filter_output.iloc[:, 0]
            else:
                raise ValueError("Filter returned a dataframe; provide estimate_key for the estimate column.")
        else:
            raw_estimates = filter_output[estimate_key]
    elif isinstance(filter_output, pd.Series):
        raw_estimates = filter_output
    elif isinstance(filter_output, tuple) and filter_output:
        raw_estimates = filter_output[0]
    else:
        raw_estimates = filter_output

    estimates = np.asarray(raw_estimates, dtype=float)
    estimates = np.squeeze(estimates)

    if estimates.ndim == 2 and estimates.shape[0] == n_samples:
        estimates = estimates[:, estimate_column]
    elif estimates.ndim != 1:
        raise ValueError("Filter estimates must be one-dimensional or an n_samples x k array.")

    if len(estimates) != n_samples:
        raise ValueError(f"Filter returned {len(estimates)} estimates for {n_samples} observations.")

    return estimates


def _score_estimates(
    dataset: GeneratedDataset,
    estimates: np.ndarray,
    *,
    target_column: str,
) -> Dict[str, float]:
    if target_column not in dataset.data:
        raise ValueError(f"target_column '{target_column}' is not present in the generated dataset.")

    truth = dataset.data[target_column].to_numpy(dtype=float)
    errors = estimates - truth

    if np.std(estimates) == 0 or np.std(truth) == 0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(estimates, truth)[0, 1])

    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "error_std": float(np.std(errors, ddof=0)),
        "max_abs_error": float(np.max(np.abs(errors))),
        "corr": corr,
    }


def _default_dataset_name(config: DatasetConfig) -> str:
    signal_name = _spec_name(config.signal)
    noise_name = _spec_name(config.noise) if config.noise is not None else "noiseless"
    return f"{signal_name}_{noise_name}_{config.noise_mode}"


def _spec_name(spec: Optional[Union[str, ModelSpec, Mapping[str, Any], RegimeSchedule]]) -> str:
    if spec is None:
        return "none"
    if isinstance(spec, RegimeSchedule):
        first = _coerce_model_spec(spec.regimes[0]).name
        return f"regime_{first}"
    if isinstance(spec, Mapping) and "regimes" in spec:
        first = _coerce_model_spec(spec["regimes"][0]).name
        return f"regime_{first}"
    return _coerce_model_spec(spec).name


def _serialize_spec(spec: Optional[Union[str, ModelSpec, Mapping[str, Any], RegimeSchedule]]) -> Any:
    if spec is None:
        return None
    if isinstance(spec, RegimeSchedule):
        return {
            "regimes": [asdict(_coerce_model_spec(item)) for item in spec.regimes],
            "switch_times": list(spec.switch_times),
            "carry_forward_state": spec.carry_forward_state,
        }
    if isinstance(spec, Mapping) and "regimes" in spec:
        coerced = _coerce_regime_schedule(spec)
        return _serialize_spec(coerced)
    return asdict(_coerce_model_spec(spec))


def _time_steps(x_values: np.ndarray) -> np.ndarray:
    if len(x_values) == 0:
        return np.array([], dtype=float)
    if len(x_values) == 1:
        return np.array([1.0], dtype=float)
    deltas = np.diff(x_values)
    fallback = float(np.median(deltas[deltas > 0])) if np.any(deltas > 0) else 1.0
    dt = np.empty(len(x_values), dtype=float)
    dt[0] = fallback
    dt[1:] = np.where(deltas > 0, deltas, fallback)
    return dt


def _invoke_user_function(function: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(function)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return function(**kwargs)

    filtered_kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}
    if filtered_kwargs:
        return function(**filtered_kwargs)
    return function(kwargs["x"])


def _coerce_signal_result(raw: Any, x_values: np.ndarray) -> SignalResult:
    if isinstance(raw, Mapping):
        values = np.asarray(raw.get("values", raw.get("signal")), dtype=float)
        latent_state = raw.get("latent_state")
        extra_columns = {
            name: np.asarray(values_array, dtype=float)
            for name, values_array in dict(raw.get("extra_columns", {})).items()
        }
        terminal_state = dict(raw.get("terminal_state", {}))
    else:
        values = np.asarray(raw, dtype=float)
        latent_state = None
        extra_columns = {}
        terminal_state = {}

    if values.shape != x_values.shape:
        raise ValueError("Signal output must have the same shape as x.")

    coerced_latent = None if latent_state is None else np.asarray(latent_state, dtype=float)
    if coerced_latent is not None and coerced_latent.shape != x_values.shape:
        raise ValueError("latent_state must have the same shape as x.")

    if "state" not in terminal_state and len(values) > 0:
        terminal_state["state"] = float(values[-1])

    return SignalResult(
        values=values,
        latent_state=coerced_latent,
        extra_columns=extra_columns,
        terminal_state=terminal_state,
    )


def _coerce_noise_result(raw: Any, x_values: np.ndarray) -> NoiseResult:
    if isinstance(raw, Mapping):
        values = np.asarray(raw.get("values", raw.get("noise")), dtype=float)
        extra_columns = {
            name: np.asarray(values_array, dtype=float)
            for name, values_array in dict(raw.get("extra_columns", {})).items()
        }
        terminal_state = dict(raw.get("terminal_state", {}))
    else:
        values = np.asarray(raw, dtype=float)
        extra_columns = {}
        terminal_state = {}

    if values.shape != x_values.shape:
        raise ValueError("Noise output must have the same shape as x.")

    if "state" not in terminal_state and len(values) > 0:
        terminal_state["state"] = float(values[-1])

    return NoiseResult(values=values, extra_columns=extra_columns, terminal_state=terminal_state)


@signal_model("callable")
def _callable_signal(
    x_values: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    function = params.get("function")
    if function is None or not callable(function):
        raise ValueError("Callable signal requires a callable 'function' parameter.")

    kwargs = dict(params)
    kwargs.pop("function", None)
    raw = _invoke_user_function(function, x=x_values, rng=rng, initial_state=initial_state, **kwargs)
    return _coerce_signal_result(raw, x_values)


@signal_model("constant")
def _constant_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = initial_state
    level = float(params.get("level", 0.0))
    values = np.full(len(x_values), level, dtype=float)
    return SignalResult(values=values, terminal_state={"state": float(values[-1])})


@signal_model("linear")
def _linear_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = initial_state
    slope = float(params.get("slope", 1.0))
    intercept = float(params.get("intercept", 0.0))
    values = intercept + slope * x_values
    return SignalResult(values=values, terminal_state={"state": float(values[-1])})


@signal_model("polynomial")
def _polynomial_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = initial_state
    coefficients = np.asarray(params.get("coefficients", (1.0, 0.0)), dtype=float)
    values = np.polyval(coefficients, x_values)
    return SignalResult(values=values, terminal_state={"state": float(values[-1])})


@signal_model("exponential")
def _exponential_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = initial_state
    amplitude = float(params.get("amplitude", 1.0))
    rate = float(params.get("rate", 0.05))
    offset = float(params.get("offset", 0.0))
    values = offset + amplitude * np.exp(rate * x_values)
    return SignalResult(values=values, terminal_state={"state": float(values[-1])})


@signal_model("logarithmic")
def _logarithmic_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = initial_state
    scale = float(params.get("scale", 1.0))
    offset = float(params.get("offset", 0.0))
    shift = float(params.get("shift", 1.0))
    base = float(params.get("base", math.e))
    shifted_x = np.clip(x_values + shift, 1e-12, None)
    values = offset + scale * np.log(shifted_x)
    if base != math.e:
        values = offset + scale * np.log(shifted_x) / math.log(base)
    return SignalResult(values=values, terminal_state={"state": float(values[-1])})


@signal_model("sinusoidal")
def _sinusoidal_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = initial_state
    amplitude = float(params.get("amplitude", 1.0))
    frequency = float(params.get("frequency", 0.05))
    phase = float(params.get("phase", 0.0))
    offset = float(params.get("offset", 0.0))
    values = offset + amplitude * np.sin(2.0 * np.pi * frequency * x_values + phase)
    return SignalResult(values=values, terminal_state={"state": float(values[-1])})


@signal_model("multi_sine")
def _multi_sine_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = initial_state
    offset = float(params.get("offset", 0.0))
    components = params.get("components")
    if not components:
        components = (
            {"amplitude": 1.0, "frequency": 0.03, "phase": 0.0},
            {"amplitude": 0.4, "frequency": 0.11, "phase": 0.5},
        )
    values = np.full(len(x_values), offset, dtype=float)
    for component in components:
        values += float(component.get("amplitude", 1.0)) * np.sin(
            2.0 * np.pi * float(component.get("frequency", 0.05)) * x_values
            + float(component.get("phase", 0.0))
        )
    return SignalResult(values=values, terminal_state={"state": float(values[-1])})


@signal_model("piecewise_linear")
def _piecewise_linear_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = initial_state
    knots = np.asarray(params.get("knots", (0.0, 50.0, 100.0)), dtype=float)
    values_at_knots = np.asarray(params.get("values", (0.0, 5.0, 1.0)), dtype=float)
    if len(knots) != len(values_at_knots):
        raise ValueError("piecewise_linear requires 'knots' and 'values' with matching lengths.")
    values = np.interp(x_values, knots, values_at_knots)
    return SignalResult(values=values, terminal_state={"state": float(values[-1])})


@signal_model("piecewise_constant")
def _piecewise_constant_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = initial_state
    change_points = np.asarray(params.get("change_points", (100.0, 200.0)), dtype=float)
    values_per_regime = np.asarray(params.get("values", (0.0, 2.5, -1.0)), dtype=float)
    if len(values_per_regime) != len(change_points) + 1:
        raise ValueError("piecewise_constant requires len(values) == len(change_points) + 1.")

    regime_index = np.searchsorted(change_points, x_values, side="left")
    values = values_per_regime[regime_index]
    return SignalResult(values=values, terminal_state={"state": float(values[-1])})


@signal_model("logistic")
def _logistic_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = initial_state
    lower = float(params.get("lower", 0.0))
    upper = float(params.get("upper", 1.0))
    midpoint = float(params.get("midpoint", np.median(x_values)))
    growth = float(params.get("growth", 0.2))
    values = lower + (upper - lower) / (1.0 + np.exp(-growth * (x_values - midpoint)))
    return SignalResult(values=values, terminal_state={"state": float(values[-1])})


@signal_model("gaussian_bump")
def _gaussian_bump_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = initial_state
    amplitude = float(params.get("amplitude", 1.0))
    center = float(params.get("center", np.mean(x_values)))
    width = float(params.get("width", max(np.std(x_values), 1.0)))
    offset = float(params.get("offset", 0.0))
    values = offset + amplitude * np.exp(-0.5 * ((x_values - center) / width) ** 2)
    return SignalResult(values=values, terminal_state={"state": float(values[-1])})


@signal_model("mean_reverting_curve")
def _mean_reverting_curve_signal(
    x_values: np.ndarray,
    _rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    mean_level = float(params.get("mean", 0.0))
    rate = float(params.get("rate", 0.2))
    initial_value = float(
        params.get("initial_value", initial_state.get("state", mean_level) if initial_state else mean_level)
    )
    start = float(params.get("start", x_values[0] if len(x_values) else 0.0))
    elapsed = np.maximum(x_values - start, 0.0)
    values = mean_level + (initial_value - mean_level) * np.exp(-rate * elapsed)
    return SignalResult(
        values=values,
        latent_state=values.copy(),
        terminal_state={"state": float(values[-1])},
    )


@signal_model("random_walk")
def _random_walk_signal(
    x_values: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    dt = _time_steps(x_values)
    drift = float(params.get("drift", 0.0))
    step_scale = float(params.get("step_scale", 1.0))
    initial_value = float(
        params.get("initial_value", initial_state.get("state", 0.0) if initial_state else 0.0)
    )

    state = np.empty(len(x_values), dtype=float)
    state[0] = initial_value
    for idx in range(1, len(x_values)):
        innovation = rng.normal(0.0, step_scale * math.sqrt(dt[idx]))
        state[idx] = state[idx - 1] + drift * dt[idx] + innovation

    return SignalResult(
        values=state,
        latent_state=state.copy(),
        terminal_state={"state": float(state[-1])},
    )


@signal_model("drifted_random_walk")
def _drifted_random_walk_signal(
    x_values: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    merged_params = dict(params)
    merged_params.setdefault("drift", 0.1)
    return _random_walk_signal(x_values, rng, merged_params, initial_state=initial_state)


@signal_model("ar1")
def _ar1_signal(
    x_values: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    _ = x_values
    phi = float(params.get("phi", 0.95))
    intercept = float(params.get("intercept", 0.0))
    innovation_scale = float(params.get("innovation_scale", 1.0))
    initial_value = float(
        params.get("initial_value", initial_state.get("state", 0.0) if initial_state else 0.0)
    )

    state = np.empty(len(x_values), dtype=float)
    state[0] = initial_value
    for idx in range(1, len(x_values)):
        innovation = rng.normal(0.0, innovation_scale)
        state[idx] = intercept + phi * state[idx - 1] + innovation

    return SignalResult(
        values=state,
        latent_state=state.copy(),
        terminal_state={"state": float(state[-1])},
    )


@signal_model("ornstein_uhlenbeck")
def _ornstein_uhlenbeck_signal(
    x_values: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    dt = _time_steps(x_values)
    theta = float(params.get("theta", 0.2))
    mean_level = float(params.get("mean", 0.0))
    sigma = float(params.get("sigma", 0.5))
    initial_value = float(
        params.get("initial_value", initial_state.get("state", mean_level) if initial_state else mean_level)
    )

    state = np.empty(len(x_values), dtype=float)
    state[0] = initial_value
    for idx in range(1, len(x_values)):
        innovation = rng.normal(0.0, sigma * math.sqrt(dt[idx]))
        state[idx] = state[idx - 1] + theta * (mean_level - state[idx - 1]) * dt[idx] + innovation

    return SignalResult(
        values=state,
        latent_state=state.copy(),
        terminal_state={"state": float(state[-1])},
    )


@signal_model("local_linear_trend")
def _local_linear_trend_signal(
    x_values: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> SignalResult:
    dt = _time_steps(x_values)
    initial_level = float(
        params.get("initial_level", initial_state.get("level", initial_state.get("state", 0.0)) if initial_state else 0.0)
    )
    initial_slope = float(params.get("initial_slope", initial_state.get("slope", 0.0) if initial_state else 0.0))
    level_scale = float(params.get("level_scale", 0.4))
    slope_scale = float(params.get("slope_scale", 0.05))

    level = np.empty(len(x_values), dtype=float)
    slope = np.empty(len(x_values), dtype=float)
    level[0] = initial_level
    slope[0] = initial_slope

    for idx in range(1, len(x_values)):
        level_noise = rng.normal(0.0, level_scale * math.sqrt(dt[idx]))
        slope_noise = rng.normal(0.0, slope_scale * math.sqrt(dt[idx]))
        level[idx] = level[idx - 1] + slope[idx - 1] * dt[idx] + level_noise
        slope[idx] = slope[idx - 1] + slope_noise

    return SignalResult(
        values=level,
        latent_state=level.copy(),
        extra_columns={"latent_slope": slope},
        terminal_state={"state": float(level[-1]), "level": float(level[-1]), "slope": float(slope[-1])},
    )


@noise_model("callable")
def _callable_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    function = params.get("function")
    if function is None or not callable(function):
        raise ValueError("Callable noise requires a callable 'function' parameter.")

    kwargs = dict(params)
    kwargs.pop("function", None)
    raw = _invoke_user_function(
        function,
        x=x_values,
        y_true=y_true,
        rng=rng,
        initial_state=initial_state,
        **kwargs,
    )
    return _coerce_noise_result(raw, x_values)


@noise_model("gaussian")
def _gaussian_noise(
    x_values: np.ndarray,
    _y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values, initial_state
    loc = float(params.get("loc", 0.0))
    scale = float(params.get("scale", 1.0))
    values = rng.normal(loc, scale, len(_y_true))
    return NoiseResult(values=values, terminal_state={"state": float(values[-1])})


@noise_model("laplace")
def _laplace_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values, initial_state
    loc = float(params.get("loc", 0.0))
    scale = float(params.get("scale", 1.0))
    values = rng.laplace(loc, scale, len(y_true))
    return NoiseResult(values=values, terminal_state={"state": float(values[-1])})


@noise_model("student_t")
def _student_t_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values, initial_state
    df = float(params.get("df", 4.0))
    loc = float(params.get("loc", 0.0))
    scale = float(params.get("scale", 1.0))
    values = loc + scale * rng.standard_t(df, len(y_true))
    return NoiseResult(values=values, terminal_state={"state": float(values[-1])})


@noise_model("uniform")
def _uniform_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values, initial_state
    low = float(params.get("low", -1.0))
    high = float(params.get("high", 1.0))
    values = rng.uniform(low, high, len(y_true))
    return NoiseResult(values=values, terminal_state={"state": float(values[-1])})


@noise_model("cauchy")
def _cauchy_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values, initial_state
    loc = float(params.get("loc", 0.0))
    scale = float(params.get("scale", 1.0))
    values = loc + scale * rng.standard_cauchy(len(y_true))
    return NoiseResult(values=values, terminal_state={"state": float(values[-1])})


@noise_model("lognormal")
def _lognormal_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values, initial_state
    mean = float(params.get("mean", 0.0))
    sigma = float(params.get("sigma", 0.5))
    loc = float(params.get("loc", 0.0))
    center = bool(params.get("center", True))
    values = rng.lognormal(mean=mean, sigma=sigma, size=len(y_true))
    if center:
        values = values - math.exp(mean + 0.5 * sigma**2)
    values = values + loc
    return NoiseResult(values=values, terminal_state={"state": float(values[-1])})


@noise_model("gamma")
def _gamma_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values, initial_state
    shape = float(params.get("shape", 2.0))
    scale = float(params.get("scale", 1.0))
    loc = float(params.get("loc", 0.0))
    center = bool(params.get("center", True))
    values = rng.gamma(shape=shape, scale=scale, size=len(y_true))
    if center:
        values = values - shape * scale
    values = values + loc
    return NoiseResult(values=values, terminal_state={"state": float(values[-1])})


@noise_model("poisson")
def _poisson_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values, initial_state
    rate = float(params.get("rate", 2.0))
    loc = float(params.get("loc", 0.0))
    center = bool(params.get("center", True))
    values = rng.poisson(lam=rate, size=len(y_true)).astype(float)
    if center:
        values = values - rate
    values = values + loc
    return NoiseResult(values=values, terminal_state={"state": float(values[-1])})


@noise_model("negative_binomial")
def _negative_binomial_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values, initial_state
    n = float(params.get("n", 5.0))
    p = float(params.get("p", 0.5))
    loc = float(params.get("loc", 0.0))
    center = bool(params.get("center", True))
    values = rng.negative_binomial(n=n, p=p, size=len(y_true)).astype(float)
    if center:
        values = values - n * (1.0 - p) / p
    values = values + loc
    return NoiseResult(values=values, terminal_state={"state": float(values[-1])})


@noise_model("contaminated_gaussian")
def _contaminated_gaussian_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values, initial_state
    loc = float(params.get("loc", 0.0))
    base_scale = float(params.get("base_scale", 1.0))
    outlier_scale = float(params.get("outlier_scale", 8.0))
    outlier_prob = float(params.get("outlier_prob", 0.05))

    values = rng.normal(loc, base_scale, len(y_true))
    outlier_mask = rng.random(len(y_true)) < outlier_prob
    values[outlier_mask] += rng.normal(0.0, outlier_scale, np.sum(outlier_mask))
    return NoiseResult(
        values=values,
        extra_columns={"is_outlier": outlier_mask.astype(float)},
        terminal_state={"state": float(values[-1])},
    )


@noise_model("impulse")
def _impulse_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values, initial_state
    spike_prob = float(params.get("spike_prob", 0.03))
    spike_scale = float(params.get("spike_scale", 10.0))
    base_scale = float(params.get("base_scale", 0.0))

    values = rng.normal(0.0, base_scale, len(y_true))
    spike_mask = rng.random(len(y_true)) < spike_prob
    values[spike_mask] += rng.normal(0.0, spike_scale, np.sum(spike_mask))
    return NoiseResult(
        values=values,
        extra_columns={"is_spike": spike_mask.astype(float)},
        terminal_state={"state": float(values[-1])},
    )


@noise_model("heteroscedastic_gaussian")
def _heteroscedastic_gaussian_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = initial_state
    base_scale = float(params.get("base_scale", 0.2))
    signal_scale = float(params.get("signal_scale", 0.05))
    x_scale = float(params.get("x_scale", 0.0))
    power = float(params.get("power", 1.0))
    scale = base_scale + signal_scale * np.abs(y_true) ** power + x_scale * np.abs(x_values) ** power
    values = rng.normal(0.0, scale)
    return NoiseResult(
        values=values,
        extra_columns={"noise_scale": scale},
        terminal_state={"state": float(values[-1])},
    )


@noise_model("ar1")
def _ar1_noise(
    x_values: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    params: Mapping[str, Any],
    *,
    initial_state: Optional[Dict[str, float]],
) -> NoiseResult:
    _ = x_values
    phi = float(params.get("phi", 0.7))
    innovation_scale = float(params.get("innovation_scale", 1.0))
    loc = float(params.get("loc", 0.0))
    initial_value = float(params.get("initial_value", initial_state.get("state", 0.0) if initial_state else 0.0))

    values = np.empty(len(y_true), dtype=float)
    values[0] = initial_value
    for idx in range(1, len(y_true)):
        innovation = rng.normal(0.0, innovation_scale)
        values[idx] = loc + phi * values[idx - 1] + innovation
    return NoiseResult(values=values, terminal_state={"state": float(values[-1])})


def _preset_linear_gaussian(*, n_samples: int, seed: Optional[int]) -> DatasetConfig:
    return DatasetConfig(
        n_samples=n_samples,
        input_spec=InputSpec(kind="grid", params={"start": 0.0, "stop": float(n_samples - 1)}),
        signal=model("linear", intercept=2.0, slope=0.08),
        noise=model("gaussian", scale=1.0),
        seed=seed,
        name="linear_gaussian",
    )


def _preset_sinusoid_gaussian(*, n_samples: int, seed: Optional[int]) -> DatasetConfig:
    return DatasetConfig(
        n_samples=n_samples,
        input_spec=InputSpec(kind="grid", params={"start": 0.0, "stop": float(n_samples - 1)}),
        signal=model("sinusoidal", amplitude=3.0, frequency=0.02, offset=1.0),
        noise=model("gaussian", scale=0.6),
        seed=seed,
        name="sinusoid_gaussian",
    )


def _preset_piecewise_laplace(*, n_samples: int, seed: Optional[int]) -> DatasetConfig:
    switch_one = max(int(0.35 * n_samples), 1)
    switch_two = max(int(0.7 * n_samples), switch_one + 1)
    return DatasetConfig(
        n_samples=n_samples,
        input_spec=InputSpec(kind="grid", params={"start": 0.0, "stop": float(n_samples - 1)}),
        signal=model(
            "piecewise_linear",
            knots=(0.0, float(switch_one), float(switch_two), float(n_samples - 1)),
            values=(0.0, 5.0, -2.0, 3.0),
        ),
        noise=model("laplace", scale=0.9),
        seed=seed,
        name="piecewise_laplace",
    )


def _preset_random_walk_measurement(*, n_samples: int, seed: Optional[int]) -> DatasetConfig:
    return DatasetConfig(
        n_samples=n_samples,
        input_spec=InputSpec(kind="grid", params={"start": 0.0, "stop": float(n_samples - 1)}),
        signal=model("random_walk", initial_value=0.0, step_scale=0.35),
        noise=model("gaussian", scale=0.5),
        seed=seed,
        name="random_walk_measurement",
    )


def _preset_ou_student_t(*, n_samples: int, seed: Optional[int]) -> DatasetConfig:
    return DatasetConfig(
        n_samples=n_samples,
        input_spec=InputSpec(kind="irregular_time", params={"start": 0.0, "mean_step": 1.0}),
        signal=model("ornstein_uhlenbeck", mean=0.0, theta=0.15, sigma=0.25),
        noise=model("student_t", df=3.5, scale=0.45),
        seed=seed,
        name="ou_student_t",
    )


def _preset_nonlinear_heteroscedastic(*, n_samples: int, seed: Optional[int]) -> DatasetConfig:
    return DatasetConfig(
        n_samples=n_samples,
        input_spec=InputSpec(kind="grid", params={"start": 0.0, "stop": 10.0}),
        signal=model("logistic", lower=-1.0, upper=4.0, midpoint=5.0, growth=1.1),
        noise=model("heteroscedastic_gaussian", base_scale=0.15, signal_scale=0.08),
        seed=seed,
        name="nonlinear_heteroscedastic",
    )


def _preset_regime_switch_signal(*, n_samples: int, seed: Optional[int]) -> DatasetConfig:
    switch_one = max(int(0.33 * n_samples), 1)
    switch_two = max(int(0.66 * n_samples), switch_one + 1)
    signal_spec = regime_schedule(
        (
            model("linear", intercept=0.0, slope=0.05),
            model("sinusoidal", amplitude=2.0, frequency=0.03, offset=4.0),
            model("random_walk", step_scale=0.25),
        ),
        switch_times=(float(switch_one), float(switch_two)),
        carry_forward_state=True,
    )
    return DatasetConfig(
        n_samples=n_samples,
        input_spec=InputSpec(kind="grid", params={"start": 0.0, "stop": float(n_samples - 1)}),
        signal=signal_spec,
        noise=model("gaussian", scale=0.4),
        seed=seed,
        name="regime_switch_signal",
    )


def _preset_regime_switch_noise(*, n_samples: int, seed: Optional[int]) -> DatasetConfig:
    switch_one = max(int(0.5 * n_samples), 1)
    noise_spec = regime_schedule(
        (
            model("gaussian", scale=0.25),
            model("contaminated_gaussian", base_scale=0.3, outlier_scale=3.5, outlier_prob=0.12),
        ),
        switch_times=(float(switch_one),),
        carry_forward_state=False,
    )
    return DatasetConfig(
        n_samples=n_samples,
        input_spec=InputSpec(kind="grid", params={"start": 0.0, "stop": float(n_samples - 1)}),
        signal=model("sinusoidal", amplitude=1.5, frequency=0.04),
        noise=noise_spec,
        seed=seed,
        name="regime_switch_noise",
    )


PRESET_BUILDERS: Dict[str, Callable[..., DatasetConfig]] = {
    "linear_gaussian": _preset_linear_gaussian,
    "sinusoid_gaussian": _preset_sinusoid_gaussian,
    "piecewise_laplace": _preset_piecewise_laplace,
    "random_walk_measurement": _preset_random_walk_measurement,
    "ou_student_t": _preset_ou_student_t,
    "nonlinear_heteroscedastic": _preset_nonlinear_heteroscedastic,
    "regime_switch_signal": _preset_regime_switch_signal,
    "regime_switch_noise": _preset_regime_switch_noise,
}


__all__ = [
    "DatasetConfig",
    "GeneratedDataset",
    "InputSpec",
    "ModelSpec",
    "NoiseResult",
    "RegimeSchedule",
    "SignalResult",
    "StandardSuiteCase",
    "available_noise_models",
    "available_signal_models",
    "dataset_summary",
    "generate_benchmark_suite",
    "generate_dataset",
    "generate_preset",
    "generate_standard_suite",
    "get_preset_config",
    "get_standard_suite_cases",
    "list_presets",
    "model",
    "plot_dataset",
    "regime_schedule",
    "test_filter_on_standard_suite",
]
