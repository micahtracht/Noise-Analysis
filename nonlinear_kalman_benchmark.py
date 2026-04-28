from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd


StateFunc = Callable[[np.ndarray], np.ndarray]
JacobianFunc = Callable[[np.ndarray], np.ndarray]
MeasurementFunc = Callable[[np.ndarray], float]
MeasurementJacobianFunc = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class NonlinearModel:
    name: str
    transition: StateFunc
    transition_jacobian: JacobianFunc
    measurement: MeasurementFunc
    measurement_jacobian: MeasurementJacobianFunc
    q: np.ndarray
    r: float


@dataclass(frozen=True)
class Regime:
    name: str
    n: int
    seeds: tuple[int, ...]
    initial_state: np.ndarray
    process_scale: np.ndarray
    measurement_scale: float
    outlier_prob: float = 0.0
    outlier_scale: float = 0.0
    switch_at: int | None = None


def build_model(strength: float, q_level: float, q_velocity: float, r: float) -> NonlinearModel:
    def transition(state: np.ndarray) -> np.ndarray:
        level, velocity = state
        next_level = level + velocity + 0.04 * strength * np.sin(1.8 * level)
        next_velocity = 0.86 * velocity - 0.025 * strength * np.tanh(level)
        return np.array([next_level, next_velocity])

    def transition_jacobian(state: np.ndarray) -> np.ndarray:
        level, _velocity = state
        sech2 = 1.0 / np.cosh(level) ** 2
        return np.array(
            [
                [1.0 + 0.072 * strength * np.cos(1.8 * level), 1.0],
                [-0.025 * strength * sech2, 0.86],
            ]
        )

    def measurement(state: np.ndarray) -> float:
        level = state[0]
        return float(level + 0.28 * strength * np.tanh(1.2 * level) + 0.035 * strength * level**2)

    def measurement_jacobian(state: np.ndarray) -> np.ndarray:
        level = state[0]
        sech2 = 1.0 / np.cosh(1.2 * level) ** 2
        derivative = 1.0 + 0.336 * strength * sech2 + 0.07 * strength * level
        return np.array([[derivative, 0.0]])

    return NonlinearModel(
        name=f"nonlinear_strength_{strength:g}",
        transition=transition,
        transition_jacobian=transition_jacobian,
        measurement=measurement,
        measurement_jacobian=measurement_jacobian,
        q=np.diag([q_level, q_velocity]),
        r=r,
    )


def generate_regime(regime: Regime, model: NonlinearModel, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = np.zeros((regime.n, 2))
    observations = np.zeros(regime.n)
    states[0] = regime.initial_state

    for idx in range(1, regime.n):
        process_scale = regime.process_scale.copy()
        if regime.switch_at is not None and idx >= regime.switch_at:
            process_scale = process_scale * np.array([1.6, 2.4])
        process_noise = rng.normal(0.0, process_scale)
        states[idx] = model.transition(states[idx - 1]) + process_noise

    for idx in range(regime.n):
        obs_noise = rng.normal(0.0, regime.measurement_scale)
        if regime.outlier_prob and rng.random() < regime.outlier_prob:
            obs_noise += rng.normal(0.0, regime.outlier_scale)
        observations[idx] = model.measurement(states[idx]) + obs_noise

    return pd.DataFrame(
        {
            "t": np.arange(regime.n),
            "level_true": states[:, 0],
            "velocity_true": states[:, 1],
            "y_obs": observations,
        }
    )


def ekf_direct(y_obs: np.ndarray, model: NonlinearModel, initial_level: float) -> np.ndarray:
    state = np.array([initial_level, 0.0])
    covar = np.eye(2)
    estimates = np.zeros(len(y_obs))

    for idx, observation in enumerate(y_obs):
        f = model.transition_jacobian(state)
        state_pred = model.transition(state)
        covar_pred = f @ covar @ f.T + model.q

        h = model.measurement_jacobian(state_pred)
        innovation = observation - model.measurement(state_pred)
        innovation_covar = h @ covar_pred @ h.T + model.r
        gain = covar_pred @ h.T / innovation_covar.item()

        state = state_pred + gain.flatten() * innovation
        covar = (np.eye(2) - gain @ h) @ covar_pred
        estimates[idx] = state[0]

    return estimates


def ekf_iterated(
    y_obs: np.ndarray,
    model: NonlinearModel,
    initial_level: float,
    max_iterations: int = 8,
    tolerance: float = 1e-9,
) -> np.ndarray:
    state = np.array([initial_level, 0.0])
    covar = np.eye(2)
    estimates = np.zeros(len(y_obs))

    for idx, observation in enumerate(y_obs):
        f = model.transition_jacobian(state)
        state_pred = model.transition(state)
        covar_pred = f @ covar @ f.T + model.q
        state_iter = state_pred.copy()

        for _ in range(max_iterations):
            h = model.measurement_jacobian(state_iter)
            innovation = observation - model.measurement(state_iter) + (h @ (state_iter - state_pred)).item()
            innovation_covar = h @ covar_pred @ h.T + model.r
            gain = covar_pred @ h.T / innovation_covar.item()
            next_state = state_pred + gain.flatten() * innovation
            if np.linalg.norm(next_state - state_iter) < tolerance:
                state_iter = next_state
                break
            state_iter = next_state

        h = model.measurement_jacobian(state_iter)
        innovation_covar = h @ covar_pred @ h.T + model.r
        gain = covar_pred @ h.T / innovation_covar.item()
        state = state_iter
        covar = (np.eye(2) - gain @ h) @ covar_pred
        estimates[idx] = state[0]

    return estimates


def ukf(y_obs: np.ndarray, model: NonlinearModel, initial_level: float) -> np.ndarray:
    state_dim = 2
    alpha = 0.35
    beta = 2.0
    kappa = 0.0
    lambda_ = alpha**2 * (state_dim + kappa) - state_dim
    weight_count = 2 * state_dim + 1
    mean_weights = np.full(weight_count, 1.0 / (2.0 * (state_dim + lambda_)))
    covar_weights = mean_weights.copy()
    mean_weights[0] = lambda_ / (state_dim + lambda_)
    covar_weights[0] = mean_weights[0] + (1.0 - alpha**2 + beta)

    state = np.array([initial_level, 0.0])
    covar = np.eye(2)
    estimates = np.zeros(len(y_obs))

    for idx, observation in enumerate(y_obs):
        sigma_points = sigma_points_for(state, covar, lambda_)
        pred_sigma = np.array([model.transition(point) for point in sigma_points])
        state_pred = np.sum(mean_weights[:, None] * pred_sigma, axis=0)
        covar_pred = model.q.copy()
        for weight, point in zip(covar_weights, pred_sigma):
            diff = point - state_pred
            covar_pred += weight * np.outer(diff, diff)

        obs_sigma = np.array([model.measurement(point) for point in pred_sigma])
        obs_pred = float(np.sum(mean_weights * obs_sigma))
        innovation_covar = model.r
        cross_covar = np.zeros(state_dim)
        for weight, point, obs_point in zip(covar_weights, pred_sigma, obs_sigma):
            state_diff = point - state_pred
            obs_diff = obs_point - obs_pred
            innovation_covar += weight * obs_diff * obs_diff
            cross_covar += weight * state_diff * obs_diff

        gain = cross_covar / innovation_covar
        state = state_pred + gain * (observation - obs_pred)
        covar = covar_pred - np.outer(gain, gain) * innovation_covar
        covar = 0.5 * (covar + covar.T)
        estimates[idx] = state[0]

    return estimates


def sigma_points_for(state: np.ndarray, covar: np.ndarray, lambda_: float) -> np.ndarray:
    state_dim = len(state)
    scaled_covar = (state_dim + lambda_) * covar
    jitter = 1e-10
    for _ in range(5):
        try:
            sqrt_covar = np.linalg.cholesky(scaled_covar + np.eye(state_dim) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        sqrt_covar = np.linalg.cholesky(scaled_covar + np.eye(state_dim) * jitter)

    points = np.zeros((2 * state_dim + 1, state_dim))
    points[0] = state
    for idx in range(state_dim):
        points[idx + 1] = state + sqrt_covar[:, idx]
        points[idx + 1 + state_dim] = state - sqrt_covar[:, idx]
    return points


def fractional_filter(y_obs: np.ndarray, model: NonlinearModel) -> np.ndarray:
    deltas = np.linspace(0.1, 0.9, 10)
    coeffs = [gl_coefficients(delta, tolerance=1e-4, max_l=180) for delta in deltas]
    max_l = max(len(c) for c in coeffs)
    states = np.full(len(deltas), y_obs[0])
    covars = np.ones(len(deltas))
    weights = np.ones(len(deltas)) / len(deltas)
    measurement_estimates = np.zeros(len(y_obs))
    level_estimates = np.zeros(len(y_obs))

    for idx, observation in enumerate(y_obs):
        if idx < max_l:
            prediction = y_obs[max(idx - 1, 0)]
            measurement_estimates[idx] = prediction
            level_estimates[idx] = invert_measurement(prediction, model, level_estimates[idx - 1] if idx else y_obs[0])
            continue

        history = y_obs[idx - 1 :: -1]
        predictions = np.array([np.dot(c, history[: len(c)]) for c in coeffs])
        fused_prediction = float(np.dot(weights, predictions))
        dynamic_r = max(float(np.var(y_obs[idx - max_l : idx])), 1e-5)

        likelihoods = np.zeros(len(deltas))
        for branch_idx, branch_prediction in enumerate(predictions):
            p_pred = covars[branch_idx] + 1e-5
            innovation = observation - branch_prediction
            gain = p_pred / (p_pred + dynamic_r)
            states[branch_idx] = branch_prediction + gain * innovation
            covars[branch_idx] = (1.0 - gain) * p_pred
            likelihoods[branch_idx] = np.exp(-0.5 * innovation**2 / (p_pred + dynamic_r)) / np.sqrt(
                2.0 * np.pi * (p_pred + dynamic_r)
            )

        weights = weights * likelihoods + 1e-4
        weights = weights / np.sum(weights)
        measurement_estimates[idx] = fused_prediction
        level_estimates[idx] = invert_measurement(
            fused_prediction,
            model,
            level_estimates[idx - 1],
        )

    return level_estimates


def gl_coefficients(delta: float, tolerance: float, max_l: int) -> np.ndarray:
    weights = [1.0]
    for k in range(1, max_l + 1):
        next_weight = weights[-1] * (1.0 - (delta + 1.0) / k)
        weights.append(next_weight)
        if abs(next_weight) < tolerance:
            break
    return -np.array(weights[1:])


def invert_measurement(target: float, model: NonlinearModel, initial_level: float) -> float:
    level = float(np.clip(initial_level, -8.0, 8.0))
    for _ in range(12):
        state = np.array([level, 0.0])
        residual = model.measurement(state) - target
        derivative = model.measurement_jacobian(state)[0, 0]
        if abs(derivative) < 1e-8:
            break
        next_level = float(np.clip(level - residual / derivative, -8.0, 8.0))
        if abs(next_level - level) < 1e-10:
            return next_level
        level = next_level
    return level


def score(estimates: np.ndarray, truth: np.ndarray, warmup: int) -> dict[str, float]:
    errors = estimates[warmup:] - truth[warmup:]
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "max_abs_error": float(np.max(np.abs(errors))),
    }


def run_benchmark() -> pd.DataFrame:
    regimes = (
        Regime(
            name="mild_nonlinear_gaussian",
            n=600,
            seeds=tuple(range(100, 115)),
            initial_state=np.array([0.15, 0.01]),
            process_scale=np.array([0.025, 0.006]),
            measurement_scale=0.08,
        ),
        Regime(
            name="strong_measurement_nonlinearity",
            n=600,
            seeds=tuple(range(200, 215)),
            initial_state=np.array([-0.25, 0.02]),
            process_scale=np.array([0.035, 0.01]),
            measurement_scale=0.10,
        ),
        Regime(
            name="switching_outlier_regime",
            n=700,
            seeds=tuple(range(300, 315)),
            initial_state=np.array([0.10, -0.01]),
            process_scale=np.array([0.025, 0.008]),
            measurement_scale=0.07,
            outlier_prob=0.04,
            outlier_scale=0.75,
            switch_at=350,
        ),
    )
    strengths = {
        "mild_nonlinear_gaussian": 0.55,
        "strong_measurement_nonlinearity": 1.35,
        "switching_outlier_regime": 0.95,
    }

    rows: list[dict[str, float | str | int]] = []
    for regime in regimes:
        model = build_model(
            strengths[regime.name],
            q_level=regime.process_scale[0] ** 2,
            q_velocity=regime.process_scale[1] ** 2,
            r=regime.measurement_scale**2,
        )
        for seed in regime.seeds:
            data = generate_regime(regime, model, seed)
            y_obs = data["y_obs"].to_numpy()
            truth = data["level_true"].to_numpy()
            initial_level = invert_measurement(y_obs[0], model, y_obs[0])
            filters = {
                "direct_second_order_ekf": lambda: ekf_direct(y_obs, model, initial_level),
                "iterated_ekf": lambda: ekf_iterated(y_obs, model, initial_level),
                "unscented_kf": lambda: ukf(y_obs, model, initial_level),
                "fractional": lambda: fractional_filter(y_obs, model),
            }

            for filter_name, runner in filters.items():
                started_at = perf_counter()
                estimates = runner()
                elapsed_ms = (perf_counter() - started_at) * 1000.0
                row = {
                    "regime": regime.name,
                    "seed": seed,
                    "filter": filter_name,
                    "runtime_ms": elapsed_ms,
                }
                row.update(score(estimates, truth, warmup=40))
                rows.append(row)

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["regime", "filter"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            bias_mean=("bias", "mean"),
            max_abs_error_mean=("max_abs_error", "mean"),
            runtime_ms_mean=("runtime_ms", "mean"),
        )
        .sort_values(["regime", "rmse_mean"])
    )
    summary["rank"] = summary.groupby("regime")["rmse_mean"].rank(method="dense")
    return summary


if __name__ == "__main__":
    benchmark_results = run_benchmark()
    benchmark_summary = summarize(benchmark_results)
    benchmark_results.to_csv("nonlinear_kalman_benchmark_results.csv", index=False)
    benchmark_summary.to_csv("nonlinear_kalman_benchmark_summary.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(benchmark_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
