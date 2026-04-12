# Noise-Analysis

## Synthetic Data Generator

`SyntheticData.py` now contains a reusable dataset generator for benchmarking
Kalman filters, particle filters, and alternative estimators against known
ground truth.

### Core API

```python
import sys
sys.path.insert(0, "Noise-Analysis")

import SyntheticData as sd

dataset = sd.generate_dataset(
    n_samples=500,
    signal=sd.model("sinusoidal", amplitude=2.0, frequency=0.03, offset=1.0),
    noise=sd.model("student_t", df=4.0, scale=0.35),
    seed=42,
)

print(dataset.data.head())
print(dataset.summary())
```

The returned dataframe includes:

- `x`: input or time axis
- `y_true`: noiseless signal
- `noise_draw`: raw draw from the configured noise model
- `noise`: realized observation error after additive/multiplicative application
- `y_obs`: observed value
- `signal_regime`: active signal regime index
- `noise_regime`: active noise regime index
- optional latent columns such as `latent_state`, `latent_slope`, `noise_scale`

### Supported Input Samplers

- `grid`
- `uniform`
- `normal`
- `mixture_gaussian`
- `irregular_time`

### Built-in Signal Models

- `callable`
- `constant`
- `linear`
- `polynomial`
- `exponential`
- `logarithmic`
- `sinusoidal`
- `multi_sine`
- `piecewise_linear`
- `piecewise_constant`
- `logistic`
- `gaussian_bump`
- `random_walk`
- `drifted_random_walk`
- `ar1`
- `ornstein_uhlenbeck`
- `local_linear_trend`

### Built-in Noise Models

- `callable`
- `gaussian`
- `laplace`
- `student_t`
- `uniform`
- `cauchy`
- `lognormal`
- `gamma`
- `poisson`
- `negative_binomial`
- `contaminated_gaussian`
- `impulse`
- `heteroscedastic_gaussian`
- `ar1`

### Regime Switching

Regime switching is configured by passing:

- a list of regimes
- a list of switch times

The schedule must satisfy `len(regimes) == len(switch_times) + 1`.

```python
signal_schedule = sd.regime_schedule(
    [
        sd.model("linear", intercept=0.0, slope=0.05),
        sd.model("sinusoidal", amplitude=2.0, frequency=0.03, offset=4.0),
        sd.model("random_walk", step_scale=0.25),
    ],
    switch_times=[100, 250],
    carry_forward_state=True,
)

noise_schedule = sd.regime_schedule(
    [
        sd.model("gaussian", scale=0.25),
        sd.model("contaminated_gaussian", base_scale=0.3, outlier_scale=3.0, outlier_prob=0.1),
    ],
    switch_times=[180],
)

dataset = sd.generate_dataset(
    n_samples=400,
    signal=signal_schedule,
    noise=noise_schedule,
    seed=5,
)
```

With the default `grid` input sampler, switch times line up naturally with
sample indices because `x` defaults to `0, 1, ..., n_samples - 1`.

### Noise Application Modes

- `additive`: `y_obs = y_true + noise_draw`
- `multiplicative`: `y_obs = y_true * (1 + noise_draw)`
- `mixed`: `y_obs = y_true + noise_draw + w * y_true * noise_draw`

### Presets

Use `sd.list_presets()` to inspect the bundled benchmark scenarios. Current
presets include:

- `linear_gaussian`
- `sinusoid_gaussian`
- `piecewise_laplace`
- `random_walk_measurement`
- `ou_student_t`
- `nonlinear_heteroscedastic`
- `regime_switch_signal`
- `regime_switch_noise`

Example:

```python
preset = sd.generate_preset("regime_switch_signal", n_samples=500, seed=1)
```

### Dependencies

`SyntheticData.py` depends on `numpy` and `pandas`. `plot_dataset(...)` also
requires `matplotlib`.
