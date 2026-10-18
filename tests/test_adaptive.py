import jax
import jax.numpy as jnp

from mmd import distributions, estimators, experiments, kernels


def test_jackknife_variance():
    x = jnp.array([[0.0], [0.2], [1.0], [2.0]])
    y = jnp.array([[0.5], [1.5], [3.0]])
    kernel = kernels.make_gaussian_kernel(1.0)
    leave_x = jnp.array([
        estimators.mmd_squared_unbiased(jnp.concatenate([x[:i], x[i + 1:]]), y, kernel)
        for i in range(len(x))
    ])
    leave_y = jnp.array([
        estimators.mmd_squared_unbiased(x, jnp.concatenate([y[:i], y[i + 1:]]), kernel)
        for i in range(len(y))
    ])
    expected = (len(x) - 1) * jnp.var(leave_x) + (len(y) - 1) * jnp.var(leave_y)
    variance = estimators.mmd_squared_unbiased_variance(x, y, kernel)
    print(f"Jackknife variance: {variance:.6f} (expected {expected:.6f})")
    assert jnp.isclose(variance, expected)

def test_naive_selection():
    x, y = distributions.gaussian_mean_shift_one_coordinate(jax.random.key(0), 8, (2,), 0.5)
    bandwidths = jnp.array([0.25, 1.0, 4.0])
    values = jnp.array([estimators.mmd_squared_unbiased(x, y, kernels.make_gaussian_kernel(sigma)) for sigma in bandwidths])
    variances = jnp.array([estimators.mmd_squared_unbiased_variance(x, y, kernels.make_gaussian_kernel(sigma)) for sigma in bandwidths])
    expected = [bandwidths[jnp.argmax(values)], bandwidths[jnp.argmax(values / jnp.sqrt(variances + 1e-8))]]
    key = jax.random.key(1)

    for selector, sigma in zip([experiments.select_max_mmd, experiments.select_max_variance_adjusted_mmd], expected):
        assert selector(x, y, bandwidths) == sigma
        result = experiments.naive_adaptive_test(x, y, bandwidths, key, 39, selector)
        fixed = experiments.permutation_test(x, y, kernels.make_gaussian_kernel(sigma), key, 39)
        print(f"{selector.__name__}: sigma={sigma:.2f}, p-value={result.p_value:.3f}")
        assert jnp.isclose(result.observed, fixed.observed)
        assert jnp.isclose(result.p_value, fixed.p_value)

    identical = jnp.ones((4, 2))
    assert jnp.isfinite(experiments.select_max_variance_adjusted_mmd(identical, identical, bandwidths))

def test_adaptive_fixed_baseline():
    sampler = distributions.make_gaussian_null_sampler(8, (2,))
    key = jax.random.key(0)
    result = experiments.estimate_adaptive_rejection_rates(sampler, key, num_trials=3, num_permutations=39, num_candidates=3)
    expected = experiments.estimate_rejection_rate(sampler, kernels.make_gaussian_kernel(1.0), key, num_trials=3, num_permutations=39)
    assert result.rate.shape == (4,)
    assert jnp.isclose(result.rate[0], expected.rate)
    assert jnp.isclose(result.standard_error[0], expected.standard_error)
    assert jnp.allclose(result.standard_error, jnp.sqrt(result.rate * (1 - result.rate) / 3))
    print("Adaptive rejection rates:", result.rate, "±", result.standard_error)


if __name__ == "__main__":
    test_jackknife_variance()
    test_naive_selection()
    test_adaptive_fixed_baseline()
