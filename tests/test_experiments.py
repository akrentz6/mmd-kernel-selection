import jax
import jax.numpy as jnp

from mmd import kernels, estimators, experiments, distributions

def test_permutation_test_matches_recomputed_gram_matrices():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)
    key = jax.random.PRNGKey(42)
    num_permutations = 50

    x = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
    y = jnp.array([[0.5, 0.0], [3.0, 1.0]])

    expected_observed = estimators.mmd_squared_biased(x, y, kernel)
    keys = jax.random.split(key, num=num_permutations)

    def statistic(permutation_key):
        permuted_x, permuted_y = experiments.permute_samples(x, y, permutation_key)
        return estimators.mmd_squared_biased(permuted_x, permuted_y, kernel)

    statistics = jax.lax.map(statistic, keys)
    expected_p_value = (1 + jnp.sum(statistics >= expected_observed)) / (num_permutations + 1)
    result = experiments.permutation_test(x, y, kernel, key, num_permutations)

    print(f"MMD: {result.observed:.6f} (expected {expected_observed:.6f})")
    print(f"p-value: {result.p_value:.6f} ± {result.standard_error:.6f} SE (expected {expected_p_value:.6f})")
    assert jnp.isclose(result.observed, expected_observed)
    assert jnp.isclose(result.p_value, expected_p_value)
    exceedance_rate = jnp.mean(statistics >= expected_observed)
    expected_se = jnp.sqrt(num_permutations * exceedance_rate * (1 - exceedance_rate)) / (num_permutations + 1)
    assert jnp.isclose(result.standard_error, expected_se)

def test_permutation_test_identical_observations():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)
    key = jax.random.PRNGKey(0)

    for m, n in [(10, 10), (7, 13), (1, 3)]:
        x = jnp.ones((m, 2))
        y = jnp.ones((n, 2))

        for num_permutations in [1, 16, 19]:
            result = experiments.permutation_test(x, y, kernel, key, num_permutations)

            print(f"Identical samples (m={m}, n={n}, permutations={num_permutations}): "
                  f"MMD={result.observed:.2e}, p-value={result.p_value:.3f} ± {result.standard_error:.3f}")
            assert jnp.isclose(result.observed, 0.0, atol=1e-6)
            assert result.p_value == 1.0
            assert result.standard_error == 0.0

def test_rate_standard_errors():
    sampler = distributions.make_gaussian_mean_shift_one_coordinate_sampler(8, (2,), 2.0)
    kernel = kernels.make_gaussian_kernel(1.0)
    key = jax.random.key(3)
    num_trials = 3
    num_permutations = 39
    rejections = []
    bandwidth_rejections = []

    for trial_key in jax.random.split(key, num_trials):
        sample_key, permutation_key = jax.random.split(trial_key)
        x, y = sampler(sample_key)
        result = experiments.permutation_test(x, y, kernel, permutation_key, num_permutations)
        rejections.append(result.p_value < 0.05)
        bandwidth_rejections.append([
            experiments.permutation_test(x, y, kernels.make_gaussian_kernel(sigma), permutation_key, num_permutations).p_value < 0.05
            for sigma in kernels.candidate_bandwidths(x, y, 3)
        ])

    rate_result = experiments.estimate_rejection_rate(sampler, kernel, key, num_trials=num_trials, num_permutations=num_permutations)
    power_result = experiments.estimate_bandwidth_power(sampler, key, num_trials=num_trials, num_permutations=num_permutations, num_candidates=3)

    for result, outcomes in [(rate_result, rejections), (power_result, bandwidth_rejections)]:
        expected_rate = jnp.mean(jnp.array(outcomes), axis=0)
        expected_se = jnp.sqrt(expected_rate * (1 - expected_rate) / num_trials)
        print("Rate:", result.rate, "±", result.standard_error)
        assert jnp.allclose(result.rate, expected_rate)
        assert jnp.allclose(result.standard_error, expected_se)


if __name__ == "__main__":
    test_permutation_test_matches_recomputed_gram_matrices()
    test_permutation_test_identical_observations()
    test_rate_standard_errors()
