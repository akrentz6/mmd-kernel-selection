import jax
import jax.numpy as jnp

from mmd import distributions, estimators, experiments, kernels


def test_sample_split_matches_manual_selection():
    x, y = distributions.gaussian_mean_shift_one_coordinate(jax.random.key(0), 16, (2,), 0.5)
    key = jax.random.key(1)
    x_key, y_key, permutation_key = jax.random.split(key, 3)
    shuffled_x = jax.random.permutation(x_key, x, axis=0)
    shuffled_y = jax.random.permutation(y_key, y, axis=0)

    for fraction in [0.25, 0.5, 0.75]:
        split = int(len(x) * fraction)
        bandwidths = kernels.candidate_bandwidths(shuffled_x[:split], shuffled_y[:split], 5)
        sigma = experiments.select_max_variance_adjusted_mmd(shuffled_x[:split], shuffled_y[:split], bandwidths)
        expected = experiments.permutation_test(shuffled_x[split:], shuffled_y[split:], kernels.make_gaussian_kernel(sigma), permutation_key, 39)
        result = experiments.sample_split_test(x, y, key, fraction, 5, 39)
        print(f"Selection fraction: {fraction:.2f}, p-value: {result.p_value:.3f}")
        assert jnp.isclose(result.observed, expected.observed)
        assert jnp.isclose(result.p_value, expected.p_value)

def test_permutation_max_matches_recomputed_search():
    x = jnp.array([[0.0], [0.2], [1.0], [2.0]])
    y = jnp.array([[0.5], [1.5], [3.0]])
    key = jax.random.key(3)
    num_permutations = 39
    bandwidths = kernels.candidate_bandwidths(x, y, 5)

    def statistic(x, y):
        return jnp.max(jnp.array([
            estimators.mmd_squared_unbiased(x, y, kernels.make_gaussian_kernel(sigma))
            for sigma in bandwidths
        ]))

    observed = statistic(x, y)
    statistics = []
    for permutation_key in jax.random.split(key, num_permutations):
        permuted_x, permuted_y = experiments.permute_samples(x, y, permutation_key)
        statistics.append(statistic(permuted_x, permuted_y))
        assert jnp.allclose(kernels.candidate_bandwidths(permuted_x, permuted_y, 5), bandwidths)
    tolerance = 8 * jnp.finfo(observed.dtype).eps
    exceedances = jnp.sum(jnp.array(statistics) >= observed - tolerance)
    expected_p_value = (1 + exceedances) / (num_permutations + 1)
    result = experiments.permutation_max_test(x, y, key, 5, num_permutations)
    print(f"Permutation max: {result.p_value:.3f} (expected {expected_p_value:.3f})")
    assert jnp.isclose(result.observed, observed, atol=1e-6)
    assert jnp.isclose(result.p_value, expected_p_value)

def test_permutation_max_ties():
    x = jnp.ones((4, 2))
    y = jnp.ones((3, 2))
    result = experiments.permutation_max_test(x, y, jax.random.key(0), 5, 39)
    print(f"Identical samples: p-value={result.p_value:.3f}")
    assert jnp.isfinite(result.observed)
    assert result.p_value == 1


if __name__ == "__main__":
    test_sample_split_matches_manual_selection()
    test_permutation_max_matches_recomputed_search()
    test_permutation_max_ties()
