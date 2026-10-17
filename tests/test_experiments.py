import jax
import jax.numpy as jnp

from mmd import kernels, estimators, experiments

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
    observed, p_value = experiments.permutation_test(x, y, kernel, key, num_permutations)

    print(f"MMD: {observed:.6f} (expected {expected_observed:.6f})")
    print(f"p-value: {p_value:.6f} (expected {expected_p_value:.6f})")
    assert jnp.isclose(observed, expected_observed)
    assert jnp.isclose(p_value, expected_p_value)

def test_permutation_test_identical_observations():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)
    key = jax.random.PRNGKey(0)

    for m, n in [(10, 10), (7, 13), (1, 3)]:
        x = jnp.ones((m, 2))
        y = jnp.ones((n, 2))

        for num_permutations in [1, 16, 19]:
            observed, p_value = experiments.permutation_test(x, y, kernel, key, num_permutations)

            print(f"Identical samples (m={m}, n={n}, permutations={num_permutations}): "
                  f"MMD={observed:.2e}, p-value={p_value:.3f}")
            assert jnp.isclose(observed, 0.0, atol=1e-6)
            assert p_value == 1.0


if __name__ == "__main__":
    test_permutation_test_matches_recomputed_gram_matrices()
    test_permutation_test_identical_observations()
