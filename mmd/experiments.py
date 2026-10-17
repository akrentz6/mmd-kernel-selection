from functools import partial

import jax
import jax.numpy as jnp

from . import estimators


@jax.jit
def permute_samples(x, y, key):
    """
    Randomly reassigns the pooled observations to the two samples.
    """
    combined = jnp.concatenate([x, y], axis=0)
    permuted = jax.random.permutation(key, combined, axis=0)

    return jnp.split(permuted, [x.shape[0]], axis=0)

@partial(jax.jit, static_argnames=("kernel", "num_permutations"))
def permutation_test(x, y, kernel, key, num_permutations=1000):
    """
    Performs an MMD permutation test.
    """
    if num_permutations < 1:
        raise ValueError("num_permutations must be at least 1.")

    observed = estimators.mmd_squared_biased(x, y, kernel)

    keys = jax.random.split(key, num=num_permutations)

    def count_exceedance(count, permutation_key):
        permuted_x, permuted_y = permute_samples(x, y, permutation_key)
        statistic = estimators.mmd_squared_biased(permuted_x, permuted_y, kernel)
        return count + (statistic >= observed), None

    exceedances, _ = jax.lax.scan(count_exceedance, jnp.asarray(0, dtype=jnp.int32), keys)
    p_value = (1 + exceedances) / (num_permutations + 1)
    return observed, p_value

@partial(jax.jit, static_argnames=("sampler", "kernel", "num_trials", "num_permutations"))
def estimate_rejection_rate(sampler, kernel, key, alpha=0.05, num_trials=500, num_permutations=500):
    """
    Estimates the rejection rate of the MMD test. If sampler generates samples from
    the null hypothesis, this estimates the Type I error rate. If sampler generates
    samples from the alternative hypothesis, this estimates the power of the test.
    """
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1.")

    keys = jax.random.split(key, num=num_trials)

    def count_rejection(count, k):
        sample_key, permutation_key = jax.random.split(k)

        x, y = sampler(sample_key)
        _, p_value = permutation_test(x, y, kernel, permutation_key, num_permutations)

        return count + (p_value < alpha), None

    rejections, _ = jax.lax.scan(count_rejection, jnp.asarray(0, dtype=jnp.int32), keys)
    return rejections / num_trials
