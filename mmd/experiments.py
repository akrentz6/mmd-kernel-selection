from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import estimators, kernels


class PermutationResult(NamedTuple):
    """
    Observed MMD, corrected p-value, and its permutation Monte Carlo SE.
    """
    observed: jax.Array
    p_value: jax.Array
    standard_error: jax.Array


class RateResult(NamedTuple):
    """
    Estimated rejection rate and binomial Monte Carlo SE across trials.
    """
    rate: jax.Array
    standard_error: jax.Array


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

    m = x.shape[0]
    n = y.shape[0]
    combined = jnp.concatenate([x, y], axis=0)
    k_combined = estimators.gram_matrix(combined, combined, kernel)
    indices = jnp.arange(m + n)

    def statistic_from_indices(indices):
        # Membership vectors avoid gathering matrix blocks for each permutation.
        in_x = jnp.zeros((m + n,), dtype=k_combined.dtype)
        in_x = in_x.at[indices[:m]].set(1)
        in_y = 1 - in_x

        k_x = k_combined @ in_x
        k_y = k_combined @ in_y

        return (in_x @ k_x) / (m * m) + (in_y @ k_y) / (n * n) - 2 * (in_x @ k_y) / (m * n)

    observed = statistic_from_indices(indices)

    keys = jax.random.split(key, num=num_permutations)

    def count_exceedance(count, permutation_key):
        permuted_indices = jax.random.permutation(permutation_key, indices)
        statistic = statistic_from_indices(permuted_indices)
        return count + (statistic >= observed), None

    # Limit matrix operations to a small batch of permutations at a time.
    exceedances, _ = jax.lax.map(partial(count_exceedance, 0), keys, batch_size=min(32, num_permutations))
    p_value = (1 + jnp.sum(exceedances)) / (num_permutations + 1)
    exceedance_rate = jnp.clip(jnp.sum(exceedances) / num_permutations, 0, 1)
    standard_error = jnp.sqrt(num_permutations * exceedance_rate * (1 - exceedance_rate)) / (num_permutations + 1)
    return PermutationResult(observed, p_value, standard_error)

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
        result = permutation_test(x, y, kernel, permutation_key, num_permutations)

        return count + (result.p_value < alpha), None

    rejections, _ = jax.lax.scan(count_rejection, jnp.asarray(0, dtype=jnp.int32), keys)
    rate = jnp.clip(rejections / num_trials, 0, 1)
    return RateResult(rate, jnp.sqrt(rate * (1 - rate) / num_trials))

@partial(jax.jit, static_argnames=("sampler", "num_trials", "num_permutations", "num_candidates"))
def estimate_bandwidth_power(sampler, key, num_trials=500, num_permutations=500, alpha=0.05, num_candidates=9):
    """
    Estimates Gaussian-kernel power at multiples of each trial's pooled median.
    """
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1.")
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1.")

    keys = jax.random.split(key, num=num_trials)

    def count_rejections(counts, trial_key):
        sample_key, permutation_key = jax.random.split(trial_key)
        x, y = sampler(sample_key)
        bandwidths = kernels.candidate_bandwidths(x, y, num_candidates)

        def reject(sigma):
            kernel = kernels.make_gaussian_kernel(sigma)
            result = permutation_test(x, y, kernel, permutation_key, num_permutations)
            return result.p_value < alpha

        rejections = jax.lax.map(reject, bandwidths)
        return counts + rejections, None

    counts = jnp.zeros(num_candidates, dtype=jnp.int32)
    counts, _ = jax.lax.scan(count_rejections, counts, keys)
    rate = jnp.clip(counts / num_trials, 0, 1)
    return RateResult(rate, jnp.sqrt(rate * (1 - rate) / num_trials))
