from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import estimators, kernels


class PermutationResult(NamedTuple):
    """
    Observed test statistic, corrected p-value, and its permutation Monte Carlo SE.
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


class PowerLossResult(NamedTuple):
    """
    Powers and their paired difference on independent evaluation trials.
    """
    oracle: RateResult
    median: RateResult
    loss: RateResult


@jax.jit
def select_max_mmd(x, y, bandwidths):
    """
    Selects the bandwidth maximising observed unbiased MMD squared.
    """
    values = jax.lax.map(lambda sigma: estimators.mmd_squared_unbiased(x, y, kernels.make_gaussian_kernel(sigma)), bandwidths)
    return bandwidths[jnp.argmax(values)]

@jax.jit
def select_max_variance_adjusted_mmd(x, y, bandwidths, regularisation=1e-8):
    """
    Selects unbiased MMD squared / sqrt(jackknife variance + regularisation).
    """
    def score(sigma):
        kernel = kernels.make_gaussian_kernel(sigma)
        value = estimators.mmd_squared_unbiased(x, y, kernel)
        variance = estimators.mmd_squared_unbiased_variance(x, y, kernel)
        return value / jnp.sqrt(variance + regularisation)

    values = jax.lax.map(score, bandwidths)
    return bandwidths[jnp.argmax(values)]

@partial(jax.jit, static_argnames=("selector", "num_permutations"))
def naive_adaptive_test(x, y, bandwidths, key, num_permutations=1000, selector=select_max_mmd):
    """
    Selects on observed labels, then holds that bandwidth fixed in permutations.
    """
    sigma = selector(x, y, bandwidths)
    return permutation_test(x, y, kernels.make_gaussian_kernel(sigma), key, num_permutations)

@partial(jax.jit, static_argnames=("selection_fraction", "num_candidates", "num_permutations"))
def sample_split_test(x, y, key, selection_fraction=0.5, num_candidates=5, num_permutations=1000):
    """
    Selects a variance-adjusted bandwidth on independent selection data.
    """
    m_select = int(x.shape[0] * selection_fraction)
    n_select = int(y.shape[0] * selection_fraction)
    if min(m_select, n_select) < 3 or min(x.shape[0] - m_select, y.shape[0] - n_select) < 2:
        raise ValueError("Each group needs at least three selection and two test samples.")
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1.")

    x_key, y_key, permutation_key = jax.random.split(key, 3)
    x = jax.random.permutation(x_key, x, axis=0)
    y = jax.random.permutation(y_key, y, axis=0)
    x_select, x_test = x[:m_select], x[m_select:]
    y_select, y_test = y[:n_select], y[n_select:]

    bandwidths = kernels.candidate_bandwidths(x_select, y_select, num_candidates)
    bandwidths = jnp.maximum(bandwidths, jnp.finfo(bandwidths.dtype).eps)
    sigma = select_max_variance_adjusted_mmd(x_select, y_select, bandwidths)
    return permutation_test(x_test, y_test, kernels.make_gaussian_kernel(sigma), permutation_key, num_permutations)

@partial(jax.jit, static_argnames=("num_candidates", "num_permutations"))
def permutation_max_test(x, y, key, num_candidates=5, num_permutations=1000):
    """
    Calibrates max unbiased MMD squared by repeating the maximum per permutation.
    """
    m = x.shape[0]
    n = y.shape[0]
    if min(m, n) < 2:
        raise ValueError("Unbiased MMD requires at least two samples per distribution.")
    if num_candidates < 1 or num_permutations < 1:
        raise ValueError("num_candidates and num_permutations must be at least 1.")

    bandwidths = kernels.candidate_bandwidths(x, y, num_candidates)
    bandwidths = jnp.maximum(bandwidths, jnp.finfo(bandwidths.dtype).eps)
    combined = jnp.concatenate([x, y], axis=0)
    gram_matrices = jax.lax.map(lambda sigma: estimators.gram_matrix(combined, combined, kernels.make_gaussian_kernel(sigma)), bandwidths)
    indices = jnp.arange(m + n)

    def statistic_from_indices(indices):
        in_x = jnp.zeros(m + n, dtype=gram_matrices.dtype).at[indices[:m]].set(1)
        in_y = 1 - in_x

        def statistic(k):
            k_x = k @ in_x
            k_y = k @ in_y
            sum_xx = in_x @ k_x - in_x @ jnp.diag(k)
            sum_yy = in_y @ k_y - in_y @ jnp.diag(k)
            return sum_xx / (m * (m - 1)) + sum_yy / (n * (n - 1)) - 2 * (in_x @ k_y) / (m * n)

        return jnp.max(jax.lax.map(statistic, gram_matrices))

    observed = statistic_from_indices(indices)

    def count_exceedance(count, permutation_key):
        permuted_indices = jax.random.permutation(permutation_key, indices)
        # Include numerical ties despite cancellation in the unbiased statistic.
        tolerance = 8 * jnp.finfo(gram_matrices.dtype).eps
        return count + (statistic_from_indices(permuted_indices) >= observed - tolerance), None

    keys = jax.random.split(key, num=num_permutations)
    exceedances, _ = jax.lax.scan(count_exceedance, jnp.asarray(0, dtype=jnp.int32), keys)
    p_value = (1 + exceedances) / (num_permutations + 1)
    rate = jnp.clip(exceedances / num_permutations, 0, 1)
    standard_error = jnp.sqrt(num_permutations * rate * (1 - rate)) / (num_permutations + 1)
    return PermutationResult(observed, p_value, standard_error)


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
def estimate_bandwidth_power(sampler, key, num_trials=500, num_permutations=500, alpha=0.05, num_candidates=9, bandwidths=None):
    """
    Estimates Gaussian-kernel power at multiples of each trial's pooled median.
    If bandwidths are supplied, evaluates that fixed grid instead.
    """
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1.")
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1.")

    keys = jax.random.split(key, num=num_trials)

    def count_rejections(counts, trial_key):
        sample_key, permutation_key = jax.random.split(trial_key)
        x, y = sampler(sample_key)
        candidates = kernels.candidate_bandwidths(x, y, num_candidates) if bandwidths is None else bandwidths

        def reject(sigma):
            kernel = kernels.make_gaussian_kernel(sigma)
            result = permutation_test(x, y, kernel, permutation_key, num_permutations)
            return result.p_value < alpha

        rejections = jax.lax.map(reject, candidates)
        return counts + rejections, None

    counts = jnp.zeros(num_candidates if bandwidths is None else len(bandwidths), dtype=jnp.int32)
    counts, _ = jax.lax.scan(count_rejections, counts, keys)
    rate = jnp.clip(counts / num_trials, 0, 1)
    return RateResult(rate, jnp.sqrt(rate * (1 - rate) / num_trials))

@partial(jax.jit, static_argnames=("sampler", "num_trials", "num_permutations"))
def estimate_power_loss(sampler, bandwidth, key, num_trials=500, num_permutations=500, alpha=0.05):
    """
    Compares a selected fixed bandwidth with the per-trial median.
    """
    if num_trials < 2:
        raise ValueError("Power loss requires at least two evaluation trials.")

    keys = jax.random.split(key, num=num_trials)

    def count_rejections(counts, trial_key):
        sample_key, permutation_key = jax.random.split(trial_key)
        x, y = sampler(sample_key)
        median = kernels.pooled_median_heuristic(x, y)
        oracle_result = permutation_test(x, y, kernels.make_gaussian_kernel(bandwidth), permutation_key, num_permutations)
        median_result = permutation_test(x, y, kernels.make_gaussian_kernel(median), permutation_key, num_permutations)
        oracle_reject = (oracle_result.p_value < alpha).astype(jnp.int32)
        median_reject = (median_result.p_value < alpha).astype(jnp.int32)
        difference = oracle_reject - median_reject
        return counts + jnp.array([oracle_reject, median_reject, difference ** 2]), None

    counts, _ = jax.lax.scan(count_rejections, jnp.zeros(3, dtype=jnp.int32), keys)
    rates = jnp.clip(counts[:2] / num_trials, 0, 1)
    standard_errors = jnp.sqrt(rates * (1 - rates) / num_trials)
    loss = (counts[0] - counts[1]) / num_trials
    loss_se = jnp.sqrt(jnp.maximum(counts[2] / num_trials - loss ** 2, 0) / (num_trials - 1))
    return PowerLossResult(
        RateResult(rates[0], standard_errors[0]),
        RateResult(rates[1], standard_errors[1]),
        RateResult(loss, loss_se),
    )

@partial(jax.jit, static_argnames=("sampler", "num_trials", "num_permutations", "num_candidates"))
def estimate_adaptive_rejection_rates(sampler, key, fixed_bandwidth=1.0, alpha=0.05, num_trials=500, num_permutations=500, num_candidates=9):
    """
    Compares fixed, pooled median, naive raw MMD, and naive variance-adjusted MMD.
    All four methods use the same samples and permutation keys in each trial.
    """
    if num_trials < 1 or num_candidates < 1:
        raise ValueError("num_trials and num_candidates must be at least 1.")

    def count_rejections(counts, trial_key):
        sample_key, permutation_key = jax.random.split(trial_key)
        x, y = sampler(sample_key)
        bandwidths = kernels.candidate_bandwidths(x, y, num_candidates)
        median = bandwidths[num_candidates // 2]
        fixed_result = permutation_test(x, y, kernels.make_gaussian_kernel(fixed_bandwidth), permutation_key, num_permutations)
        median_result = permutation_test(x, y, kernels.make_gaussian_kernel(median), permutation_key, num_permutations)
        raw_result = naive_adaptive_test(x, y, bandwidths, permutation_key, num_permutations)
        adjusted_result = naive_adaptive_test(x, y, bandwidths, permutation_key, num_permutations, select_max_variance_adjusted_mmd)
        p_values = jnp.array([fixed_result.p_value, median_result.p_value, raw_result.p_value, adjusted_result.p_value])
        return counts + (p_values < alpha), None

    keys = jax.random.split(key, num=num_trials)
    counts, _ = jax.lax.scan(count_rejections, jnp.zeros(4, dtype=jnp.int32), keys)
    rates = jnp.clip(counts / num_trials, 0, 1)
    return RateResult(rates, jnp.sqrt(rates * (1 - rates) / num_trials))

@partial(jax.jit, static_argnames=("sampler", "num_trials", "num_permutations", "num_candidates", "selection_fractions"))
def estimate_calibrated_rejection_rates(sampler, key, alpha=0.05, num_trials=500, num_permutations=500, num_candidates=5, selection_fractions=(0.25, 0.5, 0.75)):
    """
    Compares median, naive raw/adjusted, sample splits, and permutation maximum.
    """
    if num_trials < 1 or num_candidates < 1:
        raise ValueError("num_trials and num_candidates must be at least 1.")

    def count_rejections(counts, trial_key):
        sample_key, permutation_key = jax.random.split(trial_key)
        x, y = sampler(sample_key)
        bandwidths = kernels.candidate_bandwidths(x, y, num_candidates)
        median = bandwidths[num_candidates // 2]
        median_result = permutation_test(x, y, kernels.make_gaussian_kernel(median), permutation_key, num_permutations)
        raw_result = naive_adaptive_test(x, y, bandwidths, permutation_key, num_permutations)
        adjusted_result = naive_adaptive_test(x, y, bandwidths, permutation_key, num_permutations, select_max_variance_adjusted_mmd)
        split_results = [sample_split_test(x, y, permutation_key, fraction, num_candidates, num_permutations)
                         for fraction in selection_fractions]
        max_result = permutation_max_test(x, y, permutation_key, num_candidates, num_permutations)
        p_values = jnp.array([result.p_value for result in [median_result, raw_result, adjusted_result, *split_results, max_result]])
        return counts + (p_values < alpha), None

    keys = jax.random.split(key, num=num_trials)
    counts, _ = jax.lax.scan(count_rejections, jnp.zeros(4 + len(selection_fractions), dtype=jnp.int32), keys)
    rates = jnp.clip(counts / num_trials, 0, 1)
    return RateResult(rates, jnp.sqrt(rates * (1 - rates) / num_trials))
