import argparse
import math
import time

import jax

from mmd import distributions, kernels
from mmd.experiments import estimate_rejection_rate


ALPHA = 0.05
N = 100
DIMENSION = 2
SIGMA = 1.0
SHIFT = 0.5

EFFECT_SIZES = [0.0, 0.1, 0.25, 0.5, 1.0]
SAMPLE_SIZES = [25, 50, 100, 200]
BANDWIDTHS = [0.25, 0.5, 1.0, 2.0, 4.0]
DIMENSIONS = [1, 2, 5, 10, 20]


def run_test(name, value, sampler, kernel, key, trials, permutations):
    start = time.time()
    rate = estimate_rejection_rate(
        sampler,
        kernel,
        key,
        alpha=ALPHA,
        num_trials=trials,
        num_permutations=permutations,
    )
    rate = float(jax.block_until_ready(rate))
    elapsed = time.time() - start

    print(f"{name}: {value} -> {rate:.3f} ({elapsed:.1f}s)")

def type_1_error(key, trials, permutations):
    sampler = distributions.make_gaussian_null_sampler(N, (DIMENSION,))
    kernel = kernels.make_gaussian_kernel(SIGMA)
    run_test("alpha", ALPHA, sampler, kernel, key, trials, permutations)

def power_vs_effect_size(key, trials, permutations):
    kernel = kernels.make_gaussian_kernel(SIGMA)

    for i, shift in enumerate(EFFECT_SIZES):
        sampler = distributions.make_gaussian_mean_shift_sampler(N, (DIMENSION,), shift)
        run_test(
            "shift",
            shift,
            sampler,
            kernel,
            jax.random.fold_in(key, i),
            trials,
            permutations,
        )

def power_vs_sample_size(key, trials, permutations):
    kernel = kernels.make_gaussian_kernel(SIGMA)

    for i, n in enumerate(SAMPLE_SIZES):
        sampler = distributions.make_gaussian_mean_shift_sampler(n, (DIMENSION,), SHIFT)
        run_test(
            "n",
            n,
            sampler,
            kernel,
            jax.random.fold_in(key, i),
            trials,
            permutations,
        )

def power_vs_bandwidth(key, trials, permutations):
    sampler = distributions.make_gaussian_mean_shift_sampler(N, (DIMENSION,), SHIFT)

    for i, sigma in enumerate(BANDWIDTHS):
        kernel = kernels.make_gaussian_kernel(sigma)
        run_test(
            "sigma",
            sigma,
            sampler,
            kernel,
            jax.random.fold_in(key, i),
            trials,
            permutations,
        )

def power_vs_dimension(key, trials, permutations):
    for i, d in enumerate(DIMENSIONS):
        # Keep the overall size of the mean shift roughly constant.
        shift = SHIFT / math.sqrt(d)
        sampler = distributions.make_gaussian_mean_shift_sampler(N, (d,), shift)
        kernel = kernels.make_gaussian_kernel(SIGMA * math.sqrt(d))
        run_test(
            "dimension",
            d,
            sampler,
            kernel,
            jax.random.fold_in(key, i),
            trials,
            permutations,
        )


experiments = {
    "type_1": type_1_error,
    "effect_size": power_vs_effect_size,
    "sample_size": power_vs_sample_size,
    "bandwidth": power_vs_bandwidth,
    "dimension": power_vs_dimension,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["all", *experiments], default="all", nargs="?")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = experiments if args.experiment == "all" else [args.experiment]
    key = jax.random.key(args.seed)

    for i, name in enumerate(names):
        print(f"\n{name}")
        experiments[name](jax.random.fold_in(key, i), args.trials, args.permutations)
