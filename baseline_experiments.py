import argparse
import time

import jax

from mmd import distributions, kernels
from mmd.experiments import estimate_rejection_rate


ALPHA = 0.05
N = 100
DIMENSION = 2
SIGMA = 1.0

EFFECT_SIZES = [0.0, 0.1, 0.25, 0.5, 1.0]
BANDWIDTHS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
MEAN_SHIFT = 0.25
SCALE_SHIFT = 1.25
OUTLIER_PROBABILITY = 0.1
OUTLIER_SCALE = 4.0


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
    """
    Basic check that the permutation test is correctly calibrated.
    """
    print(f"Expect rejection near {ALPHA:.0%}")

    sampler = distributions.make_gaussian_null_sampler(N, (DIMENSION,))
    kernel = kernels.make_gaussian_kernel(SIGMA)

    run_test("alpha", ALPHA, sampler, kernel, key, trials, permutations)

def power_vs_effect_size(key, trials, permutations):
    """
    Power should increase with the size of the Gaussian mean shift.
    """
    print(f"Expect power to rise from around {ALPHA:.0%} as mean shift grows")

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

def power_vs_bandwidth_by_distribution(key, trials, permutations):
    """
    Compares bandwidth sensitivity across several different alternatives.
    """

    descriptions = {
        "mean_shift": "Expect power to remain high at larger bandwidths",
        "scale_shift": "Expect power to drop at very large bandwidths",
        "gaussian_vs_uniform": "Expect power to peak at intermediate bandwidths",
        "outliers": "Expect power to vary with bandwidth",
    }

    alternatives = [
        (
            "mean_shift",
            distributions.make_gaussian_mean_shift_sampler(N, (DIMENSION,), MEAN_SHIFT),
        ),
        (
            "scale_shift",
            distributions.make_gaussian_scale_shift_sampler(N, (DIMENSION,), SCALE_SHIFT),
        ),
        (
            "gaussian_vs_uniform",
            distributions.make_gaussian_vs_uniform_sampler(N, (DIMENSION,)),
        ),
        (
            "outliers",
            distributions.make_gaussian_outliers_sampler(
                N,
                (DIMENSION,),
                OUTLIER_PROBABILITY,
                OUTLIER_SCALE,
            ),
        ),
    ]

    for i, (name, sampler) in enumerate(alternatives):
        print(f"\n{name}")
        print(descriptions[name])

        # Keep the datasets/permutations fixed across bandwidths.
        alternative_key = jax.random.fold_in(key, i)

        for sigma in BANDWIDTHS:
            kernel = kernels.make_gaussian_kernel(sigma)

            run_test(
                "sigma",
                sigma,
                sampler,
                kernel,
                alternative_key,
                trials,
                permutations,
            )


experiments = {
    "type_1": type_1_error,
    "effect_size": power_vs_effect_size,
    "bandwidth_comparison": power_vs_bandwidth_by_distribution,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["all", *experiments], default="all", nargs="?")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = experiments if args.experiment == "all" else [args.experiment]
    key = jax.random.key(args.seed)

    for i, name in enumerate(names):
        print(f"\n{name}")
        experiments[name](jax.random.fold_in(key, i), args.trials, args.permutations)
