import argparse
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mmd import distributions
from mmd.experiments import estimate_bandwidth_power


ALPHA = 0.05
N = 100
DIMENSION = 2
NUM_CANDIDATES = 9
MEAN_SHIFT = 0.25
SCALE_SHIFT = 1.25
OUTLIER_PROBABILITY = 0.1
OUTLIER_SCALE = 4.0
MIXTURE_SEPARATION = 0.75

OUTPUT = "median_heuristic_power.png"


def bandwidth_sensitivity(key, trials, permutations):
    alternatives = [
        ("Mean shift", distributions.make_gaussian_mean_shift_one_coordinate_sampler(N, (DIMENSION,), MEAN_SHIFT)),
        ("Scale shift", distributions.make_gaussian_scale_shift_sampler(N, (DIMENSION,), SCALE_SHIFT)),
        ("Gaussian vs uniform", distributions.make_gaussian_vs_uniform_sampler(N, (DIMENSION,))),
        ("Outliers", distributions.make_gaussian_outliers_sampler(N, (DIMENSION,), OUTLIER_PROBABILITY, OUTLIER_SCALE)),
        ("Mixture", distributions.make_gaussian_mixture_sampler(N, (DIMENSION,), MIXTURE_SEPARATION)),
    ]

    multipliers = 2.0 ** jnp.arange(-4, 5)
    results = {}

    for i, (name, sampler) in enumerate(alternatives):
        print(f"\n{name}", flush=True)
        start = time.time()
        power = estimate_bandwidth_power(
            sampler,
            jax.random.fold_in(key, i),
            num_trials=trials,
            num_permutations=permutations,
            alpha=ALPHA,
            num_candidates=NUM_CANDIDATES,
        )
        power = jax.device_get(jax.block_until_ready(power))
        elapsed = time.time() - start
        results[name] = power

        for multiplier, rate in zip(multipliers, power):
            print(f"sigma / median: {float(multiplier):6.4g} -> {rate:.3f}")
        print(f"Time: {elapsed:.1f}s", flush=True)

    return multipliers, results


def plot_power(multipliers, results, trials, permutations):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")

    for name, power in results.items():
        ax.plot(multipliers, power, marker="o", label=name)

    ax.set_xscale("log", base=2)
    ax.set_xticks(multipliers, ["1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8", "16"])
    ax.axvline(1, color="grey", linestyle=":", label="Median heuristic")
    ax.axhline(ALPHA, color="grey", linestyle="--", label=f"alpha = {ALPHA}")
    ax.set_xlabel("Bandwidth / pooled median bandwidth")
    ax.set_ylabel("Test power")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Bandwidth sensitivity (n={N}, d={DIMENSION})\n{trials} trials, {permutations} permutations")
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Bandwidth sensitivity around the pooled median heuristic")
    print("Median recalculated per trial; bandwidth / median ranges from 1/16 to 16", flush=True)
    multipliers, results = bandwidth_sensitivity(jax.random.key(args.seed), args.trials, args.permutations)
    fig = plot_power(multipliers, results, args.trials, args.permutations)
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)
    print(f"\nPlot saved to {OUTPUT}")
