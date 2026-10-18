import argparse
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mmd import distributions, kernels
from mmd.experiments import estimate_bandwidth_power, estimate_power_loss


ALPHA = 0.05
N = 100
DIMENSION = 2
DIMENSIONS = [1, 2, 5, 10, 20, 50, 100]
NUM_CANDIDATES = 9
MEAN_SHIFT = 0.25
SCALE_SHIFT = 1.25
OUTLIER_PROBABILITY = 0.1
OUTLIER_SCALE = 4.0
MIXTURE_SEPARATION = 0.75

OUTPUT = "median_heuristic_power.png"
LOSS_OUTPUT = "median_heuristic_power_loss.png"


def make_alternatives(dimension):
    return [
        ("Mean shift", distributions.make_gaussian_mean_shift_one_coordinate_sampler(N, (dimension,), MEAN_SHIFT)),
        ("Scale shift", distributions.make_gaussian_scale_shift_sampler(N, (dimension,), SCALE_SHIFT)),
        ("Gaussian vs uniform", distributions.make_gaussian_vs_uniform_sampler(N, (dimension,))),
        ("Outliers", distributions.make_gaussian_outliers_sampler(N, (dimension,), OUTLIER_PROBABILITY, OUTLIER_SCALE)),
        ("Mixture", distributions.make_gaussian_mixture_sampler(N, (dimension,), MIXTURE_SEPARATION)),
    ]


def bandwidth_sensitivity(key, trials, permutations):
    alternatives = make_alternatives(DIMENSION)

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

        for multiplier, rate, standard_error in zip(multipliers, power.rate, power.standard_error):
            print(f"sigma / median: {float(multiplier):6.4g} -> {rate:.3f} ± {standard_error:.3f}")
        print(f"Time: {elapsed:.1f}s", flush=True)

    return multipliers, results


def plot_power(multipliers, results, trials, permutations):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")

    for name, power in results.items():
        ax.errorbar(multipliers, power.rate, yerr=power.standard_error, marker="o", capsize=3, label=name)

    ax.set_xscale("log", base=2)
    ax.set_xticks(multipliers, ["1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8", "16"])
    ax.axvline(1, color="grey", linestyle=":", label="Median heuristic")
    ax.axhline(ALPHA, color="grey", linestyle="--", label=f"alpha = {ALPHA}")
    ax.set_xlabel("Bandwidth / pooled median bandwidth")
    ax.set_ylabel("Test power (±1 SE)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Bandwidth sensitivity (n={N}, d={DIMENSION})\n{trials} trials, {permutations} permutations")
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


def power_loss_vs_dimension(key, trials, permutations, selection_trials, dimensions=DIMENSIONS):
    results = {name: [] for name, _ in make_alternatives(DIMENSION)}

    for dimension in dimensions:
        print(f"\nDimension: {dimension}", flush=True)
        for i, (name, sampler) in enumerate(make_alternatives(dimension)):
            start = time.time()
            setting_key = jax.random.fold_in(jax.random.fold_in(key, dimension), i)
            pilot_key, selection_key, evaluation_key = jax.random.split(setting_key, 3)
            x, y = sampler(pilot_key)
            bandwidths = kernels.candidate_bandwidths(x, y, NUM_CANDIDATES)
            selection = estimate_bandwidth_power(
                sampler,
                selection_key,
                num_trials=selection_trials,
                num_permutations=permutations,
                alpha=ALPHA,
                bandwidths=bandwidths,
            )
            # Prefer the bandwidth closest to the pilot median when powers tie.
            distances = jnp.abs(jnp.arange(NUM_CANDIDATES) - NUM_CANDIDATES // 2)
            best = int(jnp.argmin(jnp.where(selection.rate == jnp.max(selection.rate), distances, NUM_CANDIDATES)))
            bandwidth = bandwidths[best]
            result = estimate_power_loss(
                sampler,
                bandwidth,
                evaluation_key,
                num_trials=trials,
                num_permutations=permutations,
                alpha=ALPHA,
            )
            result = jax.device_get(jax.block_until_ready(result))
            results[name].append(result)
            elapsed = time.time() - start
            boundary = " (grid edge)" if best in (0, NUM_CANDIDATES - 1) else ""
            print(f"{name}: sigma*={float(bandwidth):.3g}{boundary}, "
                  f"oracle={result.oracle.rate:.3f} ± {result.oracle.standard_error:.3f}, "
                  f"median={result.median.rate:.3f} ± {result.median.standard_error:.3f}, "
                  f"loss={result.loss.rate:.3f} ± {result.loss.standard_error:.3f} ({elapsed:.1f}s)", flush=True)

    return results


def plot_power_loss(dimensions, results, trials, permutations, selection_trials):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for name, comparisons in results.items():
        loss = [result.loss.rate for result in comparisons]
        standard_errors = [result.loss.standard_error for result in comparisons]
        ax.errorbar(dimensions, loss, yerr=standard_errors, marker="o", capsize=3, label=name)

    ax.axhline(0, color="grey", linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks(dimensions, [str(d) for d in dimensions])
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Oracle power - median power (±1 paired SE)")
    ax.set_title(f"Median heuristic power loss (n={N})\n"
                 f"{selection_trials} selection trials, {trials} evaluation trials, {permutations} permutations")
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["bandwidth", "dimension", "all"], default="bandwidth", nargs="?")
    parser.add_argument("--selection-trials", type=int, default=500)
    parser.add_argument("--dimensions", type=int, nargs="+", default=DIMENSIONS)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.trials < 2 or args.selection_trials < 1 or args.permutations < 1:
        parser.error("Use at least 2 trials, 1 selection trial, and 1 permutation.")
    if any(d < 1 for d in args.dimensions):
        parser.error("Dimensions must be positive.")

    key = jax.random.key(args.seed)
    if args.experiment in ("bandwidth", "all"):
        print("Bandwidth sensitivity around the pooled median heuristic")
        print("Median recalculated per trial; bandwidth / median ranges from 1/16 to 16", flush=True)
        multipliers, results = bandwidth_sensitivity(key, args.trials, args.permutations)
        fig = plot_power(multipliers, results, args.trials, args.permutations)
        fig.savefig(OUTPUT, dpi=200)
        plt.close(fig)
        print(f"\nPlot saved to {OUTPUT}")

    if args.experiment in ("dimension", "all"):
        print("Median heuristic power loss across dimensions")
        print("Oracle selected on separate trials; power loss = oracle power - median power", flush=True)
        results = power_loss_vs_dimension(jax.random.fold_in(key, 1), args.trials, args.permutations,
                                          args.selection_trials, args.dimensions)
        fig = plot_power_loss(args.dimensions, results, args.trials, args.permutations, args.selection_trials)
        fig.savefig(LOSS_OUTPUT, dpi=200)
        plt.close(fig)
        print(f"\nPlot saved to {LOSS_OUTPUT}")
