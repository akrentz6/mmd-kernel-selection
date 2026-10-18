import argparse
import time

import jax
import matplotlib.pyplot as plt

from mmd import distributions
from mmd.experiments import estimate_adaptive_rejection_rates


ALPHA = 0.05
N = 100
DIMENSION = 10
SIGMA = 1.0
NUM_CANDIDATES = 9
EFFECT_SIZES = [0.0, 0.1, 0.25, 0.5, 1.0]
METHODS = ["Fixed bandwidth", "Pooled median", "Naive max MMD", "Naive variance-adjusted"]
OUTPUT = "adaptive_power.png"


def adaptive_comparison(key, trials, permutations):
    results = []
    for shift in EFFECT_SIZES:
        print(f"\nMean shift: {shift}", flush=True)
        sampler = distributions.make_gaussian_mean_shift_one_coordinate_sampler(N, (DIMENSION,), shift)
        start = time.time()
        result = estimate_adaptive_rejection_rates(
            sampler,
            key,
            fixed_bandwidth=SIGMA,
            alpha=ALPHA,
            num_trials=trials,
            num_permutations=permutations,
            num_candidates=NUM_CANDIDATES,
        )
        result = jax.device_get(jax.block_until_ready(result))
        results.append(result)
        for name, rate, standard_error in zip(METHODS, result.rate, result.standard_error):
            print(f"{name}: {rate:.3f} ± {standard_error:.3f}")
        print(f"Time: {time.time() - start:.1f}s", flush=True)

    return results


def plot_comparison(results, trials, permutations):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    null = results[0]
    labels = ["Fixed", "Median", "Raw MMD", "Adjusted"]

    for i, name in enumerate(METHODS):
        axes[0].bar(i, null.rate[i], yerr=null.standard_error[i], capsize=4, color=f"C{i}")
        rates = [result.rate[i] for result in results]
        errors = [result.standard_error[i] for result in results]
        axes[1].errorbar(EFFECT_SIZES, rates, yerr=errors, marker="o", capsize=3, label=name)

    axes[0].set_xticks(range(len(METHODS)), labels)
    axes[0].set_title("Null calibration: P = Q = N(0, I)")
    axes[0].set_ylabel("Type I rejection rate (±1 SE)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Rejection rate as mean shift grows")
    axes[1].set_xlabel("Mean shift magnitude")
    axes[1].set_ylabel("Rejection rate (±1 SE)")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    for ax in axes:
        ax.axhline(ALPHA, color="grey", linestyle="--", label=f"alpha = {ALPHA}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"Naive adaptive bandwidth selection (n={N}, d={DIMENSION})\n"
                 f"{trials} trials, {permutations} permutations; dashed line: alpha={ALPHA}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Raw MMD vs variance-adjusted bandwidth selection")
    print("Expect fixed and pooled median near 5% under the null; adaptive selection may inflate rejection")
    print("Both adaptive methods select once on observed labels; adjustment does not correct selection bias", flush=True)
    results = adaptive_comparison(jax.random.key(args.seed), args.trials, args.permutations)
    fig = plot_comparison(results, args.trials, args.permutations)
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)
    print(f"\nPlot saved to {OUTPUT}")
