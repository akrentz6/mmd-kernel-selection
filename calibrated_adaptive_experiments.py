import argparse
import time

import jax
import matplotlib.pyplot as plt

from mmd import distributions
from mmd.experiments import estimate_calibrated_rejection_rates


ALPHA = 0.05
N = 100
DIMENSION = 10
NUM_CANDIDATES = 5
SELECTION_FRACTIONS = (0.25, 0.5, 0.75)
EFFECT_SIZES = [0.0, 0.1, 0.25, 0.5, 1.0]
SCALES = [1.0, 1.05, 1.1, 1.25, 1.5]
METHODS = ["Pooled median", "Naive max MMD", "Naive variance-adjusted",
           "Split 25/75", "Split 50/50", "Split 75/25", "Permutation max"]
OUTPUT = "calibrated_adaptive_power.png"


def calibrated_comparison(key, trials, permutations):
    settings = [
        ("Mean shift", EFFECT_SIZES, distributions.make_gaussian_mean_shift_one_coordinate_sampler),
        ("Scale shift", SCALES, distributions.make_gaussian_scale_shift_sampler),
    ]
    results = {name: [] for name, _, _ in settings}
    for i, (name, values, make_sampler) in enumerate(settings):
        setting_key = jax.random.fold_in(key, i)
        for value in values:
            print(f"\n{name}: {value}", flush=True)
            sampler = make_sampler(N, (DIMENSION,), value)
            start = time.time()
            result = estimate_calibrated_rejection_rates(
                sampler,
                setting_key,
                alpha=ALPHA,
                num_trials=trials,
                num_permutations=permutations,
                num_candidates=NUM_CANDIDATES,
                selection_fractions=SELECTION_FRACTIONS,
            )
            result = jax.device_get(jax.block_until_ready(result))
            results[name].append(result)
            for method, rate, standard_error in zip(METHODS, result.rate, result.standard_error):
                print(f"{method}: {rate:.3f} ± {standard_error:.3f}")
            print(f"Time: {time.time() - start:.1f}s", flush=True)

    return results


def plot_comparison(results, trials, permutations):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    labels = ["Median", "Naive raw", "Naive adj.", "25/75", "50/50", "75/25", "Perm. max"]

    settings = [
        ("Mean shift", EFFECT_SIZES, "Mean shift magnitude", "shift = 0"),
        ("Scale shift", SCALES, "Standard deviation multiplier (variance = multiplier²)", "scale = 1"),
    ]
    for row, (name, values, xlabel, null_label) in enumerate(settings):
        null = results[name][0]
        for i, method in enumerate(METHODS):
            axes[row, 0].bar(i, null.rate[i], yerr=null.standard_error[i], capsize=4, color=f"C{i}")
            rates = [result.rate[i] for result in results[name]]
            errors = [result.standard_error[i] for result in results[name]]
            axes[row, 1].errorbar(values, rates, yerr=errors, marker="o", capsize=3, label=method, linestyle="--" if i in (1, 2) else "-")

        axes[row, 0].set_xticks(range(len(METHODS)), labels, rotation=30, ha="right")
        axes[row, 0].set_title(f"{name}: null calibration ({null_label})")
        axes[row, 0].set_ylabel("Type I rejection rate (±1 SE)")
        axes[row, 0].set_ylim(bottom=0)
        axes[row, 1].set_title(f"Rejection rate as {name.lower()} grows")
        axes[row, 1].set_xlabel(xlabel)
        axes[row, 1].set_ylabel("Rejection rate (±1 SE)")
        axes[row, 1].set_ylim(0, 1.02)
        axes[row, 1].legend()
        for ax in axes[row]:
            ax.axhline(ALPHA, color="grey", linestyle="--", label=f"alpha = {ALPHA}")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    fig.suptitle(f"Calibrated adaptive bandwidth selection (n={N}, d={DIMENSION})\n"
                 f"{trials} trials, {permutations} permutations; dashed line: alpha={ALPHA}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Sample splitting vs permutation-calibrated bandwidth selection")
    results = calibrated_comparison(jax.random.key(args.seed), args.trials, args.permutations)
    fig = plot_comparison(results, args.trials, args.permutations)
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)
    print(f"\nPlot saved to {OUTPUT}")
