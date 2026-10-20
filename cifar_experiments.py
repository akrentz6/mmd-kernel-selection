import argparse
from functools import partial
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from calibrated_adaptive_experiments import METHODS, SELECTION_FRACTIONS
from mmd import experiments, kernels


ALPHA = 0.05
SAMPLE_SIZES = [16, 32, 64, 128]
NUM_CANDIDATES = 5
NULL_LABEL = "CIFAR-10 null (Type I error)"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load_data(directory):
    """Loads the CIFAR-10 test batch and CIFAR-10.1 v6 in the same pixel order."""
    directory = Path(directory)
    with (directory / "cifar-10-batches-py" / "test_batch").open("rb") as file:
        batch = pickle.load(file, encoding="bytes")
    x = np.asarray(batch[b"data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_labels = np.asarray(batch[b"labels"])
    y = np.load(directory / "cifar10.1_v6_data.npy", allow_pickle=False)
    y_labels = np.load(directory / "cifar10.1_v6_labels.npy", allow_pickle=False)
    if y.shape != (len(y_labels), 32, 32, 3):
        raise ValueError("Expected CIFAR-10.1 images with shape (n, 32, 32, 3).")

    # One common transform, without fitting separate normalisations to the datasets.
    x = x.reshape(len(x), -1).astype(np.float32) / 255
    y = y.reshape(len(y), -1).astype(np.float32) / 255
    return x, x_labels, y, y_labels


def sample_pair(x, y, n, rng, same_distribution=False):
    """Samples without replacement within a trial."""
    if same_distribution:
        indices = rng.choice(len(x), 2 * n, replace=False)
        return x[indices[:n]], x[indices[n:]]
    return x[rng.choice(len(x), n, replace=False)], y[rng.choice(len(y), n, replace=False)]


@partial(jax.jit, static_argnames=("permutations", "num_candidates"))
def compare_tests(x, y, key, permutations, num_candidates=NUM_CANDIDATES):
    """Uses the same samples and permutation key for all seven methods."""
    bandwidths = kernels.candidate_bandwidths(x, y, num_candidates)
    bandwidths = jnp.maximum(bandwidths, jnp.finfo(bandwidths.dtype).eps)
    median = experiments.permutation_test(
        x, y, kernels.make_gaussian_kernel(bandwidths[num_candidates // 2]), key, permutations,
    )
    raw = experiments.naive_adaptive_test(x, y, bandwidths, key, permutations)
    adjusted = experiments.naive_adaptive_test(
        x, y, bandwidths, key, permutations, experiments.select_max_variance_adjusted_mmd,
    )
    splits = [experiments.sample_split_test(x, y, key, fraction, num_candidates, permutations)
              for fraction in SELECTION_FRACTIONS]
    maximum = experiments.permutation_max_test(x, y, key, num_candidates, permutations)
    return jnp.array([result.p_value for result in [median, raw, adjusted, *splits, maximum]])


def estimate_rates(x, y, n, key, trials, permutations, same_distribution=False):
    # Keep the full datasets on the host and compile only the small sampled arrays.
    sample_key, test_key = jax.random.split(key)
    rng = np.random.default_rng(int(jax.random.bits(sample_key)))
    rejections = np.zeros(len(METHODS))
    for trial in range(trials):
        sample_x, sample_y = sample_pair(x, y, n, rng, same_distribution)
        p_values = compare_tests(sample_x, sample_y, jax.random.fold_in(test_key, trial), permutations)
        rejections += np.asarray(p_values) < ALPHA
    rates = rejections / trials
    return experiments.RateResult(rates, np.sqrt(rates * (1 - rates) / trials))


def sample_size_comparison(x, x_labels, y, y_labels, key, trials, permutations,
                           sample_sizes=SAMPLE_SIZES, classes=range(10), same_distribution=False):
    groups = [("All classes", x, y)]
    groups += [(CLASSES[label], x[x_labels == label], y[y_labels == label]) for label in classes]
    results = {}
    for i, (name, group_x, group_y) in enumerate(groups):
        limit = len(group_x) // 2 if same_distribution else min(len(group_x), len(group_y))
        results[name] = {}
        pools = f"{len(group_x)} CIFAR-10 images" if same_distribution else f"{len(group_x)} CIFAR-10, {len(group_y)} CIFAR-10.1"
        print(f"\n{name} ({pools})", flush=True)
        for n in sample_sizes:
            if n > limit:
                print(f"n={n}: skipped (maximum without replacement: {limit})", flush=True)
                continue
            start = time.time()
            setting_key = jax.random.fold_in(jax.random.fold_in(key, i), n)
            result = estimate_rates(group_x, group_y, n, setting_key, trials, permutations, same_distribution)
            results[name][n] = result
            print(f"n={n} per sample", flush=True)
            for method, rate, error in zip(METHODS, result.rate, result.standard_error):
                print(f"  {method}: {rate:.3f} ± {error:.3f}")
            print(f"Time: {time.time() - start:.1f}s", flush=True)
    return results


def plot_comparison(results, trials, permutations, same_distribution=False):
    names = [name for name in results if name != NULL_LABEL]
    if NULL_LABEL in results:
        names.insert(0, NULL_LABEL)
    columns = min(3, len(results))
    rows = (len(results) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows),
                             squeeze=False, layout="constrained")
    for ax, name in zip(axes.flat, names):
        comparisons = results[name]
        sizes = list(comparisons)
        for i, method in enumerate(METHODS):
            rates = [result.rate[i] for result in comparisons.values()]
            errors = [result.standard_error[i] for result in comparisons.values()]
            ax.errorbar(sizes, rates, yerr=errors, marker="o", capsize=3, label=method,
                        linestyle="--" if i in (1, 2) else "-")
        ax.axhline(ALPHA, color="grey", linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.set_xlabel("Images per sample")
        ax.set_ylabel("Type I error (±1 SE)" if same_distribution or name == NULL_LABEL else "Rejection rate (±1 SE)")
        ax.set_ylim(0, 1.02)
        ax.set_title(name)
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize="small")
    title = "CIFAR-10 vs CIFAR-10: Type I error" if same_distribution else "CIFAR-10 vs CIFAR-10.1: power"
    fig.suptitle(f"{title}\n{trials} trials, {permutations} permutations; alpha={ALPHA}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["power", "type_1", "all"], default="all", nargs="?")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=SAMPLE_SIZES)
    parser.add_argument("--classes", choices=CLASSES, nargs="*", default=CLASSES,
                        help="Classes to examine in addition to the overall comparison; empty means overall only.")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.trials < 2 or args.permutations < 20:
        parser.error("Use at least 2 trials and 20 permutations (to allow p < 0.05).")
    if any(n < 12 for n in args.sample_sizes):
        parser.error("Use at least 12 images per sample for all three selection/test splits.")
    sizes = sorted(set(args.sample_sizes))
    classes = [CLASSES.index(name) for name in dict.fromkeys(args.classes)]
    try:
        x, x_labels, y, y_labels = load_data(args.data_dir)
    except FileNotFoundError as error:
        parser.error(f"{error}. See README.md for the expected dataset files.")

    key = jax.random.key(args.seed)
    results = {}
    for name in (["power", "type_1"] if args.experiment == "all" else [args.experiment]):
        null = name == "type_1"
        print(f"\n{name}: {'disjoint CIFAR-10 samples' if null else 'CIFAR-10 vs CIFAR-10.1'}", flush=True)
        comparison = sample_size_comparison(x, x_labels, y, y_labels, jax.random.fold_in(key, int(null)),
                                            args.trials, args.permutations, sizes, [] if null else classes, null)
        results.update({NULL_LABEL: comparison["All classes"]} if null else comparison)
    fig = plot_comparison(results, args.trials, args.permutations, args.experiment == "type_1")
    output = "cifar_type_1.png" if args.experiment == "type_1" else "cifar_power.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"\nPlot saved to {output}")
