import jax
import numpy as np

from cifar_experiments import METHODS, estimate_rates, sample_pair


def test_disjoint_null_samples():
    images = np.arange(80).reshape(40, 2)
    x, y = sample_pair(images, images, 12, np.random.default_rng(0), same_distribution=True)
    assert len(np.unique(np.concatenate([x[:, 0], y[:, 0]]))) == 24
    print("Null trial: 24 distinct images split into two groups of 12")


def test_small_shift_comparison():
    """A small power/calibration experiment, without downloading any images."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 2)).astype(np.float32)
    y = rng.normal(size=(100, 2)).astype(np.float32) + 6
    key = jax.random.key(2)
    null = estimate_rates(x, y, 16, key, trials=4, permutations=39, same_distribution=True)
    shifted = estimate_rates(x, y, 16, key, trials=4, permutations=39)
    for i, method in enumerate(METHODS):
        print(f"{method}: null={null.rate[i]:.2f}, large shift={shifted.rate[i]:.2f}")
    # Four trials illustrate the null behaviour; they cannot measure calibration precisely.
    assert shifted.rate[0] >= 0.75
    assert shifted.rate[-1] >= 0.75
    assert np.all(np.isfinite(null.standard_error))


if __name__ == "__main__":
    test_disjoint_null_samples()
    test_small_shift_comparison()
