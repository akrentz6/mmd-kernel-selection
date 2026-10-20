import jax
import matplotlib.pyplot as plt
import numpy as np
import torch

from cifar_experiments import CLASSES, NULL_LABEL, plot_comparison
from cifar_model_experiments import MEAN, STD, model_comparison, predict_logits


def test_batched_inference():
    """Batching preserves logits and uses the pretrained model's normalisation."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 10, 1), torch.nn.BatchNorm2d(10),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
    )
    images = np.random.default_rng(0).random((9, 3072), dtype=np.float32)
    batched = predict_logits(model, images, batch_size=4)
    whole = predict_logits(model, images, batch_size=9)
    pixels = torch.from_numpy(images).reshape(-1, 32, 32, 3).permute(0, 3, 1, 2)
    mean = torch.tensor(MEAN).reshape(1, 3, 1, 1)
    std = torch.tensor(STD).reshape(1, 3, 1, 1)
    with torch.inference_mode():
        expected = model((pixels - mean) / std).numpy()
    np.testing.assert_allclose(batched, whole, atol=1e-6)
    np.testing.assert_allclose(batched, expected, atol=1e-6)
    assert batched.shape == (9, 10)
    print("Batched inference: nine images, ten logits each; normalisation matches")


def test_small_logit_experiment():
    """A short null/shift comparison on synthetic ten-dimensional model outputs."""
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(10), 24)
    x = rng.normal(size=(240, 10)).astype(np.float32)
    y = rng.normal(size=(240, 10)).astype(np.float32) + 3
    results = model_comparison(x, labels, y, labels, jax.random.key(0),
                               trials=2, permutations=39, sample_sizes=[12])
    for name, comparisons in results.items():
        print(f"{name}: median rejection={comparisons[12].rate[0]:.2f}")
    fig = plot_comparison(results, 2, 39)
    assert [ax.get_title() for ax in fig.axes] == [NULL_LABEL, "All classes", *CLASSES]
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_batched_inference()
    test_small_logit_experiment()
