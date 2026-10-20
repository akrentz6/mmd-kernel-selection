import argparse
import hashlib
from pathlib import Path
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch

from cifar_experiments import (
    ALPHA, CLASSES, NULL_LABEL, SAMPLE_SIZES, load_data,
    plot_comparison, sample_size_comparison,
)


MODEL_REPO = "chenyaofo/pytorch-cifar-models"
MODEL_NAME = "cifar10_resnet56"
WEIGHTS_URL = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
OUTPUT = "cifar_model_power.png"


def load_model(directory, device):
    """Downloads model code and pretrained weights into data/torch_hub once."""
    torch.hub.set_dir(str(Path(directory) / "torch_hub"))
    model = torch.hub.load(MODEL_REPO, MODEL_NAME, pretrained=False,
                           trust_repo=True, skip_validation=True)
    weights = torch.hub.load_state_dict_from_url(
        WEIGHTS_URL, model_dir=Path(directory) / "torch_hub" / "checkpoints",
        map_location="cpu", check_hash=True, weights_only=True,
    )
    model.load_state_dict(weights)
    return model.to(device).eval()


def predict_logits(model, images, batch_size=128, device="cpu"):
    """Converts flattened [0, 1] RGB images to the model's ten output logits."""
    mean = torch.tensor(MEAN, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).reshape(1, 3, 1, 1)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start:start + batch_size])
            batch = batch.reshape(-1, 32, 32, 3).permute(0, 3, 1, 2).to(device)
            logits = model((batch - mean) / std)
            outputs.append(logits.cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def load_outputs(directory, batch_size=128, device="cpu", refresh=False):
    """Caches logits, checking the model, preprocessing and dataset contents."""
    x, x_labels, y, y_labels = load_data(directory)
    fingerprint = hashlib.sha256(f"{MODEL_REPO}/{MODEL_NAME}/{WEIGHTS_URL}/{MEAN}/{STD}".encode())
    for array in (x, x_labels, y, y_labels):
        fingerprint.update(str((array.shape, array.dtype)).encode())
        fingerprint.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    signature = fingerprint.hexdigest()
    cache = Path(directory) / f"{MODEL_NAME}_outputs.npz"
    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as saved:
            if saved["signature"].item() == signature:
                print(f"Using cached logits: {cache}", flush=True)
                return saved["x"], saved["x_labels"], saved["y"], saved["y_labels"]

    start = time.time()
    model = load_model(directory, device)
    print(f"{MODEL_NAME}: {sum(p.numel() for p in model.parameters()):,} parameters; device={device}", flush=True)
    x_logits = predict_logits(model, x, batch_size, device)
    print(f"CIFAR-10 inference complete ({len(x)} images)", flush=True)
    y_logits = predict_logits(model, y, batch_size, device)
    np.savez(cache, signature=signature, x=x_logits, x_labels=x_labels, y=y_logits, y_labels=y_labels)
    print(f"Logits saved to {cache} ({time.time() - start:.1f}s)", flush=True)
    return x_logits, x_labels, y_logits, y_labels


def model_comparison(x, x_labels, y, y_labels, key, trials, permutations,
                     sample_sizes=SAMPLE_SIZES, classes=range(10)):
    results = sample_size_comparison(x, x_labels, y, y_labels, jax.random.fold_in(key, 0),
                                     trials, permutations, sample_sizes, classes)
    null = sample_size_comparison(x, x_labels, y, y_labels, jax.random.fold_in(key, 1),
                                  trials, permutations, sample_sizes, classes=[], same_distribution=True)
    results[NULL_LABEL] = null["All classes"]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMD on pretrained CIFAR-10 ResNet-56 output logits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=SAMPLE_SIZES)
    parser.add_argument("--classes", choices=CLASSES, nargs="*", default=CLASSES,
                        help="True classes to examine in addition to the overall and null comparisons.")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--refresh-outputs", action="store_true", help="Recompute logits instead of using the cache.")
    args = parser.parse_args()
    if args.trials < 2 or args.permutations < 20:
        parser.error("Use at least 2 trials and 20 permutations (to allow p < 0.05).")
    if any(n < 12 for n in args.sample_sizes) or args.batch_size < 1:
        parser.error("Use at least 12 images per sample and a positive batch size.")
    sizes = sorted(set(args.sample_sizes))
    classes = [CLASSES.index(name) for name in dict.fromkeys(args.classes)]
    # Small 32x32 batches often run faster with fewer CPU threads.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        x, x_labels, y, y_labels = load_outputs(args.data_dir, args.batch_size, args.device, args.refresh_outputs)
    except FileNotFoundError as error:
        parser.error(f"{error}. See README.md for the expected dataset files.")
    for name, logits, labels in [("CIFAR-10", x, x_labels), ("CIFAR-10.1", y, y_labels)]:
        print(f"{name} accuracy: {np.mean(logits.argmax(axis=1) == labels):.2%}", flush=True)
    results = model_comparison(x, x_labels, y, y_labels, jax.random.key(args.seed),
                               args.trials, args.permutations, sizes, classes)
    fig = plot_comparison(results, args.trials, args.permutations)
    fig.suptitle(f"CIFAR-10 vs CIFAR-10.1: ResNet-56 output logits\n"
                 f"{args.trials} trials, {args.permutations} permutations; alpha={ALPHA}")
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)
    print(f"\nPlot saved to {OUTPUT}")
