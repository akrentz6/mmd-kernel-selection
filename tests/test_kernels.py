import jax.numpy as jnp

from mmd import kernels

def test_gaussian_kernel_same_point():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)

    x = jnp.array([1.0, 2.0, 3.0])
    value = kernel(x, x)

    print("Gaussian kernel, same point:", value)
    assert jnp.isclose(value, 1.0)

def test_gaussian_kernel_symmetry():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])

    k_xy = kernel(x, y)
    k_yx = kernel(y, x)

    print("Gaussian k(x, y):", k_xy)
    print("Gaussian k(y, x):", k_yx)
    assert jnp.isclose(k_xy, k_yx)

def test_gaussian_kernel_distance():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)

    x = jnp.array([0.0])
    y1 = jnp.array([1.0])
    y2 = jnp.array([2.0])

    close_value = kernel(x, y1)
    far_value = kernel(x, y2)

    print("Close point:", close_value)
    print("Far point:", far_value)
    assert close_value > far_value

def test_gaussian_and_rbf_match():
    sigma = 2.0
    gamma = 1 / (2 * sigma ** 2)

    gaussian = kernels.make_gaussian_kernel(sigma)
    rbf = kernels.make_rbf_kernel(gamma)

    x = jnp.array([1.0, -2.0, 0.5])
    y = jnp.array([0.0, 1.0, 2.0])

    gaussian_value = gaussian(x, y)
    rbf_value = rbf(x, y)

    print("Gaussian:", gaussian_value)
    print("RBF:", rbf_value)
    assert jnp.isclose(gaussian_value, rbf_value)

def test_median_heuristic_for_multidimensional_samples():
    x = jnp.array([[[0.0, 0.0]], [[3.0, 4.0]], [[6.0, 8.0]]])

    # Pairwise distances are 5, 10, and 5.
    median_distance = kernels.median_heuristic(x)

    print("Median heuristic:", median_distance)
    assert jnp.isclose(median_distance, 5.0)

if __name__ == "__main__":
    test_gaussian_kernel_same_point()
    test_gaussian_kernel_symmetry()
    test_gaussian_kernel_distance()
    test_gaussian_and_rbf_match()
    test_median_heuristic_for_multidimensional_samples()
