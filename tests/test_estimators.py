import jax.numpy as jnp

from mmd import kernels, estimators

def test_gram_matrix_shape():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)

    x = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    y = jnp.array([[0.0, 0.0], [1.0, 1.0]])

    k_xy = estimators.gram_matrix(x, y, kernel)

    print("Gram matrix:")
    print(k_xy)
    print("Shape:", k_xy.shape)
    assert k_xy.shape == (3, 2)

def test_gram_matrix_symmetry():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)

    x = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])

    k_xx = estimators.gram_matrix(x, x, kernel)

    print("Gram matrix:")
    print(k_xx)
    assert jnp.allclose(k_xx, k_xx.T)

def test_gram_matrix_diagonal():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)

    x = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])

    k_xx = estimators.gram_matrix(x, x, kernel)

    print("Diagonal:", jnp.diag(k_xx))
    assert jnp.allclose(jnp.diag(k_xx), 1.0)

def test_biased_mmd_same_samples():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)

    x = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

    mmd = estimators.mmd_squared_biased(x, x, kernel)

    print("Biased MMD(x, x):", mmd)
    assert jnp.isclose(mmd, 0.0, atol=1e-6)

def test_mmd_symmetry():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)

    x = jnp.array([[0.0], [1.0], [2.0]])
    y = jnp.array([[5.0], [6.0], [7.0]])

    mmd_xy = estimators.mmd_squared_biased(x, y, kernel)
    mmd_yx = estimators.mmd_squared_biased(y, x, kernel)

    print("MMD(x, y):", mmd_xy)
    print("MMD(y, x):", mmd_yx)
    assert jnp.isclose(mmd_xy, mmd_yx)

def test_mmd_different_samples():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)

    x = jnp.array([[0.0], [0.1], [-0.1]])
    y = jnp.array([[5.0], [5.1], [4.9]])

    mmd = estimators.mmd_squared_biased(x, y, kernel)

    print("MMD for clearly different samples:", mmd)
    assert mmd > 0

def test_unbiased_mmd_returns_scalar():
    kernel = kernels.make_gaussian_kernel(sigma=1.0)

    x = jnp.array([[0.0], [1.0], [2.0]])
    y = jnp.array([[0.5], [1.5], [2.5]])

    mmd = estimators.mmd_squared_unbiased(x, y, kernel)

    print("Unbiased MMD:", mmd)
    print("Shape:", mmd.shape)
    assert mmd.shape == ()


if __name__ == "__main__":
    test_gram_matrix_shape()
    test_gram_matrix_symmetry()
    test_gram_matrix_diagonal()
    test_biased_mmd_same_samples()
    test_mmd_symmetry()
    test_mmd_different_samples()
    test_unbiased_mmd_returns_scalar()