from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("kernel",))
def gram_matrix(x, y, kernel):
    """
    Computes the Gram matrix between two sets of samples:
    K[i, j] = kernel(x[i], y[j])
    """
    return jax.vmap(jax.vmap(kernel, in_axes=(None, 0)), in_axes=(0, None))(x, y)


@partial(jax.jit, static_argnames=("kernel",))
def mmd_squared_biased(x, y, kernel):
    """
    Computes the biased Maximum Mean Discrepancy (MMD) between two sets of samples.
    """
    m = x.shape[0]
    n = y.shape[0]

    k_xx = gram_matrix(x, x, kernel)
    k_yy = gram_matrix(y, y, kernel)
    k_xy = gram_matrix(x, y, kernel)

    return jnp.sum(k_xx) / (m * m) + jnp.sum(k_yy) / (n * n) - 2 * jnp.sum(k_xy) / (m * n)

@partial(jax.jit, static_argnames=("kernel",))
def mmd_squared_unbiased(x, y, kernel):
    """
    Computes the unbiased Maximum Mean Discrepancy (MMD) between two sets of samples.
    """
    m = x.shape[0]
    n = y.shape[0]

    if m < 2 or n < 2:
        raise ValueError("Unbiased MMD requires at least two samples per distribution.")

    k_xx = gram_matrix(x, x, kernel)
    k_yy = gram_matrix(y, y, kernel)
    k_xy = gram_matrix(x, y, kernel)

    sum_xx_no_diag = jnp.sum(k_xx) - jnp.trace(k_xx)
    sum_yy_no_diag = jnp.sum(k_yy) - jnp.trace(k_yy)

    return sum_xx_no_diag / (m * (m - 1)) + sum_yy_no_diag / (n * (n - 1)) - 2 * jnp.sum(k_xy) / (m * n)
