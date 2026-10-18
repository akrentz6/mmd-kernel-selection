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

@partial(jax.jit, static_argnames=("kernel",))
def mmd_squared_unbiased_variance(x, y, kernel):
    """
    Two-sample delete-one jackknife variance of unbiased MMD squared.
    """
    m = x.shape[0]
    n = y.shape[0]
    if m < 3 or n < 3:
        raise ValueError("Jackknife variance requires at least three samples per distribution.")

    k_xx = gram_matrix(x, x, kernel)
    k_yy = gram_matrix(y, y, kernel)
    k_xy = gram_matrix(x, y, kernel)
    rows_xx = jnp.sum(k_xx, axis=1) - jnp.diag(k_xx)
    rows_yy = jnp.sum(k_yy, axis=1) - jnp.diag(k_yy)
    sum_xx = jnp.sum(rows_xx)
    sum_yy = jnp.sum(rows_yy)
    sum_xy = jnp.sum(k_xy)

    # Recompute each leave-one-out statistic from sums, without new Gram matrices.
    leave_x = ((sum_xx - 2 * rows_xx) / ((m - 1) * (m - 2))
               + sum_yy / (n * (n - 1))
               - 2 * (sum_xy - jnp.sum(k_xy, axis=1)) / ((m - 1) * n))
    leave_y = (sum_xx / (m * (m - 1))
               + (sum_yy - 2 * rows_yy) / ((n - 1) * (n - 2))
               - 2 * (sum_xy - jnp.sum(k_xy, axis=0)) / (m * (n - 1)))
    return (m - 1) * jnp.var(leave_x) + (n - 1) * jnp.var(leave_y)
