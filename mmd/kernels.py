import jax.numpy as jnp

def gaussian_kernel(x, y, sigma):
    """
    Computes the Gaussian kernel between two sets of points.
    """
    squared_distance = jnp.sum((x - y) ** 2)
    return jnp.exp(-squared_distance / (2 * sigma ** 2))

def make_gaussian_kernel(sigma):
    return lambda x, y: gaussian_kernel(x, y, sigma)

def rbf_kernel(x, y, gamma):
    """
    Computes the Radial Basis Function (RBF) kernel between two sets of points.
    """
    squared_distance = jnp.sum((x - y) ** 2)
    return jnp.exp(-gamma * squared_distance)

def make_rbf_kernel(gamma):
    return lambda x, y: rbf_kernel(x, y, gamma)