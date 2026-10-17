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

def median_heuristic(x):
    """
    Computes the median pairwise distance between points in x.
    """
    if x.shape[0] < 2:
        raise ValueError("The median heuristic requires at least two samples.")

    flat_x = jnp.reshape(x, (x.shape[0], -1))
    squared_norms = jnp.sum(flat_x ** 2, axis=1)
    squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2 * (flat_x @ flat_x.T)
    distances = jnp.sqrt(jnp.maximum(squared_distances, 0))

    i, j = jnp.triu_indices(x.shape[0], k=1)
    distances = distances[i, j]

    return jnp.median(distances)

def pooled_median_heuristic(x, y):
    """
    Computes the median heuristic on the pooled samples from x and y.
    """
    combined = jnp.concatenate([x, y], axis=0)
    return median_heuristic(combined)

def candidate_bandwidths(x, y, candidates=None):
    """
    Generates a list of candidate bandwidths based on the median heuristic.
    """
    if isinstance(candidates, int):
        median_distance = pooled_median_heuristic(x, y)
        return jnp.linspace(median_distance / 2, median_distance * 2, candidates)
    else:
        median_distance = pooled_median_heuristic(x, y)
        return jnp.array([median_distance]) if candidates is None else jnp.array(candidates)
