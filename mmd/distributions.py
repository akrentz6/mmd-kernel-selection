import math
from functools import partial

import jax
import jax.numpy as jnp


SQRT_THREE = math.sqrt(3.0)


@partial(jax.jit, static_argnames=("n", "sample_shape"))
def gaussian_null(key, n, sample_shape=()):
    """
    Samples x and y independently from N(0, I).
    """
    key_x, key_y = jax.random.split(key)

    x = jax.random.normal(key_x, (n,) + tuple(sample_shape))
    y = jax.random.normal(key_y, (n,) + tuple(sample_shape))

    return x, y

def make_gaussian_null_sampler(n, sample_shape=()):
    return lambda key: gaussian_null(key, n, sample_shape)

@partial(jax.jit, static_argnames=("n", "sample_shape"))
def gaussian_mean_shift(key, n, sample_shape=(), shift=1.0):
    """
    Samples x from N(0, I) and y from N(shift, I).
    """
    key_x, key_y = jax.random.split(key)

    x = jax.random.normal(key_x, (n,) + tuple(sample_shape))
    y = jax.random.normal(key_y, (n,) + tuple(sample_shape)) + shift

    return x, y

def make_gaussian_mean_shift_sampler(n, sample_shape=(), shift=1.0):
    return lambda key: gaussian_mean_shift(key, n, sample_shape, shift)

@partial(jax.jit, static_argnames=("n", "sample_shape"))
def gaussian_mean_shift_one_coordinate(key, n, sample_shape=(), shift=1.0):
    """
    Samples x from N(0, I) and y from N(shift * e_1, I), where e_1 is the
    first standard basis vector. This keeps the Euclidean distance between
    the means to be abs(shift) in any dimension.
    """
    key_x, key_y = jax.random.split(key)

    x = jax.random.normal(key_x, (n,) + tuple(sample_shape))
    y = jax.random.normal(key_y, (n,) + tuple(sample_shape))
    y = y.at[(slice(None),) + (0,) * len(sample_shape)].add(shift)

    return x, y

def make_gaussian_mean_shift_one_coordinate_sampler(n, sample_shape=(), shift=1.0):
    return lambda key: gaussian_mean_shift_one_coordinate(key, n, sample_shape, shift)

@partial(jax.jit, static_argnames=("n", "sample_shape"))
def gaussian_scale_shift(key, n, sample_shape=(), scale=2.0):
    """
    Samples x from N(0, I) and y from N(0, scale^2 * I).
    """
    key_x, key_y = jax.random.split(key)

    x = jax.random.normal(key_x, (n,) + tuple(sample_shape))
    y = scale * jax.random.normal(key_y, (n,) + tuple(sample_shape))

    return x, y

def make_gaussian_scale_shift_sampler(n, sample_shape=(), scale=2.0):
    return lambda key: gaussian_scale_shift(key, n, sample_shape, scale)

@partial(jax.jit, static_argnames=("n", "sample_shape"))
def gaussian_vs_uniform(key, n, sample_shape=(), minval=-SQRT_THREE, maxval=SQRT_THREE):
    """
    Samples x from N(0, I) and y from U[-sqrt(3), sqrt(3)].

    The default uniform distribution has mean 0 and variance 1,
    matching the first two moments of the standard Gaussian.
    """
    key_x, key_y = jax.random.split(key)

    x = jax.random.normal(key_x, (n,) + tuple(sample_shape))
    y = jax.random.uniform(key_y, (n,) + tuple(sample_shape), minval=minval, maxval=maxval)

    return x, y

def make_gaussian_vs_uniform_sampler(n, sample_shape=(), minval=-SQRT_THREE, maxval=SQRT_THREE):
    return lambda key: gaussian_vs_uniform(key, n, sample_shape, minval, maxval)

@partial(jax.jit, static_argnames=("n", "sample_shape"))
def gaussian_mixture(key, n, sample_shape=(), separation=2.0):
    """
    Samples x from N(0, I) and y from 0.5 * N(-separation, I) + 0.5 * N(separation, I).
    """
    key_x, key_noise, key_components = jax.random.split(key, 3)
    shape = (n,) + tuple(sample_shape)

    x = jax.random.normal(key_x, shape)
    noise = jax.random.normal(key_noise, shape)
    components = jax.random.bernoulli(key_components, p=0.5, shape=(n,))

    signs = jnp.where(components, 1.0, -1.0)
    # give each observation the same mixture-component shift
    signs = signs.reshape( (n,) + (1,) * len(sample_shape))

    y = noise + signs * separation

    return x, y

def make_gaussian_mixture_sampler(n, sample_shape=(), separation=2.0):
    return lambda key: gaussian_mixture(key, n, sample_shape, separation)

@partial(jax.jit, static_argnames=("n", "sample_shape"))
def gaussian_outliers(key, n, sample_shape=(), outlier_probability=0.1, outlier_scale=5.0):
    """
    Samples x N(0, I) and y from a mixture of N(0, I) and N(0, outlier_scale^2 * I).

    y is mostly standard Gaussian, but a fraction of observations have a much larger scale.
    """
    key_x, key_y, key_outliers = jax.random.split(key, 3)
    shape = (n,) + tuple(sample_shape)

    x = jax.random.normal(key_x, shape)
    y = jax.random.normal(key_y, shape)

    is_outlier = jax.random.bernoulli(key_outliers, p=outlier_probability, shape=(n,))
    scales = jnp.where(is_outlier, outlier_scale, 1.0)
    scales = scales.reshape((n,) + (1,) * len(sample_shape))

    y = y * scales

    return x, y

def make_gaussian_outliers_sampler(n, sample_shape=(), outlier_probability=0.1, outlier_scale=5.0):
    return lambda key: gaussian_outliers(key, n, sample_shape, outlier_probability, outlier_scale)
