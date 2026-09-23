import numpy as np
import jax
import jax.numpy as jnp
from itertools import product
from abc import ABC, abstractmethod


class AbstractPositions(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def cast_to_tuple(self, x):
        if isinstance(x, (list, np.ndarray, jnp.ndarray)):
            return tuple(x)
        elif isinstance(x, tuple):
            return x
        else:  # For scalars, wrap it in a tuple
            return (x,)


class Positions(AbstractPositions):
    def __init__(self, positions):
        self.positions = positions

    def __call__(self, *args):
        return self.positions

    def to_dict(self):
        return {"positions": self.positions}


class GridPositions(AbstractPositions):
    def __init__(self, domain):
        self.domain = self.cast_to_tuple(domain)

    def __call__(self, shape):
        shape = self.cast_to_tuple(shape)
        if len(shape) != len(self.domain):
            raise ValueError("Shape and domain must have the same length")
        grids = []
        for s, n in zip(self.domain, shape):
            offset = (s / n) / 2  # Offset to center the grid
            grids.append(jnp.linspace(0 + offset, s + offset, n, endpoint=False))
        positions = list(product(*grids))
        return positions

    def to_dict(self):
        return {"domain": self.domain}


class RandomPositions(AbstractPositions):
    def __init__(self, domain, key=None):
        super().__init__()
        self.domain = self.cast_to_tuple(domain)
        # If no key is provided, just generate one from a random seed
        if key is None:
            key = jax.random.PRNGKey(42)
        self.key = key

    def __call__(self, shape, sort=True):
        """
        Generates random (x, y, ...) positions within 'domain' for the given 'shape'.
        Returns the positions in sorted order if sort=True
        Returns a JAX array of shape (N, D), where N = prod(shape) and D = len(domain).
        """
        shape = self.cast_to_tuple(shape)
        total_positions = np.prod(shape)

        coords = []
        key = self.key

        # For each dimension s in domain, sample total_positions from Uniform(0, s)
        for s in self.domain:
            key, subkey = jax.random.split(key)
            arr = jax.random.uniform(subkey, shape=(total_positions,)) * s
            coords.append(arr)

        # coords is a list of D arrays [ (N,), (N,), ... ]
        # Stack along axis=1 to get shape (N, D)
        coords = jnp.stack(coords, axis=1)

        if sort:
            # Sort the positions along the first dimension
            coords = coords[jnp.lexsort((coords[:, 0], coords[:, 1]))]

        # Update this object's key to avoid reusing it
        self.key = key
        return coords

    def to_dict(self):
        return {"domain": self.domain}


class ClusteredPositions(AbstractPositions):
    def __init__(self, center, radius, key=None):
        self.center = self.cast_to_tuple(center)
        self.radius = radius
        self._init_key = jax.random.PRNGKey(42) if key is None else key

    def __call__(self, shape, key=None):
        shape = self.cast_to_tuple(shape)
        total_positions = int(np.prod(shape))
        # Use the provided key or fall back to _init_key
        if key is None:
            key = self._init_key

        key, subkey_theta = jax.random.split(key)
        theta = jax.random.uniform(
            subkey_theta, shape=(total_positions,), minval=0.0, maxval=2 * jnp.pi
        )

        key, subkey_r = jax.random.split(key)
        r = self.radius * jnp.sqrt(
            jax.random.uniform(
                subkey_r, shape=(total_positions,), minval=0.0, maxval=1.0
            )
        )

        x = self.center[0] + r * jnp.cos(theta)
        y = self.center[1] + r * jnp.sin(theta)

        # Return positions as a JAX array, along with the updated key
        positions = jnp.stack([x, y], axis=1)
        return positions, key

    def to_dict(self):
        return {"center": self.center, "radius": self.radius}
