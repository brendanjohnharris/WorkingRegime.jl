import numpy as np
import jax
import jax.numpy as jnp
from itertools import product
from abc import ABC, abstractmethod


# class AbstractPositions(ABC):
#     @abstractmethod
#     def __call__(self, *args, **kwargs):
#         pass

#     def cast_to_tuple(self, x):
#         if isinstance(x, (list, np.ndarray, jnp.ndarray)):
#             return tuple(x)
#         elif isinstance(x, tuple):
#             return x
#         else:  # For scalars, wrap it in a tuple
#             return (x,)


# class GridPositions(AbstractPositions):
#     def __init__(self, domain):
#         self.domain = self.cast_to_tuple(domain)

#     def __call__(self, shape):
#         shape = self.cast_to_tuple(shape)
#         if len(shape) != len(self.domain):
#             raise ValueError("Shape and size must have the same length")
#         grids = []
#         for s, n in zip(self.domain, shape):
#             offset = (s / n) / 2  # Offset to center the grid
#             grids.append(jnp.linspace(0 + offset, s + offset, n, endpoint=False))
#         positions = list(product(*grids))
#         return positions

#     def to_dict(self):
#         return {"domain": self.domain}


# class RandomPositions(AbstractPositions):
#     def __init__(self, domain):
#         self.domain = self.cast_to_tuple(domain)

#     def __call__(self, shape):
#         shape = self.cast_to_tuple(shape)
#         total_positions = jnp.prod(shape)
#         positions = []
#         for s in self.domain:
#             positions.append(np.random.uniform(0, s, total_positions))
#         positions = list(zip(*positions))
#         return positions

#     def to_dict(self):
#         return {"domain": self.domain}


class AbstractPositions(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def cast_to_tuple(self, x):
        if isinstance(x, (list, jnp.ndarray)):
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


@jax.tree_util.register_pytree_node_class
class ClusteredPositions(AbstractPositions):
    def __init__(self, center, radius):
        self.center = self.cast_to_tuple(center)
        self.radius = radius

    def __call__(self, shape, key):
        shape = self.cast_to_tuple(shape)
        key_theta, key_r = jax.random.split(key)

        # Generate random angles uniformly between 0 and 2π
        theta = jax.random.uniform(key_theta, shape=shape, minval=0, maxval=2 * jnp.pi)
        # Generate random radii with uniform distribution over disc area
        r = self.radius * jnp.sqrt(jax.random.uniform(key_r, shape=shape))

        # Convert polar coordinates to Cartesian coordinates
        x = self.center[0] + r * jnp.cos(theta)
        y = self.center[1] + r * jnp.sin(theta)

        positions = jnp.stack([x, y], axis=-1)
        return positions

    def to_dict(self):
        return {"center": self.center, "radius": self.radius}

    # Methods to make the class compatible with JAX transformations
    def tree_flatten(self):
        children = (self.center, self.radius)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        center, radius = children
        return cls(center, radius)
