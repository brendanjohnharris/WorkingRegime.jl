import brainpy as bp
import brainpy.math as bm
from brainpy.connect import TwoEndConnector
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import networkx as nx
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax import lax
import copy
import os

from typing import Union, Callable, Optional, Sequence, Any
from functools import partial
from .utils import *


def euclidean_distance(pos1, pos2):
    return jnp.sqrt(jnp.sum((pos1 - pos2) ** 2))


def gumbel_sample_top_k(rng_key, log_weights, k):
    """
    Gumbel-max trick to sample k items without replacement,
    proportionally to exp(log_weights).

    Returns:
    -------
    (top_scores, top_indices)
    """
    # Sample Gumbels
    gumbels = jax.random.gumbel(rng_key, shape=log_weights.shape)
    # Score = log_weight + gumbel
    scores = log_weights + gumbels
    # Grab top k
    top_scores, top_indices = jax.lax.top_k(scores, k)
    return top_scores, top_indices


class DistanceDependent(TwoEndConnector):
    """
    Synaptic connector that connects neurons based on their distances.
    Optionally enforces an exact total number of connections, `num_connections`.
    """

    def __init__(
        self,
        kernel,
        num_connections,
        domain,
        positions_pre,
        positions_post=None,
        distance_metric=euclidean_distance,
        seed=None,
        **kwargs,
    ):
        super(DistanceDependent, self).__init__(**kwargs)
        self.positions_pre = jnp.asarray(positions_pre)
        if positions_post is None:
            self.positions_post = self.positions_pre
        else:
            self.positions_post = jnp.asarray(positions_post)
        self.num_pre_neurons = self.positions_pre.shape[0]
        self.num_post_neurons = self.positions_post.shape[0]
        self.dimensions = self.positions_pre.shape[1]
        self.domain = jnp.asarray(domain)

        if distance_metric is None:
            raise ValueError("A distance metric function must be specified.")
        self.distance_metric = distance_metric

        if kernel is None:
            raise ValueError("A kernel function must be specified.")
        self.kernel = kernel

        self.num_connections = num_connections

        if seed is not None:
            self.key = seed
        else:
            self.key = jax.random.PRNGKey(np.random.randint(0, 2**32))

    def build_csr(self):
        """
        Build a CSR matrix with exactly `num_connections` total edges.
        The chosen edges are drawn proportionally to the distance-based
        connection probabilities given by `self.kernel`.
        """
        num_connections = self.num_connections

        # ------------------------------
        # 1) Compute the distance matrix for all (i, j).
        #    shape: (num_pre_neurons, num_post_neurons)
        def compute_distance_with_boundary(pos_pre, pos_post):
            # Periodic boundary adjustment
            delta = pos_pre - pos_post
            delta_periodic = (
                jnp.mod(delta + self.domain / 2, self.domain) - self.domain / 2
            )
            adjusted = pos_post + delta_periodic
            return self.distance_metric(adjusted, pos_post)

        def distance_for_one_pre(pre_pos):
            return jax.vmap(lambda ppos: compute_distance_with_boundary(pre_pos, ppos))(
                self.positions_post
            )

        # For demonstration, compute all distances in one shot (no chunking):
        distances = jax.vmap(distance_for_one_pre)(self.positions_pre)
        # Probability matrix
        probs = self.kernel(distances)  # shape = (num_pre, num_post)

        # ------------------------------
        # 2) Flatten the probability matrix to shape (M,).
        #    M = num_pre * num_post
        #    We'll sample exactly num_connections edges from these M with probability ∝ probs.
        M = self.num_pre_neurons * self.num_post_neurons
        p_flat = probs.reshape((M,))
        # For numerical stability, take log of probabilities,
        # but clamp at some small epsilon to avoid -inf
        eps = 1e-20
        log_p = jnp.log(jnp.maximum(p_flat, eps))

        # If user requests more connections than total pairs, clamp it
        # num_connections = jnp.minimum(num_connections, M)

        # ------------------------------
        # 3) Weighted Gumbel top-k sampling
        self.key, subkey = jax.random.split(self.key)
        _, top_indices = gumbel_sample_top_k(subkey, log_p, num_connections)

        # Now we have the `num_connections` chosen pairs.
        # Convert back to (row, col)
        rows = top_indices // self.num_post_neurons
        cols = top_indices % self.num_post_neurons

        # ------------------------------
        # 4) Sort chosen edges by (row, then col) to build a valid CSR.
        #    We'll do an argsort of the pairs. We can combine row, col into
        #    a single "lex order" index for sorting, or we can do a stable sort
        #    by row then col. For simplicity, do a single pass:
        def sort_by_row_then_col(r, c):
            """
            Return r, c sorted lexicographically by r, then c.
            """
            # Make a single key for each pair: key = r*(max_col+1) + c
            # or use jnp.lexsort with [c, r]
            # We'll do jnp.lexsort for clarity:
            sort_keys = jnp.stack([c, r], axis=0)
            # jnp.lexsort sorts by the last row first, so we stack as [c, r].
            # That means the sorting is primarily by r, breaks ties by c.
            idx_sorted = jnp.lexsort(sort_keys)
            return r[idx_sorted], c[idx_sorted]

        sorted_rows, sorted_cols = sort_by_row_then_col(rows, cols)

        # ------------------------------
        # 5) Build row_counts and indptr from sorted_rows
        #    row_counts[i] = how many chosen edges belong to presyn neuron i
        #    so we can do a scatter_add of 1 for each row index
        row_counts = jnp.zeros((self.num_pre_neurons,), dtype=jnp.int32)
        row_counts = row_counts.at[sorted_rows].add(1)  # scatter

        # Build the indptr array: length = (num_pre_neurons + 1)
        indptr = jnp.cumsum(
            jnp.concatenate([jnp.array([0], dtype=row_counts.dtype), row_counts])
        )

        # ------------------------------
        # 6) Pre-allocate arrays for final CSR
        #    post_ids has length = num_connections
        #    (We won't store pre_ids as a separate array; the row partition is in indptr.)
        post_ids = jnp.asarray(sorted_cols, dtype=get_idx_type())
        indptr = indptr.astype(get_idx_type())

        return post_ids, indptr


class AbstractKernel(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass


class GaussianKernel(AbstractKernel):
    """
    Gaussian kernel for a distance-dependent connector.

    Parameters
    ----------
    sigma : float, keyword-only
        Width of the Gaussian.
    p_max : float, optional
        Maximum probability of connection.
    """

    def __init__(
        self,
        sigma,
        p_max=1.0,
    ):
        super().__init__()
        self.sigma = sigma
        self.p_max = p_max

    def mass2pmax(omega, sigma):
        """
        Convert the total integrated distribution to the height at the origin. Only for 2D.
        """
        return omega / (2 * np.pi * sigma**2)

    def pmax2mass(p_max, sigma):
        """
        Convert the height at the origin to the total integrated distribution. Only for 2D.
        """
        return p_max * (2 * np.pi * sigma**2)

    def __call__(self, distance):
        return self.p_max * jnp.exp(-(distance**2) / (2 * self.sigma**2))

    def to_dict(self):
        return {"sigma": self.sigma, "p_max": self.p_max}


class ExponentialKernel(AbstractKernel):
    """
    Exponential kernel for a distance-dependent connector.

    Parameters
    ----------
    sigma : float, keyword-only
        Decay constant (length scale) for the exponential kernel.
    p_max : float, optional
        Maximum probability of connection (default is 1.0).
    """

    def __init__(self, sigma, p_max=1.0):
        super().__init__()
        self.sigma = sigma
        self.p_max = p_max

    def __call__(self, distance):
        """
        Compute the connection probability given a distance.

        Parameters
        ----------
        distance : float or array-like
            The distance(s) between pre- and post-synaptic neurons.

        Returns
        -------
        float or array-like
            The connection probability.
        """
        return self.p_max * jnp.exp(-distance / self.sigma)

    def mass2pmax(omega, sigma):
        """
        Convert the total integrated distribution to the height at the origin. Only for 2D.
        """
        return omega / (2 * np.pi * sigma**2)

    def pmax2mass(p_max, sigma):
        """
        Convert the height at the origin to the total integrated distribution. Only for 2D.
        """
        return p_max * (2 * np.pi * sigma**2)

    def to_dict(self):
        """
        Return a dictionary representation of the kernel's parameters.
        """
        return {"sigma": self.sigma, "p_max": self.p_max}


class FixedProb(TwoEndConnector):
    def __init__(
        self,
        prob,
        pre_ratio=1.0,
        include_self=True,
        seed=None,
        **kwargs,
    ):
        super(FixedProb, self).__init__(**kwargs)
        assert 0.0 <= prob <= 1.0
        assert 0.0 <= pre_ratio <= 1.0
        self.prob = prob
        self.pre_ratio = pre_ratio
        self.include_self = include_self
        self.seed = seed
        self._jaxrand = bm.random.default_rng(self.seed)

    def _iii(
        self,
    ):
        if (not self.include_self) and (self.pre_num != self.post_num):
            raise bp.ConnectorError(
                f"We found pre_num != post_num ({self.pre_num} != {self.post_num}). "
                f"But `include_self` is set to True."
            )

        if self.pre_ratio < 1.0:
            pre_num_to_select = int(self.pre_num * self.pre_ratio)
            pre_ids = self._jaxrand.choice(
                self.pre_num, size=(pre_num_to_select,), replace=False
            )
        else:
            pre_num_to_select = self.pre_num
            pre_ids = jnp.arange(self.pre_num)

        post_num_total = self.post_num
        post_num_to_select = int(self.post_num * self.prob)

        selected_post_ids = self._jaxrand.randint(
            0, post_num_total, (pre_num_to_select, post_num_to_select)
        )

        return (
            pre_num_to_select,
            post_num_to_select,
            bm.as_jax(selected_post_ids),
            bm.as_jax(pre_ids),
        )

    def build_csr(self):
        pre_num_to_select, post_num_to_select, selected_post_ids, pre_ids = self._iii()
        pre_nums = jnp.ones(pre_num_to_select) * post_num_to_select
        if not self.include_self:
            true_ids = selected_post_ids == jnp.reshape(pre_ids, (-1, 1))
            pre_nums -= jnp.sum(true_ids, axis=1)
            selected_post_ids = selected_post_ids.flatten()[
                jnp.logical_not(true_ids).flatten()
            ]
        else:
            selected_post_ids = selected_post_ids.flatten()
        selected_pre_inptr = jnp.cumsum(jnp.concatenate([jnp.zeros(1), pre_nums]))
        return selected_post_ids.astype(get_idx_type()), selected_pre_inptr.astype(
            get_idx_type()
        )
