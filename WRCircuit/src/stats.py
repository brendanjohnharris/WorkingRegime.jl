import brainpy as bp
import brainpy.math as bm
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
from tqdm import trange, tqdm

import json
import pickle

from brainpy import share
from typing import Union, Callable, Optional, Sequence, Any, Dict, Tuple, List
from brainpy.types import ArrayType
from functools import partial


def create_run(
    model,
    fixed_params,
    monitors,
    duration,
    transient=0.0,
    concrete_out=False,
):
    monitor_names = [m if isinstance(m, str) else m[0] for m in monitors]
    transient_idx = int(transient / bp.share["dt"])

    def run(swept_params):
        m = model(**fixed_params, **swept_params)
        runner = bp.DSRunner(m, monitors=monitors, numpy_mon_after_run=concrete_out)
        runner.run(duration=duration)
        return {m: runner.mon[m][transient_idx:, :] for m in monitor_names}

    return run


def create_stats_run(run, stats):
    """
    run is a create_run function returning a given set of monitors.
    stats is a Dict of statname=>function pairs. Each function should accept the dict
    of create_run as an input, select the relevant monitor, and return the desired
    statistic. Assuems you are using the bp.share["dt"]
    """

    def stats_run(swept_params):
        results = run(swept_params)  # Dict of monitor outputs
        calc_stats = {key: jax.tree_map(func, results) for key, func in stats.items()}
        return calc_stats

    return stats_run


def save(file, d):
    with open(file, "wb") as f:
        pickle.dump(d, f)
    return file


def load(file):
    with open(file, "rb") as f:
        d = pickle.load(f)
    return d


def progress_vmap(
    func: callable,
    batch_size: int,
    in_axes=0,
    clear_buffer: bool = True,
):
    def _vmap(arguments: Union[Dict[str, ArrayType], Sequence[ArrayType]]):
        if not isinstance(arguments, (dict, tuple, list)):
            raise TypeError(
                f'"arguments" must be sequence or dict, but we got {type(arguments)}'
            )

        # Convert to appropriate array type
        array_func = np.array if clear_buffer else jnp.array
        arguments = jax.tree.map(array_func, arguments)

        # Flatten the pytree
        flat_args, tree_def = jax.tree.flatten(arguments)

        # Verify uniform length
        lengths = [len(ele) for ele in flat_args]
        if len(np.unique(lengths)) != 1:
            raise ValueError(
                f"All elements in parameters should have the same length. "
                f"But we got {jax.tree.unflatten(tree_def, lengths)}"
            )

        n_samples = lengths[0]

        # Create vmapped function - define once
        vfunc = vmap(func, in_axes=in_axes)

        # Process in batches
        all_batch_results = []
        res_tree = None

        for i in trange(0, n_samples, batch_size):
            # * Do we need to regenerate the vfunc each loop?
            # Create batch
            batch_slice = [ele[i : i + batch_size] for ele in flat_args]
            batch_args = jax.tree.unflatten(tree_def, batch_slice)

            # Process batch
            batch_result = vfunc(batch_args)

            # Extract values and structure
            batch_values, batch_tree = jax.tree.flatten(
                batch_result, is_leaf=lambda a: isinstance(a, bm.Array)
            )

            # Store structure from first batch
            if res_tree is None:
                res_tree = batch_tree

            # Store batch results
            if clear_buffer:
                all_batch_results.append([np.asarray(val) for val in batch_values])
                bm.clear_buffer_memory()
            else:
                all_batch_results.append(batch_values)

        # Handle empty results
        if not all_batch_results:
            return None

        # Transpose to group by value index instead of batch
        n_values = len(all_batch_results[0])
        value_batches = [[] for _ in range(n_values)]

        for batch_values in all_batch_results:
            for i, val in enumerate(batch_values):
                value_batches[i].append(np.asarray(val) if clear_buffer else val)

        # Concatenate each value's batches
        concat_func = np.concatenate if clear_buffer else jnp.concatenate
        final_values = [concat_func(batches, axis=0) for batches in value_batches]

        # Clear buffer if requested
        if clear_buffer:
            bm.clear_buffer_memory()

        # Return reconstructed result
        return jax.tree.unflatten(res_tree, final_values), arguments

    return _vmap


from typing import Union, Sequence, Dict, List, Any

ArrayType = Any  # For simplicity


def partial_vmap(
    func: callable,
    batch_size: int,
    static_argnames: List[str] = [],
    in_axes=0,
    clear_buffer: bool = True,
):
    def _vmap(arguments: Dict[str, ArrayType]):
        if not isinstance(arguments, dict):
            raise TypeError("partial_vmap requires a dictionary of arguments")

        # Convert inputs to the appropriate array type
        array_func = np.array if clear_buffer else jnp.array
        arguments = jax.tree.map(array_func, arguments)

        # Split into dynamic and static arguments
        dynamic_args = {k: v for k, v in arguments.items() if k not in static_argnames}
        static_args = {k: arguments[k] for k in static_argnames if k in arguments}

        # Check that we have dynamic arguments
        if not dynamic_args:
            raise ValueError("No dynamic arguments found. All arguments are static.")

        # Determine the number of samples from any dynamic argument
        n_samples = len(next(iter(dynamic_args.values())))

        # Create lookup for sample indices by group
        sample_groups = {}  # Maps sample index to group index
        group_indices = {}  # Maps group index to list of sample indices
        group_static = {}  # Maps group index to static args for that group
        next_group = 0

        for i in range(n_samples):
            # Extract the "static" slice for this sample
            sample_static = {}
            for k, v in static_args.items():
                sample_static[k] = v[i].item()

            # Check if this combination already exists in a group
            found_group = False
            for group_idx, existing_static in group_static.items():
                # Compare each static argument value
                if all(
                    (
                        np.array_equal(sample_static[k], existing_static[k])
                        if isinstance(sample_static[k], np.ndarray)
                        else sample_static[k] == existing_static[k]
                    )
                    for k in sample_static
                ):
                    sample_groups[i] = group_idx
                    group_indices[group_idx].append(i)
                    found_group = True
                    break

            # If no matching group, create a new group
            if not found_group:
                group_idx = next_group
                next_group += 1
                sample_groups[i] = group_idx
                group_indices[group_idx] = [i]
                group_static[group_idx] = sample_static

        # Process each group
        all_results = []
        res_tree = None

        # print(static_args)
        # print(group_indices)
        for group_idx, indices in tqdm(group_indices.items()):
            # Extract static args for this group
            static_vals = group_static[group_idx]
            # Extract dynamic args for these samples
            group_dynamic = {k: v[indices] for k, v in dynamic_args.items()}

            # Define function with static args baked in
            def run_with_static(dyn_args):
                return func({**dyn_args, **static_vals})

            # Create vmapped function
            vfunc = vmap(run_with_static, in_axes=in_axes)

            # Process in batches
            group_batch_results = []
            for i in trange(0, len(indices), batch_size, leave=False):
                # if clear_buffer:  # ? Why??
                #     vfunc = vmap(run_with_static, in_axes=in_axes)

                # Slice the batch
                batch_args = {
                    k: v[i : i + batch_size] for k, v in group_dynamic.items()
                }

                # Process batch
                batch_result = vfunc(batch_args)

                # Extract structure for the first batch
                if res_tree is None:
                    batch_values, batch_tree = jax.tree.flatten(
                        batch_result, is_leaf=lambda a: isinstance(a, bm.Array)
                    )
                    batch_values = [array_func(val) for val in batch_values]
                    res_tree = batch_tree
                else:
                    batch_values, _ = jax.tree.flatten(
                        batch_result, is_leaf=lambda a: isinstance(a, bm.Array)
                    )
                    batch_values = [array_func(val) for val in batch_values]

                group_batch_results.append(batch_values)

                # Clear memory if requested
                if clear_buffer:
                    bm.clear_buffer_memory()

            # Skip empty groups
            if not group_batch_results:
                continue

            # Transpose to group by value index instead of by batch
            n_values = len(group_batch_results[0])
            value_batches = [[] for _ in range(n_values)]
            for batch_values in group_batch_results:
                for i, val in enumerate(batch_values):
                    value_batches[i].append(np.asarray(val) if clear_buffer else val)

            # Concatenate each value across its batches
            concat_func = np.concatenate if clear_buffer else jnp.concatenate
            group_values = [concat_func(batches, axis=0) for batches in value_batches]

            # Store results along with the sample indices that produced them
            all_results.append((indices, group_values))

            if clear_buffer:
                bm.clear_buffer_memory()

        # If everything is empty, return None
        if not all_results:
            return None

        # Allocate final arrays and fill
        n_values = len(all_results[0][1])
        final_values = []
        for i in range(n_values):
            shape = all_results[0][1][i].shape[1:]  # remove batch dimension
            dtype = all_results[0][1][i].dtype
            final_values.append(np.zeros((n_samples,) + shape, dtype=dtype))

        # Place group results in the correct location of final arrays
        for indices, values in all_results:
            for i, idx in enumerate(indices):
                for v_idx, val in enumerate(values):
                    final_values[v_idx][idx] = val[i]

        # Reconstruct the output from flattened form
        return jax.tree.unflatten(res_tree, final_values), arguments

    return _vmap


@partial(jax.jit, static_argnames=("bin_size", "axis"))
def coarsegrain(spikes, bin_size, axis=0):
    # Move the specified axis to the front
    spikes = jnp.moveaxis(spikes, axis, 0)
    n_steps = spikes.shape[0]
    n_bins = n_steps // bin_size
    truncated_spikes = spikes[: n_bins * bin_size]
    reshaped_spikes = jnp.reshape(truncated_spikes, (n_bins, bin_size, -1))
    binned_spikes = jnp.sum(reshaped_spikes, axis=1)
    # Move the axis back to its original position
    binned_spikes = jnp.moveaxis(binned_spikes, 0, axis)
    return binned_spikes


# * Some stats functions
@jax.jit
def firing_rate(spikes, dt=bp.share["dt"]):
    """
    Calculate the firing rate of a neuron given its spike train. If 'spikes' is not a boolean array, return an array of nans. Should be jax compatible.
    """
    # * If spikes is not a boolean array, return an array of nans
    if spikes.dtype != bool:
        return jnp.full(spikes.shape[1], np.nan)  # Drop the time axis

    total_spikes = jnp.sum(
        spikes, axis=0
    )  # The first axis is the batch, the second axis is the time

    total_time = spikes.shape[0] * (dt / 1000.0)
    mean_rate = total_spikes / total_time
    return mean_rate


def susceptibility(bin, dt=bp.share["dt"]):
    @jax.jit
    def _susceptibility(spikes):
        # * If spikes is not a boolean array, return an array of nans
        if spikes.dtype != bool:
            return jnp.nan

        # * Bin over time
        if bin is not None:
            bin_size = int(bin / dt)
            spikes = coarsegrain(spikes, bin_size, axis=0)  # Coarse grain over time

        active_neurons = spikes > 0

        # * Fraction of active neurons at each time step
        rho = jnp.mean(active_neurons, axis=1)  # mean across neurons

        # * Susceptibility
        chi = jnp.mean(rho**2) - jnp.mean(rho) ** 2
        return chi

    return _susceptibility


def spike_spectrum(n_segments=1):
    @jax.jit
    def _spike_spectrum(spikes):
        """
        Compute the Bartlett spectrum for each neuron (column) of `data`
        and then average across neurons.

        Args:
            data: A (T, N) array of time-series data, one column per neuron.
                E.g., discrete spike counts or continuous signals.
            n_segments: Number of segments to split the time axis into
                        for Bartlett's method. Must be treated as static
                        for JIT compilation.
            dt: Sampling interval (if you need frequency in Hz, etc.).

        Returns:
            A 1D array of length `seg_size`, representing the Bartlett-averaged
            power spectral density (PSD), further averaged across all neurons.
            Frequencies can be obtained separately with jnp.fft.fftfreq(seg_size, d=dt).
        """

        T, N = spikes.shape
        seg_size = T // n_segments  # integer division; remainder is discarded if any

        if spikes.dtype != bool:
            return jnp.full(seg_size, np.nan)

        # Reshape data into (n_segments, seg_size, N)
        # so each segment is [segment_index, time_in_segment, neuron].
        data_reshaped = spikes[: seg_size * n_segments, :].reshape(
            n_segments, seg_size, N
        )

        # Compute the FFT of each segment along the time-in-segment axis (axis=1).
        # Shape: (n_segments, seg_size, N)
        fft_segments = jnp.fft.fft(data_reshaped, axis=1)

        # Periodogram for each segment/neuron: (1/seg_size)*|FFT|^2.
        # Still shape: (n_segments, seg_size, N)
        periodograms = (1.0 / seg_size) * jnp.abs(fft_segments) ** 2

        # Average over segments -> shape (seg_size, N)
        psd_per_neuron = jnp.mean(periodograms, axis=0)

        # Now average over neurons -> shape (seg_size,)
        psd_avg = jnp.mean(psd_per_neuron, axis=1)

        return psd_avg

    return _spike_spectrum


@jax.jit
def temporal_average(V):
    # # * If V is not an array of floats, return NaN
    # if (V.dtype != jnp.float32) or (V.dtype != jnp.float64):
    #     return jnp.full(V.shape[1], np.nan)  # Drop the time axis

    mean_V = jnp.mean(V, axis=0)
    return mean_V


def grand_distribution(n_bins):
    @jax.jit
    def _grand_distribution(X):
        """
        Calculate the total distribution of a given variable X.
        """
        lower = jnp.float32(jnp.min(X))
        upper = jnp.float32(jnp.max(X))
        bin_size = (upper - lower) / n_bins
        bin_edges = jnp.linspace(lower, upper, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        indices = jnp.floor((X - lower) / bin_size).astype(int)
        indices = jnp.clip(indices, 0, n_bins - 1)
        hist = jnp.bincount(indices.flatten(), length=n_bins)
        return hist, bin_centers

    return _grand_distribution


def mua(bin, dt=bp.share["dt"]):
    """
    Calculate the multi-unit activity (MUA) of a neuron given its spike train. If 'spikes' is not a boolean array, return an array of nans. Should be jax compatible.
    """

    @jax.jit
    def _mua(spikes):
        # * If spikes is not a boolean array, return an array of nans
        if spikes.dtype != bool:
            return jnp.full(spikes.shape[0], np.nan)  # Drop the neuron axis

        # * Bin over time
        if bin is not None:
            bin_size = int(bin / dt)
            spikes = coarsegrain(spikes, bin_size, axis=0)  # Coarse grain over time

        mua = jnp.sum(spikes, axis=1)  # Sum over neurons

        return mua

    return _mua


def select(func, idxs):

    def _select(spikes):
        return func(spikes[:, idxs])

    return _select


def efficiency(bin_indices, tau):

    def _efficiency(spikes):
        nbins_x = bin_indices.shape[0]
        nbins_y = bin_indices.shape[1]
        # * Sum the spike counts for each neuron over the time interval
        tbinned = coarsegrain(spikes, int(tau / bp.share["dt"]))

        # * group the spike counts from each bin for each time bin
        xbinned = np.empty((nbins_x, nbins_y, tbinned.shape[0]), dtype=int)
        for i in range(nbins_x):
            for j in range(nbins_y):
                xbinned[i, j] = tbinned[:, bin_indices[i, j]].sum(axis=1)

        xprob = xbinned / np.sum(xbinned, axis=0, keepdims=True)
        # * Now calcualte the entropy for each time bin
        H = np.zeros(xprob.shape[-1])
        for i in range(xprob.shape[-1]):
            H[i] = -np.sum(xprob[:, :, i] * np.log(xprob[:, :, i] + 1e-10))  # ok??
        H

        # * Calculate a energy cost equal to the number of spikes emitted at each time step
        C = np.sum(tbinned, axis=1)  # Sum over neurons

        # efficiency is the ratio of the two
        n = spikes.shape[1]
        eta = n * H / C
        return eta

    return _efficiency


def radial_autocorrelation(
    positions: List[Tuple[float, float]],
    dr: float = 0.05,
) -> Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Build a callable that returns the radial (isotropic) autocorrelation
    of binary activity on a rectangular lattice.

    Parameters
    ----------
    positions : list of (x, y)
        Flat list of X × Y coordinates for each site/neuron/pixel.
    dr : float, optional
        Shell width for the radial average (same units as `positions`).

    Returns
    -------
    f(S) -> (g_r, r_bins)
        S : array, shape (T, N=X*Y)
            0/1 (or count) raster — one row per time frame.
        g_r : array, shape (len(r_bins),)
            Radially averaged autocorrelation; g_r[0] == 1.
        r_bins : array
            Left edges of the radial shells.
    """

    # ------------------------------------------------------------------ grid checks
    xs = jnp.unique(jnp.array(sorted([x for x, _ in positions])))
    ys = jnp.unique(jnp.array(sorted([y for _, y in positions])))
    X, Y = len(xs), len(ys)

    if X * Y != len(positions):
        raise ValueError("Positions do not form a full rectangular grid.")

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    if not (jnp.allclose(jnp.diff(xs), dx) and jnp.allclose(jnp.diff(ys), dy)):
        raise ValueError("Coordinates are not evenly spaced.")

    # ---------------------------------------- map flat index <-> (i, j) on the grid
    pos2idx = {
        (float(x), float(y)): (i, j) for i, x in enumerate(xs) for j, y in enumerate(ys)
    }
    ix = jnp.array([pos2idx[(float(x), float(y))][0] for x, y in positions])
    iy = jnp.array([pos2idx[(float(x), float(y))][1] for x, y in positions])

    # --------------------------------------------------------- lag‑space distance grid
    lag_x = jnp.fft.fftfreq(X) * X * dx  # Δx with wrap‑around
    lag_y = jnp.fft.fftfreq(Y) * Y * dy  # Δy with wrap‑around
    dxg, dyg = jnp.meshgrid(lag_x, lag_y, indexing="ij")
    dist = jnp.sqrt(dxg**2 + dyg**2)  # (X, Y)
    flat_dist = dist.ravel()

    max_r = flat_dist.max()
    r_bins = jnp.arange(0.0, max_r + dr, dr)

    # which radial shell each lag element belongs to
    shell_idx = jnp.digitize(flat_dist, r_bins) - 1  # length X*Y
    counts_per_shell = jnp.bincount(shell_idx, length=r_bins.size)

    # --------------------- helper: radial profile of one flattened 2‑D array --------
    def radial_profile(flat_vals):
        sums = jnp.bincount(shell_idx, weights=flat_vals, length=r_bins.size)
        return sums / jnp.maximum(counts_per_shell, 1)  # avoid 0/0

    # --------------------------------------------------------------------- main call
    def compute(S: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Parameters
        ----------
        S : array, shape (T, N=X*Y)

        Returns
        -------
        g_r, r_bins
        """
        T, N = S.shape
        if N != X * Y:
            raise ValueError(f"Expected N={X*Y}, got {N}")

        # ---------- reshape each frame to (X, Y) grid
        def to_grid(row):
            grid = jnp.zeros((X, Y))
            return grid.at[ix, iy].set(row)

        frames = vmap(to_grid)(S)  # (T, X, Y)

        # ---------- 2‑D autocorrelation per frame + valid flag
        def ac2d_with_flag(frame):
            f0 = frame - jnp.mean(frame)
            var = jnp.mean(f0**2)

            def do_autocorr(_):
                F = jnp.fft.fft2(f0)
                C = jnp.fft.ifft2(jnp.abs(F) ** 2).real / (X * Y)
                return C / C[0, 0]  # C(0)=1

            C_norm = jax.lax.cond(
                var > 0, do_autocorr, lambda _: jnp.zeros_like(frame), operand=None
            )
            return C_norm, var > 0  # (X, Y), scalar bool

        AC_all, valid_flags = vmap(ac2d_with_flag)(frames)  # (T, X, Y), (T,)

        # keep only frames whose variance > 0
        valid = valid_flags.astype(bool)
        n_valid = jnp.sum(valid)

        jax.debug.print("radial_autocorrelation: kept {}/{} frames", n_valid, T)

        if n_valid == 0:
            raise ValueError(
                "All frames were empty / variance‑zero — nothing to average."
            )

        AC_valid = AC_all[valid]  # (n_valid, X, Y)
        AC_flat = AC_valid.reshape(n_valid, -1)  # (n_valid, X*Y)

        # ---------- radial average, then mean across valid frames
        profiles = vmap(radial_profile)(AC_flat)  # (n_valid, len(r_bins))
        g_r = jnp.mean(profiles, axis=0)

        return g_r, r_bins

    return compute


def monitor(mon):
    return mon
