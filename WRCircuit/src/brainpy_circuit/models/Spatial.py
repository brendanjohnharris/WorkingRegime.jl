import warnings
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
from scipy.integrate import dblquad

from brainpy.dyn.neurons import GradNeuDyn
from brainpy.dyn import NeuDyn
from brainpy.initialize import (
    ZeroInit,
    OneInit,
    Uniform,
    variable_,
    noise as init_noise,
)
from brainpy import share
from brainpy.types import Shape, ArrayType
from brainpy.check import is_initializer
from brainpy import odeint, sdeint, JointEq
from brainpy._src.connect.base import get_idx_type
from typing import Union, Callable, Optional, Sequence, Any
from functools import partial
from ..utils import *
from ..distances import *
from ..neurons import FNSNeuron, PoissonGroup
from ..positions import *
from ..synapses import *


class Spatial(bp.Network):
    """
    A spatially embedded network of FNS neurons.
    """

    def _validate_connectivity_parameters(self, ne, ni):
        """
        Validate that connectivity parameters K_* are compatible with network size.

        Args:
            ne: Number of grid points in each dimension (total E neurons = ne^2)
            ni: Total number of inhibitory neurons

        Raises:
            ValueError: If any K parameter is too large for the network size
        """
        ne_total = ne * ne  # Total excitatory neurons

        # For each connection type, check that requested connections don't exceed maximum possible
        errors = []

        # E2E: K_ee * ne connections among ne^2 neurons, max = ne^4
        if self.K_ee * ne > ne_total * ne_total:
            errors.append(
                f"K_ee={self.K_ee} is too large for network with ne={ne} ({ne_total} E neurons). "
                f"Maximum K_ee = {ne_total * ne_total // ne} for this network size."
            )

        # E2I: K_ei * ni connections from ne^2 E neurons to ni I neurons, max = ne^2 * ni
        if self.K_ei * ni > ne_total * ni:
            errors.append(
                f"K_ei={self.K_ei} is too large for network with ne={ne}, ni={ni}. "
                f"Maximum K_ei = {ne_total} for this network size."
            )

        # I2E: K_ie * ne connections from ni I neurons to ne^2 E neurons, max = ni * ne^2
        if self.K_ie * ne > ni * ne_total:
            errors.append(
                f"K_ie={self.K_ie} is too large for network with ne={ne}, ni={ni}. "
                f"Maximum K_ie = {ni * ne_total // ne} for this network size."
            )

        # I2I: K_ii * ni connections among ni neurons, max = ni^2
        if self.K_ii * ni > ni * ni:
            errors.append(
                f"K_ii={self.K_ii} is too large for network with ni={ni} I neurons. "
                f"Maximum K_ii = {ni} for this network size."
            )

        if errors:
            error_msg = (
                "Connectivity parameters incompatible with network size:\n  "
                + "\n  ".join(errors)
                + f"\n\nSuggestion: Increase rho (currently {self.rho}) or dx (currently {self.dx}), "
                f"or reduce the K_* parameters to match your desired network size."
            )
            raise ValueError(error_msg)

    def __init__(
        self,
        rho=20000,  # Density of Exc. neurons (neurons per mm^2)
        dx=0.5,  # Width of the spatial domain (mm)
        sigma_ee=0.06,  # Width of the distance-dependent connectivity kernel (mm)
        sigma_ei=0.07,
        sigma_ie=0.14,
        sigma_ii=0.14,
        gamma=4,  # Ratio of num. Exc. to num. Inh. neurons
        K_ee=260,
        K_ei=340,
        K_ie=225,
        K_ii=290,
        delta=4.0,
        nu=10.0,  # External population firing rate
        n_ext=100,  # Number of external synapses per Exc. neuron
        J_ee=0.00105,  # Mean weight of excitatory synapses to excitatory neurons
        J_ei=0.00145,
        tau_r_e=1.0,
        tau_r_i=2.0,
        tau_d_e=5.0,  # * Excitatory synapse decays more slowly than inhibitory
        tau_d_i=4.5,  # 3.0 for yifan, # 4.5 for shencong
        V_rev_e=0.0,  # Reversal potential for excitatory synapses
        V_rev_i=-80.0,  # Makes the inhibitory synapses inhibitory
        e_delay=1.5,  # Synaptic delay. Shencong uses uniform dist. between 0.5 and 2.5
        i_delay=1.5,
        Delta_g_K=0.002,  # Adaptation strength for excitatory neurons
        tau_K=40.0,
        kernel=ExponentialKernel,
        method="exp_auto",
        key=None,
        copy_conn=False,  # Whether to copy connectivity from the provided WRCircuit
    ):
        super().__init__()

        self.rho = rho
        self.dx = dx
        self.sigma_ee = sigma_ee
        self.sigma_ei = sigma_ei
        self.sigma_ie = sigma_ie
        self.sigma_ii = sigma_ii
        self.gamma = gamma
        self.nu = nu
        self.n_ext = n_ext

        self.delta = delta
        self.J_ee = J_ee
        self.J_ei = J_ei

        self.method = method
        self.kernel = kernel

        if key is None:
            # Add a warning here about non-reproducibility
            warnings.warn(
                "No random seed provided. Results may not be reproducible.", UserWarning
            )
            self.key = jax.random.PRNGKey(np.random.randint(0, 2**32))
        elif isinstance(key, int):
            self.key = jax.random.PRNGKey(key)
        else:
            self.key = key

        self.K_ee = K_ee
        self.K_ei = K_ei
        self.K_ie = K_ie
        self.K_ii = K_ii

        self.omega_ee = self.required_omega("ee")
        self.omega_ei = self.required_omega("ei")
        self.omega_ie = self.required_omega("ie")
        self.omega_ii = self.required_omega("ii")
        self.p_ee = kernel.mass2pmax(self.omega_ee, self.sigma_ee)
        self.p_ei = kernel.mass2pmax(self.omega_ei, self.sigma_ei)
        self.p_ie = kernel.mass2pmax(self.omega_ie, self.sigma_ie)
        self.p_ii = kernel.mass2pmax(self.omega_ii, self.sigma_ii)

        self.tau_r_e = tau_r_e
        self.tau_r_i = tau_r_i
        self.tau_d_e = tau_d_e
        self.tau_d_i = tau_d_i
        self.V_rev_e = V_rev_e
        self.V_rev_i = V_rev_i
        self.e_delay = e_delay
        self.i_delay = i_delay
        self.Delta_g_K = Delta_g_K

        # geometry
        A = dx**2
        ne = round(np.sqrt(rho * A))  # Number of grid points in each dimension
        ni = round((ne**2) / gamma)

        # Validate that K values are compatible with network size
        # Maximum connections occur when every neuron connects to every other neuron
        self._validate_connectivity_parameters(ne, ni)

        # dx bounds the grid, assuming left bottom corner is at (0, 0)
        exc_positions = GridPositions((dx, dx))

        self.key, subkey = jax.random.split(self.key)
        inh_positions = RandomPositions((dx, dx), subkey)

        # neurons
        self.key, subkey = jax.random.split(self.key)
        self.E = FNSNeuron(
            size=[ne, ne],
            C=0.25,
            g_L=0.0167,
            V_L=-70.0,
            V_th=-50.0,
            V_rt=-70.0,
            tau_ref=4.0,
            V_K=-85.0,
            tau_K=tau_K,
            Delta_g_K=Delta_g_K,
            V_initializer=bp.init.Uniform(-70.0, -50.0, subkey),
            method=method,
            embedding=exc_positions,
        )

        # Shencong uses independent parameter delta_e and delta_i, decoupling J_ie and J_ii.
        # We can account for this by tuning Ks later
        self.J_ie = (J_ee * K_ee) * self.delta / K_ie
        self.J_ii = (J_ei * K_ei) * self.delta / K_ii

        # Create a population of inhibitory neurons
        self.key, subkey = jax.random.split(self.key)
        self.I = FNSNeuron(
            size=ni,
            C=0.25,
            g_L=0.025,
            V_L=-70.0,
            V_th=-50.0,
            V_rt=-70.0,
            tau_ref=4.0,
            V_K=-85.0,
            tau_K=tau_K,
            Delta_g_K=0.0,  # No adaptation for inhibitory neurons
            V_initializer=bp.init.Uniform(-70.0, -50.0, subkey),
            method=method,
            embedding=inh_positions,
        )

        # Connectivity topology

        # External population
        p_ext = np.sqrt(self.n_ext / self.E.num)  # !!! Check !!!
        N_ext = int(np.round(self.E.num * p_ext))

        def copy_connectivity(proj):
            # if isinstance(proj, bp.Projection):
            #     indices = copy.deepcopy(proj.comm.indices)
            #     inptr = copy.deepcopy(proj.comm.indptr)
            # else:
            indices = copy.deepcopy(proj["indices"])
            inptr = copy.deepcopy(proj["indptr"])

            return CSRConn(indices, inptr)

        if copy_conn:
            if isinstance(copy_conn, Spatial):
                copy_conn = copy_conn.get_connectivity()

            conn_ee = copy_connectivity(copy_conn["E2E"])
            conn_ei = copy_connectivity(copy_conn["E2I"])
            conn_ie = copy_connectivity(copy_conn["I2E"])
            conn_ii = copy_connectivity(copy_conn["I2I"])
            conn_exte = copy_connectivity(copy_conn["ext2E"])
            conn_exti = copy_connectivity(copy_conn["ext2I"])
            self.N_e = copy_conn["N_e"]
            self.N_i = copy_conn["N_i"]

        else:
            self.N_e = self.E.size
            self.N_i = self.I.size
            ne = self.E.size[0] * self.E.size[1]
            ni = self.I.size[0]  # 1 element
            self.key, subkey = jax.random.split(self.key)
            conn_ee = DistanceDependent(
                num_connections=self.K_ee * ne,  # Freezes total connections
                kernel=kernel(sigma=self.sigma_ee, p_max=self.p_ee),
                domain=self.E.embedding.domain,
                positions_pre=self.E.positions,
                positions_post=self.E.positions,
                seed=subkey,
            )
            self.key, subkey = jax.random.split(self.key)
            conn_ei = DistanceDependent(
                num_connections=self.K_ei * ni,
                kernel=kernel(sigma=self.sigma_ei, p_max=self.p_ei),
                domain=self.E.embedding.domain,
                positions_pre=self.E.positions,
                positions_post=self.I.positions,
                seed=subkey,
            )
            self.key, subkey = jax.random.split(self.key)
            conn_ie = DistanceDependent(
                num_connections=self.K_ie * ne,
                kernel=kernel(sigma=self.sigma_ie, p_max=self.p_ie),
                domain=self.I.embedding.domain,
                positions_pre=self.I.positions,
                positions_post=self.E.positions,
                seed=subkey,
            )
            self.key, subkey = jax.random.split(self.key)
            conn_ii = DistanceDependent(
                num_connections=self.K_ii * ni,
                kernel=kernel(sigma=self.sigma_ii, p_max=self.p_ii),
                domain=self.I.embedding.domain,
                positions_pre=self.I.positions,
                positions_post=self.I.positions,
                seed=subkey,
            )

            self.key, subkey = jax.random.split(self.key)
            conn_exte = FixedProb(prob=p_ext, seed=subkey)
            self.key, subkey = jax.random.split(self.key)
            conn_exti = FixedProb(prob=p_ext, seed=subkey)

        # Synapses
        self.key, subkey = jax.random.split(self.key)
        self.E2E = Synapse(
            pre=self.E,
            post=self.E,
            delay=e_delay,
            conn=conn_ee,
            tau_d=tau_d_e,
            tau_r=bp.init.Normal(
                tau_r_e, 0.05 * tau_r_e, subkey
            ),  # To mimic random conduction delays
            g_max=0.0,  # This gets updated later when we call reinit_weights
            V_rev=V_rev_e,
        )

        self.key, subkey = jax.random.split(self.key)
        self.E2I = Synapse(
            pre=self.E,
            post=self.I,
            delay=e_delay,
            conn=conn_ei,
            tau_d=tau_d_e,
            tau_r=bp.init.Normal(tau_r_e, 0.05 * tau_r_e, subkey),
            g_max=0.0,  # This gets updated later when we call reinit_weights
            V_rev=V_rev_e,
        )

        self.key, subkey = jax.random.split(self.key)
        self.I2E = Synapse(
            pre=self.I,
            post=self.E,
            delay=i_delay,
            conn=conn_ie,
            tau_d=tau_d_i,
            tau_r=bp.init.Normal(tau_r_i, 0.05 * tau_r_i, subkey),
            g_max=0.0,  # This gets updated later when we call reinit_weights
            V_rev=V_rev_i,
        )

        self.key, subkey = jax.random.split(self.key)
        self.I2I = Synapse(
            pre=self.I,
            post=self.I,
            delay=i_delay,
            conn=conn_ii,
            tau_d=tau_d_i,
            tau_r=bp.init.Normal(tau_r_i, 0.05 * tau_r_i, subkey),
            g_max=0.0,  # This gets updated later when we call reinit_weights
            V_rev=V_rev_i,
        )

        # External population
        self.key, subkey = jax.random.split(self.key)
        self.ext = PoissonGroup(
            size=N_ext,
            freqs=self.nu,
            keep_size=False,
            sharding=None,
            spk_type=None,
            name=None,
            mode=None,
            seed=subkey,
        )
        self.key, subkey = jax.random.split(self.key)
        self.ext2E = Synapse(
            pre=self.ext,
            post=self.E,
            conn=conn_exte,
            delay=e_delay,
            tau_d=tau_d_e,
            g_max=self.J_ee,
            tau_r=bp.init.Normal(tau_r_e, 0.05 * tau_r_e, subkey),
            V_rev=V_rev_e,
        )
        self.key, subkey = jax.random.split(self.key)
        self.ext2I = Synapse(
            pre=self.ext,
            post=self.I,
            conn=conn_exti,
            delay=e_delay,
            tau_d=tau_d_e,
            g_max=self.J_ei,
            tau_r=bp.init.Normal(tau_r_e, 0.05 * tau_r_e, subkey),
            V_rev=V_rev_e,
        )

        # define input variables given to E/I populations
        self.Ein = bp.dyn.InputVar(self.E.varshape)
        self.Iin = bp.dyn.InputVar(self.I.varshape)
        self.E.add_inp_fun("", self.Ein)
        self.I.add_inp_fun("", self.Iin)

        # * Posthoc weight updates to maintain mean_weight = 1/sqrt(in-degree) per neuron
        self.reinit_weights(self.delta, (self.J_ee, self.J_ei))

    def reinit_weights(self, delta=None, J_e=None):
        if delta is not None:
            self.delta = delta
        if J_e is not None:
            self.J_ee = J_e[0]
            self.J_ei = J_e[1]

        self.J_ie = self.J_ee * self.delta
        self.J_ii = self.J_ei * self.delta
        self.key, subkey = jax.random.split(self.key)
        self.E2E.proj.comm.weight = correlate_weights(
            self.E2E.proj,
            self.J_ee,
            self.N_e,
            subkey,  # ? Need to pass N_ee to keep this function jittable.
        )
        self.key, subkey = jax.random.split(self.key)
        self.E2I.proj.comm.weight = correlate_weights(
            self.E2I.proj, self.J_ei, self.N_i, subkey
        )
        self.key, subkey = jax.random.split(self.key)
        self.I2E.proj.comm.weight = correlate_weights(
            self.I2E.proj, self.J_ie, self.N_e, subkey
        )
        self.key, subkey = jax.random.split(self.key)
        self.I2I.proj.comm.weight = correlate_weights(
            self.I2I.proj, self.J_ii, self.N_i, subkey
        )
        self.key, subkey = jax.random.split(self.key)
        self.ext2E.proj.comm.weight = correlate_weights(
            self.ext2E.proj, self.J_ee, self.N_e, subkey
        )
        self.key, subkey = jax.random.split(self.key)
        self.ext2I.proj.comm.weight = correlate_weights(
            self.ext2I.proj, self.J_ei, self.N_i, subkey
        )
        self.reset_state()  ## or bp.reset_state(self)??

    def reinit_nu(self, nu):
        self.nu = nu
        self.ext.freqs = nu
        self.reset_state()

    def get_connectivity(self):
        # Extract just the connectivity indices and inptrs
        return {
            "E2E": {
                "indices": self.E2E.proj.comm.indices,
                "indptr": self.E2E.proj.comm.indptr,
            },
            "E2I": {
                "indices": self.E2I.proj.comm.indices,
                "indptr": self.E2I.proj.comm.indptr,
            },
            "I2E": {
                "indices": self.I2E.proj.comm.indices,
                "indptr": self.I2E.proj.comm.indptr,
            },
            "I2I": {
                "indices": self.I2I.proj.comm.indices,
                "indptr": self.I2I.proj.comm.indptr,
            },
            "ext2E": {
                "indices": self.ext2E.proj.comm.indices,
                "indptr": self.ext2E.proj.comm.indptr,
            },
            "ext2I": {
                "indices": self.ext2I.proj.comm.indices,
                "indptr": self.ext2I.proj.comm.indptr,
            },
            "N_e": self.N_e,
            "N_i": self.N_i,
        }

    def get_input_params(self):
        keys = [
            "rho",
            "dx",
            "sigma_ee",
            "sigma_ei",
            "sigma_ie",
            "sigma_ii",
            "gamma",
            "delta",
            "nu",
            "n_ext",
            "J_e",
            "kernel",
            "method",
            "key",
        ]
        _params = {key: value for key, value in self.__dict__.items() if key in keys}
        if len(_params) != len(keys):
            missings = set(keys) - set(_params.keys())
            raise ValueError("Missing parameters: {}".format(missings))
        return _params

    def expected_indegree(self, pop="ee", approx=True):
        """
        Compute the mean indegree.
        Should converge to `2 * np.pi * sigma**2 * p_max * rho` in the limit of large
        dx/small sigma.
        """
        dx = self.dx
        if pop[0] == "e":
            rho = self.rho
        else:
            rho = self.rho / self.gamma
        if pop == "ee":
            p_max = self.p_ee
            sigma = self.sigma_ee
        elif pop == "ei":
            p_max = self.p_ei
            sigma = self.sigma_ei
        elif pop == "ie":
            p_max = self.p_ie
            sigma = self.sigma_ie
        elif pop == "ii":
            p_max = self.p_ii
            sigma = self.sigma_ii

        if approx:
            result = self.kernel.pmax2mass(p_max, sigma)  # 2 * np.pi * sigma**2 * p_max
        else:

            def integrand(y, x, dx, kernel_func, p_max):
                # y is integrated first, x second (dblquad's calling convention).
                rx = min(x, dx - x)
                ry = min(y, dx - y)
                r = np.sqrt(rx * rx + ry * ry)
                return p_max * self.kernel(r)

            result, error_est = dblquad(
                integrand,
                0,
                dx,  # outer integral range for x
                lambda x: 0,  # lower limit for y
                lambda x: dx,  # upper limit for y
                args=(dx, sigma, p_max),
            )

        return rho * result

    def required_omega(self, pop):
        """
        Calculate the expected omega for a given population.
        """
        if pop == "ee":
            return self.K_ee / self.rho
        elif pop == "ei":
            return self.K_ei / self.rho
        elif pop == "ie":
            return self.K_ie / (self.rho / self.gamma)
        elif pop == "ii":
            return self.K_ii / (self.rho / self.gamma)
        else:
            raise ValueError(f"Unknown population connection type: {pop}")

    def calculate_zeta(self):
        """
        Calculate the effective IE ratio for inhibitory and excitatory populations.
        The ratio represents total inhibitory strength divided by
        total excitatory strength (for the average neuron) per neuron type.
        """

        projs = [self.E2E, self.E2I, self.I2E, self.I2I]
        weights = [p.proj.comm.weight for p in projs]
        indices = [p.proj.comm.indices for p in projs]
        indptrs = [p.proj.comm.indptr for p in projs]
        mats = [bp.connect.csr2coo((indices[i], indptrs[i])) for i in range(4)]

        assert len(mats[2][0]) == len(weights[2])

        Ne = np.prod(self.E.size)
        Ni = np.prod(self.I.size)

        w_E2E = np.bincount(mats[0][1], weights=weights[0], minlength=Ne)
        w_E2I = np.bincount(mats[1][1], weights=weights[1], minlength=Ni)
        w_I2E = np.bincount(mats[2][1], weights=weights[2], minlength=Ne)
        w_I2I = np.bincount(mats[3][1], weights=weights[3], minlength=Ni)

        IE_e = np.mean(w_I2E) / np.mean(w_E2E)
        IE_i = np.mean(w_I2I) / np.mean(w_E2I)
        return IE_e, IE_i

    def expected_zeta(self):
        """
        Calculate the effective IE ratio for inhibitory and excitatory populations.
        The ratio represents total inhibitory strength divided by
        total excitatory strength (for the average neuron) per neuron type.
        """
        # * An approximation to the distance kernel that is not valid with periodic
        # * boundaries when the kernel is too wide
        zeta_e = (
            self.delta
            * (self.sigma_ie**2 * self.p_ie)
            / (self.gamma * self.sigma_ee**2 * self.p_ee)
        )

        zeta_i = (
            self.delta
            * (self.sigma_ii**2 * self.p_ii)
            / (self.gamma * self.sigma_ei**2 * self.p_ei)
        )

        return zeta_e, zeta_i

    def copy(self):
        _params = self.get_input_params()
        new_model = self.__class__(copy_conn=self, **_params)
        bp.reset_state(new_model)
        return new_model

    def update_copy(self, **params):
        _params = self.get_input_params()
        params = {**_params, **params}
        new_model = self.__class__(copy_conn=self, **params)
        new_model.reinit_weights(params["delta"], params["J_e"])
        new_model.reinit_nu(params["nu"])
        bp.reset_state(new_model)
        return new_model

    def sweep_deltas(
        self, deltas, duration=1000.0, monitors=["E.spike"], num_parallel=20
    ):

        def copy_run(
            self, duration=1000.0, monitors=["E.spike"], key=42, concrete_out=False
        ):
            conn = self.get_connectivity()
            conn = pytree_to_numpy(conn)  # Freeze connectivity
            params = self.get_input_params()
            params["key"] = key

            def run(delta):
                new_model = self.__class__(copy_conn=conn, **params)
                new_model.reinit_weights(delta)
                bp.reset_state(new_model)
                runner = bp.DSRunner(
                    new_model, monitors=monitors, numpy_mon_after_run=concrete_out
                )
                runner.run(duration=duration)
                return [runner.mon[m] for m in monitors]

            return run

        key = np.array(self.key)
        print(key)
        # If gpu available, use vmap
        if jax.lib.xla_bridge.get_backend().platform == "gpu":
            run = copy_run(
                self, duration=duration, monitors=monitors, concrete_out=False, key=key
            )
            print("Vectorizing on GPU")
            res = bp.running.jax_vectorize_map(
                run, [deltas], num_parallel=num_parallel, clear_buffer=False
            )
        else:
            run = copy_run(
                self, duration=duration, monitors=monitors, concrete_out=False, key=key
            )
            print("Parallelizing on CPU")
            # Make sure to set bp.math.set_host_device_count(os.cpu_count()) in the calling script
            res = bp.running.jax_parallelize_map(
                run, [deltas], num_parallel=num_parallel, clear_buffer=False
            )
        return res

    def to_dict(
        self,
        keys=[
            "rho",
            "dx",
            "sigma_ee",
            "sigma_ei",
            "sigma_ie",
            "sigma_ii",
            "gamma",
            "K_ee",
            "K_ei",
            "K_ie",
            "K_ii",
            "delta",
            "nu",
            "n_ext",
            "J_ee",
            "J_ei",
            "tau_r_e",
            "tau_r_i",
            "tau_d_e",
            "tau_d_i",
            "V_rev_e",
            "V_rev_i",
            "e_delay",
            "i_delay",
            "Delta_g_K",
            "kernel",
            "method",
            "key",
            "copy_conn",
        ],
    ):
        out = {
            "parameters": {
                key: maybe_initializer(value)
                for key, value in self.__dict__.items()
                if key in keys
            }
        }
        out["populations"] = {
            "E": self.E.to_dict(),
            "I": self.I.to_dict(),
        }
        out["synapses"] = {
            "E2E": self.E2E.to_dict(),
            "E2I": self.E2I.to_dict(),
            "I2E": self.I2E.to_dict(),
            "I2I": self.I2I.to_dict(),
        }
        return {self.__class__.__name__: out}
