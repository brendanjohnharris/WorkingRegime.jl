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
from brainpy._src.dyn.utils import get_spk_type
from typing import Union, Callable, Optional, Sequence, Any
from functools import partial
from ..utils import *
from ..distances import *
from ..neurons import FNSNeuron, PoissonGroup
from ..positions import *
from ..synapses import *
from ..stats import *


class Nonspatial(bp.Network):
    """
    A spatially independent network of FNS neurons.
    """

    def __init__(
        self,
        N_e=2000,  # Number of Exc. neurons
        K_ee=60,
        K_ei=60,
        K_ie=60,
        K_ii=80,
        gamma=4,  # Ratio of num. Exc. to num. Inh. neurons
        delta=4,  # Per-neuron synaptic weight I:E ratio
        nu=10.0,  # External population firing rate
        n_ext=100,  # Number of external synapses per Exc. neuron
        J_ee=0.00105,
        J_ei=0.00145,
        tau_r_e=1.0,
        tau_r_i=2.0,
        tau_d_e=5.0,  # * Excitatory synapse decays more slowly than inhibitory
        tau_d_i=4.0,  # 3.0 for yifan, # 4.5 for shencong
        Delta_g_K=0.003,  # Adaptation strength for excitatory neurons
        method="exp_auto",
        key=jax.random.PRNGKey(np.random.randint(0, 2**32)),
        copy_conn=False,  # Whether to copy connectivity from the provided network
    ):
        super().__init__()
        self.gamma = gamma
        self.delta = delta
        self.nu = nu
        self.n_ext = n_ext
        self.J_ee = J_ee
        self.J_ei = J_ei
        self.method = method
        self.key = key

        self.N_e = N_e
        self.N_i = N_e // gamma

        self.K_ee = K_ee
        self.K_ei = K_ei
        self.K_ie = K_ie
        self.K_ii = K_ii
        self.omega_ee = self.required_omega("ee")
        self.omega_ie = self.required_omega("ie")
        self.omega_ei = self.required_omega("ei")
        self.omega_ii = self.required_omega("ii")

        self.tau_r_e = tau_r_e
        self.tau_r_i = tau_r_i
        self.tau_d_e = (
            tau_d_e  # * Excitatory synapse decays more slowly than inhibitory
        )
        self.tau_d_i = tau_d_i  # 3.0 for yifan, # 4.5 for shencong

        self.J_ie = (J_ee * K_ee) * self.delta / K_ie
        self.J_ii = (J_ei * K_ei) * self.delta / K_ii
        self.Delta_g_K = Delta_g_K

        self.key, subkey = jax.random.split(self.key)
        exc_positions = ClusteredPositions((-1.5, 0), 1, key=subkey)

        self.key, subkey = jax.random.split(self.key)
        inh_positions = ClusteredPositions((1.5, 0), 1, key=subkey)

        # neurons
        self.key, subkey = jax.random.split(self.key)
        self.E = FNSNeuron(
            size=N_e,
            C=0.25,
            g_L=0.0167,
            V_L=-70.0,
            V_th=-50.0,
            V_rt=-70.0,
            tau_ref=4.0,
            V_K=-85.0,
            tau_K=60.0,
            Delta_g_K=Delta_g_K,  # 0.003 for guazhang, 0.002 for shencong
            V_initializer=bp.init.Uniform(-55.0, -50.0, subkey),
            method=method,
            embedding=exc_positions,
        )

        # Create a population of inhibitory neurons
        self.key, subkey = jax.random.split(self.key)
        self.I = FNSNeuron(
            size=self.N_i,
            C=0.25,
            g_L=0.025,
            V_L=-70.0,
            V_th=-50.0,
            V_rt=-70.0,
            tau_ref=4.0,
            V_K=-85.0,
            tau_K=60.0,
            Delta_g_K=0.0,  # No adaptation for inhibitory neurons
            V_initializer=bp.init.Uniform(-55.0, -50.0, subkey),
            method=method,
            embedding=inh_positions,
        )

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
            if isinstance(copy_conn, FNS):
                copy_conn = copy_conn.get_connectivity()

            conn_ee = copy_connectivity(copy_conn["E2E"])
            conn_ei = copy_connectivity(copy_conn["E2I"])
            conn_ie = copy_connectivity(copy_conn["I2E"])
            conn_ii = copy_connectivity(copy_conn["I2I"])
            conn_exte = copy_connectivity(copy_conn["ext2E"])
            conn_exti = copy_connectivity(copy_conn["ext2I"])

        else:

            self.key, *subkeys = jax.random.split(self.key, 7)
            conn_ee = FixedProb(prob=self.omega_ee, seed=subkeys[0])
            conn_ei = FixedProb(prob=self.omega_ei, seed=subkeys[1])
            conn_ie = FixedProb(prob=self.omega_ie, seed=subkeys[2])
            conn_ii = FixedProb(prob=self.omega_ii, seed=subkeys[3])

            conn_exte = FixedProb(prob=p_ext, seed=subkeys[4])
            conn_exti = FixedProb(prob=p_ext, seed=subkeys[5])

        # Synapses
        V_rev_e = 0.0
        V_rev_i = -80.0  # ? Makes the inhibitory synapses inhibitory

        e_delay = 1.5
        i_delay = 2.0

        self.key, subkey = jax.random.split(self.key)
        self.E2E = Synapse(
            pre=self.E,
            post=self.E,
            delay=e_delay,
            conn=conn_ee,
            tau_d=tau_d_e,
            tau_r=bp.init.Normal(tau_r_e, 0.05 * tau_r_e, subkey),
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

    def reinit_weights(self, delta, J_e):
        if delta is not None:
            self.delta = delta
        if J_e is not None:
            self.J_ee = J_e[0]
            self.J_ei = J_e[1]

        self.J_ie = self.J_ee * self.delta
        self.J_ii = self.J_ei * self.delta

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

        # # ! Need to fix
        # K_ee = indegrees_static(self.E2E.proj.comm.indices, self.N_e)
        # K_ei = indegrees_static(self.E2I.proj.comm.indices, self.N_i)
        # K_ie = indegrees_static(self.I2E.proj.comm.indices, self.N_e)
        # K_ii = indegrees_static(self.I2I.proj.comm.indices, self.N_i)
        # K_ext_e = indegrees_static(self.ext2E.proj.comm.indices, self.N_e)
        # K_ext_i = indegrees_static(self.ext2I.proj.comm.indices, self.N_i)

        # w_ee = draw_lognormal(self.J_ee, self.J_ee / 4, jnp.sum(K_ee))
        # w_ei = draw_lognormal(self.J_ei, self.J_ei / 4, jnp.sum(K_ei))
        # w_ie = draw_lognormal(self.J_ie, self.J_ie / 4, jnp.sum(K_ie))
        # w_ii = draw_lognormal(self.J_ii, self.J_ii / 4, jnp.sum(K_ii))
        # w_ext_e = draw_lognormal(self.J_ee, self.J_ee / 4, jnp.sum(K_ext_e))
        # w_ext_i = draw_lognormal(self.J_ei, self.J_ei / 4, jnp.sum(K_ext_i))

        # w_ee = sorted_block_assignment(w_ee, K_ee)
        # w_ei = sorted_block_assignment(w_ei, K_ei)
        # w_ie = sorted_block_assignment(w_ie, K_ie)
        # w_ii = sorted_block_assignment(w_ii, K_ii)
        # w_ext_e = sorted_block_assignment(w_ext_e, K_ee)
        # w_ext_i = sorted_block_assignment(w_ext_i, K_ei)

        # self.E2E.proj.comm.weight = w_ee
        # self.E2I.proj.comm.weight = w_ei
        # self.I2E.proj.comm.weight = w_ie
        # self.I2I.proj.comm.weight = w_ii
        # self.ext2E.proj.comm.weight = w_ext_e
        # self.ext2I.proj.comm.weight = w_ext_i

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
        }

    def get_input_params(self):
        keys = [
            "N_e",
            "omega_ee",
            "omega_ei",
            "omega_ie",
            "omega_ii",
            "gamma",
            "delta",
            "nu",
            "n_ext",
            "J_e",
            "method",
            "key",
        ]
        _params = {key: value for key, value in self.__dict__.items() if key in keys}
        if len(_params) != len(keys):
            missings = set(keys) - set(_params.keys())
            raise ValueError("Missing parameters: {}".format(missings))
        return _params

    def expected_indegree(self, pop="ee"):
        """
        Calculate the expected in-degree for a given population.
        """
        if pop == "ee":
            # Subtract 1 if self-connections are excluded
            return self.N_e * self.omega_ee
        elif pop == "ei":
            return self.N_e * self.omega_ei
        elif pop == "ie":
            return self.N_i * self.omega_ie
        elif pop == "ii":
            return self.N_i * self.omega_ii
        else:
            raise ValueError(f"Unknown population connection type: {pop}")

    def required_omega(self, pop):
        """
        Calculate the expected in-degree for a given population.
        """
        if pop == "ee":
            return self.K_ee / self.N_e
        elif pop == "ei":
            return self.K_ei / self.N_e
        elif pop == "ie":
            return self.K_ie / self.N_i
        elif pop == "ii":
            return self.K_ii / self.N_i
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

        w_E2E = np.bincount(mats[0][1], weights=weights[0], minlength=self.N_e)
        w_E2I = np.bincount(mats[1][1], weights=weights[1], minlength=self.N_i)
        w_I2E = np.bincount(mats[2][1], weights=weights[2], minlength=self.N_e)
        w_I2I = np.bincount(mats[3][1], weights=weights[3], minlength=self.N_i)

        IE_e = np.mean(w_I2E) / np.mean(w_E2E)
        IE_i = np.mean(w_I2I) / np.mean(w_E2I)
        return IE_e, IE_i

    def expected_zeta(self):
        """
        Calculate the effective IE ratio for inhibitory and excitatory populations.
        The ratio represents total inhibitory strength divided by
        total excitatory strength (for the average neuron) per neuron type.
        See also `calculate_zeta`.
        """
        # For excitatory neurons: (N_i/gamma * omega_ie * J_i) / (N_e * omega_ee * J_e)
        # Since J_i = J_e * delta and N_i = N_e / gamma:
        IE_e = (self.omega_ie * self.delta) / (self.gamma * self.omega_ee)

        # For inhibitory neurons: (N_i/gamma * omega_ii * J_i) / (N_e * omega_ei * J_e)
        # Similar simplification:
        IE_i = (self.omega_ii * self.delta) / (self.gamma * self.omega_ei)

        return IE_e, IE_i

    def expected_sum_of_weights(self, pop="ee"):
        if pop == "ee":
            j = self.J_ee
        elif pop == "ei":
            j = self.J_ei
        elif pop == "ie":
            j = self.J_ee * self.delta
        elif pop == "ii":
            j = self.J_ei * self.delta
        else:
            raise ValueError(f"Unknown population connection type: {pop}")
        return j * self.expected_indegree(pop)

    # def nu_next_thresh(self):  # !!! WROOOOONG
    #     """
    #     Calculate the minimum threshold for the product nu*n_ext to achieve spontaneous
    #     firing of external and inhibitory populations
    #     """
    #     thr_e = self.E.g_L * (self.E.V_th - self.E.V_L) / self.J_e
    #     thr_i = self.I.g_L * (self.I.V_th - self.I.V_L) / self.J_e
    #     return thr_e, thr_i

    def linear_subthreshold_equilibrium(self):
        Ve = self.n_ext * self.nu * self.J_e / self.E.g_L
        Vi = self.n_ext * self.nu * self.J_e / self.I.g_L

        return Ve, Vi

    def subthreshold_equilibrium_potential(self, alpha_e=0.4202, alpha_i=0.4507):
        """
        Calculate the mean voltage for a given population due solely to background input.
        Corrected by filter_factor which you shoudl estimate from the subthreshold transfer function.
        """
        Ve, Vi = self.linear_subthreshold_equilibrium()
        Ve = self.E.V_L + Ve * alpha_e
        Vi = self.I.V_L + Vi * alpha_i

        return Ve, Vi

    def estimate_filter_factor(
        self, nu_n_ext=range(50, 450, 10), num_parallel=32
    ):  # E.g. pop = self.I
        # Estimate the filter factor for the steady-state membrane potential by simulating
        # many instances of a single neuron
        def run(nu_n_ext):
            nu = nu_n_ext / self.n_ext
            disconnect = DisconnectedFNS(
                N_e=self.N_e,
                gamma=self.gamma,
                delta=self.delta,
                nu=nu,
                n_ext=self.n_ext,
                J_e=self.J_e,
                method=self.method,
                key=jax.random.PRNGKey(np.random.randint(0, 2**32)),
            )
            duration = 1000.0
            monitors = ["E.V", "I.V"]
            # Run the simulation
            runner = bp.DSRunner(
                disconnect, monitors=monitors, numpy_mon_after_run=False
            )
            runner.run(duration=duration)
            Ve = runner.mon["E.V"].view()
            Vi = runner.mon["I.V"].view()
            out = {
                "E": {"mean": bp.math.mean(Ve), "std": bp.math.std(Ve)},
                "I": {"mean": bp.math.mean(Vi), "std": bp.math.std(Vi)},
            }
            return out

        res = bp.running.jax_vectorize_map(
            run, [nu_n_ext], num_parallel=num_parallel, clear_buffer=True
        )
        return res

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
        # If gpu available, use vmap
        if jax.lib.xla_bridge.get_backend().platform == "gpu":
            run = copy_run(
                self, duration=duration, monitors=monitors, concrete_out=False, key=key
            )
            print("Vectorizing on GPU")
            res = bp.running.jax_vectorize_map(
                run, [deltas], num_parallel=num_parallel, clear_buffer=True
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
            "N_e",
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
