import brainpy as bp
import brainpy.math as bm
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.synapses import TwoEndConn, SynOut, SynSTP
from brainpy.initialize import Initializer
from brainpy._src.initialize.base import _InterLayerInitializer
from brainpy.synouts import CUBA
from typing import Union, Dict, Callable, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import networkx as nx
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import jit, vmap
import uuid

import brainpy as bp
import brainpy.math as bm

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
from brainpy.initialize import parameter
from typing import Union, Callable, Optional, Sequence, Any
from functools import partial


def maybe_initializer(x, exclude=["rng"]):
    if isinstance(x, bp.init.Initializer):
        name = x.__class__.__name__
        d = {keys: value for keys, value in x.__dict__.items() if keys not in exclude}
        if not d:
            return name
        else:
            return {name: d}
    else:
        return x


class ScaledInitializer(_InterLayerInitializer):
    """A wrapper that scales the output of any initializer by a constant factor.

    Parameters
    ----------
    initializer : callable
        The base initializer to wrap.
    scale_factor : float
        The constant factor by which to scale the initializer's output.
    """

    def __init__(self, initializer, scale_factor):
        super(ScaledInitializer, self).__init__()
        self.initializer = initializer
        self.scale_factor = scale_factor

    def __call__(self, shape, dtype=None):
        # Call the wrapped initializer to get the initial weights.
        if callable(self.initializer):
            weights = self.initializer(shape, dtype=dtype)
        elif isinstance(self.initializer, (int, float)):
            # Create an array filled with the constant initializer value.
            weights = jnp.full(shape, self.initializer, dtype=dtype or jnp.float32)
        else:
            raise ValueError("Initializer must be callable or a constant float value.")
        # Scale the weights by the constant factor.
        return self.scale_factor * weights

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(initializer={self.initializer}, "
            f"scale_factor={self.scale_factor})"
        )


class Synapse(bp.Projection):
    def __init__(
        self,
        pre: NeuDyn,
        post: NeuDyn,
        conn: Union[TwoEndConnector],
        delay: Union[float, ArrayType, Initializer, Callable] = 0.0,
        g_max: Union[float, ArrayType, Initializer, Callable] = 1.0,
        tau_d: Union[float, ArrayType, Initializer, Callable] = 5.0,
        tau_r: Union[float, ArrayType, Initializer, Callable] = 1.0,
        V_rev: Union[float, ArrayType, Initializer, Callable] = 0.0,
        alpha: Union[float, ArrayType, Initializer, Callable] = 1.0,
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
    ):
        self.delay = delay
        self.g_max = g_max
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.V_rev = V_rev
        self.alpha = alpha

        super().__init__(name=name, mode=mode)

        # scaled_g_max = ScaledInitializer(  # Only for dual exponential synapse
        #     initializer=g_max, scale_factor=1.0 / (tau_d - tau_r)
        # )
        # A = 1/ (tau_d - tau_r)  # Setting A to this scales the integral of the conductance change to g_max. The default A means g_max is the peak conductance change.
        self.proj = bp.dyn.FullProjAlignPreSDMg(
            pre=pre,
            delay=self.delay,
            # # T_dur and T describe a rectangular pulse with unitary area, of duration T=tau_r
            # # and height T = 1/tau_r. Alpha is the binding constant, generally set to 1
            # syn=bp.dyn.AMPA.desc(
            #     pre.num, alpha=1 / tau_r, beta=1 / tau_d, T=tau_r, T_dur=tau_r
            # ),
            # !!! YIFAN USES AMPA SYNAPSE, SHENCONG USES DUAL EXPONENTIAL SYNAPSE.
            syn=bp.dyn.DualExponV2.desc(
                pre.num,
                tau_decay=tau_d,
                tau_rise=tau_r,
            ),
            # A=A),
            comm=bp.dnn.CSRLinear(
                conn=conn(pre.size, post.size),
                # weight=1.0,
                weight=g_max,  # Scales the output to g_max, the weight of the synapse
            ),  ### IMPORTANT DON"T USE EVENTCSRLINEAR MESSES WITH THE SYNAPSES. See https://github.com/brainpy/BrainPy/issues/654#issuecomment-2008556824
            out=bp.dyn.COBA(E=V_rev),
            post=post,
        )

    def to_dict(self):
        return {
            "pre": self.proj.pre.name,
            "post": self.proj.post.name,
            "delay": maybe_initializer(self.delay),
            "g_max": maybe_initializer(self.g_max),
            "tau_d": maybe_initializer(self.tau_d),
            "tau_r": maybe_initializer(self.tau_r),
            "V_rev": maybe_initializer(self.V_rev),
            "alpha": maybe_initializer(self.alpha),
        }


class DeltaSynapse(bp.Projection):
    def __init__(
        self,
        pre: NeuDyn,
        post: NeuDyn,
        conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
        delay: Union[float, ArrayType, Initializer, Callable] = 0.0,
        g_max: Union[float, ArrayType, Initializer, Callable] = 1.0,
    ):
        self.delay = delay
        self.g_max = g_max

        super().__init__()
        self.proj = bp.dyn.FullProjDelta(
            pre=pre,
            post=post,
            delay=self.delay,
            comm=bp.dnn.EventCSRLinear(
                conn(pre_size=pre.size, post_size=post.size), g_max
            ),
        )


# !!! This was much faster... why?
# class DeltaSynapse(TwoEndConn):
#     def __init__(
#         self,
#         pre: NeuDyn,
#         post: NeuDyn,
#         conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
#         output: SynOut = CUBA(target_var="input"),
#         stp: Optional[SynSTP] = None,
#         comp_method: str = "sparse",
#         g_max: Union[float, ArrayType, Initializer, Callable] = 1.0,
#         delay_step: Union[float, ArrayType, Initializer, Callable] = None,
#         post_ref_key: str = None,
#         name: str = None,
#         mode: bm.Mode = None,
#         stop_spike_gradient: bool = False,
#     ):
#         super().__init__(
#             name=name, pre=pre, post=post, conn=conn, output=output, stp=stp, mode=mode
#         )
#         # parameters
#         self.stop_spike_gradient = stop_spike_gradient
#         self.post_ref_key = post_ref_key
#         if post_ref_key:
#             self.check_post_attrs(post_ref_key)
#         self.comp_method = comp_method

#         # connections and weights
#         self.g_max, self.conn_mask = self._init_weights(
#             g_max, comp_method=comp_method, sparse_data="csr"
#         )

#         # register delay
#         self.delay_step = delay_step
#         self.pre.register_local_delay("spike", self.name, delay_step=self.delay_step)

#     def update(self, pre_spike=None):
#         # pre-synaptic spikes
#         if pre_spike is None:
#             pre_spike = self.pre.get_local_delay("spike", self.name)
#         pre_spike = bm.as_jax(pre_spike)
#         if self.stop_spike_gradient:
#             pre_spike = jax.lax.stop_gradient(pre_spike)

#         # update sub-components
#         if self.stp is not None:
#             self.stp.update(pre_spike)

#         # synaptic values onto the post
#         if isinstance(self.conn, All2All):
#             syn_value = bm.asarray(pre_spike, dtype=bm.float_)
#             if self.stp is not None:
#                 syn_value = self.stp(syn_value)
#             post_vs = self._syn2post_with_all2all(syn_value, self.g_max)
#         elif isinstance(self.conn, One2One):
#             syn_value = bm.asarray(pre_spike, dtype=bm.float_)
#             if self.stp is not None:
#                 syn_value = self.stp(syn_value)
#             post_vs = self._syn2post_with_one2one(syn_value, self.g_max)
#         else:
#             if self.comp_method == "sparse":
#                 if self.stp is not None:
#                     syn_value = self.stp(pre_spike)
#                     f = lambda s: bm.sparse.csrmv(
#                         self.g_max,
#                         self.conn_mask[0],
#                         self.conn_mask[1],
#                         s,
#                         shape=(self.pre.num, self.post.num),
#                         transpose=True,
#                     )
#                 else:
#                     syn_value = pre_spike
#                     f = lambda s: bm.event.csrmv(
#                         self.g_max,
#                         self.conn_mask[0],
#                         self.conn_mask[1],
#                         s,
#                         shape=(self.pre.num, self.post.num),
#                         transpose=True,
#                     )
#                 if isinstance(self.mode, bm.BatchingMode):
#                     f = jax.vmap(f)
#                 post_vs = f(syn_value)
#             else:
#                 syn_value = bm.asarray(pre_spike, dtype=bm.float_)
#                 if self.stp is not None:
#                     syn_value = self.stp(syn_value)
#                 post_vs = self._syn2post_with_dense(
#                     syn_value, self.g_max, self.conn_mask
#                 )
#         if self.post_ref_key:
#             post_vs = post_vs * (1.0 - getattr(self.post, self.post_ref_key))

#         # update outputs
#         return self.output(post_vs)

#     def _init_weights(
#         self,
#         weight: Union[float, ArrayType, Callable],
#         comp_method: str,
#         sparse_data: str = "csr",
#     ) -> Tuple[Union[float, ArrayType], ArrayType]:
#         if comp_method not in ["sparse", "dense"]:
#             raise ValueError(
#                 f'"comp_method" must be in "sparse" and "dense", but we got {comp_method}'
#             )
#         if sparse_data not in ["csr", "ij", "coo"]:
#             raise ValueError(
#                 f'"sparse_data" must be in "csr" and "ij", but we got {sparse_data}'
#             )
#         if self.conn is None:
#             raise ValueError(
#                 f'Must provide "conn" when initialize the model {self.name}'
#             )

#         # connections and weights
#         if isinstance(self.conn, One2One):
#             weight = parameter(weight, (self.pre.num,), allow_none=False)
#             conn_mask = None

#         elif isinstance(self.conn, All2All):
#             weight = parameter(weight, (self.pre.num, self.post.num), allow_none=False)
#             conn_mask = None

#         else:
#             if comp_method == "sparse":
#                 if sparse_data == "csr":
#                     conn_mask = self.conn.require("pre2post")
#                 elif sparse_data in ["ij", "coo"]:
#                     conn_mask = self.conn.require("post_ids", "pre_ids")
#                 else:
#                     ValueError(f"Unknown sparse data type: {sparse_data}")
#                 weight = parameter(weight, conn_mask[0].shape, allow_none=False)
#             elif comp_method == "dense":
#                 weight = parameter(
#                     weight, (self.pre.num, self.post.num), allow_none=False
#                 )
#                 conn_mask = self.conn.require("conn_mat")
#             else:
#                 raise ValueError(f"Unknown connection type: {comp_method}")

#         # training weights # !!! EDIT IF WE WANT TO TRAIN
#         # if isinstance(self.mode, bm.TrainingMode):
#         #     weight = bm.TrainVar(weight)
#         return weight, conn_mask

#     def to_dict(self):
#         return {
#             "pre": self.pre.name,
#             "post": self.post.name,
#             "delay_step": self.delay_step,
#             "g_max": maybe_initializer(self.g_max),
#         }
