from ..neurons import FNSNeuron, LIFNeuron
from ..synapses import Synapse, maybe_initializer, DeltaSynapse
from ..positions import ClusteredPositions, Positions
import numpy as np
import jax.numpy as jnp
import jax
import types
from typing import Tuple

import brainpy as bp
from abc import ABC, abstractmethod


def fixedprob_to_dict(self):
    return {"prob": self.prob}


bp.connect.FixedProb.to_dict = fixedprob_to_dict


class FNSPopulations(bp.Network):
    r"""
    See Brunel 2000, "Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory
    Spiking Neurons"

    Uses a fixed probability epsilon to connect all populations
    """

    def __init__(
        self, N, epsilon=0.1, D=1.5, nu_hat=2, g=5, J=0.1, key=jax.random.key(42)
    ):
        super().__init__()

        # * Note that nu_thr = theta / (epsilon * Ne * J * tau)

        self.epsilon = epsilon
        self.D = D
        self.nu_hat = nu_hat  # Normalized input rate
        self.g = g
        self.J = J

        num_inh = N // 5
        num_exc = N - num_inh

        theta = 20  # mV
        tau = 20  # ms
        tau_ref = 2.0  # ms
        self.nu_thr = (
            1000 * theta / (epsilon * num_exc * J * tau)
        )  # Tau is in milliseconds. *1000 to convert to Hz
        self.nu = self.nu_hat * self.nu_thr

        # geometry
        exc_positions = ClusteredPositions((-1.5, 0), 1)
        inh_positions = ClusteredPositions((1.5, 0), 1)

        Ekey, Ikey = jax.random.split(key)
        # neurons
        self.E = LIFNeuron(
            size=num_exc,
            embedding=exc_positions,
            V_rest=0.0,  # For simple IF neuron in paper
            V_th=theta,
            V_reset=10.0,
            R=1,
            tau=tau,
            tau_ref=tau_ref,
            V_initializer=bp.init.Normal(0, 1.0),
            key=Ekey,
        )

        # Create a population of inhibitory neurons
        self.I = LIFNeuron(
            size=num_inh,
            embedding=inh_positions,
            V_rest=0.0,
            V_th=theta,
            V_reset=10.0,
            R=1,
            tau=tau,
            tau_ref=tau_ref,
            V_initializer=bp.init.Normal(0, 1.0),
            key=Ikey,
        )

        # Synapses
        delay_step = D // bp.share["dt"]
        delay_step = int(delay_step)

        JE = self.J
        JI = -g * self.J

        self.E2E = DeltaSynapse(
            self.E,
            self.E,
            bp.connect.FixedProb(prob=epsilon),
            delay_step=delay_step,
            g_max=JE,
        )
        self.E2I = DeltaSynapse(
            self.E,
            self.I,
            bp.connect.FixedProb(prob=epsilon),
            delay_step=delay_step,
            g_max=JE,
        )
        self.I2E = DeltaSynapse(
            self.I,
            self.E,
            bp.connect.FixedProb(prob=epsilon),
            delay_step=delay_step,
            g_max=JI,
        )
        self.I2I = DeltaSynapse(
            self.I,
            self.I,
            bp.connect.FixedProb(prob=epsilon),
            delay_step=delay_step,
            g_max=JI,
        )

        # External population
        self.ext = bp.dyn.PoissonGroup(
            self.E.num,  # So that the average number of connections to each population matches Ce
            self.nu,
            keep_size=False,
            sharding=None,
            spk_type=None,
            name=None,
            mode=None,
            seed=None,
        )
        self.ext2E = DeltaSynapse(
            self.ext,
            self.E,
            bp.connect.FixedProb(prob=epsilon),
            delay_step=delay_step,
            g_max=JE,
        )
        self.ext2I = DeltaSynapse(
            self.ext,
            self.I,
            bp.connect.FixedProb(prob=epsilon),
            delay_step=delay_step,
            g_max=JE,
        )

        # define input variables given to E/I populations
        self.Ein = bp.dyn.InputVar(self.E.varshape)
        self.Iin = bp.dyn.InputVar(self.I.varshape)
        self.E.add_inp_fun("", self.Ein)
        self.I.add_inp_fun("", self.Iin)

    def to_dict(self):
        return {
            self.__class__.__name__: {
                "epsilon": self.epsilon,
                "D": self.D,
                "nu": maybe_initializer(self.nu),
                "g": self.g,
                "J": self.J,
                "populations": {
                    "E": self.E.to_dict(),
                    "I": self.I.to_dict(),
                },
                "synapses": {
                    "E2E": self.E2E.to_dict(),
                    "E2I": self.E2I.to_dict(),
                    "I2E": self.I2E.to_dict(),
                    "I2I": self.I2I.to_dict(),
                },
            }
        }
