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

import brainpy as bp
import brainpy.math as bm


def _run(model, T, monitors, fixed_parameters, jit=True, **kwargs):
    def __run(**parameters):
        net = model(**parameters, **fixed_parameters)
        runner = bp.DSRunner(net, monitors=monitors, jit=jit, **kwargs)
        runner.run(T)
        t = runner.mon["ts"].view()
        X = {m: runner.mon[m].view() for m in monitors}
        return [t, X]

    return __run


def run_parallel(
    model, parameters, fixed_parameters, num_parallel, clear_buffer=False, **kwargs
):
    parameters = {k: v for k, v in parameters.items()}
    print(parameters)
    Xs = bp.running.jax_vectorize_map(
        _run(model, fixed_parameters=fixed_parameters, **kwargs),
        parameters,
        num_parallel,
        clear_buffer,
    )
    return Xs
