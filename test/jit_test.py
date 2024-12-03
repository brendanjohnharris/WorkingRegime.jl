import jax
import brainpy as bp
import sys
import os

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_dir)
import src


def simple_jit_test():
    @jax.jit
    def add(a, b):
        return a + b

    add(1, 2)
    return True


def jit_positions():
    positions = src.positions.ClusteredPositions((-1.5, 0), 1)
    jax.jit(positions.__call__)
    key = jax.random.key(42)
    p = positions([10, 10], key)
    return True


def jit_LIFNeurons():
    positions = src.positions.ClusteredPositions((-1.5, 0), 1)
    positions = bp.math.jit(positions.__call__, static_argnums=0)
    key = jax.random.key(42)
    init = bp.init.Normal(0, 1.0)
    En = 100
    E = src.neurons.LIFNeuron(
        size=En,
        embedding=positions,
        V_rest=0.0,  # For simple IF neuron in paper
        V_th=20,
        V_reset=10.0,
        R=1,
        tau=20,
        tau_ref=2,
        V_initializer=bp.math.Variable(init(En)),
        key=key,
    )

    def run(T):
        runner = bp.DSRunner(E, monitors=("spike",))
        runner.run(T)
        return runner.mon.ts

    run = bp.math.jit(run)
    ts = run(1000.0)
    return True


def main():
    simple_jit_test()
    jit_positions()
    jit_LIFNeurons()


if __name__ == "__main__":
    main()
