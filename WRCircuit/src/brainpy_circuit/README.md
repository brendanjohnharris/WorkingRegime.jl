# WRCircuit in BrainPy

A Python reimplementation of the WRCircuit spiking model on [BrainPy](https://github.com/brainpy/BrainPy) (JAX). It was the original backend, replaced by native [Dewdrop.jl](https://github.com/brendanjohnharris/Dewdrop.jl). It is kept as a snapshot for comparison; WRCircuit no longer loads it.

| File | Contents |
|---|---|
| `models/Spatial.py`, `models/Nonspatial.py` | the spatial (sheet) and non-spatial circuits |
| `neurons.py`, `synapses.py` | FNS neuron, Poisson input, synapse dynamics |
| `positions.py`, `distances.py` | neuron placement and distance-dependent connectivity |
| `running.py` | `DSRunner` wrappers, including a vectorised parameter map |
| `stats.py`, `plots.py`, `utils.py` | summary statistics and plotting helpers |

Requires `brainpy`, `jax`, `numpy`, `scipy`, `networkx` and `matplotlib`.

```python
import sys; sys.path.insert(0, "WRCircuit/src")
from brainpy_circuit import models
net = models.Spatial()
```
