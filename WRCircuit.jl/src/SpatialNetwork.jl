# * The spatial FNS "working-regime" E/I network, built from FIRST-CLASS Dewdrop components -------------
# The native replacement for the old `Dewdrop.spatial_fns` (now removed): a thin composition of the Dewdrop
# builder over generic primitives --- FNSNeuron populations on a periodic sheet, distance-fixed-count
# recurrent connectivity with in-degree `correlate_weights`, frozen-current `FrozenDualExpSynapse`, and a
# streaming Poisson `drive!` per population. Frozen-current COBA, matching the BrainPy `sum_current_inputs`
# scheme (the synaptic current `g·(Erev−V)` is injected without shunting the membrane time constant). The
# external drive stays an independent per-population stream (BrainPy shares one pool across E and I), so this
# is close to --- not a bit-for-bit --- BrainPy. The whole model is just builder calls; no engine patching.

# Independent sub-seeds off the master seed (golden-ratio mix; one per random structure).
_subseed(seed::Unsigned, tag::Integer) = (seed % UInt64) ⊻ ((tag % UInt64) * 0x9e3779b97f4a7c15)

"""
    build_spatial(; rho, dx, gamma, sigma_*, K_*, delta, J_ee, J_ei, nu, n_ext, tau_*, V_rev_*,
                  e_delay, i_delay, Delta_g_K, tau_K, seed, arch=DEWDROP_BACKEND()) -> Dewdrop.FrozenBuilder

Assemble the spatial FNS E/I working-regime network from first-class Dewdrop components (see file header)
as an **unmaterialised spec** --- a `Dewdrop.FrozenBuilder` that renders the full population/projection tree
*cheaply* (it holds the recipes, not the millions-of-edges connectome) and materialises into a
`DewdropNetwork` only at `Dewdrop.solve`/`build`, with the run's real `tspan`/`dt`. `rho`/`dx` set the E grid
(`ne = round(√rho·dx)`, `NE = ne²`); `gamma` the E:I ratio; `sigma_*`/`K_*` the distance kernels and
per-target in-degrees; `J_*`/`delta` the in-degree-scaled weights; `nu`/`n_ext` the external Poisson drive.
The synapse is frozen-current COBA (`Dewdrop.FrozenDualExpSynapse`), matching the BrainPy `sum_current_inputs`
scheme. `T` is the simulation float type (default `Float32` --- halves the state /
recorded-trace / connectome footprint vs `Float64`, which matters for large populations × long runs).
"""
function build_spatial(;
        rho = 20000, dx = 0.5, gamma = 4,
        sigma_ee = 0.06, sigma_ei = 0.07, sigma_ie = 0.14, sigma_ii = 0.14,
        K_ee = 260, K_ei = 340, K_ie = 225, K_ii = 290,
        delta = 4.0, J_ee = 0.00105, J_ei = 0.00145, nu = 10.0, n_ext = 100,
        tau_r_e = 1.0, tau_r_i = 2.0, tau_d_e = 5.0, tau_d_i = 4.5,
        V_rev_e = 0.0, V_rev_i = -80.0, e_delay = 1.5, i_delay = 1.5,
        Delta_g_K = 0.002, tau_K = 40.0,
        seed = 0x05fd, arch::Dewdrop.AbstractArchitecture = DEWDROP_BACKEND(),
        T::Type{<:AbstractFloat} = Float32, tspan = (0.0, 1.0)
    )
    seed = UInt64(seed)
    # --- geometry: E on a cell-centred grid, I uniform-random, on a periodic [0,dx]² sheet ---
    ne = round(Int, sqrt(rho) * dx)
    ne ≥ 1 || throw(ArgumentError("rho·dx² too small: ne = round(√rho·dx) = $ne < 1"))
    NE = ne^2
    NI = round(Int, NE / gamma)
    NI ≥ 1 || throw(ArgumentError("NE/gamma too small: NI = $NI < 1"))
    period = (Float64(dx), Float64(dx))
    posE = Dewdrop.grid_positions(ne, ne; spacing = dx / ne, centered = true)
    posI = Dewdrop.random_positions(NI, (dx, dx); seed = _subseed(seed, 1))
    # --- neurons: E adapts (conductance gK), I does not ---
    E = Dewdrop.FNSNeuron(;
        C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -70.0,
        tref = 4.0, τK = tau_K, ΔgK = Delta_g_K
    )
    I = Dewdrop.FNSNeuron(;
        C = 0.25, gL = 0.025, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -70.0,
        tref = 4.0, τK = tau_K, ΔgK = 0.0
    )
    # --- exact-COBA synapses: excitatory (Erev=V_rev_e) from E, inhibitory (Erev=V_rev_i) from I ---
    exc() = Dewdrop.FrozenDualExpSynapse(; τr = tau_r_e, τd = tau_d_e, Erev = V_rev_e)
    inh() = Dewdrop.FrozenDualExpSynapse(; τr = tau_r_i, τd = tau_d_i, Erev = V_rev_i)
    J_ie = J_ee * delta
    J_ii = J_ei * delta             # I weights δ-amplified; sign carried by the reversal potential
    cw(J, tag) = Dewdrop.correlate_weights(J; seed = _subseed(seed, tag))
    # --- builder: populations → four recurrent distance-fixed-count paths → external drive ---
    nb = Dewdrop.network(; tspan = tspan, arch = arch)
    Dewdrop.population!(nb, :E, E, NE; positions = posE)
    Dewdrop.population!(nb, :I, I, NI; positions = posI)
    Dewdrop.project!(
        nb, :E => :E, exc(); kernel = Dewdrop.exponential_kernel(sigma_ee), count = K_ee * NE,
        weight = 1.0, delay = e_delay, seed = _subseed(seed, 2), allow_self = true, period = period, adjust = cw(J_ee, 6)
    )
    Dewdrop.project!(
        nb, :E => :I, exc(); kernel = Dewdrop.exponential_kernel(sigma_ei), count = K_ei * NI,
        weight = 1.0, delay = e_delay, seed = _subseed(seed, 3), allow_self = false, period = period, adjust = cw(J_ei, 7)
    )
    Dewdrop.project!(
        nb, :I => :E, inh(); kernel = Dewdrop.exponential_kernel(sigma_ie), count = K_ie * NE,
        weight = 1.0, delay = i_delay, seed = _subseed(seed, 4), allow_self = false, period = period, adjust = cw(J_ie, 8)
    )
    Dewdrop.project!(
        nb, :I => :I, inh(); kernel = Dewdrop.exponential_kernel(sigma_ii), count = K_ii * NI,
        weight = 1.0, delay = i_delay, seed = _subseed(seed, 5), allow_self = true, period = period, adjust = cw(J_ii, 9)
    )
    # --- external streaming Poisson drive: ~n_ext inputs per neuron, independent per population ---
    N_ext = round(Int, sqrt(n_ext * NE))
    p_ext = sqrt(n_ext / NE)
    Dewdrop.drive!(
        nb, :E, exc(); rate = nu, n_ext = N_ext, p = p_ext, weight = 1.0, delay = e_delay,
        seed = _subseed(seed, 10), adjust = cw(J_ee, 12)
    )
    Dewdrop.drive!(
        nb, :I, exc(); rate = nu, n_ext = N_ext, p = p_ext, weight = 1.0, delay = e_delay,
        seed = _subseed(seed, 11), adjust = cw(J_ei, 13)
    )
    # Return the FROZEN BUILDER (an unmaterialised spec), NOT the built network: it renders the full
    # population/projection tree cheaply (no connectome) and materialises into a `DewdropNetwork` only at
    # `solve`/`build`, with the run's real `tspan`/`dt` (the `tspan` above is just the placeholder default).
    # Everything above is built in convenient `Float64`; switch the whole spec to `T` in ONE pass
    # (`convertfloat` recurses models/synapses/weights/positions), instead of wrapping every literal.
    spec = Dewdrop.freeze(nb)
    return Dewdrop.convertfloat(T, spec)
end
