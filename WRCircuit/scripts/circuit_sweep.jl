#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 --handle-signals=yes -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate :WRCircuit
using WRCircuit
import AcademicClusters.NCIGadi: runscripts

delta = round.(range(2.5, 5, length = 51); sigdigits = 3)
Delta_g_K = round.(range(0, 0.005, length = 26); sigdigits = 3)
sigma_ee = round.(range(0.03, 0.075, length = 19); sigdigits = 3)
tau_r_e = round.(range(0.5, 2.0, length = 31); sigdigits = 3)
tau_d_e = round.(range(4, 6, length = 41); sigdigits = 3)

seeds = [1] # 1:10

PLANES = [
    # (:tau_r_e => tau_r_e, :tau_d_e => tau_d_e),        # τ_syn
    (:delta => delta, :tau_d_e => tau_d_e),           # δ/τ_d
    # (:delta => delta, :Delta_g_K => Delta_g_K),       # δ × Δg_K
    # (:delta => delta, :sigma_ee => sigma_ee),         # δ × σ_ee
    # (:Delta_g_K => Delta_g_K, :sigma_ee => sigma_ee), # Δg_K × σ_ee
]

if contains(gethostname(), "gadi")
    batch = 16 # Ok for v100
    @info "Submitting sweep jobs to Gadi"
    setup = quote
        using DrWatson
        @quickactivate :WRCircuit
    end
    jobs = runscripts(
        vec([:(send_sweep($p1, $p2, $seed; batch = $batch)) for (p1, p2) in PLANES, seed in seeds]);
        setup, queue = "gpuvolta", project = `$(projectdir())`,
    )
    @info "Submitted $(length(jobs)) sweep jobs"
else
    batch = 32 # Ok for l40s
    for seed in seeds, (p1, p2) in PLANES
        send_sweep(p1, p2, seed; batch)
    end
end
