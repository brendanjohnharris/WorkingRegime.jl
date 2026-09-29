# WorkingRegime.jl
[![Zenodo](https://img.shields.io/badge/Zenodo-1682D4?logo=zenodo&logoColor=fff&style=for-the-badge)](https://doi.org/10.5281/zenodo.23033328)
[![Figshare](https://img.shields.io/badge/figshare-556472?logo=figshare&logoColor=fff&style=for-the-badge)](https://doi.org/10.6084/m9.figshare.34021314)

WorkingRegime.jl reproduces the analyses and figures of "_An adaptive fractional state links circuit mechanisms to cortical dynamics across the visual hierarchy_". We measure three scaling exponents of cortical dynamics in neural recordings, a spiking circuit model, and a fractional neural sampling theory:
- the diffusion exponent `a`, from the mean absolute deviation of increments;
- the spectral exponent `b`, the aperiodic slope of the power spectrum;
- the variability exponent `c`, from the scaling of spike-count Fano factors.

The repository is a Julia workspace of three modules and a root package:
- [`WRExperiment/`](WRExperiment/) measures the exponents in local field potentials and single units from mouse visual cortex, using the Allen Visual Behavior Neuropixels dataset with the Visual Coding functional-connectivity cohort for replication. The module's scripts read the data through [AllenNeuropixelsBase.jl](https://github.com/brendanjohnharris/AllenNeuropixelsBase.jl), fit exponents with [TimeseriesTools.jl](https://github.com/brendanjohnharris/TimeseriesTools.jl) and build surrogate nulls with [TimeseriesSurrogates.jl](https://github.com/JuliaDynamics/TimeseriesSurrogates.jl).
- [`WRCircuit/`](WRCircuit/) simulates a spatial spiking circuit on GPU with [Dewdrop.jl](https://github.com/brendanjohnharris/Dewdrop.jl) and sweeps its parameters.
- [`WRTheory/`](WRTheory/) simulates fractional neural samplers with [FractionalNeuralSampling.jl](https://github.com/brendanjohnharris/FractionalNeuralSampling.jl).
- The root package reproduces all figures in the paper. The scripts in [`scripts/`](scripts/) use CairoMakie with [Fathom.jl](https://github.com/brendanjohnharris/Fathom.jl) and share helpers from [`src/WorkingRegime.jl`](src/WorkingRegime.jl).

To reproduce the figures, run each module's calculation scripts in the order given under [Producing the data](#producing-the-data), then run `bash make_plots`. To skip the calculations, download the figure inputs from [figshare](https://doi.org/10.6084/m9.figshare.34021314) instead (see [Producing the figures](#producing-the-figures)).

# Setup

The scripts call Julia 1.13 as `julia +1.13`, which selects the 1.13 channel of [juliaup](https://github.com/JuliaLang/juliaup). From the repository root:
```bash
juliaup add 1.13
julia +1.13 --project -e 'using Pkg; Pkg.instantiate()'
```
`Pkg.instantiate()` resolves and installs the root project and all three workspace members, which share one `Manifest.toml`.

The experiment scripts read Allen Neuropixels data through [AllenNeuropixelsBase.jl](https://github.com/brendanjohnharris/AllenNeuropixelsBase.jl). Set the Allen data cache with the `datadir` preference in a root `LocalPreferences.toml`:
```toml
[AllenNeuropixelsBase]
datadir = "/path/to/allen/cache"
```

Every script has a `bash` header and activates its own project, and can be run from any directory as `bash path/to/script.jl`. Hand-made figure inputs (the brain illustration, the visual-cortex map and the mean-field schematic) are in [`assets/`](assets/).

# Producing the data

The three groups below are independent, but the steps within each group must be run in order. Figure inputs, the files whose *Used by* entry names a figure, are written to `data/<Module>/` at the repository root. Intermediate files stay in each module's own `data/` directory. Steps marked *cluster* use cluster jobs or workers on the USyd Physics cluster (or NCI Gadi, for the circuit sweep) and local workers elsewhere. Steps marked *GPU* need a CUDA device.

## Experiment ([`WRExperiment/scripts/`](WRExperiment/scripts/))

| Step | Script | Writes | Used by |
|---|---|---|---|
| 1 | `select_sessions.jl` | `session_table.jld2` | steps 2, 3, 5 |
| 2 | `run_calculations.jl` (*cluster*) | `calculations/`, one file per session, stimulus and area | step 3 |
| 3 | `collect_calculations.jl` | `WRExperiment.jld2`, `traces.jld2`, `increment_histograms.jld2` | Fig 1, Fig 4, step 4 |
| 4 | `variability_variation.jl` | `variability_variation/variability_exponents.jld2` | Fig 4 |
| 5 | `run_surrogates.jl` (*cluster*) | `surrogates_ft/` | Fig 4 |
| 6 | `run_calculations_visual_coding.jl` (*cluster*) | `calculations_visual_coding_fc/` | step 7 |
| 7 | `collect_calculations_visual_coding.jl` | `visual_coding_fc.jld2`, `visual_coding_increments_fc.jld2` | Fig S1 |

Steps 6 and 7 carry their own session list and do not need step 1.

## Circuit model ([`WRCircuit/scripts/`](WRCircuit/scripts/))

| Step | Script | Writes | Used by |
|---|---|---|---|
| 1 | `demo_run.jl` (*GPU*) | `demo_run.jld2`, `demo_run_stats.jld2` | Fig 1, Fig S4, Video S1, step 2 |
| 2 | `plot_demo_run.jl` | `circuit_curves.jld2` | Fig 1 |
| 3 | `circuit_sweep.jl` (*GPU*, *cluster*; several days) | `circuit_sweep/` | step 4 |
| 4 | `circuit_exponents.jl` (*cluster*) | `circuit_exponents.jld2` | Fig 4 |
| 5 | `synchrony_surrogates.jl` (*GPU*) | `synchrony_surrogates/`, `synchrony_surrogates.jld2` | Fig 1 |

## Theory ([`WRTheory/scripts/`](WRTheory/scripts/))

| Step | Script | Writes | Used by |
|---|---|---|---|
| 1 | `bFNS_sweep.jl` (*cluster*) | `bFNS_sweep/` | Fig 4, step 2 |
| 2 | `bFNS_data.jl` | `bFNS_data.jld2` | Fig 2, Fig S2/3 |
| 3 | `Circuit/mean_field_sweep.jl` (*cluster*) | `mean_field_sweep/` | step 4 |
| 4 | `effective_theory_data.jl` | `Fig3_effective_theory.jld2` | Fig 3 |

# Producing the figures

The figure inputs are deposited on [figshare](https://doi.org/10.6084/m9.figshare.34021314) as a single archive, `WorkingRegime_data.zip`. Unzipping it at the repository root restores `data/` and replaces the calculation steps above. With the data in place, draw the figures with:
```bash
bash make_plots
```
`make_plots` runs every figure script in [`scripts/`](scripts/), in figure order. Each script writes its figure or video to `plots/<script name>/`, and each figure script also writes the source data for every panel there as tab-separated files.

| Script | Figure |
|---|---|
| `Fig1_combined_curves.jl` | Fig 1 |
| `Fig2_bFNS.jl` | Fig 2 |
| `Fig3_effective_theory.jl` | Fig 3 |
| `Fig4_hierarchical_variation.jl` | Fig 4 |
| `FigS1_visual_coding.jl` | Fig S1 |
| `FigS23_summaries.jl` | Fig S2, Fig S3 |
| `FigS4_input_parameters.jl` | Fig S4 |
| `VideoS1_input_field.jl` | Video S1 |
