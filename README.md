# WorkingRegime.jl
<!-- Structure: summary (headline; each module and the packages behind it; how to reproduce), then setup, data production in run order, figure production. -->

WorkingRegime.jl reproduces the analyses and figures of the working-regime study, which measures three scaling exponents of cortical dynamics in neural recordings, a spiking circuit model and a fractional neural sampling theory. The exponents are the diffusion exponent `a`, from the mean absolute deviation of increments; the spectral exponent `b`, the aperiodic power-spectrum slope; and the variability exponent `c`, the scaling of spike-count Fano factors. The repository is a Julia workspace of three modules and a root package. [`WRExperiment/`](WRExperiment/) measures the exponents in local field potentials and single units from mouse visual cortex (Allen Visual Behavior Neuropixels, with the Visual Coding functional-connectivity cohort as a replication), reading the data through [AllenNeuropixelsBase.jl](https://github.com/brendanjohnharris/AllenNeuropixelsBase.jl), fitting exponents with [TimeseriesTools.jl](https://github.com/brendanjohnharris/TimeseriesTools.jl) and building surrogate nulls with [TimeseriesSurrogates.jl](https://github.com/JuliaDynamics/TimeseriesSurrogates.jl). [`WRCircuit/`](WRCircuit/) simulates a spatial spiking circuit on GPU with [Dewdrop.jl](https://github.com/brendanjohnharris/Dewdrop.jl) and sweeps its parameters. [`WRTheory/`](WRTheory/) simulates fractional neural samplers with [FractionalNeuralSampling.jl](https://github.com/brendanjohnharris/FractionalNeuralSampling.jl). The root package draws the figures: the scripts in [`scripts/`](scripts/) use CairoMakie with [Fathom.jl](https://github.com/brendanjohnharris/Fathom.jl), and share helpers from [`src/WorkingRegime.jl`](src/WorkingRegime.jl). To reproduce the analyses, (a) run each module's calculation scripts in the order given under [Producing the data](#producing-the-data), then (b) run `bash make_plots`, which draws every figure from the data those scripts write.

# Setup

The scripts need Julia 1.13, which they call as `julia +1.13` through [juliaup](https://github.com/JuliaLang/juliaup). From the repository root:
```bash
julia +1.13 --project -e 'using Pkg; Pkg.instantiate()'
```
`Pkg.instantiate()` resolves and installs the root project and all three workspace members, which share one `Manifest.toml`.

The experimental steps read the Allen data through [AllenNeuropixelsBase.jl](https://github.com/brendanjohnharris/AllenNeuropixelsBase.jl), whose cache directory is set as the `datadir` preference in a root `LocalPreferences.toml`:
```toml
[AllenNeuropixelsBase]
datadir = "/path/to/allen/cache"
```

Since every script is executable through its `bash` header and activates its own project, each can be run as `bash path/to/script.jl` from anywhere. Hand-made figure inputs (the brain illustration, the visual-cortex map and the mean-field schematic) live in [`assets/`](assets/).

# Producing the data

The three groups below are independent of one another; within a group, run the steps in order. Files read by a figure (those whose *Used by* names one) are written to `data/<Module>/` at the repository root; the rest are intermediates, kept in each package's own `data/` directory. Steps marked *cluster* use cluster jobs or workers when run on the USyd Physics cluster (or NCI Gadi, for the circuit sweep) and local workers otherwise; steps marked *GPU* need a CUDA device.

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

Since steps 6 and 7 carry their own session list, they do not need step 1.

## Circuit model ([`WRCircuit/scripts/`](WRCircuit/scripts/))

| Step | Script | Writes | Used by |
|---|---|---|---|
| 1 | `demo_run.jl` (*GPU*) | `demo_run.jld2`, `demo_run_stats.jld2` | Fig 1, Fig S4, Video S1, step 2 |
| 2 | `plot_demo_run.jl` | `circuit_curves.jld2` | Fig 1 |
| 3 | `circuit_sweep.jl` (*GPU*, *cluster*; several days) | `circuit_sweep/` | step 4 |
| 4 | `circuit_exponents.jl` (*cluster*) | `circuit_exponents.jld2` | Fig 4 |

## Theory ([`WRTheory/scripts/`](WRTheory/scripts/))

| Step | Script | Writes | Used by |
|---|---|---|---|
| 1 | `bFNS_sweep.jl` (*cluster*) | `bFNS_sweep/` | Fig 4, step 2 |
| 2 | `bFNS_data.jl` | `bFNS_data.jld2` | Fig 2, Fig S2/3 |
| 3 | `Circuit/mean_field_sweep.jl` (*cluster*) | `mean_field_sweep/` | step 4 |
| 4 | `effective_theory_data.jl` | `Fig3_effective_theory.jld2` | Fig 3 |

# Producing the figures

The figure inputs are also deposited on Zenodo (DOI to be added), one archive per module; unpacking each archive into `data/` replaces running the steps above. Once the data exist, draw the figures with:
```bash
bash make_plots
```
`make_plots` runs every figure script in [`scripts/`](scripts/) in figure order. Each script writes to `plots/<script name>/`: the figure (or video), and, for figures, the source data behind each panel as tab-separated files.

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
