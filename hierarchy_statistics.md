# Hierarchy correlation: statistical methods

The construction behind **Figure 4c** and **Figure S1f**, from the cached exponent matrices to the
drawn markers. Written to be transcribed into the manuscript Methods and into `datamap.md`; every
number below is read from an output file, and the file is named.

LAST REFRESHED: 2026-09-08. Analysis repo: `~/code/DDC/WorkingRegime.jl` on `cartman`.

---

## 1. What is estimated, and the difficulty

Each of the three dynamical exponents is measured once per (session, area, layer) block, and we ask
whether it varies systematically along the anatomical hierarchy. The obstacle is that the exponents
are not comparable across sessions: recording quality, electrode placement, and behavioural state
shift an entire session's exponents together, so between-session variance dominates the
between-area variance that carries the hierarchy signal.

We measured this directly, decomposing the variance of each `(session x area)` matrix into a session
main effect, an area main effect, and a residual. Across the twelve (exponent, layer) cells of
Figure 4c, the session effect accounts for 22-61% of the total variance and 64-95% of the variance
attributable to the two main effects together; for the spectral exponent at L2/3 those figures are
38% and 65%.

A correlation that pools every (hierarchy, exponent) point across sessions spends most of its
concordant and discordant pairs on that nuisance variance, and is attenuated as a result. We
therefore correlate **within** each session, over that session's own areas, and summarise the
resulting per-session correlations across sessions. Ranking within a session removes the session
offset exactly, because Kendall's tau depends only on the ordering of the values inside the session.

The attenuation is measurable. Taking the pooled correlation over the identical matrices:

| exponent | layer | within-session (median) | pooled |
|---|---|---|---|
| a | L2/3 | -0.400 | -0.274 |
| a | L4 | -0.133 | -0.154 |
| a | L5 | +0.067 | +0.043 |
| a | L6 | +0.333 | +0.176 |
| b | L2/3 | +0.600 | +0.380 |
| b | L4 | +0.467 | +0.324 |
| b | L5 | +0.333 | +0.205 |
| b | L6 | -0.067 | +0.015 |
| c | L2/3 | +0.200 | +0.162 |
| c | L4 | +0.200 | +0.134 |
| c | L5 | +0.333 | +0.233 |
| c | L6 | +0.200 | +0.118 |

Pooling shrinks the estimate towards zero in ten of the twelve cells; the two exceptions (a at L4,
b at L6) are cells where the effect is weak or absent under either construction.

---

## 2. Inputs

**Hierarchy scores.** The six visual areas carry the anatomical hierarchy scores of Siegle et al.
(2021), held in `WRExperiment.hierarchy_scores`: VISp -0.357, VISl -0.093, VISrl -0.059, VISal
0.152, VISpm 0.327, VISam 0.441. These are fixed covariates, identical for every session, and are
never re-estimated here.

**Exponents.** Three per block, none of them recomputed by this analysis:

| symbol | quantity | source |
|---|---|---|
| `a` | diffusion (MAD) exponent | `coeffs_median`, `WRExperiment/data/WRExperiment.jld2` |
| `b` | spectral (aperiodic) exponent | `spectral_exponents`, same file |
| `c` | variability (Fano) exponent | `WRExperiment/data/variability_variation/variability_exponents.jld2` |

**Layers.** L2/3, L4, L5 and L6. Layers rather than cortical depths, because the variability
exponent is defined only per layer (its unit Fano curves carry no finer binning), so this is the
finest grid all three exponents share. L1 is excluded: too few channels per block.

**Cohorts.** Figure 4c uses Allen Visual Behaviour under the spontaneous condition. Figure S1f uses
the Allen Visual Coding `functional_connectivity` cohort, from
`WRExperiment/data/visual_coding_fc.jld2` (23 sessions, 106 blocks), as an independent replication.

---

## 3. Assembling the session-by-area matrix

The estimator operates on one `(session x area)` matrix per (exponent, layer) cell, with `NaN`
marking a missing block. Building that matrix is where the analysis is most easily got wrong, so it
is worth stating explicitly.

**Areas must be aligned by session identifier, never by position.** The six areas do not share a
session list: in Visual Behaviour, VISp carries 68 sessions against the other five areas' 69, and
the session VISp lacks (`1062755779`) sits at position 29 of 69 in the others. Pairing the *i*-th
entry of each area's vector therefore misaligns every row from position 29 onwards, correlating one
session's VISp value against a different session's value in the other five areas: 40 of 68 rows.
`sessionmatrix` (in `WRExperiment/src/SessionKendall.jl`) does the alignment on the sorted
intersection of the identifiers, and `test/runtests.jl` pins the behaviour.

The three exponents arrive in different shapes, so alignment applies unevenly:

- `a` is stored per area, each area with its own `SessionID` lookup, and needs `sessionmatrix`.
- `b` is stored as a single `(SessionID x Structure x layer)` array, already intersected upstream to
  68 sessions when it was assembled, and needs no further alignment.
- `c` is stored as one `(session x area)` matrix per layer, likewise already aligned.

After alignment the Visual Behaviour matrices are complete: every retained session records all six
areas, with row counts of 68 for `a` and `b` and 46 / 55 / 65 / 62 for `c` at L2/3 / L4 / L5 / L6.
The intersection rather than the union is taken so that all three exponents rest on the same
construction; the one session recovered by a union changes nothing beyond the odd/even convention
for the median (it moves `a` at L2/3 from -0.400 to -0.333, a single grid step).

Visual Coding is genuinely ragged, which is what the missing-data handling is for. Of its 23
sessions, 4 record all six areas at L2/3, most record four or five, and three record only three.
Across the twelve cells of Figure S1f, the retained per-session correlations rest on four areas 87
times, five areas 99 times, and six areas 37 times (`plots/FigS1_visual_coding/panelF_sessions.tsv`,
column `n_areas`).

---

## 4. The estimator

`sessionkendall(x, Y; minareas = 4, N = 10_000, seed = 42)`, in
`WRExperiment/src/SessionKendall.jl`. It is included by both figure scripts rather than duplicated,
so the two figures cannot drift apart.

### 4.1 Per-session correlation

For each session *i*, we take the areas that session actually recorded (the non-`NaN` entries of row
*i*) and compute Kendall's tau between their hierarchy scores and their exponents. Kendall rather
than Spearman or Pearson: with at most six points per session we want a statistic that depends only
on concordance and is not distorted by the spacing of the hierarchy scores, which are themselves an
ordinal construct.

### 4.2 Minimum coverage

A session contributes only if it retains at least `minareas = 4` areas. Below that the correlation
is degenerate: with two areas tau can only be -1 or +1, and with three it takes five values, so a
sparse session would enter the summary with an extreme value that reflects coverage rather than
hierarchy. The gate is inert on Visual Behaviour (every aligned session is complete) and does real
work on Visual Coding, where it removes the three sessions carrying only three areas.

### 4.3 Point estimate

The reported estimate is the **median** of the per-session tau. The median rather than the mean
because the per-session values are bounded, discrete, and occasionally saturated at +-1, so a mean
is pulled by sessions whose ordering happens to be perfect.

### 4.4 Interval

A percentile bootstrap of the median, resampling **sessions** (10,000 resamples, 95% interval, seeded
`MersenneTwister(42)`). Sessions are the independent unit; resampling individual area-points would
treat the six values within a session as exchangeable with values from other sessions, which is
exactly the assumption the within-session construction exists to avoid.

The interval is a percentile interval, not a bias-corrected and accelerated one. Earlier captions
claimed BCa because `WRExperiment` re-exports TimeseriesTools' BCa `bootstrapmedian`, while the
figure scripts each defined a shadowing local percentile version. The local copies are now named
`percentilebootmedian`, and the interval drawn is the interval described.

Note that the interval is grid-valued. Each session's tau lives on a grid of spacing 1/15 (six areas
give C(6,2) = 15 pairs), so the median and every bootstrap replicate of it land on grid points, and a
bootstrap endpoint can coincide with the point estimate. Four of the twelve cells in Figure 4c show
this. It is a property of the statistic, not a defect in the interval, and the panels now draw the
per-session values so a reader can see it (Section 5).

### 4.5 The test

**Null.** The hierarchy carries no information about the exponent, within any session. We realise it
by permuting the hierarchy labels **within each session, among only the areas that session
recorded**, independently per session, and recomputing every session's tau. Permuting within session
preserves the number of areas each session contributes and the marginal distribution of its
exponents, so coverage cannot leak into the null; permuting across sessions would not.

**Test statistic: the mean of the per-session tau, not the median.** This is the one place where the
reported estimate and the tested quantity differ, and it is deliberate. Because tau is confined to a
1/15 grid, the median of 68 such values is grid-valued, and its permutation null collapses onto a
handful of atoms. Measured on the observed data with 4,000 permutations, the null distribution of the
median takes **7 distinct values**, and for `a` at L4 **44.3% of null draws land exactly on the
observed value**. The p-value is then decided by the tie convention rather than by the data: 0.913
counting ties against the null, 0.027 excluding them.

Calibration confirms the problem. Over 300 synthetic datasets generated under the null:

| test statistic | rate(p < 0.05) | rate(p < 0.10) | median p |
|---|---|---|---|
| median tau | 0.023 | 0.023 | 0.905 |
| mean tau | 0.060 | 0.110 | 0.53 |
| *nominal* | *0.05* | *0.10* | *0.5* |

The median-statistic p-values are so heavily lumped that no dataset at all falls between 0.05 and
0.10. The mean of the same per-session values is not grid-locked, is calibrated, and orders the
twelve cells identically, so we use it as the test statistic while continuing to report and draw the
median. `test/runtests.jl` asserts this calibration, and the assertion fails if the statistic is
changed back.

**Why not the alternatives.** A one-sample Wilcoxon signed-rank test of the per-session tau against
zero assumes a symmetric null distribution, which the grid and the +-1 saturation make awkward, and
it is not the class of test the retired pooled path used. We report it as a cross-check only, and it
agrees with the permutation test on every one of the twelve cells of Figure 4c at `PTHR`:

| exponent | L2/3 | L4 | L5 | L6 |
|---|---|---|---|---|
| a | 2.1e-10 | 1.4e-3 | 0.13 | 2.8e-8 |
| b | 8.1e-13 | 2.1e-11 | 5.8e-9 | 0.57 |
| c | 7.7e-6 | 1.9e-3 | 8.8e-9 | 5.7e-5 |

(uncorrected, normal approximation with tie correction, which is what `SignedRankTest` uses once
ranks are tied). The signed-rank values are systematically smaller than the permutation p-values
because the permutation floor is 1/(N + 1); the ordering of the cells is the same, and the same three
cells fail `PTHR` under both. `test/runtests.jl` pins the agreement in both directions on synthetic
data. A Mann-Whitney rank-sum test of the observed tau against pooled surrogates (as in the
superseded `mediankendallpvalue`) is mis-specified: it compares `n_sessions` observations against
`N x n_sessions` surrogates as two independent samples, which is not a test of "median tau = 0" and
discards the pairing the surrogates were built to respect.

### 4.6 The p-value itself

We use the add-one estimator

    p = (1 + #{ |tau*| >= |tau_obs| }) / (N + 1)

over N = 10,000 permutations, seeded `MersenneTwister(43)`. The naive proportion returns exactly zero
when no permutation exceeds the observation, which is not a valid p-value; the add-one estimator is
bounded below by 1/(N + 1) and is the correct Monte Carlo estimate. With N = 10,000 the floor is
9.999e-5, and cells at the floor are reported in prose as **p < 1e-4**. Ties count against rejection,
which is conservative.

The same correction has been applied to the superseded pooled path, whose permutation p-values were
previously reported as literal zeros.

### 4.7 Multiple comparisons

Each panel draws 12 cells (3 exponents x 4 layers), and we control the false discovery rate across
those 12 with Benjamini-Hochberg. The correction is applied by the figure script, not inside
`sessionkendall`, because the estimator cannot know what family it belongs to; the function returns
the raw p and the script reports both. `bhadjust` is pinned against `MultipleTesting.adjust` in the
test suite.

A cell is called significant when its adjusted p falls below `PTHR = 1e-2`, the threshold used
throughout the package.

### 4.8 Reproducibility

Every stochastic step is seeded: the bootstrap from `MersenneTwister(seed)` and the permutation from
`MersenneTwister(seed + 1)`, with `seed = 42` by default, drawn from separate streams so that the
interval does not depend on whether the test ran. Two calls with the same seed return identical
output, which the test suite asserts. The jitter in the panels is seeded separately
(`MersenneTwister(7)`) so the figure is byte-reproducible.

---

## 5. What the panels draw

Both panels use the same conventions:

- A **filled marker** at the median tau where the BH-adjusted p is below `PTHR`, an **open marker**
  where it is not. Significance is decided by the test, not by whether the interval clears zero.
- A **whisker** showing the 95% percentile bootstrap interval, drawn in both cases: it describes the
  precision of the median and is no longer what decides significance.
- A **light jittered strip** of the per-session tau behind each interval. This is what makes the 1/15
  granularity visible; without it a bootstrap endpoint coinciding with the point estimate reads as a
  typographical error rather than as a property of a discrete statistic.
- The full tau axis, -1 to 1. The strip reaches the bounds (34 of the 772 per-session values in
  Figure 4c lie beyond +-0.85), so a narrower axis would clip the distribution it exists to show.
- Colours: `a` blue (`baikal`), `b` red (`bermejo`), `c` teal (`glas`), identical in both figures.

---

## 6. Results

### Figure 4c, Visual Behaviour (`plots/Fig4_hierarchical_variation/panelC_stats.tsv`)

| exponent | layer | n | median tau | 95% CI | mean tau | p | p (BH) | sig. |
|---|---|---|---|---|---|---|---|---|
| a | L2/3 | 68 | -0.400 | [-0.467, -0.333] | -0.365 | <1e-4 | 1.5e-4 | yes |
| a | L4 | 68 | -0.133 | [-0.200, -0.067] | -0.147 | 9.0e-4 | 1.2e-3 | yes |
| a | L5 | 68 | +0.067 | [0.000, 0.200] | +0.069 | 0.118 | 0.129 | no |
| a | L6 | 68 | +0.333 | [0.200, 0.333] | +0.251 | <1e-4 | 1.5e-4 | yes |
| b | L2/3 | 68 | +0.600 | [0.467, 0.600] | +0.537 | <1e-4 | 1.5e-4 | yes |
| b | L4 | 68 | +0.467 | [0.333, 0.600] | +0.439 | <1e-4 | 1.5e-4 | yes |
| b | L5 | 68 | +0.333 | [0.200, 0.467] | +0.314 | <1e-4 | 1.5e-4 | yes |
| b | L6 | 68 | -0.067 | [-0.133, 0.067] | -0.008 | 0.872 | 0.872 | no |
| c | L2/3 | 46 | +0.200 | [0.200, 0.467] | +0.270 | <1e-4 | 1.5e-4 | yes |
| c | L4 | 55 | +0.200 | [0.067, 0.200] | +0.139 | 3.4e-3 | 4.1e-3 | yes |
| c | L5 | 65 | +0.333 | [0.200, 0.467] | +0.319 | <1e-4 | 1.5e-4 | yes |
| c | L6 | 62 | +0.200 | [0.067, 0.333] | +0.191 | <1e-4 | 1.5e-4 | yes |

The per-session values behind these medians are in `panelC.tsv`, long format, one row per (exponent,
layer, session) with the area count that produced it.

The diffusion exponent falls with hierarchy in the supragranular layers and rises in L6; the spectral
exponent rises with hierarchy in L2/3 through L5 and is indistinguishable from zero in L6; the
variability exponent rises with hierarchy at every layer.

### Figure S1f, Visual Coding functional connectivity (`plots/FigS1_visual_coding/panelF.tsv`)

| exponent | layer | n | median tau | 95% CI | p | p (BH) | sig. |
|---|---|---|---|---|---|---|---|
| a | L2/3 | 20 | -0.400 | [-0.633, -0.267] | <1e-4 | 1.2e-3 | yes |
| a | L4 | 20 | -0.200 | [-0.433, 0.000] | 0.080 | 0.192 | no |
| a | L5 | 20 | 0.000 | [-0.133, 0.333] | 0.399 | 0.532 | no |
| a | L6 | 20 | +0.100 | [-0.100, 0.333] | 0.301 | 0.452 | no |
| b | L2/3 | 20 | +0.333 | [0.200, 0.600] | 4.0e-4 | 2.4e-3 | yes |
| b | L4 | 20 | 0.000 | [-0.333, 0.200] | 0.959 | 0.959 | no |
| b | L5 | 20 | 0.000 | [-0.333, 0.200] | 0.567 | 0.680 | no |
| b | L6 | 20 | 0.000 | [-0.200, 0.267] | 0.827 | 0.902 | no |
| c | L2/3 | 14 | +0.200 | [-0.067, 0.333] | 0.271 | 0.452 | no |
| c | L4 | 17 | +0.333 | [0.200, 0.600] | 1.5e-3 | 4.5e-3 | yes |
| c | L5 | 18 | +0.333 | [0.167, 0.600] | 1.1e-3 | 4.4e-3 | yes |
| c | L6 | 14 | +0.100 | [0.000, 0.467] | 0.255 | 0.452 | no |

The L2/3 result replicates for both `a` and `b`, with the same sign and comparable magnitude. The
deeper layers do not replicate, which is consistent with the cohort's coverage rather than
surprising: 20 sessions against 68, most contributing four or five areas rather than six.

---

## 7. What changed, and why

Three changes separate this construction from the one previously drawn.

**A session-alignment defect is fixed.** The previous per-session code built its matrix by
truncating each area's session vector to the shortest length and pairing by position, which
misaligned 40 of 68 rows for the diffusion exponent. Three of its four `a` cells move as a result:
L2/3 from -0.333 to -0.400, L4 from -0.067 to -0.133, and L5 from +0.200 to +0.067. The `b` and `c`
rows are unchanged, because those matrices were already assembled on a shared session set upstream.

**Significance testing is restored.** The reworked per-session estimator had dropped both the test
and the multiple-comparison correction that the retired pooled path carried, leaving the filled and
open markers decided by whether the interval happened to exclude zero. Two markers change under the
test: `a` at L5 becomes open (adjusted p = 0.129, having been drawn filled), and `a` at L4 becomes
filled (adjusted p = 1.2e-3, having been drawn open because its interval touched zero).

Note that `a` at L5 is affected by both changes at once. Under the misaligned matrix it read +0.200
and was drawn as a hierarchy effect; correctly aligned it is +0.067 with p = 0.12, which is not
distinguishable from the pooled estimate of +0.043 for the same cell. The within-session construction
is still the right one, but this particular cell should no longer be cited as evidence for it. The
spectral exponent at L2/3 (0.600 within-session against 0.380 pooled) remains a clean example.

**Four overlapping implementations are consolidated into one.** `sessionkendall` is now the only
per-session hierarchy correlation in the repository. `hierarchicalkendall(..., ::Val{:group})`,
`hierarchicalkendall(..., ::Val{:individual})` and `mediankendallpvalue` carry deprecation notices
pointing at it and survive only because `WRExperiment/scripts/plots/madev.jl` and
`WRExperiment/scripts/plots/hierarchy_tau.jl` still run against them.

---

## 8. Assumptions and limitations

**Sessions are weighted equally.** A session contributing four areas counts as much as one
contributing six in the median, despite the larger sampling variance of its tau. Provided coverage is
independent of hierarchy this costs a little power and biases nothing. A coverage-weighted
combination is possible; we judged the extra machinery not to be worth it, and the `n_areas` column
in the panel source data lets a reader check the composition.

**Mice are not treated as a cluster.** The Visual Behaviour cohort's 68 sessions come from fewer
mice, and neither the bootstrap nor the permutation accounts for that nesting. The bootstrap is
clustered by session, not by animal, so its intervals may be slightly narrow if exponents are
correlated within an animal. A cluster bootstrap by mouse is the natural robustness check and is not
yet run.

**Tau is coarsely discretised.** With six areas per session it takes 15 values. This affects the
interval (Section 4.4) and the choice of test statistic (Section 4.5), and it means the panels should
be read as a distribution over a grid, which is why the per-session strip is drawn.

**The test is one-sided in magnitude only.** We test |tau| against the null, so the reported p-values
are two-sided with respect to the sign of the correlation.

**Nothing here re-estimates the exponents.** The analysis operates on cached matrices; the estimator
definitions for `a`, `b` and `c` are documented separately.

---

## 9. Files

| what | where |
|---|---|
| estimator | `WRExperiment/src/SessionKendall.jl` |
| tests | `WRExperiment/test/runtests.jl` (`julia --project=WRExperiment WRExperiment/test/runtests.jl`) |
| Figure 4c | `scripts/Fig4_hierarchical_variation.jl` |
| Figure 4c summary | `plots/Fig4_hierarchical_variation/panelC_stats.tsv` |
| Figure 4c per-session | `plots/Fig4_hierarchical_variation/panelC.tsv` |
| Figure S1f | `scripts/FigS1_visual_coding.jl` |
| Figure S1f summary | `plots/FigS1_visual_coding/panelF.tsv` |
| Figure S1f per-session | `plots/FigS1_visual_coding/panelF_sessions.tsv` |
| superseded paths | `WRExperiment/src/Patch.jl` (`hierarchicalkendall`, `mediankendallpvalue`) |

### Manuscript passages this supersedes

The following are known stale and must be rewritten from the tables in Section 6 rather than patched:

- `main.tex:455`, `main.tex:458`, `main.tex:462`: Results, describing 20 common depths and
  group-level correlations at 22%, 75% and 82% cortical depth.
- `main.tex:427`: Figure 4 caption; says "across cortical depths", gives the wrong colours, claims
  BCa intervals, and omits the variability exponent.
- `main.tex:649-650`: Methods, "Hierarchical correlation analysis", describing the retired pooled
  construction end to end.
- `supp.tex`, Figure S1 caption: says "Group-level Kendall's tau", which it is not.
- `datamap.md:203`: the row citing the old depth-based numbers, and the "p = 0 (0 against 1e4
  permutations)" entry that the add-one estimator now makes impossible.
