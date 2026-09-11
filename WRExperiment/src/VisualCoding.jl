# The Allen Visual Coding cohorts, as a second source of the same observables Visual Behaviour
# supplies. Nothing here re-implements an estimator: `send_madev` runs unchanged on these sessions
# (the two cohort differences it used to trip on --- overlapping unit/channel column names and the
# unit-id column, see `unitids` --- are handled in Patch.jl), so a Visual Coding run is a session
# list plus an output directory.
#
# Two protocols are covered. `functional_connectivity` is the one the supplementary figure uses: it
# contains a single contiguous ~1800 s spontaneous block, against Visual Behaviour's much shorter
# interleaved rests. `brain_observatory_1.1` is the larger, shorter-epoch protocol, included so the
# duration difference can be controlled for.
#
# Both record LFP at 1250 Hz (`probes.csv` advertises 2500 Hz, but that is the acquisition rate and
# `lfp_temporal_subsampling_factor = 2` halves it on the way into the NWB) and store every fourth
# channel, so an area yields roughly 24 channels rather than Visual Behaviour's ~90.

"""
Visual Coding `functional_connectivity` session identifiers: every session of that protocol in the
Allen release. Coverage of the six visual areas is ragged: only 4 of the 23 sessions carry all six
at L2/3, most carry four or five, and three carry only three. This is why the hierarchy correlation
correlates within each session over the areas that session has, gated on a minimum coverage, rather
than requiring a complete design; see [`sessionkendall`](@ref).
"""
const VISUAL_CODING_FC = [766640955, 767871931, 768515987, 771160300, 771990200, 774875821,
    778240327, 778998620, 779839471, 781842082, 786091066, 787025148, 789848216, 793224716,
    794812542, 816200189, 819186360, 819701982, 821695405, 829720705, 831882777, 835479236,
    839068429, 840012044, 847657808]

"""
Visual Coding `brain_observatory_1.1` session identifiers. Same recording configuration as
[`VISUAL_CODING_FC`](@ref) but a different stimulus protocol, whose spontaneous epochs are short and
interleaved rather than one long block.
"""
const VISUAL_CODING_BO = [715093703, 719161530, 721123822, 732592105, 737581020, 739448407,
    742951821, 743475441, 744228101, 746083955, 750332458, 750749662, 751348571, 754312389,
    754829445, 755434585, 756029989, 757216464, 757970808, 758798717, 759883607, 760345702,
    760693773, 761418226, 762120172, 762602078, 763673393, 773418906, 791319847, 797828357,
    798911424, 799864342]

"The only stimulus these cohorts are analysed under; the pipeline's other two are behavioural."
const VISUAL_CODING_STIMULUS = "spontaneous"

"""
    visual_coding_sessions(cohort = :functional_connectivity)

Session identifiers for a Visual Coding `cohort`, one of `:functional_connectivity` (`:fc`) or
`:brain_observatory` (`:bo`). Visual Behaviour's equivalent is the `ecephys_session_id` column of
`datadir("session_table.jld2")`; these lists are fixed by the release, so they are held here rather
than rediscovered from the cache on every run.
"""
function visual_coding_sessions(cohort::Symbol = :functional_connectivity)
    cohort in (:functional_connectivity, :fc) && return copy(VISUAL_CODING_FC)
    cohort in (:brain_observatory, :bo) && return copy(VISUAL_CODING_BO)
    throw(ArgumentError("unknown cohort $(cohort); expected :functional_connectivity or :brain_observatory"))
end

"""
    visual_coding_calcdir(cohort = :functional_connectivity)

Where [`send_madev`](@ref) output for a Visual Coding `cohort` lives, alongside Visual Behaviour's
`datadir("calculations")`. Kept separate so `calcquality` on either directory describes one cohort.
"""
function visual_coding_calcdir(cohort::Symbol = :functional_connectivity)
    cohort in (:functional_connectivity, :fc) &&
        return DrWatson.datadir("calculations_visual_coding_fc")
    cohort in (:brain_observatory, :bo) &&
        return DrWatson.datadir("calculations_visual_coding_bo")
    throw(ArgumentError("unknown cohort $(cohort); expected :functional_connectivity or :brain_observatory"))
end
