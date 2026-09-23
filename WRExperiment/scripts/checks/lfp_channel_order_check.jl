#! /bin/bash
#=
exec julia +1.13 --project="$(dirname "${BASH_SOURCE[0]}")/../.." "${BASH_SOURCE[0]}" "$@"
=#
# Asserts that the LFP channel axis lines up with its labels, against the only reference that does
# not share the read pipeline: the NWB file itself. The ElectricalSeries electrode region is the
# identity, so data row r belongs to the electrode `AN.getlfpchannels(...)[r]`.
#
# History, because this is easy to get wrong in both directions. `AN.sortbydepth` used to permute the
# `Chan` lookup but not the data, so every LFP came back mirrored against its own labels; and
# `set(x, Chan => Depth(depths))` reverses data whenever the two lookups run in opposite directions
# (DimensionalData >= 0.30). The two cancelled, and the pipeline was accidentally correct. Both are
# now fixed: AllenNeuropixelsBase sorts data and lookup together, and `send_madev` relabels with
# `chan2depth` (swapdims), which preserves the data. Neither may be reverted alone.
#
# A layer/depth self-consistency check cannot catch this: sorting depths and reading off layer names
# tests labels against labels and passes happily while the data is mirrored. Only the file can tell.
using WRExperiment
using Statistics
using TimeseriesTools
using Unitful
import AllenNeuropixelsBase as AN

sid, structure = 1048189115, "VISp"
session = AN.Session(sid)
probeid = first([p for (p, ss) in pairs(AN.getprobestructures(session)) if structure in ss])
fileids = AN.getlfpchannels(session, probeid)
ts = AN.getlfptimes(session, probeid)

LFP = AN.formatlfp(session; tol = 3, sessionid = sid, epoch = :longest,
                   stimulus = "spontaneous", structure)
chans = collect(lookup(LFP, AN.Chan))
X = Float64.(ustrip.(parent(ustripall(LFP))))
i0 = argmin(abs.(ts .- ustrip(first(times(LFP)))))
n = 5000
REF = AN.h5open(AN.getlfppath(session, probeid)) do f
    r = "probe_$(probeid)_lfp"
    Float64.(f["acquisition"][r][r * "_data"]["data"][:, i0:(i0 + n - 1)])
end
m = min(size(X, 1), n)
own = mean([cor(X[1:m, j], REF[findfirst(==(c), fileids), 1:m]) for (j, c) in enumerate(chans)])
@info "formatlfp columns vs the NWB file" own
own > 0.99 ||
    error("""formatlfp columns do not match their channel labels (mean cor $own).
             The channel axis is mirrored somewhere. Check `AN.sortbydepth`, and check that
             `send_madev` still relabels with `chan2depth` rather than `set`.""")

depths = AN.getchanneldepths(session, LFP; method = :probe)
issorted(depths) || error("formatlfp channels are not ordered surface-to-depth")

# The relabel used by send_madev must not touch the data.
S = WRExperiment.chan2depth(LFP, depths)
parent(S) === parent(LFP) || error("chan2depth reordered the data; it must only relabel")

@info "OK: LFP columns match their labels, depths run surface-to-depth, chan2depth preserves data"
