module WorkingRegime
# Helpers shared by the figure scripts, which load them with `@quickactivate :WorkingRegime`.

using DelimitedFiles: writedlm
using Random: MersenneTwister
using Statistics: median, quantile
using Makie: Label, rich, translate!
using TimeseriesBase: At, lookup # DimensionalData, re-exported
import Fathom

export mit, unitlabel, mathify, blocklabel, centre_on_box!, _select, percentilebootmedian, writegrid

# ──────────────────────────────────────────────────────────────────────────────
# Math labels
# ──────────────────────────────────────────────────────────────────────────────

const MATH_IT = joinpath(Fathom.stixbase, "STIXTwoText-Italic.ttf")

"A variable, in the math italic face: `mit(\"X\")`, `mit(\"|ΔI|\")`."
mit(s) = rich(s; font = MATH_IT)

"""
    unitlabel(name, u)

`name` followed by the unit `u` in parentheses. Only `name` may carry the math face; the unit stays
in the figure's sans font, so `unitlabel(mit("X"), "mm")` gives an italic *X* against a plain "mm".
`name` may be a plain string (a word) or a `rich` (e.g. `mit("X")` for a variable).
"""
unitlabel(name, u) = rich(name, " ($u)")

"""
    mathify(str)

`str` with any Greek letter set in the math face and everything else left in the figure's sans
font: `mathify("α=2, β=1")`. For labels that arrive as plain strings from the data.
"""
function mathify(str::AbstractString)
    parts, buf = Any[], IOBuffer()
    flush!() = (t = String(take!(buf)); isempty(t) || push!(parts, t))
    for ch in str
        if ch in ('α', 'β', 'γ', 'η', 'ν', 'κ')
            flush!()
            push!(parts, mit(string(ch)))
        else
            print(buf, ch)
        end
    end
    flush!()
    return rich(parts...)
end

# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────

"""
    blocklabel(gp, text)

A vertical bold `Label` at grid position `gp` naming a block of panels, usually placed in a column 0
beside the block's group box.
"""
blocklabel(gp, text) = Label(
    gp, text; rotation = pi / 2, font = :bold, fontsize = 18, tellheight = false
)

"""
    centre_on_box!(lab, box)

Translate the label `lab` vertically onto the centre of `box`. A label beside a box is centred on the
label's layout cell, whereas the box reaches past that cell by different amounts above and below.
Call after `addlabels!`, which re-solves the layout.
"""
centre_on_box!(lab, box) = let b = lab.layoutobservables.computedbbox[],
        d = box.layoutobservables.computedbbox[]

    translate!(
        lab.blockscene, 0,
        (d.origin[2] + d.widths[2] / 2) - (b.origin[2] + b.widths[2] / 2), 0
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

"""
    _select(x, name => value, ...)

`x` indexed at `value` (an `At` lookup) along each dimension called `name`. Arrays reloaded through
`toolsarray_typemap` carry custom dimensions (`Structure`, `SessionID`, `α`, ...) as generic
`Dim{:name}` with lookups intact, so those arrays are indexed by name.
"""
_select(x, selectors::Pair...) = getindex(x; (Symbol(n) => At(v) for (n, v) in selectors)...)

"""
    percentilebootmedian(x; N = 10_000, α = 0.05)

Median of `x`, with missing and `NaN` values dropped, and a percentile-bootstrap `1 - α` interval
from `N` resamples: returns `(median, (lower, upper))`, or `(NaN, (NaN, NaN))` if no values remain.
The seed is fixed, so a figure redraws with the same interval.
"""
function percentilebootmedian(x; N = 10_000, α = 0.05)
    x = filter(!isnan, collect(skipmissing(x)))
    isempty(x) && return (NaN, (NaN, NaN))
    rng = MersenneTwister(42)
    n = length(x)
    meds = [median(x[rand(rng, 1:n, n)]) for _ in 1:N]
    return median(x), Tuple(quantile(meds, (α / 2, 1 - α / 2)))
end

"""
    writegrid(path, X)

Write the `(α × β)` grid `X` to `path` as a tab-separated table: a header row of β values under the
corner cell `alpha\\beta`, then one row per α value. `X` is permuted to `(α, β)` first, so the file
layout is independent of the dimension order in which `X` was built.
"""
function writegrid(path, X)
    X = permutedims(X, (:α, :β))
    return writedlm(
        path,
        vcat(
            hcat("alpha\\beta", permutedims(collect(lookup(X, :β)))),
            hcat(collect(lookup(X, :α)), parent(X))
        ), '\t'
    )
end

end
