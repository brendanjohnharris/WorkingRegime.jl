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
