
mutable struct Stats
    num::Dict{String, Int}
    sum::Dict{String, Float64}
end

Stats() = Stats(Dict{String, Int}(), Dict{String, Float64}())

function init_stat!(s::Stats, k::String)
    s.num[k] = 0
    s.sum[k] = 0.0
    return s
end

function add_obs!(s::Stats, name::String, value::Number, batchsize::Int=1)
    if !haskey(s.num, name)
        init_stat!(s, name)
    end
    s.num[name] += batchsize
    s.sum[name] += value * batchsize
end

function Base.getindex(s::Stats, k::String)
    return s.sum[k] / s.num[k]
end

Base.haskey(s::Stats, k::String) = haskey(s.num, k)
Base.length(s::Stats) = length(s.num)
Base.keys(s::Stats) = keys(s.num)
Base.values(s::Stats) = (s[k] for k in keys(s))
Base.pairs(s::Stats) = (k => s[k] for k in keys(s))
Base.isempty(s::Stats) = isempty(s.num)

function Statistics.mean(s::Stats)
    return Dict(k => s[k] for (k, v) in pairs(s))
end

function Base.empty!(s::Stats)
    empty!(s.num)
    empty!(s.sum)
end
