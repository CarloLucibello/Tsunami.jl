
@kwdef mutable struct Stats
    num::Dict{String, Int} = Dict{String, Int}()
    sum::Dict{String, Float64} = Dict{String, Float64}()
    last::Dict{String, Number} = Dict{String, Number}()
    mvavg::Dict{String, Float64} = Dict{String, Float64}()
    mvavg_inertia::Float64 = 0.9
end

function init_stat!(s::Stats, k::String, value = NaN)
    s.num[k] = 0
    s.sum[k] = 0.0
    s.mvavg[k] = value
    s.last[k] = value
    return s
end

function add_obs!(s::Stats, name::String, value::Number, batchsize::Int=1)
    if !haskey(s.num, name)
        init_stat!(s, name, value)
    end
    s.num[name] += batchsize
    s.sum[name] += value * batchsize
    s.last[name] = value
    s.mvavg[name] = s.mvavg_inertia * s.mvavg[name] + (1 - s.mvavg_inertia) * value
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
    empty!(s.mvavg)
    empty!(s.last)
end
