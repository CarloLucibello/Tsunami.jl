
"""
    Stats() <: OnlineStat
"""
mutable struct Stats <: OnlineStats.OnlineStat{AbstractDict}
    _stats::Dict{String, OnlineStats.OnlineStat}
end

Stats() = Stats(Dict{String, OnlineStats.OnlineStat}())

Base.getindex(s::Stats, k::String) = getindex(s._stats, k)
Base.keys(s::Stats) = keys(s._stats)
Base.pairs(s::Stats) = pairs(s._stats)
Base.setindex!(s::Stats, v, k::String) = setindex!(s._stats, v, k)
Base.haskey(s::Stats, k::String) = haskey(s._stats, k)
Base.length(s::Stats) = length(s._stats)
Base.values(s::Stats) = values(s._stats)
Base.isempty(s::Stats) = isempty(s._stats)

function OnlineStats._fit!(s::Stats, x::AbstractDict)
    for (k, v) in pairs(x)
        if !haskey(s, k)
            init_stat!(s, k)
        end
        OnlineStats.fit!(s[k], v)
    end
end

function init_stat!(s::Stats, k::String)
    # stat() = OnlineStats.Mean(weight=OnlineStats.ExponentialWeight(0.1))
    s[k] = OnlineStats.Mean()
    return s
end

function OnlineStats.nobs(s::Stats)
    if length(s) > 0 
        ns = [OnlineStats.nobs(v) for v in values(s)]
        if allequal(ns)
            return ns[1]
        else
            return Dict(k => OnlineStats.nobs(v) for (k,v) in pairs(s))
        end
    else 
        return 0
    end
end

OnlineStats.value(s::Stats) = Dict(k => OnlineStats.value(v) for (k, v) in pairs(s))

# HAD TO OVERLOAD PRIVATE METHOD TO WORK WITH 
# vector return from nobs
function OnlineStats.OnlineStatsBase.nobs_string(o::Stats)
    n = string(OnlineStats.nobs(o))
    return n
end

