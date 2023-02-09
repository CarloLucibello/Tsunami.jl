
"""
    Stats() <: OnlineStat
"""
mutable struct Stats <: OnlineStats.OnlineStat{NamedTuple}
    _stats::NamedTuple
end

Stats() = Stats((;))

Base.getindex(s::Stats, k::Symbol) = getindex(s._stats, k)
Base.keys(s::Stats) = keys(s._stats)
Base.pairs(s::Stats) = pairs(s._stats)
Base.setindex!(s::Stats, v, k::Symbol) = setindex!(s._stats, v, k)
Base.haskey(s::Stats, k::Symbol) = haskey(s._stats, k)
Base.length(s::Stats) = length(s._stats)
Base.values(s::Stats) = values(s._stats)
Base.isempty(s::Stats) = isempty(s._stats)

function Base.getproperty(s::Stats, k::Symbol)
    if hasfield(Stats, k)
        return getfield(s, k)
    else
        x = getfield(s, :_stats)[k].stats
        return OnlineStats.value(x)
    end
end

function OnlineStats._fit!(s::Stats, x::NamedTuple)
    if isempty(s)
        init_stat!(s, x)
    end
    for (k, v) in pairs(x)
        OnlineStats.fit!(s[k], v)
    end
end

function init_stat!(s, x::NamedTuple)
    stat() = OnlineStats.Mean(weight=OnlineStats.ExponentialWeight(0.1))
    # stat() = OnlineStats.Mean(weight=OnlineStats.ExponentialWeight(0.1))
    s._stats = map(x -> stat(), x)
    return s
end

OnlineStats.nobs(s::Stats) = length(s._stats) > 0 ? OnlineStats.nobs(first(s._stats)) : 0

OnlineStats.value(s::Stats) = map(x -> OnlineStats.value(x), s._stats)
