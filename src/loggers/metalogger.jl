



### MetaLogger ###########################

mutable struct MetaLogger
    loggers::Vector
    train_epoch_stats::Stats # accumulates the values during the training epoch
    val_epoch_stats::Stats # accumulates the values during the validation epoch or test epoch
    values_for_train_progressbar::Dict{String, Any}
    values_for_val_progressbar::Dict{String, Any}
    last_step_printed::Int
end

function MetaLogger(loggers)
    return MetaLogger(loggers, Stats(), Stats(), Dict{String, Any}(), Dict{String, Any}(), -1)
end

function log_step(metalogger::MetaLogger, name::AbstractString, value, fit_state)
    @unpack step, epoch = fit_state
    for logger in metalogger.loggers
        log_scalar(logger, name, value; step)
    end
    if step > metalogger.last_step_printed
        metalogger.last_step_printed = step
        for logger in metalogger.loggers
            log_scalar(logger, "epoch", epoch; step)
        end
    end
end

function accumulate_epoch!(metalogger::MetaLogger, name::AbstractString, value, stage, batchsize)
    stats = stage ∈ (:training, :train_epoch_end)  ? metalogger.train_epoch_stats : 
                                                     metalogger.val_epoch_stats

    add_obs!(stats, name, value, batchsize)
end

function log_epoch(metalogger::MetaLogger, fit_state)
    @unpack stage, step, epoch = fit_state
    stats = stage ∈ (:training, :train_epoch_end)  ? metalogger.train_epoch_stats : 
                                                     metalogger.val_epoch_stats

    for (name, value) in pairs(stats)
        for logger in metalogger.loggers
            log_scalar(logger, name, value; step)
        end
    end
    dict_stats = Dict(pairs(stats))
    clean_stats!(metalogger, stage) # TODO make sure that this is always called
    return dict_stats
end

function clean_stats!(metalogger::MetaLogger, stage)
    if stage ∈ (:training, :train_epoch_end)
        empty!(metalogger.train_epoch_stats)
    else # stage ∈ (:validation, :testing, :val_epoch_end, :test_epoch_end)
        empty!(metalogger.val_epoch_stats)
    end
end

function store_for_train_prog_bar!(metalogger::MetaLogger, name::AbstractString, value)
    metalogger.values_for_train_progressbar[name] = value
end

function store_for_val_prog_bar!(metalogger::MetaLogger, name::AbstractString, value)
    metalogger.values_for_val_progressbar[name] = value
end

function values_for_train_progbar(metalogger::MetaLogger)
    dict = metalogger.values_for_train_progressbar
    ks = sort(collect(keys(dict)))
    return [(k, roundval(dict[k])) for k in ks]
end

function values_for_val_progressbar(metalogger::MetaLogger)
    stats = metalogger.val_epoch_stats
    ks = sort(collect(keys(stats)))
    return [(k, roundval(stats[k])) for k in ks]
end

function set_run_dir!(metalogger::MetaLogger, run_dir)
    for logger in metalogger.loggers
        set_run_dir!(logger, run_dir)
    end
end

ChainRulesCore.@non_differentiable log_epoch(::Any...)
EnzymeCore.EnzymeRules.inactive_noinl(::typeof(log_epoch), args...; kws...) = nothing
ChainRulesCore.@non_differentiable log_step(::Any...)
EnzymeCore.EnzymeRules.inactive_noinl(::typeof(log_step), args...; kws...) = nothing
ChainRulesCore.@non_differentiable accumulate_epoch!(::Any...)
EnzymeCore.EnzymeRules.inactive_noinl(::typeof(accumulate_epoch!), args...; kws...) = nothing
