mutable struct GradScaler
    init_scale
    growth_factor
    backoff_factor
    growth_interval::Int
    scale
    growth_tracker
    init_growth_tracker
end

function GradScaler(
            init_scale = 2.0^16,
            growth_factor = 2.0,
            backoff_factor = 0.5,
            growth_interval = 2000)

    @assert growth_factor > 1.0 "The growth factor must be > 1.0."
    @assert backoff_factor < 1.0 "The backoff factor must be < 1.0."
    init_growth_tracker = 0
    growth_tracker = nothing
    scale = nothing
    return GradScaler(init_scale, growth_factor, backoff_factor, growth_interval, scale, growth_tracker, init_growth_tracker)
end

function scale(scaler::GradScaler, x)
    if scaler.scale === nothing
        scaler.scale = scaler.init_scale
        scaler.growth_tracker = scaler.init_growth_tracker
    end
    fmap()
    return x .* scaler.scale
end

unscale_grad(grad, scale)

