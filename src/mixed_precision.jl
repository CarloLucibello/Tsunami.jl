"""
    GradScaler(; init_scale=2.0f0^16, growth_factor=2.0f0, 
               backoff_factor=0.5f0, growth_interval=2000, min_scale=1.0f0)

A gradient scaler for mixed precision training, similar to `torch.cuda.amp.GradScaler`.

Dynamically scales the loss to prevent gradient underflow when using lower precision
(e.g., Float16). The scale is adjusted based on whether inf/nan values are detected
in the gradients.

# Features
- `scale(loss)`: Scale the loss by the current scale factor
- `unscale!(model)`: Unscale gradients in the model
- `update!()`: Update the scale based on overflow detection
- Dynamic loss scale with growth/backoff

# Arguments
- `init_scale`: Initial scale factor (default: 2^16)
- `growth_factor`: Multiply scale by this when no overflow (default: 2.0)
- `backoff_factor`: Multiply scale by this on overflow (default: 0.5)
- `growth_interval`: Number of steps without overflow before growing scale (default: 2000)
- `min_scale`: Minimum allowed scale (default: 1.0)

# Example
```julia
using Zygote

scaler = GradScaler()
for batch in dataloader
    loss, grads = Zygote.withgradient(model) do m
        out = compute_loss(m, batch)
        scaler.scale(out)  # Scale the loss
    end
    
    if !unscale!(scaler, grads[1])
        # Skip this step due to inf/nan
        scaler.update!()
        continue
    end
    
    Optimisers.update!(opt, model, grads[1])
    scaler.update!()
end
```
"""
@kwdef mutable struct GradScaler
    scale::Float32 = 2.0f0^16
    growth_factor::Float32 = 2.0f0
    backoff_factor::Float32 = 0.5f0
    growth_interval::Int = 2000
    min_scale::Float32 = 1.0f0
    # bookkeeping
    growth_tracker::Int = 0
    found_inf::Bool = false
end

""" 
    scale(scaler::GradScaler, loss)

Scale the loss by the current scale factor.
"""
function scale(scaler::GradScaler, loss::AbstractFloat)
    if loss isa Float16
        @warn """Scaling Float16 loss. When using mixed precision training, it's recommended \
        to compute the loss in Float32 precision since reductions like sum or mean can cause \
        numerical instability in Float16.
        
        Here's an example for a classification task:
        ```julia
        logits = model(x)                                  # x and logits in Float16
        loss = Flux.logitcrossentropy(Float32.(logits), y) # loss in Float32
        ```
        """
    end
    return loss * scaler.scale
end

function _has_inf_or_nan(grads)
    found = false
    Functors.fmap(grads) do x
        if Optimisers.isnumeric(x) && !all(isfinite, x)
            found = true
        end
        return x
    end
    return found
end

"""
    unscale!(scaler::GradScaler, grads)

Unscale gradients by dividing by the current scale.
Returns `false` if inf/nan detected, `true` otherwise.
"""
function unscale!(scaler::GradScaler, grads)
    # Check for inf/nan first
    scaler.found_inf = _has_inf_or_nan(grads)
    
    scaler.found_inf && return false
    
    # Unscale gradients in-place
    inv_scale = 1.0f0 / scaler.scale
    Functors.fmap!(grads) do x
        if Optimisers.isnumeric(x)
            x .*= inv_scale
        end
        return x
    end
    return true
end


function Optimisers.update!(scaler::GradScaler, opt_state, model, grads)
    scaler.found_inf = unscale!(scaler, grads)

    if scaler.found_inf
        # Overflow detected: decrease scale
        new_scale = max(scaler.scale * scaler.backoff_factor, scaler.min_scale)
        scaler.scale = new_scale
        scaler.growth_tracker = 0
    else
        opt_state, model = Optimisers.update!(opt_state, model, grads)

        # No overflow: maybe grow scale
        scaler.growth_tracker += 1
        if scaler.growth_tracker >= scaler.growth_interval
            scaler.scale *= scaler.growth_factor
            scaler.growth_tracker = 0
        end
    end

    return opt_state, model
end
