module TsunamiEnzymeExt

using Tsunami: Tsunami, train_step, Trainer
using Enzyme
using Functors: Functors
using Optimisers: Optimisers

# function Tsunami.pullback_train_step(model::Duplicated,  trainer::Trainer, batch, batch_idx::Int)
#     make_zero!(model.dval)
#     ad = Enzyme.set_runtime_activity(Enzyme.ReverseSplitWithPrimal)
#     args = (model, Const(trainer), Const(batch), Const(batch_idx))
#     forward, reverse = autodiff_thunk(ad, Const{typeof(train_step)}, Active, map(typeof, args)...)
#     tape, loss, _  = forward(Const(train_step), args...)
#     pb = () -> reverse(Const(train_step), args..., one(loss), tape)
#     return loss, pb
# end

function Tsunami.pullback_train_step(model::Duplicated, trainer::Trainer, batch, batch_idx::Int)
    make_zero!(model.dval)
    ad = Enzyme.set_runtime_activity(ReverseWithPrimal)
    args = (model, Const(trainer), Const(batch), Const(batch_idx))
    ret = Enzyme.autodiff(ad, Const(train_step), Active, args...)
    pb = () -> model.dval
    return ret[2], pb
end

function Tsunami.gradient_train_step(model::Duplicated, trainer::Trainer, batch, batch_idx::Int)
    make_zero!(model.dval)
    ad = Enzyme.set_runtime_activity(ReverseWithPrimal)
    args = (model, Const(trainer), Const(batch), Const(batch_idx))
    ret = Enzyme.autodiff(ad, Const(train_step), Active, args...)
    return ret[2], model.dval
end

# We can't use Enzyme.make_zero! to reset Duplicated, as it complains about e.g. LayerNorm having immutable differentiable fields
make_zero!(model) = Functors.fmapstructure(make_zero_inner!, model)

function make_zero_inner!(x::AbstractArray{<:Number})
  Optimisers.isnumeric(x) || return
  Optimisers.maywrite(x) || error("can't handle this")
  fill!(x, zero(eltype(x)))
  nothing
end

make_zero_inner!(x) = nothing  # any other Functors leaf type

end # module