module Tsunami

using Base: @kwdef
import BSON
using ChainRulesCore: ChainRulesCore, @non_differentiable
using Crayons
using CUDA
using Flux
using Flux: onecold, onehotbatch, DataLoader
import Functors
# import ImageMagick # for image logging
using MLUtils
import Optimisers
# import ParameterSchedulers
# @static if Sys.isapple()
#     import QuartzImageIO # for image logging
# end
using Random
using Statistics
using TensorBoardLogger: TBLogger, tb_append
import TensorBoardLogger as TensorBoardLoggers
using UnPack: @unpack
using Zygote

CUDA.allowscalar(false)

include("ProgressMeter/ProgressMeter.jl")
using .ProgressMeter

include("utils.jl")
# export accuracy

include("stats.jl")
# export Stats

include("show.jl")

include("fluxmodule.jl")
export FluxModule
        #  train_step,
        #  val_step,
        #  test_step,
        #  predict_step,
        #  configure_optimizers

include("hooks.jl")
# export  on_before_update,
#         on_before_pullback_call,
#         on_train_epoch_start,
#         on_train_epoch_end,
#         on_val_epoch_start,
#         on_val_epoch_end
#         on_test_epoch_start,
#         on_test_epoch_end,


include("loggers/metalogger.jl") # export MetaLogger
include("loggers/tensorboard.jl") # export TensorBoardLogger


include("callbacks.jl")
export AbstractCallback

include("checkpointer.jl")
export Checkpointer, load_checkpoint

include("foil.jl")
# export Foil

include("trainer.jl")
export Trainer

include("log.jl")
# export log

include("deprecated.jl")

end # module