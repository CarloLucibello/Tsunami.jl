module Tsunami

using Base: @kwdef
import BSON
using ChainRulesCore: ChainRulesCore, @non_differentiable
using Crayons
using CUDA
using Flux
using Flux: onecold, onehotbatch, DataLoader
import Functors
using Dates
# import ImageMagick # for image logging
using Logging
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
        #  training_step,
        #  validation_step,
        #  test_step,
        #  predict_step,
        #  configure_optimizers


include("trainer.jl")
export Trainer

include("logging.jl")
include("loggers/tensorboard.jl")

include("callbacks.jl")
export AbstractCallback

include("checkpointer.jl")
export Checkpointer, load_checkpoint

include("deprecated.jl")

end # module