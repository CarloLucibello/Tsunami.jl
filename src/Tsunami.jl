module Tsunami

using Base: @kwdef
import BSON
import ChainRulesCore
using CUDA
using Flux
using Flux: onecold, onehotbatch, DataLoader
import Functors
using Dates
# import ImageMagick # for image logging
using Logging
using ProgressMeter
import OnlineStats
import Optimisers
# import ParameterSchedulers
# @static if Sys.isapple()
#     import QuartzImageIO # for image logging
# end
using Random
using Statistics
using TensorBoardLogger: TensorBoardLogger, TBLogger, tb_append
using UnPack: @unpack
import Zygote
using Crayons

CUDA.allowscalar(false)

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

include("callbacks.jl")
export AbstractCallback

include("checkpointer.jl")
export Checkpointer, load_checkpoint

end # module