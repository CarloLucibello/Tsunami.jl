module Flurry

using Base: @kwdef
import BSON
import ChainRulesCore
using CUDA
using Flux
using Flux: onecold, onehotbatch, DataLoader
import Functors
using Glob: glob
using Dates
import ImageMagick # for image logging
using Logging
using MLUtils
using ProgressMeter
import OnlineStats
import Optimisers
@static if Sys.isapple()
    import QuartzImageIO # for image logging
end
using Random
using Statistics
using TensorBoardLogger: TBLogger, tb_append
using UnPack: @unpack
import Zygote

CUDA.allowscalar(false)

include("utils.jl")
export accuracy

include("stats.jl")
export Stats


include("fluxmodule.jl")
export FluxModule#,
        #  training_step,
        #  validation_step,
        #  test_step,
        #  predict_step,
        #  training_epoch_end,
        #  validation_epoch_end,
        #  test_epoch_end,
        #  predict_epoch_end,
        #  configure_optimizers

include("checkpointer.jl")
export Checkpointer

include("trainer.jl")
export Trainer

end # module