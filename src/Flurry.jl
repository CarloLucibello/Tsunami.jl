module Flurry

using Flux
using Flux: onecold, onehotbatch, DataLoader
import Optimisers
using Random, Statistics
using UnPack: @unpack
import Zygote
import ChainRulesCore
using Base: @kwdef
using ProgressMeter
import OnlineStats
import BSON
using Glob: glob
import Functors
using CUDA
using MLUtils
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