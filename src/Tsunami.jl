module Tsunami

using Base: @kwdef, PkgId, UUID
using BFloat16s: BFloat16
using BSON: BSON
using ChainRulesCore: ChainRulesCore
using Compat: @compat
using Crayons
using EnzymeCore: EnzymeCore
using Flux
using Flux: onecold, onehotbatch, DataLoader
using Functors: Functors
# import ImageMagick # for image logging
using JLD2: JLD2
using MLUtils
using MLDataDevices: get_device, gpu_device, cpu_device, 
                 CPUDevice, AMDGPUDevice, CUDADevice, MetalDevice
using Optimisers: Optimisers, trainable
# @static if Sys.isapple()
#     import QuartzImageIO # for image logging
# end
using Random
using Statistics
using TensorBoardLogger: TBLogger, tb_append
import TensorBoardLogger as TensorBoardLoggers
using UnPack: @unpack
using Zygote


include("ProgressMeter/ProgressMeter.jl")
using .ProgressMeter

include("stats.jl")
# export Stats

include("loggers/metalogger.jl") # export MetaLogger

include("loggers/tensorboard.jl")
@compat(public, (TensorBoardLogger,))

include("foil.jl")
export Foil
@compat(public, (setup,))

include("trainer.jl")
export Trainer
@compat(public, (FitState,))

include("utils.jl")
@compat(public, accuracy)

include("fluxmodule.jl")
export FluxModule
@compat(public, (train_step,
                 val_step,
                 test_step,
                 predict_step,
                 configure_optimisers,
                ))

include("show.jl")

include("hooks.jl")
@compat(public, (on_before_update,
                 on_train_epoch_start,
                 on_train_epoch_end,
                 on_val_epoch_start,
                 on_val_epoch_end,
                 on_test_epoch_start,
                 on_test_epoch_end,
                ))

include("callbacks.jl")
export AbstractCallback

include("checkpointer.jl")
export Checkpointer, load_checkpoint

include("fit.jl")
@compat(public, (fit!,))

include("log.jl")
@compat(public, (log,))

include("deprecated.jl")

end # module