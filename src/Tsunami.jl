module Tsunami

using Base: @kwdef, PkgId, UUID
using BSON: BSON
using ChainRulesCore: ChainRulesCore, @non_differentiable
using Compat: @compat
using Crayons
using Flux
using Flux: onecold, onehotbatch, DataLoader
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

include("utils.jl")
@compat(public, accuracy)

include("stats.jl")
# export Stats

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
                 on_before_backprop,
                 on_train_epoch_start,
                 on_train_epoch_end,
                 on_val_epoch_start,
                 on_val_epoch_end,
                 on_test_epoch_start,
                 on_test_epoch_end,
                ))


include("loggers/metalogger.jl") # export MetaLogger

include("loggers/tensorboard.jl")
@compat(public, (TensorBoardLogger,))

include("callbacks.jl")
export AbstractCallback

include("checkpointer.jl")
export Checkpointer, load_checkpoint

include("foil.jl")
export Foil
@compat(public, (setup, setup_batch))

include("trainer.jl")
export Trainer

include("log.jl")
@compat(public, (log,))

include("deprecated.jl")

end # module