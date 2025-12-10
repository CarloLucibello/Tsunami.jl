using TestItemRunner
using Test

## Uncomment below and in test_module.jl to change the default test settings
# ENV["TSUNAMI_TEST_CPU"] = "false"
# ENV["TSUNAMI_TEST_CUDA"] = "true"
# ENV["TSUNAMI_TEST_AMDGPU"] = "true"
# ENV["TSUNAMI_TEST_Metal"] = "true"

TEST_CPU = get(ENV, "TSUNAMI_TEST_CPU", "true") == "true"
TEST_GPU = get(ENV, "TSUNAMI_TEST_CUDA", "false") == "true" ||
           get(ENV, "TSUNAMI_TEST_AMDGPU", "false") == "true" ||
           get(ENV, "TSUNAMI_TEST_Metal", "false") == "true"
TEST_ENZYME = VERSION < v"1.12" # Enzyme.jl only supports up to Julia 1.11 as of December 2025

if TEST_CPU
    @run_package_tests filter = ti -> :gpu ∉ ti.tags && :enzyme ∉ ti.tags
end
if TEST_GPU
    @run_package_tests filter = ti -> :gpu ∈ ti.tags && :enzyme ∉ ti.tags
end
if TEST_ENZYME
    @run_package_tests filter = ti -> :enzyme ∈ ti.tags && :gpu ∉ ti.tags
    if TEST_GPU
        @run_package_tests filter = ti -> :enzyme ∈ ti.tags && :gpu ∈ ti.tags
    end
end

@testset "Examples" begin
   include(joinpath(@__DIR__, "..", "examples/MLP_MNIST/mlp_mnist.jl"))
end
