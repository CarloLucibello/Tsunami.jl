# using TestItemRunner
using Test
using ParallelTestRunner
using Tsunami

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

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

args = parse_args(ARGS)

if filter_tests!(testsuite, args)

    delete!(testsuite, "test_module")
    if !TEST_ENZYME
        delete!(testsuite, "ext_enzyme")
    end
    for test in keys(testsuite)
        if !TEST_GPU
            startswith(test, "gpu/") && delete!(testsuite, test)
        end
        if !TEST_CPU
            !startswith(test, "gpu/") && delete!(testsuite, test)
        end
    end
end

const init_code = quote
    include(joinpath(@__DIR__, "test_module.jl"))
end

runtests(Tsunami, args; testsuite, init_code)

# @testset "Examples" begin
#    include(joinpath(@__DIR__, "..", "examples/MLP_MNIST/mlp_mnist.jl"))
# end
