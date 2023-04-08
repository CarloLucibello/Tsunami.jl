"""
    accuracy(ŷ::AbstractMatrix, y)

Compute the classification accuracy of a batch of predictions `ŷ` against true labels `y`.
`y` can be either a vector or a matrix. 
If `y` is a vector, it is assumed that the labels are integers in the range `1:K` 
where `K == size(ŷ, 1)` is the number of classes.
"""
accuracy(ŷ::AbstractMatrix, y::AbstractVector) = mean(onecold(ŷ) .== y)
accuracy(ŷ::AbstractMatrix, y::AbstractMatrix) = mean(onecold(ŷ) .== onecold(y))


# function accuracy(dataset, m)
#     num = sum(sum(onecold(m(x)) .== onecold(y)) for (x,y) in dataset)
# 	den = sum(size(x, ndims(x)) for (x,y) in dataset)
# 	# @show dataset
# 	# @show typeof(dataset) length(dataset)
# 	# sum(size(x, ndims(x)) for (x,y) in dataset)
#     return num / den
# end


@non_differentiable accuracy(::Any...)

roundval(x::Float64) = round(x, sigdigits=3)
roundval(x::AbstractFloat) = roundval(Float64(x))
roundval(x::Int) = x
roundval(x::NamedTuple) = map(roundval, x)


"""
    dir_with_version(dir::String)

Append a version number to `dir`.
"""
function dir_with_version(dir)
    i = 1
    outdir = dir * "_$i"
    while isdir(outdir)
        i += 1
		outdir = dir * "_$i"
    end
    return outdir
end

"""
    seed!(seed::Int)

Seed the RNGs of both CPU and GPU.
"""
function seed!(seed::Int)
    Random.seed!(seed)
    if CUDA.functional()
        CUDA.seed!(seed)
    end
end


unref(x::Ref) = x[]
unref(x) = x
