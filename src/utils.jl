	
function accuracy(dataset, m)
    num = sum(sum(onecold(m(x)) .== onecold(y)) for (x,y) in dataset)
	den = sum(size(x, ndims(x)) for (x,y) in dataset)
	# @show dataset
	# @show typeof(dataset) length(dataset)
	# sum(size(x, ndims(x)) for (x,y) in dataset)
    return num / den
end

accuracy(ŷ::AbstractMatrix, y::AbstractVector) = mean(onecold(ŷ) .== y)
accuracy(ŷ::AbstractMatrix, y::AbstractMatrix) = mean(onecold(ŷ) .== onecold(y))

ChainRulesCore.@non_differentiable accuracy(::Any...)

round4(x::Float64) = round(x, digits=4)
round4(x::AbstractFloat) = round4(Float64(x))
round4(x::Int) = x
round4(x::NamedTuple) = map(round4, x)