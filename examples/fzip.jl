using Functors, StructWalk

struct FunctorStyle <: StructWalk.WalkStyle end
# StructWalk.children(::FunctorStyle, x) = () # every object is a leaf by default (with few exceptions: tuples, arrays, etc.) 
StructWalk.children(::FunctorStyle, x::AbstractArray{<:Number}) = ()

StructWalk.scan(identity, FunctorStyle(), model) do x
    @show typeof(x)
end