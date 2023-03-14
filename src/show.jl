
function tsunami_big_show(io::IO, obj, indent::Int=0, name=nothing)
    pre, post = obj isa Chain{<:AbstractVector} ? ("([", "])") : ("(", ")")
    children = Flux._show_children(obj)
    if all(Flux._show_leaflike, children)
        Flux._layer_show(io, obj, indent, name)
    else
        println(io, isnothing(name) ? "" : "$name = ", nameof(typeof(obj)), pre)
        if obj isa Chain{<:NamedTuple} && children == getfield(obj, :layers)
            # then we insert names -- can this be done more generically? 
            for k in Base.keys(obj)
                Flux._big_show(io, obj[k], indent+2, k)
            end
        elseif obj isa Parallel{<:Any, <:NamedTuple} || obj isa PairwiseFusion{<:Any, <:NamedTuple}
            Flux._big_show(io, obj.connection, indent+2)
            for k in Base.keys(obj)
                Flux._big_show(io, obj[k], indent+2, k)
            end
        else
            for c in children
                Flux._big_show(io, c, indent+2)
            end
        end
        # if indent == 0  # i.e. this is the outermost container
            print(io, " "^indent, rpad(post, 2))
            Flux._big_finale(io, obj)
        # else
        #     println(io, " "^indent, post, ",")
        # end
    end
end

function container_show(io::IO, m::T; exclude=[]) where T
    if get(io, :compact, false)
        return print(io, "$T()")
    end
    print(io, "$T:")
    for f in sort(fieldnames(T) |> collect)
        startswith(string(f), "_") && continue
        f in exclude && continue
        v = getfield(m, f)
        print(io, "\n  $f = ")
        show(IOContext(io, :compact=>true), v)
    end
end

function compact_show(io::IO, x)
    show(IOContext(stdout, :limit => true, :compact => true), x)
end