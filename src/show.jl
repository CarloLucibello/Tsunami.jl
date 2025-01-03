

function shortshow(io::IO, x::T) where T
    # Tstripped = Base.typename(T).wrapper # remove all parameters. From https://discourse.julialang.org/t/stripping-parameter-from-parametric-types/8293/16
    str = string(T.name.name)
    print(io, str * "()")
end

container_show(m) = container_show(stdout, m)

function container_show(io::IO, m::T; exclude=[], brief=[]) where T
    Tname = compact_typename(T)
    if get(io, :compact, false)
        return print(io, "$Tname()")
    end
    print(io, "$Tname:")
    for f in sort(fieldnames(T) |> collect)
        startswith(string(f), "_") && continue
        f in exclude && continue
        v = getfield(m, f)
        print(io, "\n  $f = ")
        if f in brief
            print(io, "...")
        else
            compact_show(io, v)
        end
    end
end

function compact_show(io::IO, x)
    show(IOContext(io, :limit => true, :compact => true), x)
end

# """
#     compact_typename(x::T) -> String
#     compact_typename(T) -> String

# Return a compact string representation of the type `T` of `x`.
# Keep only the name and `T`'s parameters, discarding their own parameters.
# """
compact_typename(x::T) where T = compact_typename(T)

function compact_typename(T::DataType)
    name = T.name.name
    params = T.parameters
    pnames = map(S -> S.name.name, params)
    if isempty(pnames)
        str = "$name"
    else
        str = "$(name){$(join(pnames, ", "))}"
    end
    return str
end
