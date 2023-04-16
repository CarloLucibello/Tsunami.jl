function fluxshow(io::IO, m::MIME"text/plain", x)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        Flux._big_show(io, x)
    elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        Flux._layer_show(io, x)
    else
        show(io, x)
    end
end

function container_show(io::IO, m::T; exclude=[]) where T
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
        compact_show(io, v)
    end
end

function compact_show(io::IO, x)
    show(IOContext(io, :limit => true, :compact => true), x)
end