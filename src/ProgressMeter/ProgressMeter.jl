# Had to copy and past the entire ProgressMeter package here 
# since the mantainer is not responsive. See:
# https://github.com/timholy/ProgressMeter.jl/pull/261
# The changes with respect to the original package are:
# - merged PR #261
# - removed docstring so that Documenter does not complain they are not in the docs
# - removed ProgressThresh and ProgressUnknown since not needed
# - removed @showprogress
# - added the `keep` keyword argument to Progress
# - added `rewind` method
# - remove `finish!`, simplify updateProgress

module ProgressMeter

using Printf: @sprintf
using Distributed

export Progress, BarGlyphs, next!, update!, cancel, ijulia_behavior


abstract type AbstractProgress end

mutable struct BarGlyphs
    leftend::Char
    fill::Char
    front::Union{Vector{Char}, Char}
    empty::Char
    rightend::Char
end

function BarGlyphs(s::AbstractString)
    glyphs = (s...,)
    if !isa(glyphs, NTuple{5,Char})
        error("""
            Invalid string in BarGlyphs constructor.
            You supplied "$s".
            Note: string argument must be exactly 5 characters long, e.g. "[=> ]".
        """)
    end
    return BarGlyphs(glyphs...)
end

mutable struct Progress <: AbstractProgress
    n::Int
    reentrantlocker::Threads.ReentrantLock
    dt::Float64
    counter::Int
    tinit::Float64
    tsecond::Float64           # ignore the first loop given usually uncharacteristically slow
    tlast::Float64
    printed::Bool              # true if we have issued at least one status update
    desc::String               # prefix to the percentage, e.g.  "Computing..."
    barlen::Union{Int,Nothing} # progress bar size (default is available terminal width)
    barglyphs::BarGlyphs       # the characters to be used in the bar
    color::Symbol              # default to green
    output::IO                 # output stream into which the progress is written
    offset::Int                # position offset of progress bar (default is 0)
    numprintedvalues::Int      # num values printed below progress in last iteration
    start::Int                 # which iteration number to start from
    enabled::Bool              # is the output enabled
    showspeed::Bool            # should the output include average time per iteration
    check_iterations::Int
    prev_update_count::Int
    threads_used::Vector{Int}
    keep::Bool                 # whether to keep the progress bar after completion

    function Progress(n::Integer;
                      dt::Real=0.1,
                      desc::AbstractString="Progress: ",
                      color::Symbol=:green,
                      output::IO=stderr,
                      barlen=nothing,
                      barglyphs::BarGlyphs=BarGlyphs('|','█', Sys.iswindows() ? '█' : ['▏','▎','▍','▌','▋','▊','▉'],' ','|',),
                      offset::Integer=0,
                      start::Integer=0,
                      enabled::Bool = true,
                      showspeed::Bool = false,
                      keep::Bool = (offset == 0),
                     )
        CLEAR_IJULIA[] = clear_ijulia()
        reentrantlocker = Threads.ReentrantLock()
        counter = start
        tinit = tsecond = tlast = time()
        printed = false
        new(n, reentrantlocker, dt, counter, tinit, tsecond, tlast, printed, desc, barlen, barglyphs, color, output, offset, 0, start, enabled, showspeed, 1, 1, Int[], keep)
    end
end

Progress(n::Integer, dt::Real, desc::AbstractString="Progress: ",
         barlen=nothing, color::Symbol=:green, output::IO=stderr;
         offset::Integer=0) =
    Progress(n, dt=dt, desc=desc, barlen=barlen, color=color, output=output, offset=offset)

Progress(n::Integer, desc::AbstractString, offset::Integer=0) = Progress(n, desc=desc, offset=offset)


#...length of percentage and ETA string with days is 29 characters, speed string is always 14 extra characters
function tty_width(desc, output, showspeed::Bool)
    full_width = displaysize(output)[2]
    desc_width = length(desc)
    eta_width = 29
    speed_width = showspeed ? 14 : 0
    return max(0, full_width - desc_width - eta_width - speed_width)
end

# Package level behavior of IJulia clear output
@enum IJuliaBehavior IJuliaWarned IJuliaClear IJuliaAppend

const IJULIABEHAVIOR = Ref(IJuliaWarned)

function ijulia_behavior(b)
    @assert b in [:warn, :clear, :append]
    b == :warn && (IJULIABEHAVIOR[] = IJuliaWarned)
    b == :clear && (IJULIABEHAVIOR[] = IJuliaClear)
    b == :append && (IJULIABEHAVIOR[] = IJuliaAppend)
end

# Whether or not to use IJulia.clear_output
const CLEAR_IJULIA = Ref{Bool}(false)
running_ijulia_kernel() = isdefined(Main, :IJulia) && Main.IJulia.inited
clear_ijulia() = (IJULIABEHAVIOR[] != IJuliaAppend) && running_ijulia_kernel()

function calc_check_iterations(p, t)
    if t == p.tlast
        # avoid a NaN which could happen because the print time compensation makes an assumption about how long printing
        # takes, therefore it's possible (but rare) for `t == p.tlast`
        return p.check_iterations
    end
    # Adjust the number of iterations that skips time check based on how accurate the last number was
    iterations_per_dt = (p.check_iterations / (t - p.tlast)) * p.dt
    return round(Int, clamp(iterations_per_dt, 1, p.check_iterations * 10))
end

# update progress display
function updateProgress!(p::Progress; showvalues = (), 
                        truncate_lines = false, 
                        valuecolor = :blue,
                        offset::Integer = p.offset, 
                        keep = p.keep, 
                        desc::Union{Nothing,AbstractString} = nothing,
                        ignore_predictor = false, 
                        final::Union{Nothing, Bool} = nothing)
    !p.enabled && return
    if p.counter == 2 # ignore the first loop given usually uncharacteristically slow
        p.tsecond = time()
    end
    if desc !== nothing && desc !== p.desc
        if p.barlen !== nothing
            p.barlen += length(p.desc) - length(desc) #adjust bar length to accommodate new description
        end
        p.desc = desc
    end
    p.offset = offset
    p.keep = keep
    
    if final === nothing
        final = p.counter == p.n
    end
    if final || ignore_predictor || predicted_updates_per_dt_have_passed(p)
        t = time()
        if p.counter > 2
            p.check_iterations = calc_check_iterations(p, t)
        end
        if t > p.tlast+p.dt || p.counter == p.n || final
            barlen = p.barlen isa Nothing ? tty_width(p.desc, p.output, p.showspeed) : p.barlen
            percentage_complete = 100.0 * p.counter / p.n
            bar = barstring(barlen, percentage_complete, barglyphs=p.barglyphs)
            elapsed_time = t - p.tinit
            est_total_time = elapsed_time * (p.n - p.start) / (p.counter - p.start)
            if 0 <= est_total_time <= typemax(Int)
                eta_sec = round(Int, est_total_time - elapsed_time )
                eta = durationstring(eta_sec)
            else
                eta = "N/A"
            end
            spacer = endswith(p.desc, " ") ? "" : " "
            msg = @sprintf "%s%s%3u%%%s  ETA: %s" p.desc spacer round(Int, percentage_complete) bar eta
            if p.showspeed
                sec_per_iter = elapsed_time / (p.counter - p.start)
                msg = @sprintf "%s (%s)" msg speedstring(sec_per_iter)
            end
            !CLEAR_IJULIA[] && print(p.output, "\n" ^ (p.offset + p.numprintedvalues))
            move_cursor_up_while_clearing_lines(p.output, p.numprintedvalues)
            printover(p.output, msg, p.color)
            printvalues!(p, showvalues; color = valuecolor, truncate = truncate_lines)
            if !final
                !CLEAR_IJULIA[] && rewind(p)
            else 
                if keep
                    println(p.output)
                else
                    !CLEAR_IJULIA[] && rewind(p)
                end
            end
            flush(p.output)
            # Compensate for any overhead of printing. This can be
            # especially important if you're running over a slow network
            # connection.
            p.tlast = t + 2*(time()-t)
            p.printed = true
            p.prev_update_count = p.counter
        end
    end
    return nothing
end


predicted_updates_per_dt_have_passed(p::AbstractProgress) = 
    p.counter <= 2 || # otherwise the first 2 are never printed, independently of dt
    p.counter - p.prev_update_count >= p.check_iterations

function is_threading(p::AbstractProgress)
    Threads.nthreads() == 1 && return false
    length(p.threads_used) > 1 && return true
    if !in(Threads.threadid(), p.threads_used)
        push!(p.threads_used, Threads.threadid())
    end
    return length(p.threads_used) > 1
end

function lock_if_threading(f::Function, p::AbstractProgress)
    if is_threading(p)
        lock(p.reentrantlocker) do
            f()
        end
    else
        f()
    end
end

function next!(p::Union{Progress}; step::Int = 1, options...)
    lock_if_threading(p) do
        p.counter += step
        updateProgress!(p; ignore_predictor = step == 0, options...)
    end
end

function next!(p::Union{Progress}, color::Symbol; step::Int = 1, options...)
    lock_if_threading(p) do
        p.color = color
        p.counter += step
        updateProgress!(p; ignore_predictor = step == 0, options...)
    end
end


function update!(p::Union{Progress}, counter::Int=p.counter, color::Symbol=p.color; options...)
    lock_if_threading(p) do
        counter_changed = p.counter != counter
        p.counter = counter
        p.color = color
        updateProgress!(p; ignore_predictor = !counter_changed, options...)
    end
end


function cancel(p::AbstractProgress, msg::AbstractString = "Aborted before all tasks were completed", color = :red; 
            showvalues = (), truncate_lines = false, valuecolor = :blue, 
            offset = p.offset, keep = p.keep)
    lock_if_threading(p) do
        p.offset = offset
        p.keep = keep
        if p.printed
            print(p.output, "\n" ^ (p.offset + p.numprintedvalues))
            move_cursor_up_while_clearing_lines(p.output, p.numprintedvalues)
            printover(p.output, msg, color)
            printvalues!(p, showvalues; color = valuecolor, truncate = truncate_lines)
            if p.keep
                println(p.output)
            else
                print(p.output, "\r\u1b[A" ^ (p.offset + p.numprintedvalues))
            end
        end
    end
    return
end


# Internal method to print additional values below progress bar
function printvalues!(p::AbstractProgress, showvalues; color = :normal, truncate = false)
    length(showvalues) == 0 && return
    maxwidth = maximum(Int[length(string(name)) for (name, _) in showvalues])

    p.numprintedvalues = 0

    for (name, value) in showvalues
        msg = "\n  " * rpad(string(name) * ": ", maxwidth+2+1) * string(value)
        max_len = (displaysize(p.output)::Tuple{Int,Int})[2]
        # I don't understand why the minus 1 is necessary here, but empircally
        # it is needed.
        msg_lines = ceil(Int, (length(msg)-1) / max_len)
        if truncate && msg_lines >= 2
            # For multibyte characters, need to index with nextind.
            printover(p.output, msg[1:nextind(msg, 1, max_len-1)] * "…", color)
            p.numprintedvalues += 1
        else
            printover(p.output, msg, color)
            p.numprintedvalues += msg_lines
        end
    end
    p
end

# Internal method to print additional values below progress bar (lazy-showvalues version)
printvalues!(p::AbstractProgress, showvalues::Function; kwargs...) = printvalues!(p, showvalues(); kwargs...)

function move_cursor_up_while_clearing_lines(io, numlinesup)
    if numlinesup > 0 && CLEAR_IJULIA[]
        Main.IJulia.clear_output(true)
        if IJULIABEHAVIOR[] == IJuliaWarned
            @warn "ProgressMeter by default refresh meters with additional information in IJulia via `IJulia.clear_output`, which clears all outputs in the cell. \n - To prevent this behaviour, do `ProgressMeter.ijulia_behavior(:append)`. \n - To disable this warning message, do `ProgressMeter.ijulia_behavior(:clear)`."
        end
    else
        for _ in 1:numlinesup
            print(io, "\r\u1b[K\u1b[A")
        end
    end
end

function printover(io::IO, s::AbstractString, color::Symbol = :color_normal)
    print(io, "\r")
    printstyled(io, s; color=color)
    if isdefined(Main, :IJulia)
        Main.IJulia.stdio_bytes[] = 0 # issue #76: circumvent IJulia I/O throttling
    elseif isdefined(Main, :ESS) || isdefined(Main, :Atom)
    else
        print(io, "\u1b[K")     # clear the rest of the line
    end
end

function compute_front(barglyphs::BarGlyphs, frac_solid::AbstractFloat)
    barglyphs.front isa Char && return barglyphs.front
    idx = round(Int, frac_solid * (length(barglyphs.front) + 1))
    return idx > length(barglyphs.front) ? barglyphs.fill :
           idx == 0 ? barglyphs.empty :
           barglyphs.front[idx]
end

function barstring(barlen, percentage_complete; barglyphs)
    bar = ""
    if barlen>0
        if percentage_complete == 100 # if we're done, don't use the "front" character
            bar = string(barglyphs.leftend, repeat(string(barglyphs.fill), barlen), barglyphs.rightend)
        else
            n_bars = barlen * percentage_complete / 100
            nsolid = trunc(Int, n_bars)
            frac_solid = n_bars - nsolid
            nempty = barlen - nsolid - 1
            bar = string(barglyphs.leftend,
                         repeat(string(barglyphs.fill), max(0,nsolid)),
                         compute_front(barglyphs, frac_solid),
                         repeat(string(barglyphs.empty), max(0, nempty)),
                         barglyphs.rightend)
        end
    end
    bar
end

function durationstring(nsec)
    days = div(nsec, 60*60*24)
    r = nsec - 60*60*24*days
    hours = div(r,60*60)
    r = r - 60*60*hours
    minutes = div(r, 60)
    seconds = floor(r - 60*minutes)

    hhmmss = @sprintf "%u:%02u:%02u" hours minutes seconds
    if days>9
        return @sprintf "%.2f days" nsec/(60*60*24)
    elseif days>0
        return @sprintf "%u days, %s" days hhmmss
    end
    hhmmss
end

function speedstring(sec_per_iter)
    if sec_per_iter == Inf
        return "  N/A  s/it"
    end
    ns_per_iter = 1_000_000_000 * sec_per_iter
    for (divideby, unit) in (
        (1, "ns"),
        (1_000, "μs"),
        (1_000_000, "ms"),
        (1_000_000_000, "s"),
        (60 * 1_000_000_000, "m"),
        (60 * 60 * 1_000_000_000, "hr"),
        (24 * 60 * 60 * 1_000_000_000, "d")
    )
        if round(ns_per_iter / divideby) < 100
            return @sprintf "%5.2f %2s/it" (ns_per_iter / divideby) unit
        end
    end
    return " >100  d/it"
end

"""
    rewind(p::AbstractProgress)

Rewinds the cursor to the beginning of the progress bar.
"""
function rewind(p::AbstractProgress)
    print(p.output, "\r\u1b[A" ^ (p.offset + p.numprintedvalues))
end

end # module