using Tsunami.ProgressMeter
function test()
    r = 1:10
    p = Progress(length(r))
    for i in r
        sleep(0.1)
        i == 5 && break
        next!(p)
    end;
    ProgressMeter.finish!(p)
    println("done!")
end

test()
