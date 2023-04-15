using Tsunami.ProgressMeter

@testset "baseline" begin
    r = 1:10
    p = Progress(length(r))
    for i in r
        sleep(0.2)
        next!(p)
    end
    println("done!")
end

@testset "loops ends earlier" begin
    r = 1:10
    p = Progress(length(r))
    for i in r
        sleep(0.2)
        final = i == 5
        next!(p; final)
        final && break
    end
    println("done!")
end

@testset "unknown length" begin
    r = 1:10
    p = Progress()
    for i in r
        sleep(0.2)
        next!(p; final = i == length(r))
    end
    println("done!")
end

@testset "showvalues" begin
    r = 1:10
    p = Progress()
    for i in r
        sleep(0.2)
        next!(p, showvalues = [("i", i), ("a", 1)], final = i == length(r))
    end
    println("done!")
end