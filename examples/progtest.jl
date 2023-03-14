using ProgressMeter

p1 = Progress(10,1,"a"; offset=0);

for i = 1:10
    sleep(0.1)
    next!(p1)
    
    for 
    p2 = Progress(10,1,"b"; offset=1);

    sleep(0.1)
    next!(p2)
end
