# To analyze the result file
function getlas(ARGS)
    fname = ARGS[1] # filename
    trains = Any[]; devs = Any[]; isdone=false;
    for line in eachline(fname)
        if contains(line, "acc")
            toparse = split(line)
            if length(toparse) < 9
                d0 = parse(toparse[end]); t0 = parse(toparse[end-2])
                if length(ARGS) > 1
                    println("Initial tracc $t0 devacc $d0")
                end
            else
                tracc = parse(toparse[end-3]); push!(trains, tracc)
                devacc = parse(toparse[end]); push!(devs, devacc)
            end
        end
        contains(line, "done") && (isdone=true)
    end
    if isempty(trains)
        println("No enough epoch"); return;
    end
    trmax, devmax = findmax(trains), findmax(devs)
    if isdone
        println("tr$(trmax[2]) $(trmax[1]) | dev$(devmax[2]) $(devmax[1]) DONE.")
    else
        println("tr$(trmax[2]) $(trmax[1]) | dev$(devmax[2]) $(devmax[1]) continued...")
    end
end
!isinteractive() && getlas(ARGS)
