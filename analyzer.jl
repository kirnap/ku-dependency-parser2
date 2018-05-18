# To analyze the result file
function getlas(ARGS)
    fname = ARGS[1] # filename
    trains = Any[]; devs = Any[];
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
    end
    trmax, devmax = findmax(trains), findmax(devs)
    println("tr$(trmax[2]) $(trmax[1]) | dev$(devmax[2]) $(devmax[1])")
end
!isinteractive() && getlas(ARGS)
