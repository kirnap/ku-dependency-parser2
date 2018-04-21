# Put all the old model related

# model initialization and parameter selections
postagv(m)=m[1]; deprelv(m)=m[2]; lcountv(m)=m[3]; rcountv(m)=m[4]; distancev(m)=m[5]; parserv(m)=m[6];


function makepmodel1(d; GPUFEATS=false)
    m = ([d["postagv"],d["deprelv"],d["lcountv"],d["rcountv"],d["distancev"],d["parserv"]],
         [d["postago"],d["deprelo"],d["lcounto"],d["rcounto"],d["distanceo"],d["parsero"]])

    if gpu() >= 0
        if GPUFEATS
            return map2gpu(m)
        else
            m = map2cpu(m)
            m[1][6] = map2gpu(m[1][6])
            m[2][6] = map2gpu(m[2][6])
            return m
        end
    else
        return map2cpu(m)
    end
end


function makepmodel2(o, s; ftype=Float32, intype=:normal)
    model = Any[]
    dpostag, ddeprel, dcount = o[:embed]
    for (k,n,d) in ((:postag,17,dpostag),(:deprel,37,ddeprel),(:lcount,10,dcount),(:rcount,10,dcount),(:distance,10,dcount))
        if intype == :normal
            push!(model, [ initr(d) for i=1:n ])
            #@msg "Random normal initialization" # dbg purposes
        else
            push!(model, [ initzeros(d) for i=1:n ])
            #@msg "Zero initialization" # dbg purposes
        end
    end
    p = o[:arctype](s)
    f = features([p], o[:feats], model)
    mlpdims = (length(f), o[:hidden]..., p.nmove)
    info("mlpdims=$mlpdims")
    parser = Any[]
    for i=2:length(mlpdims)
        push!(parser, initx(mlpdims[i],mlpdims[i-1]))
        push!(parser, initx(mlpdims[i],1))
    end
    push!(model,parser)
    optim = optimizers(model,o[:optimization])
    return model,optim
end


function splitmodel(pmodel)
    mlpmodel = Any[]
    mlptemp = parserv(pmodel)
    for i=1:length(mlptemp); push!(mlpmodel, mlptemp[i]); end
    featmodel = Array{Any}(5) # Array(Any,5) deprecated
    for k in 1:5 # (:postag,:deprel,:lcount,:rcount,:distance)
        featmodel[k] = Any[]
        pmodel_k = pmodel[k]
        for i in 1:length(pmodel_k)
            push!(featmodel[k], pmodel_k[i])
        end
    end
    return (featmodel,mlpmodel)
end


makepmodel(d, o, s) = (haskey(d, "parserv") ? makepmodel1(d) : makepmodel2(o, s; intype=:normal))

