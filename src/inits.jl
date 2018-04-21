# model initializations (normal or xaiver)


# xavier initialization
function initx(d...; ftype=Float32)
    if gpu() >=0
        KnetArray{ftype}(xavier(d...))
    else
        Array{ftype}(xavier(d...))
    end
end


# random normal initialization
function initr(d...; ftype=Float32, GPUFEATS=false)
    if GPUFEATS && gpu() >=0
        KnetArray{ftype}(0.1*randn(d...))
    else
        Array{ftype}(0.1*randn(d...))
    end
end


# zero initialization
function initzeros(d...; ftype=Float32, GPUFEATS=false)
    if GPUFEATS && gpu() >=0
        KnetArray{ftype}(zeros(d...))
    else
        Array{ftype}(zeros(d...))
    end
end


# initialize buffer lstm
# Possible bufemebed: fvec o bvec o wvec o upos (o xpos o feats)
function initbuf(bufembed, bufhidden; buftype=:lstm, numLayers=1)
    r, wr = rnninit(bufembed, bufhidden, rnnType=buftype, numLayers=numLayers)
end
