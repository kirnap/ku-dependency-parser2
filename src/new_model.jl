# Parameter iterators
postagv(m)=m[1];
deprelv(m)=m[2];
lcountv(m)=m[3];
rcountv(m)=m[4];
distancev(m)=m[5];
bufferm(m)=m[6];
parserv(m)=m[7];


# To initialize parser model from scratch
function initmodel1(o, s; ftype=Float32)
    model = Any[]

    # vector-level initialization
    dpostag, ddeprel, dcount = o[:embed]
    for (k,n,d) in ((:postag,17,dpostag),(:deprel,37,ddeprel),(:lcount,10,dcount),(:rcount,10,dcount),(:distance,10,dcount))
        push!(model, [ initr(d) for i=1:n ])
    end

    # buffer-lstm initialization
    r_b, wr_b = initbuf(o[:bufembed], o[:bufhidden]) # you may change lstm - gru with kwargs
    push!(model, [r_b, wr_b])

    # decision module initialization
    p = o[:arctype](s)
    f = features([p], o[:feats], model)
    mlpdims = (length(f)+r_b.hiddenSize, o[:hidden]..., p.nmove)
    info("mlpdims: $mlpdims")
    decider = Any[]
    for i=2:length(mlpdims)
        push!(decider, initx(mlpdims[i], mlpdims[i-1]))
        push!(decider, initx(mlpdims[i], 1))
    end
    push!(model, decider)
    optims = optimizers(model, o[:optimization])
    return model, optims
end


# To seperate feature vectors and all others
function splitmodel(pmodel)
    # feat-model
    featmodel = Array{Any}(5)
    for k in 1:5 # (:postag,:deprel,:lcount,:rcount,:distance)
        featmodel[k] = Any[]
        pmodel_k = pmodel[k]
        for i in 1:length(pmodel_k)
            push!(featmodel[k], pmodel_k[i])
        end
    end

    # buffer-lstm
    bufmodel = Any[]; buftemp = bufferm(pmodel); 
    for i in 1:length(buftemp); push!(bufmodel, buftemp[i]);end;

    # mlp-model
    mlpmodel = Any[]; mlptemp = parserv(pmodel);
    for i in 1:length(mlptemp); push!(mlpmodel, mlptemp[i]); end;
    return (featmodel, bufmodel, mlpmodel)
end


function mlp(w,x; pdrop=(0,0))
    x = dropout(x,pdrop[1])
    for i=1:2:length(w)-2
        x = relu.(w[i]*x .+ w[i+1])
        x = dropout(x,pdrop[2])
    end
    return w[end-1]*x .+ w[end]
end


function scan_buffer(bufmodel, featmodel, parser)
    rbuf, wbuf, pvecs = bufmodel[1], bufmodel[2], postagv(featmodel)
    sentence = parser.sentence
    seqe = length(sentence)
    seqs = parser.wptr
    range = seqe:-1:seqs
    #instate = zeros(rbuf.dataType, rbuf.hiddenSize, 1)
    instate = zeros(Float32, 256, 1)
    ybuf = hout = cout = (gpu() >=0 ? KnetArray(instate) : instate)
    for i in range
        input = vcat(sentence.cavec[i], pvecs[sentence.postag[i]])
        input = (gpu() >=0 ? KnetArray(input) : input)
        ybuf, hout, cout, _ = rnnforw(rbuf, wbuf, input, hout, cout, hy=true, cy=true)
    end
    return ybuf
end


function scan_bufbatch(bufmodel, featmodel, parsers)
    # Todo: we may cache some parsers in future
    yalls = Any[]
    for p in parsers
        ybuf = scan_buffer(bufmodel, featmodel, p)
        push!(yalls, ybuf)
    end
    yalls = cat1d(yalls...)
    ncols = length(parsers)
    nrows = div(length(yalls), ncols)
    return reshape(yalls, nrows, ncols)
end


function oracleloss(allmodel, sentences, arctype, feats; losses=nothing, pdrop=(0.0, 0.0))
    parsers =  map(arctype, sentences)
    mcosts  =  Array{Cost}(parsers[1].nmove)
    parserdone = falses(length(parsers))
    featmodel, bufmodel, mlpmodel = splitmodel(allmodel)
    totalloss = 0.0

    while !all(parserdone)
        fmatrix = features(parsers, feats, featmodel)
        if gpu() >= 0
            fmatrix = KnetArray(fmatrix)
        end
        ybufs = scan_bufbatch(bufmodel, featmodel, parsers)
        input = vcat(fmatrix, ybufs)
        global omer = (mlpmodel, input, pdrop)
        scores = mlp(mlpmodel, input, pdrop=pdrop)
        logprobs = logp(scores, 1)
        for (i, p) in enumerate(parsers)
            if parserdone[i]
                continue
            end
            movecosts(p, p.sentence.head, p.sentence.deprel, mcosts)
            goldmove = indmin(mcosts)

            if mcosts[goldmove] == typemax(Cost)
                parserdone[i] = true
                p.sentence.parse = p
            else
                totalloss -= logprobs[goldmove, i]
                move!(p, goldmove)
                if losses != nothing
                    loss1 = -getval(logprob)[goldmove,i]
                    losses[1] += loss1
                    losses[2] += 1
                    if losses[2] < 1000
                        losses[3] = losses[1]/losses[2]
                    else
                        losses[3] = 0.999 * losses[3] + 0.001 * loss1
                    end                   
                end
            end
        end
    end
    return totalloss / length(sentences)
end


oraclegrad = grad(oracleloss)


# Implement test related

# endofparse to use in test
endofparse(p)=(p.sptr == 1 && p.wptr > p.nword)


# TODO: You need to modify score calculations
function oracletest(pmodel, corpus, arctype, feats, batchsize; pdrop=(0.0, 0.0))
    sentbatches = minibatch(corpus, batchsize)
    featmodel,mlpmodel = splitmodel(pmodel)

    for sentences in sentbatches
        parsers = map(arctype, sentences)
        mcosts = Array{Cost}(parsers[1].nmove) #Array(Cost, parsers[1].nmove)
        parserdone = falses(length(parsers))
        while !all(parserdone)
            fmatrix = features(parsers, feats, featmodel)
            if gpu()>=0
                fmatrix = KnetArray(fmatrix)
            end
            scores = mlp(mlpmodel, fmatrix; pdrop=pdrop)
            logprob = Array(logp(scores, 1))
            for (i,p) in enumerate(parsers)
                if parserdone[i]
                    continue
                end

                isorted = sortperm(logprob[:, i], rev=true) # ith column for ith instance
                for m in isorted # take the best&valid action
                    if moveok(p, m)
                        move!(p, m)
                        break
                    end
                end
                if endofparse(p)
                    parserdone[i] = true
                    p.sentence.parse = p
                end
            end
        end
    end
    return las(corpus)
end


# labeled-attachment score calculator
function las(corpus)
    nword = ncorr = 0
    for s in corpus
        p = s.parse
        nword += length(s)
        ncorr += sum((s.head .== p.head) .& (s.deprel .== p.deprel))
    end
    ncorr / nword
end
