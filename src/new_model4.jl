# lefts->2:2:(p.nmove-1) even
# rights->3:2:p.nmove odd

# To accompany embedding trees
wordW(m)=m[1];
postagv(m)=m[2];
bufferm(m)=m[3];
stackm(m)=m[4];
actm(m)=m[5];
actembed(m)=m[6];
parserv(m)=m[7];
deprelv(m)=m[8]
treeNN(m)=m[9]

function initmodel1(o, s; ftype=Float32)
    model = Any[]
    
    # wordembedding # 1
    push!(model, initx(o[:wembed], 950))

    # postag-level initialization, KnetArray initialization
    dpostag = o[:posembed] # 2
    for (k, n, d) in ((:postag, 17, dpostag),)
        push!(model, Any[ initr(d, GPUFEATS=true) for i=1:n ])
    end

    # buffer-lstm initialization # 3
    r_b, wr_b = initbuf(o[:bufembed], o[:bufhidden]) # you may change lstm - gru with kwargs
    push!(model, [r_b, wr_b])

    # stack-lstm initialization # 4
    r_s, wr_s = initbuf(o[:stembed], o[:sthidden])
    push!(model, [r_s, wr_s])

    # action-lstm initialization # 5
    r_a, wr_a = initbuf(o[:actembed], o[:acthidden])
    push!(model, [r_a, wr_a])

    # action embeddings # 6
    p = o[:arctype](s)
    push!(model, initx(o[:actembed], p.nmove))

    # decision module initialization # 7
    mlpdims = (r_s.hiddenSize + r_b.hiddenSize + r_a.hiddenSize, o[:hidden]..., p.nmove)
    info("mlpdims: $mlpdims")
    decider = Any[]
    for i=2:length(mlpdims)
        push!(decider, initx(mlpdims[i], mlpdims[i-1]))
        push!(decider, initx(mlpdims[i], 1))
    end
    push!(model, decider)

    # deprel embeddings # 8
    depreldim = o[:deprel]
    push!(model, Any[ initr(depreldim, GPUFEATS=true) for i=1:p.nmove-1 ])
    
    # tree representer # 9
    rt, wrt = rnninit(o[:deprel]+o[:wembed], o[:wembed], rnnType=o[:treeType])
    push!(model, [rt, wrt])

    optims = optimizers(model, o[:optimization])
    return model, optims
    
end


function cache_wvecs(model, parsers)
    W = wordW(model)
    B = length(parsers)    
    walls = Array{Any}(B)

    for i in 1:B
        p = parsers[i]
        x = (gpu()>=0 ? KnetArray(hcat(p.sentence.cavec...)) : Array(hcat(p.sentence.cavec...)))
        X = W * x # TODO : try one-by-one multiplication
        T = length(p.sentence); w_i = Array{Any}(T)
        for t in 1:T
            temp   = X[:, t]
            w_i[t] = reshape(temp, length(temp), 1) # do we need that? or just temp
        end
        walls[i] = w_i
    end
    return walls
end


####### Buffer-Related

# youts[i] -> output of taking wi as an input
# y2use -> youts[p.wptr]
function scan_buffer(model, wvecs, parser, hidden)
    t0 = zeros(Float32, hidden, 1) # takes single instance
    t0 = (gpu()>=0 ? KnetArray(t0): Array(t0))
    if length(parser.sentence) == 1 # ugly test time!
        y0 = t0;
        c0 = reshape(t0, length(t0), 1, 1)
        return y0, c0
    end

    bufmodel, pvecs = bufferm(model), postagv(model)
    rbuf, wbuf = bufmodel[1], bufmodel[2]
    sentence = parser.sentence
    T = length(sentence)
    seqs = parser.wptr
    range_ = T:-1:seqs

    # There is no way to be done in single rnnforw call, cell required!!!
    # No need to store hiddens, reshape youts
    # isapprox(reshape(hyouts[i], 128,), youts[i]) is true
    h0 = c0 = reshape(t0, hidden, 1, 1)
    youts, cyouts = Array{Any}(T), Array{Any}(T)
    for i in range_
        i1 = cat1d(wvecs[i], pvecs[sentence.postag[i]])
        y1, h0, c0, = rnnforw(rbuf, wbuf, i1, h0, c0, hy=true, cy=true)
        youts[i], cyouts[i] = y1, c0 # fill the ith words output hidden
    end
    return youts, cyouts
end


function cache_bufbatch(model, walls, parsers, hidden)
    yalls = Array{Any}(length(parsers)) # i for ith parser
    B = length(parsers)
    for i in 1:B
        wvecs = walls[i];
        p = parsers[i]
        youts, cyouts = scan_buffer(model, wvecs, p, hidden)
        yalls[i] = Any[youts, cyouts]
    end
    return yalls
end


function scan_bufbatch(yalls, parsers, hidden)
    B = length(parsers)
    instate = zeros(Float32, hidden, 1)
    ybuf = (gpu() >=0 ? KnetArray(instate) : Array(instate))

    ybatch = Any[]
    for i in 1:B
        p = parsers[i]
        if p.wptr > p.nword # buffer empty
            push!(ybatch, ybuf)
        else
            youts, _ = yalls[i]
            push!(ybatch, youts[p.wptr]) # 2use
        end
    end
    ybatch = cat1d(ybatch...)
    nrow = div(length(ybatch), B)
    return reshape(ybatch, nrow, B)
end

####### Stack-related

# You are given new set of wvecs and re-calculate each hidden again
function scan_stack(model, wvecs, parser, hidden, embdim)
    sentence = parser.sentence
    T = parser.sptr
    if length(1:T) < 1 # empty stack
        instate = zeros(Float32, hidden, 1) # B 1
        t0 = (gpu() >= 0 ? KnetArray(instate) : Array(instate))
        return t0
    end

    stmodel, pvecs = stackm(model), postagv(model)
    rst, wst = stmodel[1], stmodel[2]

      
    inputs = Any[]
    for i in 1:T
        wi = parser.stack[i]
        i1 = cat1d(wvecs[wi], pvecs[sentence.postag[wi]])
        push!(inputs, i1)
    end
    allin = reshape(hcat(inputs...), embdim, 1, T)

    _, hyout, = rnnforw(rst, wst, allin, hy=true) # final-hidden
    return reshape(hyout, hidden, 1)
end


# Be careful walls and parsers should have the same order
function scan_stackbatch(model, walls, parsers, hidden, embdim)
    B = length(parsers)
    yalls = Any[]
    for i in 1:B
        p = parsers[i]; wvecs = walls[i];
        yst = scan_stack(model, wvecs, p, hidden, embdim)
        push!(yalls, yst)
    end
    yalls = cat1d(yalls...)
    ncols = B
    nrows = div(length(yalls), ncols)
    return reshape(yalls, nrows, ncols)
end


###### Action related

# compute batched actions
function scan_action(model, actions, state)
    acemb, actmodel = actembed(model), actm(model)
    yout0, hout0, cout0 = state[1], state[2], state[3]
    input = acemb[:, actions]
    yout, hout, cout = rnnforw(actmodel[1], actmodel[2], input, hout0, cout0, hy=true, cy=true)
    return (yout, hout, cout)
end



####### Embedding-related
# 1. Left-move : even move-numbers, pair, hw = p.wptr;dw = p.stack[p.sptr]
# 2. Right-move: odd move-numbers, pair,  hw = p.stack[p.sptr-1]; dw = p.stack[p.sptr]
function update_cache!(model, wvecs, cacbufs, p, m)
    depvecs, trNN = deprelv(model), treeNN(model)
    rt, wt = trNN[1], trNN[2]

    if m == 1
        return wvecs
    elseif m % 2 == 0 # left
        hw, dw = leftmovepair(p)
        wold = wvecs[hw]
        r  = depvecs[m-1] # -1 to reduce shift
        x1 = cat1d(wvecs[dw], r) # [x;r]
        input = reshape(x1, length(x1), 1) # batchsize 1

        # update wvec
        wnew, = rnnforw(rt, wt, input, wold)
        wvecs[hw] = wnew

        # update cachedbufs
        bufmodel, pvecs = bufferm(model), postagv(model)
        rbuf, wbuf = bufmodel[1], bufmodel[2]
        ybuf, cbuf = cacbufs
        in2rnn = cat1d(wnew, pvecs[p.sentence.postag[hw]])
        yold, cold = cacbufs
        hyin, cyin = yold[hw], cold[hw]

        # No way to get rid of hidden returning
        # updates the hidden vectors
        yold[hw], _, cold[hw], = rnnforw(rbuf, wbuf, wnew, hyin, cyin, cy=true)
        
    else # right 
        hw, dw = rightmovepair(p)
        wold = wvecs[hw]
        r  = depvecs[m-1] # -1 to reduce shift
        x1 = cat1d(wvecs[dw], r)
        input = reshape(x1, length(x1), 1)
        wnew, = rnnforw(rt, wt, input, wold)
        wvecs[hw] = wnew
    end
    #return wvecs # no need to return it only updates related .caches
end




function oracleloss(allmodel, sentences, arctype, hiddens, embdim;losses=nothing, pdrop=(0.0, 0.0))
    hidstack, hidact, hidbuf = hiddens
    parsers = map(arctype, sentences)
    mcosts = Array{Cost}(parsers[1].nmove)
    parserdone  = falses(length(parsers))
    mlpmodel = parserv(allmodel)
    
    # cache 
    walls = cache_wvecs(allmodel, parsers)
    cachedbufs = cache_bufbatch(allmodel, walls, parsers, hidbuf)

    # init
    B = length(parsers) # bathcsize
    t0 = zeros(Float32, hidact, B, 1)
    h0 = c0 = (gpu()>=0 ? KnetArray(t0) : Array(t0))
    y0 = reshape(h0, hidact, B)
    totalloss = 0.0
    nstate = (y0, h0, c0)
    inacts = map(t->1, 1:B)

    while !all(parserdone)
        yact, hact, cact = scan_action(allmodel, inacts, nstate)
        nstate = (yact, hact, cact);
        ysts   = scan_stackbatch(allmodel, walls, parsers, hidstack, embdim)
        ybufs  = scan_bufbatch(cachedbufs, parsers, hidbuf)
        encoded = vcat(yact, ysts, ybufs)
        scores  = mlp(mlpmodel, encoded, pdrop=pdrop)
        logprobs = logp(scores, 1)

        inacts = Int[] # to get the next step's actions
        # iterative loss-val
        for (i, p) in enumerate(parsers)
            if parserdone[i]
                push!(inacts, 1) # not to change batchsize
                continue
            end
            movecosts(p, p.sentence.head, p.sentence.deprel, mcosts)
            goldmove = indmin(mcosts)
            
            if mcosts[goldmove] == typemax(Cost)
                parserdone[i] = true
                p.sentence.parse = p
                push!(inacts, 1) # not to change batchsize
            else
                totalloss -= logprobs[goldmove, i]
                # update before moving to the next step
                update_cache!(allmodel, walls[i], cachedbufs[i], p, goldmove)
                move!(p, goldmove); push!(inacts, goldmove);
                if losses != nothing
                    loss1 = -getval(logprobs)[goldmove,i]
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


oraclegrad = grad(oracleloss) # less is more :)

# Implement test related

# endofparse to use in test
endofparse(p)=(p.sptr == 1 && p.wptr > p.nword)


function oracletest(allmodel, corpus, arctype, hiddens, batchsize, embdim; pdrop=(0.0, 0.0))
    sentbatches = minibatch(corpus, batchsize)
    hidstack, hidact, hidbuf = hiddens
    mlpmodel = parserv(allmodel)

    for sentences in sentbatches
        parsers = map(arctype, sentences)
        B = length(parsers) # bathcsize
        mcosts  = Array{Cost}(parsers[1].nmove)
        parserdone = map(endofparse, parsers)

        if all(parserdone) # shows how careless they prepared the dataset!
            for p in parsers
                p.sentence.parse = p
            end
            continue
        end

        # cache
        walls = cache_wvecs(allmodel, parsers)
        cachedbufs = cache_bufbatch(allmodel, walls, parsers, hidbuf)

        t0 = zeros(Float32, hidact, B, 1)
        h0 = c0 = (gpu()>=0 ? KnetArray(t0) : Array(t0))
        y0 = reshape(h0, hidact, B)
        totalloss = 0.0
        nstate = (y0, h0, c0)
        inacts = map(t->1, 1:B)

        while !all(parserdone)
            yact, hact, cact = scan_action(allmodel, inacts, nstate)
            nstate = (yact, hact, cact);
            ysts   = scan_stackbatch(allmodel, walls, parsers, hidstack, embdim)
            ybufs  = scan_bufbatch(cachedbufs, parsers, hidbuf)
            encoded = vcat(yact, ysts, ybufs)
            scores  = mlp(mlpmodel, encoded, pdrop=pdrop)
            logprobs = Array(logp(scores, 1))

            inacts = Int[]
            for (i, p) in enumerate(parsers)
                if parserdone[i]
                    if p.sentence.parse == nothing # to avoid from the first time
                        p.sentence.parse = p
                    end
                    push!(inacts, 1) # not to break batchsize
                    continue
                end
                isorted = sortperm(logprobs[:,i], rev=true) # ith col ith inst.
                for m in isorted # best&val action
                    if moveok(p, m)
                        update_cache!(allmodel, walls[i], cachedbufs[i], p, m)
                        move!(p, m)
                        push!(inacts, m) # not to break batchsize
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
