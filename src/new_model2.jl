# Second version of the new_model.jl
# Parameter iterators
postagv(m)=m[1];
bufferm(m)=m[2];
stackm(m)=m[3];
actm(m)=m[4];
actembed(m)=m[5];
parserv(m)=m[6];



# To initialize parser model from scratch
function initmodel1(o, s; ftype=Float32)
    model = Any[]

    # postag-level initialization, KnetArray initialization
    dpostag = o[:posembed] # 1
    for (k, n, d) in ((:postag, 17, dpostag),)
        push!(model, [ initr(d, GPUFEATS=true) for i=1:n ])
    end

    # buffer-lstm initialization # 2
    r_b, wr_b = initbuf(o[:bufembed], o[:bufhidden]) # you may change lstm - gru with kwargs
    push!(model, [r_b, wr_b])

    # stack-lstm initialization # 3
    r_s, wr_s = initbuf(o[:stembed], o[:sthidden])
    push!(model, [r_s, wr_s])

    # action-lstm initialization # 4
    r_a, wr_a = initbuf(o[:actembed], o[:acthidden])
    push!(model, [r_a, wr_a])

    # action embeddings
    p = o[:arctype](s)
    push!(model, initx(o[:actembed], p.nmove)) # 5

    # decision module initialization # 6
    mlpdims = (r_s.hiddenSize + r_b.hiddenSize + r_a.hiddenSize, o[:hidden]..., p.nmove)
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


function mlp(w,x; pdrop=(0,0))
    x = dropout(x,pdrop[1])
    for i=1:2:length(w)-2
        x = relu.(w[i]*x .+ w[i+1])
        x = dropout(x,pdrop[2])
    end
    return w[end-1]*x .+ w[end]
end


function scan_buffer(model, parser, hidden)
    bufmodel, pvecs = bufferm(model), postagv(model)
    rbuf, wbuf = bufmodel[1], bufmodel[2]
    sentence = parser.sentence
    seqe = length(sentence)
    seqs = parser.wptr
    range = seqe:-1:seqs
    instate = zeros(Float32, hidden, 1) # TODO: not to hardcode hidden
    ybuf = hout = cout = (gpu() >=0 ? KnetArray(instate) : Array(instate))
    for i in range
        #word = sentence.word[i]; print(" $word"); # dbg line
        x = (gpu() >=0 ? KnetArray(sentence.cavec[i]) : Array(sentence.cavec[i]))
        input = cat1d(x, pvecs[sentence.postag[i]])
        ybuf, hout, cout, _ = rnnforw(rbuf, wbuf, input, hout, cout, hy=true, cy=true)
    end
    return ybuf
end


function scan_bufbatch(model, parsers, hidden)
    # Todo: we may cache some parsers in future
    yalls = Any[]
    for p in parsers
        ybuf = scan_buffer(model, p, hidden)
        push!(yalls, ybuf)
    end
    yalls = cat1d(yalls...)
    ncols = length(parsers)
    nrows = div(length(yalls), ncols)
    return reshape(yalls, nrows, ncols)
end


function scan_stack(model, parser, hstack)
    stmodel, pvecs = stackm(model), postagv(model)
    rst, wst = stmodel[1], stmodel[2]
    sentence = parser.sentence
    seqe = parser.sptr
    instate = zeros(Float32, hstack, 1)
    yst = hout = cout = (gpu() >= 0 ? KnetArray(instate) : Array(instate))
    range = 1:seqe # words in stack p.stack[1:p.sptr]
    for i in range
        indx  = parser.stack[i]
        #word = sentence.word[indx]; print(" $word") # dbg line
        x = (gpu() >= 0 ? KnetArray(sentence.cavec[indx]) : Array(sentence.cavec[indx]))
        input = cat1d(x, pvecs[sentence.postag[indx]])
        yst, hout, cout, _ = rnnforw(rst, wst, input, hout, cout, hy=true, cy=true)
    end
    return yst
end


function scan_stackbatch(model, parsers, hstack)
    yalls = Any[]
    for p in parsers
        yst = scan_stack(model, p, hstack)
        push!(yalls, yst)
    end
    yalls = cat1d(yalls...)
    ncols = length(parsers)
    nrows = div(length(yalls), ncols)
    return reshape(yalls, nrows, ncols)
end


# compute batched actions
function scan_action(model, actions, state)
    acemb, actmodel = actembed(model), actm(model)
    yout0, hout0, cout0 = state[1], state[2], state[3]
    input = acemb[:, actions]
    yout, hout, cout = rnnforw(actmodel[1], actmodel[2], input, hout0, cout0, hy=true, cy=true)
    return (yout, hout, cout)
end


function oracleloss(allmodel, sentences, arctype, hiddens; losses=nothing, pdrop=(0.0, 0.0))
    hidstack, hidact, hidbuf = hiddens
    parsers = map(arctype, sentences)
    mcosts  = Array{Cost}(parsers[1].nmove)
    parserdone = falses(length(parsers))
    mlpmodel = parserv(allmodel)
    B = length(parsers) # bathcsize
    t0 = zeros(Float32, hidact, B, 1)
    h0 = c0 = (gpu()>=0 ? KnetArray(t0) : Array(t0))
    y0 = reshape(h0, hidact, B)
    totalloss = 0.0

    # action lstm-init
    nstate = (y0, h0, c0);
    inacts = map(t->1, 1:B)

    while !all(parserdone)
        # batched score 
        yact, hact, cact = scan_action(allmodel, inacts, nstate)
        nstate = (yact, hact, cact); 
        ybufs    = scan_bufbatch(allmodel, parsers, hidbuf)
        ysts     = scan_stackbatch(allmodel, parsers, hidstack)
        encoded  = vcat(yact, ybufs, ysts) 
        scores   = mlp(mlpmodel, encoded, pdrop=pdrop)
        logprobs = logp(scores, 1)

        inacts = Int[] # to get the next step's actions
        # iterative loss-val
        for (i, p) in enumerate(parsers)
            if parserdone[i]
                push!(inacts, 1); # not changing batchsize
                continue
            end
            movecosts(p, p.sentence.head, p.sentence.deprel, mcosts)
            goldmove = indmin(mcosts)

            if mcosts[goldmove] == typemax(Cost)
                parserdone[i] = true
                p.sentence.parse = p
                push!(inacts, 1); # not changing batchsize
            else
                totalloss -= logprobs[goldmove, i]
                move!(p, goldmove);
                push!(inacts, goldmove);
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


oraclegrad = grad(oracleloss)


# Implement test related

# endofparse to use in test
endofparse(p)=(p.sptr == 1 && p.wptr > p.nword)


function oracletest(allmodel, corpus, arctype, hiddens, batchsize; pdrop=(0.0, 0.0))
    sentbatches = minibatch(corpus, batchsize)
    hidstack, hidact, hidbuf = hiddens
    mlpmodel = parserv(allmodel)

    for sentences in sentbatches
        parsers = map(arctype, sentences)
        B = length(parsers) # bathcsize
        mcosts  = Array{Cost}(parsers[1].nmove)
        parserdone = map(endofparse, parsers)
        #parserdone = falses(length(parsers))
        t0 = zeros(Float32, hidact, B, 1)
        h0 = c0 = (gpu()>=0 ? KnetArray(t0) : Array(t0))
        y0 = reshape(h0, hidact, B)
        nstate = (y0, h0, c0);
        inacts = map(t->1, 1:B)
        yact, hact, cact = scan_action(allmodel, inacts, nstate)
        iters = 1#dbg
        while !all(parserdone)
            yact, hact, cact = scan_action(allmodel, inacts, nstate)
            nstate = (yact, hact, cact); 
            ybufs    = scan_bufbatch(allmodel, parsers, hidbuf)
            ysts     = scan_stackbatch(allmodel, parsers, hidstack)
            encoded  = vcat(yact, ybufs, ysts) 
            scores   = mlp(mlpmodel, encoded, pdrop=pdrop)
            logprobs = Array(logp(scores, 1))

            inacts = Int[] # to get the next step's actions
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


function empty_parses!(corpus)
    for s in corpus
        s.parse = nothing
    end
end
