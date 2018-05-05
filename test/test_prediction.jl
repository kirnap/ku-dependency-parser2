# To test the prediction with the previous implementation
# You need to write oraclepred, which predicts similar to oracleloss,
# Do not use invalid actions as well as gold actions
include("test_header.jl")

function load_data()
    d = load(language_model); v = create_vocab(d); wmodel = makewmodel(d);
    corpus = load_conllu(small_data_file, v)
    # newly added
    dev    = load_conllu(small_dev_file, v)
    corpora = []; push!(corpora, corpus); push!(corpora, dev); cc=vcat(corpora...);
    fillvecs!(wmodel, cc, v)
    
    # ppl = fillvecs!(wmodel, corpus, v) old
    unk = unkrate(corpus)
    info("Caching lm vectors...")
    map(cachelmvec!,corpora)
    #cachelmvec!(corpora[1]);  old

    info("Model initialization...")
    pmodel, poptim = initmodel1(odict, corpus[1])
    #p2 = odict[:arctype](corpus[2901]); pvecs = postagv(old_pmodel);
    return corpora, pmodel, poptim
end


function test_pred()
    corpora, model, poptims = load_data(); corpus=corpora[1];
    pdrop = (0.5, 0.6); batchsize = 16;

    @msg "Drop out probabilities $pdrop"
    info("calculating initial accuracies")
    acc1 = oracletest(model, corpora[1], odict[:arctype], odict[:feats], batchsize; pdrop=(0.0, 0.0))
    acc2 = oracletest(model, corpora[2], odict[:arctype], odict[:feats], batchsize; pdrop=(0.0, 0.0))
    @msg "Initial tracc $acc1 devacc $acc2"
    sentbatches = minibatch(corpus, batchsize, maxlen=64, minlen=2, shuf=false) # shuffling imp. acc
    nsent = sum(map(length,sentbatches)); nsent0 = length(corpus)
    nword = sum(map(length,vcat(sentbatches...))); nword0 = sum(map(length,corpus))
    for epoch=1:20
        @msg("nsent=$nsent/$nsent0 nword=$nword/$nword0")
        nwords = StopWatch()
        losses = Any[0,0,0]
        @time for sentences in sentbatches
            grads = oraclegrad(model, sentences, odict[:arctype], odict[:feats]; losses=losses, pdrop=pdrop)
            update!(model, grads, poptims)
            nw = sum(map(length,sentences))
            if (speed = inc(nwords, nw)) != nothing
                date("$(nwords.ncurr) words $(round(Int,speed)) wps $(losses[3]) avgloss")
                gc();Knet.gc();gc();
                #free_KnetArray()
            end

        end
        empty_parses!(corpus)
        acc = oracletest(model, corpus, odict[:arctype], odict[:feats], batchsize; pdrop=(0.0, 0.0))
        acc2 = oracletest(model, corpora[2], odict[:arctype], odict[:feats], batchsize; pdrop=(0.0, 0.0))
        
        println()
        println("epoch $epoch train acc $acc dev acc $acc2");
    end
#    p2 = odict[:arctype](corpus[2901]); pvecs = postagv(model);
#    idx = [ rand(1:length(corpus)) for i in 1:10 ];
end
!isinteractive() && test_pred()
