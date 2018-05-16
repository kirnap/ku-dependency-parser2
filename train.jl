using JLD, Knet, ArgParse, TimerOutputs
include("src/header.jl")

const language_model  = "/scratch/users/okirnap/ud-treebanks-v2.2/chmodel_converted/english_chmodel.jld" 
const small_data_file = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-train.conllu"
const small_dev_file  = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-dev.conllu"
const odict = Dict{Symbol, Any}();
odict[:embed] = (128, 32, 16); odict[:optimization] = Adam; odict[:hidden] = [2048]; odict[:arctype] = ArcHybridR1;
odict[:stembed]=odict[:bufembed]=128+950; odict[:sthidden]=odict[:bufhidden]=odict[:acthidden]=128; odict[:actembed]=32; odict[:posembed]=128;
odict[:lstmhiddens] = (128, 128, 128)

function main()
    d = load(language_model); v = create_vocab(d); wmodel = makewmodel(d);
    corpus = load_conllu(small_data_file, v)
    dev    = load_conllu(small_dev_file, v)
    corpora = []; push!(corpora, corpus); push!(corpora, dev); cc=vcat(corpora...);
    fillvecs!(wmodel, cc, v)
    info("Caching lm vectors...")
    map(cachelmvec!,corpora)


    #const to = TimerOutput()

    info("Model initialization...")
    model, optims = initmodel1(odict, corpus[1])

    # hyper - parameters
    pdrop = (0.4, 0.5); batchsize = 8; odict[:pdrop] = pdrop;

    println("opts=",[(k,v) for (k,v) in odict]...)

    info("calculating initial accuracies")
    acc1 = oracletest(model, corpora[1], odict[:arctype], odict[:lstmhiddens],batchsize; pdrop=(0.0, 0.0))
    acc2 = oracletest(model, corpora[2], odict[:arctype], odict[:lstmhiddens], batchsize; pdrop=(0.0, 0.0))
    @msg "Initial tracc $acc1 devacc $acc2"

    sentbatches = minibatch(corpus, batchsize, maxlen=64, minlen=2, shuf=true)
    nsent = sum(map(length,sentbatches)); nsent0 = length(corpus)
    nword = sum(map(length,vcat(sentbatches...))); nword0 = sum(map(length,corpus))
    for epoch=1:100
        shuffle!(sentbatches)
        @msg("nsent=$nsent/$nsent0 nword=$nword/$nword0")
        nwords = StopWatch()
        losses = Any[0,0,0]
        @time for sentences in sentbatches
            grads = oraclegrad(model, sentences, odict[:arctype], odict[:lstmhiddens], losses=losses, pdrop=pdrop)
            update!(model, grads, optims)
            nw = sum(map(length,sentences))
            if (speed = inc(nwords, nw)) != nothing
                date("$(nwords.ncurr) words $(round(Int,speed)) wps $(losses[3]) avgloss")
            end
        end
        empty_parses!(corpus); gc(); Knet.gc();gc();
        acc1 = oracletest(model, corpus, odict[:arctype], odict[:lstmhiddens],batchsize; pdrop=(0.0, 0.0))
        acc2 = oracletest(model, corpora[2], odict[:arctype], odict[:lstmhiddens], batchsize; pdrop=(0.0, 0.0))
        println()
        @msg("epoch $epoch train acc $acc1 dev acc $acc2")
    end
end
!isinteractive() && main()
