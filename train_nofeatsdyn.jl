using JLD, Knet, ArgParse
include("src/header.jl")
const mfile = "src/model_nofeat_dyn.jl" # dynamic-oracle
include(mfile)
@msg mfile

# lm-related constants(fixed!)
# const lmdir  = "/scratch/users/okirnap/ud-treebanks-v2.2/chmodel_converted"

# To run everything on aws
const lmdir  = "../ud-treebanks-v2.2/chmodel_converted"
#const datadir = "/scratch/users/okirnap/ud-treebanks-v2.2"


function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "Koc University dependency parser (c) Omer Kirnap, 2018."
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--lmfile"; help="Language model file to load pretrained language mode")
        ("--loadfile"; help="Initialize model from file/ language model or all model")
        ("--datafiles"; nargs='+'; help="Input in conllu format. If provided, use the first for training, last for dev. If single file use both for train and dev.")
        ("--bestfile"; help="To save the best model file")
        ("--output"; help="Output parse of first datafile in conllu format")
        
        ("--epochs"; arg_type=Int; default=100; help="Epochs of training.")
        ("--optimization"; default="Adam"; help="Optimization algorithm and parameters.")
        ("--batchsize"; arg_type=Int; default=8; help="Number of sequences to train on in parallel.")
        ("--transfer"; action = :store_true; help="Whether to transfer or not")

        ("--dropout"; nargs='+'; arg_type=Float64; default=[0.6, 0.6]; help="Dropout probabilities.")
        ("--gclip"; arg_type=Float64; default=0.0; help="Grad clip")
        ("--savefile"; help="To save the final model file")

        ("--hidden"; nargs='+'; arg_type=Int; default=[2048]; help="MLP dims for final layer")
        ("--lstmhiddens"; nargs='+'; arg_type=Int; default=[256,256,256]; help="lstm dims (stack-1),(buff-2),(act-3)")
        ("--embed"; nargs='+'; arg_type=Int; default=[128, 128, 128];help="embedding sizes for postag(17),xpostag(?),feats(?) default 128,?,?.")
        ("--actembed"; arg_type=Int; default=64; help="action space embeddings")
        ("--wembed"; arg_type=Int; default=512; help="Word Embeddings dimension")
        ("--fembed"; arg_type=Int; default=128; help="Features Embeddings dimension")
        ("--deprel"; arg_type=Int; default=128; help="DepRel Embed")
        ("--treeType"; default=:tanh; help="Tree embedding function")

    end
    isa(args, AbstractString) && (args=split(args))
    odict = parse_args(args, s; as_symbols=true)
    println(s.description); flush(STDOUT);

    # to use in parsing
    testmode = (odict[:epochs] <= 0)

    odict[:ldrops] = (0.5, 0.5, (0.5, 0.5))
    # Set-Up    
    odict[:arctype] = ArcHybridR1; odict[:posembed]=odict[:embed][1];
    odict[:embdim] = odict[:wembed]+odict[:posembed] # no features
    odict[:optimization] = eval(parse(odict[:optimization]))

    odict[:sthidden], odict[:acthidden], odict[:bufhidden] = odict[:lstmhiddens]
    odict[:stembed]=odict[:bufembed]=odict[:embdim]
    # hyper - parameters
    pdrop = odict[:dropout]; batchsize = odict[:batchsize];
    
    (odict[:lmfile] !=nothing) && (odict[:loadfile] != nothing) && error("Can not provide both files once")
    if odict[:lmfile] != nothing
        language_model = joinpath(lmdir, odict[:lmfile])
        @msg language_model
        d = load(language_model); v = create_vocab(d); wmodel = makewmodel(d);
    end

    odict[:transfer] && (odict[:loadfile]==nothing) && error("You have to provide load file for transfer learning")
    if odict[:loadfile] !=nothing # continue training
        @msg "loading model from $(odict[:loadfile])"
        bundle = load(odict[:loadfile])
        model, optims      = bundle["allmodel"], bundle["optims"]
        wmodel, v          = bundle["wordmodel"], bundle["vocab"]
        featdict, xposdict = bundle["featdict"], bundle["xposdict"]
        # Todo : effective way of using odict
    end
    
    
    corpora = []
    if length(odict[:datafiles]) > 1 # train
        f = odict[:datafiles][1] # train
        if !odict[:transfer] # creat featdict 
            c, xposdict, featdict = load_conllu3(f, v)
            fnav = createfnav(featdict)
            shift_cfeats2!(c, fnav)
        else # load fnav from previous model file
            @msg :transfer_learning
            c, _ = load_conllu3(f, v, featdict, xposdict)
            fnav = bundle["fnav"]
            shift_cfeats2!(c, fnav)
        end
        push!(corpora, c)
        
        f = odict[:datafiles][2] # dev
        c, _ = load_conllu3(f, v, featdict, xposdict)
        shift_cfeats2!(c, fnav)
        push!(corpora, c)
    else # test or generate
        f = odict[:datafiles][1]
        c, _ = load_conllu3(f, v, featdict, xposdict)
        fnav = bundle["fnav"]
        shift_cfeats2!(c, fnav)
        push!(corpora, c)
    end

    cc=vcat(corpora...);
    @msg("context embeddings...")
    fillvecs!(wmodel, cc, v)
    @msg("Caching lm vectors...")
    map(cachelmvec!,corpora)


    odict[:featdim] = featdim(featdict)

    if !testmode
        
        if odict[:loadfile] == nothing # if provided no need to reinit
            @msg("Model initialization...")
            model, optims = initmodel1(odict, corpora[1][1])
        end

        println("opts=",[(k,v) for (k,v) in odict]...)
        @msg("calculating initial accuracies"); flush(STDOUT)
        acc1 = oracletest(model, corpora[1], odict[:arctype], odict[:lstmhiddens], odict[:batchsize], odict[:embdim]; pdrop=(0.0, 0.0))
        acc2 = oracletest(model, corpora[2], odict[:arctype], odict[:lstmhiddens], odict[:batchsize], odict[:embdim]; pdrop=(0.0, 0.0))
        bestlas = acc2; bestepoch=0;
        @msg "Initial tracc $acc1 devacc $acc2"
        nsentdev = length(corpora[2]); nworddev = sum(map(length, corpora[2]));
        @msg "nsentdev/nworddev=$nsentdev/$nworddev"
        sentbatches = minibatch(corpora[1], odict[:batchsize], maxlen=64, minlen=2, shuf=true)
        nsent = sum(map(length,sentbatches)); nsent0 = length(corpora[1])
        nword = sum(map(length,vcat(sentbatches...))); nword0 = sum(map(length,corpora[1]))
    end

    #error()
    # train-loop
    for epoch=1:odict[:epochs]
        shuffle!(sentbatches)
        @msg("nsent=$nsent/$nsent0 nword=$nword/$nword0")
        nwords = StopWatch()
        losses = Any[0,0,0]
        @time for sentences in sentbatches
            grads = oraclegrad(model, sentences, odict[:arctype], odict[:lstmhiddens], odict[:embdim],losses=losses, pdrop=pdrop, ldrops=odict[:ldrops])
            update!(model, grads, optims)
            nw = sum(map(length,sentences))
            if (speed = inc(nwords, nw)) != nothing
                date("$(nwords.ncurr) words $(round(Int,speed)) wps $(losses[3]) avgloss")
            end
        end
        empty_parses!(corpora[1]); empty_parses!(corpora[2]); #gc(); Knet.gc();gc();
        acc1 = oracletest(model, corpora[1], odict[:arctype], odict[:lstmhiddens],odict[:batchsize], odict[:embdim]; pdrop=(0.0, 0.0))
        acc2 = oracletest(model, corpora[2], odict[:arctype], odict[:lstmhiddens], odict[:batchsize], odict[:embdim]; pdrop=(0.0, 0.0))
        println()
        @msg "epoch $epoch train acc $acc1 dev acc $acc2"

        currlas = acc2
        if currlas > bestlas
            bestepoch = epoch
            bestlas = currlas
            if odict[:bestfile] != nothing
                JLD.save(odict[:bestfile],
                         "allmodel", model,
                         "optims", optims,
                         "featdict", featdict,
                         "xposdict", xposdict,
                         "wordmodel", map2cpu(wmodel),
                         "vocab", v,
                         "odict", odict,
                         "fnav", fnav
                         )
            end
        end
        if 9 < bestepoch < epoch-10
            @msg "bestlas $bestlas in $bestepoch"
            break
        end
    end

    # test-related
    if testmode # given for testing
        @assert length(corpora) == 1
        empty_parses!(corpora[1])
        odict2 = bundle["odict"]
        @msg :Testing
        testacc = oracletest(model, corpora[1], odict2[:arctype], odict2[:lstmhiddens], odict2[:batchsize], odict2[:embdim]; pdrop=(0.0, 0.0))
        @msg "Test acc $testacc"
    end

    # parsing
    if odict[:output] != nothing
        @msg :parsing
        writeconllu(corpora[1], odict[:datafiles][1], odict[:output])
    end
    @msg :done
end
!isinteractive() && main(ARGS)
