using JLD, Knet, ArgParse
include("src/header.jl")
include("src/new_model4.jl")
@msg "src/new_model4.jl"

# lm-related constants(fixed!)
const lmdir  = "/scratch/users/okirnap/ud-treebanks-v2.2/chmodel_converted"
#const datadir = "/scratch/users/okirnap/ud-treebanks-v2.2"


function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "Koc University dependency parser (c) Omer Kirnap, 2018."
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--lmfile"; help="Language model file to load pretrained language mode")
        ("--loadfile"; help="Initialize model from file/ language model or all model")
        ("--datafiles"; nargs='+'; help="Input in conllu format. If provided, use the first for training, last for dev. If single file use both for train and dev.")
        ("--output"; help="Output parse of first datafile in conllu format")
        ("--lstmhiddens"; nargs='+'; arg_type=Int; default=[256,256,256]; help="lstm dims (stack-1),(buff-2),(act-3)")
        ("--actembed"; arg_type=Int; default=32; help="action space embeddings")
        ("--batchsize"; arg_type=Int; default=8; help="Number of sequences to train on in parallel.")
        ("--epochs"; arg_type=Int; default=100; help="Epochs of training.")
        ("--dropout"; nargs='+'; arg_type=Float64; default=[0.6, 0.6]; help="Dropout probabilities.")
        ("--embed"; nargs='+'; arg_type=Int; default=[128, 128, 128];help="embedding sizes for postag(17),xpostag(?),feats(?) default 128,?,?.")
        ("--hidden"; nargs='+'; default=[2048]; help="MLP dims for final layer")
        ("--optimization";  default="Adam"; help="Optimization algorithm and parameters.")
        ("--savefile"; help="To save the final model file")
        ("--bestfile"; help="To save the best model file")
        ("--wembed"; arg_type=Int; default=512; help="Word Embeddings dimension")
        ("--deprel"; arg_type=Int; default=128; help="DepRel Embed")
        ("--treeType"; default=:tanh; help="Tree embedding function")

    end
    isa(args, AbstractString) && (args=split(args))
    odict = parse_args(args, s; as_symbols=true)
    println(s.description); flush(STDOUT);

    # Set-Up 
    odict[:arctype] = ArcHybridR1; odict[:posembed]=odict[:embed][1];
    odict[:sthidden], odict[:acthidden], odict[:bufhidden] = odict[:lstmhiddens][1], odict[:lstmhiddens][2], odict[:lstmhiddens][3]
    odict[:embdim] = odict[:wembed]+odict[:posembed]
    odict[:optimization] = eval(parse(odict[:optimization]))


    odict[:stembed]=odict[:bufembed]=odict[:wembed]+odict[:posembed] 
    # hyper - parameters
    pdrop = odict[:dropout]; batchsize = odict[:batchsize];


    if odict[:lmfile] != nothing
        language_model = joinpath(lmdir, odict[:lmfile])
        @msg language_model
    end
    d = load(language_model); v = create_vocab(d); wmodel = makewmodel(d);
    
    corpora = []
    for f in odict[:datafiles];
        @msg f
        c = load_conllu(f, v)
        push!(corpora, c)
    end
    cc=vcat(corpora...);
    fillvecs!(wmodel, cc, v)
    @msg("Caching lm vectors...")
    map(cachelmvec!,corpora)

    @msg("Model initialization...")
    model, optims = initmodel1(odict, corpora[1][1])

    println("opts=",[(k,v) for (k,v) in odict]...)

    @msg("calculating initial accuracies"); flush(STDOUT)
    acc1 = oracletest(model, corpora[1], odict[:arctype], odict[:lstmhiddens], odict[:batchsize], odict[:embdim]; pdrop=(0.0, 0.0))
    acc2 = oracletest(model, corpora[2], odict[:arctype], odict[:lstmhiddens], odict[:batchsize], odict[:embdim]; pdrop=(0.0, 0.0))
    bestlas = acc2; bestepoch=0;
    @msg "Initial tracc $acc1 devacc $acc2"

    nsentdev = length(corpora[2]); nworddev = sum(map(length, corpora[2]));
    @msg "nsentdev/nworddev=$nsentdev/$nworddev"
    
    sentbatches = minibatch(corpora[1], batchsize, maxlen=64, minlen=2, shuf=true)
    nsent = sum(map(length,sentbatches)); nsent0 = length(corpora[1])
    nword = sum(map(length,vcat(sentbatches...))); nword0 = sum(map(length,corpora[1]))
    for epoch=1:odict[:epochs]
        shuffle!(sentbatches)
        @msg("nsent=$nsent/$nsent0 nword=$nword/$nword0")
        nwords = StopWatch()
        losses = Any[0,0,0]
        @time for sentences in sentbatches
            grads = oraclegrad(model, sentences, odict[:arctype], odict[:lstmhiddens], odict[:embdim],losses=losses, pdrop=pdrop)
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
                # TODO: add save method for trained model
            end
        end
        if 9 < bestepoch < epoch-15
            @msg "bestlas $bestlas in $bestepoch"
            break
        end
    end
    @msg :done
end
!isinteractive() && main(ARGS)
