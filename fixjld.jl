# To fix the broken jld
using JLD, Knet, ArgParse
include("src/header.jl")
const mfile = "src/new_model_feat3.jl"
include(mfile)

# old-style input
function fixjld(jldin, jldout, dfile)
    bundle = load(jldin)
    if haskey(bundle, "fnav")
        println("cp $jldin $jldout | it is fixed")
        return
    end
    
    model, optims      = bundle["allmodel"], bundle["optims"]
    wmodel, v          = bundle["wordmodel"], bundle["vocab"]
    #featdict, xposdict = bundle["featdict"], bundle["xposdict"]
    odict              = bundle["odict"]
    c, xposdict2, featdict2 = load_conllu3(dfile, v);
    
    fnav = createfnav(featdict2); # you need to save that dictionary for test-time usage
    wmodel2 = map2cpu(wmodel) # idiot
    JLD.save(jldout,
             "allmodel", model,    "optims", optims,
             "featdict", featdict2, "xposdict", xposdict2,
             "wordmodel", wmodel2, "vocab", v,
             "odict",      odict,  "fnav", fnav
             )
end


function main(args=ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--input";  help="Input jld")
        ("--output"; help="Output jld")
        ("--tfile"; help="To initialize train file")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    fixjld(o[:input], o[:output], o[:tfile])
end

!isinteractive() && main(ARGS)
