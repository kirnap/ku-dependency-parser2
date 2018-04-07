# To be able to test the language model, i.e., context and word embeddings
using JLD, Knet

const language_model = "/ai/data/nlp/conll17/competition/chmodel_converted/english_chmodel.jld"
const data_file = "/ai/data/nlp/conll17/ud-treebanks-v2.0/UD_English/en-ud-train.conllu"

include("../src/types.jl")
include("../src/lm_utils.jl")
include("../src/preprocess.jl")



function test_load_conllu(filename)
    d = load(language_model); v = create_vocab(d); wmodel = makewmodel(d);
    corpus = load_conllu(filename, v)
    ppl = fillvecs!(wmodel, corpus, v)
    unk = unkrate(corpus)
    @assert (ppl == 43.70551674466947); info("perplexity test passed!");
    @assert (unk[1]/unk[2] == 0.0938094190678691); info("unkrate test passed!");
end
!isinteractive() && test_load_conllu(data_file)



# perplexity=43.70551674466947 unkrate=0.0938094190678691
