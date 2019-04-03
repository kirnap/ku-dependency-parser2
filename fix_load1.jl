# on julia 0.6
using JLD, Knet;include("src/header.jl")
language_model = "/kuacc/users/okirnap/ud-treebanks-v2.2/chmodel_converted/english_chmodel.jld"
d = load(language_model);
word_vocab2 = Dict{String, Int64}();
for (k,v) in d["word_vocab"]; word_vocab2[k]=v;end;
new_d = Dict{String, Any}();for (k,v) in d; (k =="word_vocab") ? new_d[k]=word_vocab2 : new_d[k] =v;end;
using JLD2
JLD2.@save "english_chmodel.jld2" new_d

# on julia 1.0
# Make sure that you are on branch julia1
using JLD2,Knet;include("src/header.jl")
JLD2.@load "english_chmodel.jld2" new_d; # now you have it!