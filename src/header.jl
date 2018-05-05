using JLD, Knet
using AutoGrad: cat1d
include("types.jl")
include("lm_utils.jl")
include("preprocess.jl")
include("parser.jl")
include("helper.jl")
include("inits.jl")
include("old_features.jl")
#include("old_model.jl")
#include("new_model.jl")
include("new_model2.jl")

const MAXSENT=64       # skip longer sentences during training
const MINSENT=2        # skip shorter sentences during training
