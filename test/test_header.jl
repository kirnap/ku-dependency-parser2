# Those will be necessary for all test related files

# default feature set used in parser
FEATS=["s1c","s1v","s1p","s1A","s1a","s1B","s1b",
       "s1rL", # "s1rc","s1rv","s1rp",
       "s0c","s0v","s0p","s0A","s0B","s0a","s0b","s0d",
       "s0rL", # "s0rc","s0rv","s0rp",
       "n0lL", # "n0lc","n0lv","n0lp",
       "n0c","n0v","n0p","n0A","n0a",
       "n1c","n1v","n1p",
       ]

# first step to close buffer related features
FEATS1=["s1c","s1v","s1p","s1A","s1a","s1B","s1b",
       "s1rL", # "s1rc","s1rv","s1rp",
       "s0c","s0v","s0p","s0A","s0B","s0a","s0b","s0d",
       "s0rL" # "s0rc","s0rv","s0rp",
       #"n0lL", # "n0lc","n0lv","n0lp",
       #"n0c","n0v","n0p","n0A","n0a",
       #"n1c","n1v","n1p",
       ]



# second step to close stack related features
FEATS2=[#"s1c","s1v","s1p","s1A","s1a","s1B","s1b",
       #"s1rL", # "s1rc","s1rv","s1rp",
       #"s0c","s0v","s0p","s0A","s0B","s0a","s0b","s0d",
       #"s0rL", # "s0rc","s0rv","s0rp",
       "n0lL", # "n0lc","n0lv","n0lp",
       "n0c","n0v","n0p","n0A","n0a",
       "n1c","n1v","n1p",
       ]

using JLD, Knet
include("../src/header.jl")

const language_model  = "/scratch/users/okirnap/ud-treebanks-v2.2/chmodel_converted/english_chmodel.jld" 
const small_data_file = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-train.conllu"
const small_dev_file  = "/scratch/users/okirnap/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-dev.conllu"

const odict = Dict{Symbol, Any}();
odict[:embed] = (128, 32, 16); odict[:optimization] = Adam; odict[:hidden] = [2048]; odict[:feats]=FEATS1; odict[:arctype] = ArcHybridR1;
odict[:stembed]=odict[:bufembed]=128+950; odict[:sthidden]=odict[:bufhidden]=odict[:acthidden]=256; odict[:actembed]=32; odict[:posembed]=128;

