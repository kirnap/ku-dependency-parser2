# Reads the conllu formatted file w/o xpostag
function load_conllu(file,v::Vocab)
    corpus = Any[]
    s = Sentence(v)
    for line in eachline(file)
        if line == ""
            push!(corpus, s)
            s = Sentence(v)
        elseif (m = match(r"^\d+\t(.+?)\t.+?\t(.+?)\t.+?\t.+?\t(.+?)\t(.+?)(:.+)?\t", line)) != nothing # modify that to use different columns
            #                id   word   lem  upos   xpos feat head   deprel

            word = m.captures[1]
            push!(s.word, word)
            
            postag = get(v.postags, m.captures[2], 0)
            if postag==0
                Base.warn_once("Unknown postags")
            end
            push!(s.postag, postag)
            
            head = tryparse(Position, m.captures[3])
            head = isnull(head) ? -1 : head.value
            if head==-1
                Base.warn_once("Unknown heads")
            end
            push!(s.head, head)

            deprel = get(v.deprels, m.captures[4], 0)
            if deprel==0
                Base.warn_once("Unknown deprels")
            end
            push!(s.deprel, deprel)
        end
    end
    return corpus
end


# add xpos-tagged version
function load_conllu2(file,v::Vocab)
    corpus = Any[]; xpos = Dict{String, Int}();
    s = Sentence2(v)
    for line in eachline(file)
        if line == ""
            push!(corpus, s)
            s = Sentence2(v)
        elseif (m = match(r"^\d+\t(.+?)\t.+?\t(.+?)\t(.+?)\t.+?\t(.+?)\t(.+?)(:.+)?\t", line)) != nothing # modify that to use different columns
            #                id   word   lem  upos   xpos feat head   deprel

            word = m.captures[1]
            push!(s.word, word)
            
            postag = get(v.postags, m.captures[2], 0)
            if postag==0
                Base.warn_once("Unknown postags")
            end
            push!(s.postag, postag)

            xpostag = get!(xpos, m.captures[3], 1+length(xpos))
            push!(s.xpostag, xpostag)
            
            
            head = tryparse(Position, m.captures[4])
            head = isnull(head) ? -1 : head.value
            if head==-1
                Base.warn_once("Unknown heads")
            end
            push!(s.head, head)

            deprel = get(v.deprels, m.captures[5], 0)
            if deprel==0
                Base.warn_once("Unknown deprels")
            end
            push!(s.deprel, deprel)
        end
    end
    (length(xpos) == 1) && Base.warn("No xpostag $file")
    return corpus, xpos
end


# Takes the features of a single word, and parses them into 1-by-1 features
# Modifies fdict returns the (features, val) pair, i.e., ("Case", 2), ("Person", 1)
function parse_feats!(fdict::Dict{String, Vector{String}}, feats; testmode=false)
    getfun = (testmode ? get : get!) # modify fdict in train

    res = []
    if  feats == "_" # underscore for no feats
        return res
    end
    
    farray = split(feats, "|")
    for fs in farray
        fkey, fval = split(fs, "=")
        falls = getfun(fdict, fkey, Vector{String}())
        idx = findfirst(x->x==fval, falls)
        if idx == 0 && !testmode
            push!(falls, fval)
            idx = length(falls)
        end
        push!(res, [fkey, idx])
    end
    return res
end


# Shift morphological features to right columns for embedding mtrx implementation
function shift_cfeats!(corpus, fdict)
    fnav = Dict{String, Int}() # Shifting values
    counter = 1;for (k,v) in fdict; fnav[k]=counter; counter+=length(v);end;
    f(x) = (x[2] += fnav[x[1]]-1) # helper to shift features
    for sentence in corpus
        for fgivens in sentence.feats; map(x->f(x), fgivens);end;
        modify_feats!(sentence) # test them
    end
end


# Caches all the feats as an int array
function modify_feats!(s3::Sentence3)
    fs=[]
    for feat in s3.feats
        fset = Int[]
        for f in feat
            push!(fset, f[2])
        end
        push!(fs, fset)
    end
    s3.feats = fs
end


# numof columns in feature embedding matrix
featdim(f::Dict{String,Array{String,1}}) = mapreduce(length, +, 0, values(f))

# xpos-tag and feats version
function load_conllu3(file, v::Vocab,
                      fdict=nothing,
                      xpos=nothing, testmode=true
                      )

    corpus = Any[]; 
    if fdict == nothing # create train feats dictionary
        fdict = Dict{String, Vector{String}}()
        testmode = false
    end

    getfun = get # not to modify xpos dict in devset
    if xpos == nothing # use train-file's xpos dict
        xpos = Dict{String, Int}();
        getfun = get!
    end
    s = Sentence3(v)
    for line in eachline(file)
        if line == ""
            push!(corpus, s)
            s = Sentence3(v)
        elseif (m = match(r"^\d+\t(.+?)\t.+?\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)(:.+)?\t", line)) != nothing # modify that to use different columns
            #                id   word   lem  upos   xpos  feat   head   deprel

            word = m.captures[1]
            push!(s.word, word)
            
            postag = get(v.postags, m.captures[2], 0)
            if postag==0
                Base.warn_once("Unknown postags")
            end
            push!(s.postag, postag)

            xpostag = getfun(xpos, m.captures[3], 1+length(xpos))
            push!(s.xpostag, xpostag)

            wordfeats = m.captures[4]
            feats = parse_feats!(fdict, wordfeats, testmode=testmode)
            push!(s.feats, feats)
            
            
            head = tryparse(Position, m.captures[5])
            head = isnull(head) ? -1 : head.value
            if head==-1
                Base.warn_once("Unknown heads")
            end
            push!(s.head, head)

            deprel = get(v.deprels, m.captures[6], 0)
            if deprel==0
                Base.warn_once("Unknown deprels")
            end
            push!(s.deprel, deprel)
        end
    end
    (length(xpos) == 1) && Base.warn("No xpostag $file")
    return corpus, xpos, fdict
end


# To create vocabulary from pre-trained lstm model, modify that to use different cols
function create_vocab(d)
    Vocab(d["char_vocab"],
          Dict{String, Int}(),
          d["word_vocab"],
          d["sosword"],
          d["eosword"],
          d["unkword"],
          d["sowchar"],
          d["eowchar"],
          d["unkchar"],
          get(d, "postags", UPOSTAG),
          get(d, "deprels", UDEPREL)
          )
end


# After filling word, forward and backward vectors cache the concated version
function cachelmvec!(corpus)
    for sent in corpus
        for i in 1:length(sent)
            push!(sent.cavec, vcat(sent.wvec[i], sent.fvec[i], sent.bvec[i]))
        end
    end
end



function minibatch(corpus, batchsize; maxlen=typemax(Int), minlen=1, shuf=false)
    data = Any[]
    sorted = sort(corpus, by=length)
    i1 = findfirst(x->(length(x) >= minlen), sorted)
    if i1==0; error("No sentences >= $minlen"); end
    i2 = findlast(x->(length(x) <= maxlen), sorted)
    if i2==0; error("No sentences <= $maxlen"); end
    for i in i1:batchsize:i2
        j = min(i2, i+batchsize-1)
        push!(data, sorted[i:j])
    end
    if shuf
        data=shuffle(data)
    end
    return data
end


