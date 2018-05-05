# To test new_model2.jl
include("test_header.jl")

function test_oracletest()
    
end

function load_data()
    d = load(language_model); v = create_vocab(d); wmodel = makewmodel(d);

    #corpus = load_conllu(small_data_file, v) # no need for train data

    dev    = load_conllu(small_dev_file, v); 
    corpora = []; push!(corpora, dev); cc=vcat(corpora...)
    # push!(corpora, corpus); push!(corpora, dev); cc=vcat(corpora...);
    fillvecs!(wmodel, cc, v)
    
    # ppl = fillvecs!(wmodel, corpus, v) old
    # unk = unkrate(corpus)
    info("Caching lm vectors...")
    map(cachelmvec!,corpora)

    ss = filter(x->length(x)==8, corpora[1]);
    s1 = ss[1]; p1 = odict[:arctype](s1);
    return corpora, p1, s1
end

function test_pred()
    corpora, p1, s1 = load_data()
    #parsers = map(t->p1, 1:10)    
    model, optims = initmodel1(odict, s1)

    # check-acts
    indx = [19, 115, 885, 62]
    sentences = corpora[1][indx]
    parsers = map(odict[:arctype], sentences)
    exp_sets = map(get_goldset, map(odict[:arctype],sentences))
    global omerdict = Dict{}()
    return model, sentences, exp_sets, corpora
end

function get_goldset(parser)
    p1 = copy(parser)
    ret = Int[]
    t = get_goldmove(parser)
    while t != typemax(Cost)
        move!(p1, t);push!(ret, t)
        t = get_goldmove(p1)
    end
    return ret
end

function get_goldmove(parser)
    mc1 = movecosts(parser , parser.sentence.head, parser.sentence.deprel)
    if minimum(mc1) != typemax(Cost)
        return indmin(mc1)
    end
    return typemax(Cost) # no more goldmoves
end

function get_validmove(parser)
    mc1 = movecosts(parser , parser.sentence.head, parser.sentence.deprel)
    valids = find(x->x!=typemax(Cost), mc1)
    (isempty(valids) && return false)
    return valids[rand(1:length(valids))]
end


# How to go along parser's stack
# 1. p.sptr index of the top word of stack, p.stack is the word for stack indices
# We need to find which words stays in the stack, be careful when the word reduced from the stack
# we can truncate the words in stack by p.stack[1:p.sptr]
# There are 3 types of words:
# The ones in the buffer, the ones in the stack and the ones come together with their heads (neither in stack nor in buffer)
# Related move action works with the word p.sptr
function vis_stack(parser)
    p1 = copy(parser)
end
