# functions that are used for other purposes than machine learning or preprocessing


macro msg(_x)
    :(join(STDOUT,[Dates.format(now(),"HH:MM:SS"), $(esc(_x)),'\n'],' '); flush(STDOUT))
end

date(x)=(join(STDOUT,[Dates.format(now(),"HH:MM:SS"), x,'\n'],' '); flush(STDOUT))


type StopWatch
    tstart # start time
    nstart # start token
    ncurr  # current token
    nnext  # next token
    StopWatch()=new(time(), 0, 0, 1000)
end


function inc(s::StopWatch, n, step=1000)
    s.ncurr += n
    if s.ncurr >= s.nnext
        tcurr = time()
        dt = tcurr - s.tstart
        dn = s.ncurr - s.nstart
        s.tstart = tcurr
        s.nstart = s.ncurr
        s.nnext += step
        return dn/dt
    end
end


# labeled-attachment score calculator
function las(corpus)
    nword = ncorr = 0
    for s in corpus
        p = s.parse
        nword += length(s)
        ncorr += sum((s.head .== p.head) .& (s.deprel .== p.deprel))
    end
    return ncorr / nword
end


function empty_parses!(corpus)
    for s in corpus
        s.parse = nothing
    end
end


# To generat .conllu formatted files
# To preprocess the files from erenay,
# Replace the head and deprel fields w/ gold ones
function writeconllu1(goldfile, outputfile, erenayfile, v)
    sentences = load_conllu(goldfile, v) # gold-sentences
    out = open(outputfile, "w")
    deprels = Array{String}(length(v.deprels))
    for (k, v) in v.deprels; deprels[v]=k; end;
    s = ph = pd = nothing
    ns = nw = nl = 0
    for line in eachline(erenayfile)
        nl += 1
        if ismatch(r"^\d+\t", line)
            if s == nothing
                s = sentences[ns+1]
                ph = s.head
                pd = s.deprel
            end
            f = split(line, '\t')
            nw += 1
            if f[1] != "$nw"; error();end;
            if f[2] != s.word[nw]; error(); end
            f[7] = string(ph[nw])
            f[8] = deprels[pd[nw]]
            print(out, join(f, "\t"))
            print(out, "\n")
        else
            if line == ""
                if s == nothing; error(); end
                if nw != length(s.word); error(); end
                ns += 1; nw = 0
                s = ph = pd = nothing
            end
            print(out, line)
            print(out, "\n")
        end
    end
    if ns != length(sentences); error("#of sentences different");end;
    close(out)
end


# To save model related
import JLD: writeas, readas
import Knet: RNN
type RNNJLD; inputSize; hiddenSize; numLayers; dropout; inputMode; direction; mode; algo; dataType; end
writeas(r::RNN) = RNNJLD(r.inputSize, r.hiddenSize, r.numLayers, r.dropout, r.inputMode, r.direction, r.mode, r.algo, r.dataType)
readas(r::RNNJLD) = rnninit(r.inputSize, r.hiddenSize, numLayers=r.numLayers, dropout=r.dropout, skipInput=(r.inputMode==1), bidirectional=(r.direction==1), rnnType=(:relu,:tanh,:lstm,:gru)[1+r.mode], algo=r.algo, dataType=r.dataType)[1]
type KnetJLD; a::Array; end
writeas(c::KnetArray) = KnetJLD(Array(c))
readas(d::KnetJLD) = (gpu() >= 0 ? KnetArray(d.a) : d.a)


# There may be no need for the following methods
map2cpu(x)=(if isbits(x); x; else; map2cpu2(x); end)
map2cpu(x::KnetArray)=Array(x)
map2cpu(x::Tuple)=map(map2cpu,x)
map2cpu(x::AbstractString)=x
map2cpu(x::DataType)=x
map2cpu(x::Array)=map(map2cpu,x)
map2cpu{T<:Number}(x::Array{T})=x
map2cpu(x::Associative)=(y=Dict();for (k,v) in x; y[k] = map2cpu(x[k]); end; y)
map2cpu2(x)=(y=deepcopy(x); for f in fieldnames(x); setfield!(y,f,map2cpu(getfield(x,f))); end; y)

map2gpu(x)=(if isbits(x); x; else; map2gpu2(x); end)
map2gpu(x::KnetArray)=x
map2gpu(x::AbstractString)=x
map2gpu(x::DataType)=x
map2gpu(x::Tuple)=map(map2gpu,x)
map2gpu(x::Array)=map(map2gpu,x)
map2gpu{T<:AbstractFloat}(x::Array{T})=KnetArray(x)
map2gpu(x::Associative)=(y=Dict();for (k,v) in x; y[k] = map2gpu(x[k]); end; y)
map2gpu2(x)=(y=deepcopy(x); for f in fieldnames(x); setfield!(y,f,map2gpu(getfield(x,f))); end; y)
