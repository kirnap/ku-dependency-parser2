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
