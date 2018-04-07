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
