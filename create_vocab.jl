function vocab_from_conllu(file)
    vocab = Dict{String, Integer}()
    for line in eachline(file)
        if (m = match(r"^\d+\t(.+?)\t.+?\t(.+?)\t.+?\t.+?\t(.+?)\t(.+?)(:.+)?\t", line)) != nothing
            word = m.captures[1]
            get!(vocab, word, 1+length(vocab))
        end
    end
    return vocab
end

"""
ARGS[1]: .conllu file to create vocabulary
ARGS[2]: .vocab file to create vocabulary
"""
function main(ARGS)
    println("Reading file $(ARGS[1])...")
    vocab = vocab_from_conllu(ARGS[1])
    open(ARGS[2], "w") do f
        for (k, v) in vocab
            write(f, "$k\n")
        end
    end
    println("Writing to vocab file $(ARGS[2])...")
end

!isinteractive() && main(ARGS)
