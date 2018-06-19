using JSON
# TODO: include the final train script
# We need to preprocess the tira files, this probably will be the final .julia file

# Sample inputdir
# "/media/training-datasets/universal-dependency-learning/conll18-ud-development-2018-05-06"
# You may made it up your sample outdir
const PARSERS = "/media/data/parser" # Put goldmodels there
const ERENAYDIR # TODO: You need to create internal directory, where erenay creates its own file
function main(inputDir, outputDir)
    PARSERMODELS = readdir(PARSERS)
    metadir = joinpath(inputDir, "metadata.json")
    metadata = JSON.parsefile(metadir)
    
    for met in metadata # iterate through it

        # to parse
        outfile = joinpath(outputDir, met["outfile"]) 

        # give it to erenay
        erenayfile = joinpath(inputDir, met["psegmorfile"]) 

        langcode=met["lcode"]; trcode=met["tcode"]
        fullcode=string(langcode, "_", trcode) # give it to erenay
        # example erenay code to run

        if string(fullcode, ".jld") in PARSERMODELS # we have a trained model
            # inputfile = joinpath(ERENAYDIR, ) # TODO: fix this problem with erenay
            parser =  joinpath(PARSERS,  string(fullcode, ".jld"))
            println("I would look $parser")
        elseif string(langcode, ".jld") in PARSERMODELS
            
        else
            # TODO : erase those lines
            parser =  joinpath(PARSERS,  string(fullcode, ".jld"))
            println("I would look $parser")
        end
    end
end


# This is an example code to run
# bash srun2.sh ar_padt media/training-datasets/universal-dependency-learning/conll18-ud-trial-2018-05-06/ar_padt-udpipe.conllu /home/Kparse/mnet/myouts
