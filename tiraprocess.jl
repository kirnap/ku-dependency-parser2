# TODO: include the final train script
# We need to preprocess the tira files, this probably will be the final .julia file

# Sample inputdir
# "/media/training-datasets/universal-dependency-learning/conll18-ud-development-2018-05-06"
# You may made it up your sample outdir
const PARSERS = "/media/data/parser" # Put goldmodels there
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

        if string(fullcode, ".jld") in PARSERMODELS # we have a trained model
            parser =  joinpath(PARSERS,  string(fullcode, ".jld"))
            println("I would look $parser")
        elseif string(langcode, ".jld") in PARSERMODELS
            
        else
            println("I would look $parser")
        end
    end
end
