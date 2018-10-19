# Koç University Dependency Parser 
Dependency parser implementation used by [KParse](http://universaldependencies.org/conll18/results.html) team in CoNLL18 shared task. The model that we implemented explained in our paper titled as [Tree-stack LSTM in Transition Based Dependency Parsing](http://universaldependencies.org/conll18/proceedings/pdf/K18-2012.pdf)

### Prerequisites
We use text files tokenized by [UDPipe](http://ufal.mff.cuni.cz/udpipe), please make sure that you have installed it from their official [repository](https://github.com/ufal/udpipe).
All this code is working  with julia 0.6.2 current versions not supported yet.

### Installing
Clone the repository to install the parser and dependencies:

```sh
git clone https://github.com/kirnap/ku-dependency-parser2.git && cd ku-dependency-parser2
```
### Code Structure


#### Bi-LSTM Language Model
We used our pre-trained language model from CoNLL17 shared task and the code for that is given under our [CoNLL17 repository](https://github.com/kirnap/ku-dependency-parser) section LM


#### Parser Model Files
Since this is a research repository, code structure is a bit messy. Let's walk through the code structure. As we explained in the paper, we use morphological features only for some languages. Following command prints the dictionary where the true labels indicate that  morpohological features are used for that language.


##### Training
```sh
cat use_feats.jl
```
For example if we want to *train* en_lines here are the steps for that: 

 - 1. ```cat use_feats.jl | grep en_lines``` which gives true, therefore we need to train with

 - 2. ```julia train_feats3.jl --lmfile your/path/to/english_chmodel.jld --datafiles /your-path-to/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-train.conllu  /your/path/to/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-dev.conllu --bestfile your_model_file.jld```
 - 3. Suppose we want to train hu_szeged which is not using morphological features, thus we need tthe followin command
 - 4.```julia train_nofeats.jl --lmfile your/path/to/hu_szeged.jld --datafiles /your-path-to/hu_szeged.train.conllu  /your-path-to/hu_szeged.dev.conllu --bestfile your_model_file.jld```

##### Testing
Let's dive into the testing case
Suppose we want to test the performance of our en_lines model that we trained in the previous section
```sh
	julia train_feats3.jl --datafiles your-path-to/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-dev.conllu --loadfile your-path-to/en_lines.jld --epochs 0 --output your_testfile.conllu
```

Similarly if you want to test a model trained without morphological features (e.g. hu_szeged)
```sh
	julia train_nofeats.jl --datafiles your-path-to/ud-treebanks-v2.2/UD_Hungarian/hu_szeged.conllu --loadfile your-path-to/hu_szeged.jld --epochs 0 --output your_testfile.conllu
```
Please not that these commands creates .conllu formatted files with predicted 'head' and deprel columns


### Code details

In order to understand the code structure here is a brief explanation of some model files under ```src/``` directory:

1. src/_model_feat3_1.jl : contains the most current version of our model using morphological features as well.
2. src/model_nofeat1.jl : contains the most current version of our model not using morphological features 
3. src/model_nofeat_dyn.jl : contains the model which not uses morphological features and trained with dynamic oracle training that we explained in our paper

To better understand the code start from ```src/header.jl``` file, please note that you have to provide .conllu formatted file to our system.


## Additional help
For more help, you are welcome to [open an issue](https://github.com/kirnap/ku-dependency-parser/issues/new), or directly contact [okirnap@ku.edu.tr](mailto:okirnap@ku.edu.tr).
