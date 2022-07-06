# KvD systems for redundancy and local coherence in summaries

We introduced two extractive summarization systems, TreeKvD and GraphKvD, based on the Micro-Macro Structure Theory of human reading comprehension (Kintsch & van Dijk, 1978), a.k.a. 'KvD'.  
This repository contains code to replicate the results presented in our [paper](https://arxiv.org/abs/2205.10192)  

```
@misc{https://doi.org/10.48550/arxiv.2205.10192,
  doi = {10.48550/ARXIV.2205.10192},
  url = {https://arxiv.org/abs/2205.10192},
  author = {Cardenas, Ronald and Galle, Matthias and Cohen, Shay B.},
  title = {On the Trade-off between Redundancy and Local Coherence in Summarization},
  publisher = {arXiv},
  year = {2022}
}
```


## Installation

Create a python environment with Python 3.7 and the libraries in `requirements.txt`.


## Dataset files

We provide the preprocessed files for the arXiv and PubMed datasets [here](https://uoe-my.sharepoint.com/:u:/g/personal/s1987051_ed_ac_uk/EZ-cKo_ROE5Jn5TwQYiHmFgBqDfqvCB-VRgdSqlhnPnIJA?e=cWT7kZ).  
Each dataset split is a line-wise pickle file, each sample with the format

```
{
	"id": Str, # original ID in Cohan dataset
	"section_names": List[Str], # List of section names
	"section_sents": List[List[Str]] # List of sentences, grouped by section
	"section_ptrees": List[List[(int,Dict)]] # List of tuples (tree root, proposition tree) corresponding to each source sentence, grouped by section
	"doc_props": Dict{int:Proposition} # Dictionary of propositions in source document, mapping their id to their Proposition object.
	"abs_sents": List[str] # List of sentences in the abstract
	"abs_ptrees": List[(int,Dict)] # # List of tuples (tree root, proposition tree) corresponding to each target (abstract) sentence
	"abs_props": Dict{int:Proposition} # Dictionary of propositions in the abstract
}
```

## Preprocessing ArXiv and PubMed from scratch

Download Cohan dataset from [here](https://github.com/armancohan/long-summarization) and unzip them.  
The preprocessing scripts assume the following directory structure
```
datasets/
	arxiv/
		train.jsonl
		valid.jsonl
		test.jsonl
	pubmed/
		train.jsonl
		valid.jsonl
		test.jsonl
redundancy-kvd/ (this repository)
```

You will need [UDPipe 1.2.0](https://ufal.mff.cuni.cz/udpipe/1) dependency parser, [here](https://github.com/ufal/udpipe/releases/download/v1.2.0/udpipe-1.2.0-bin.zip), and its pretrained model for English, [here](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/english-ewt-ud-2.5-191206.udpipe?sequence=17&isAllowed=y).

Then, refer to `data_preprocessing/preprocessing_steps.sh`.


### Running the KvD Summarizers

The systems load their hyperparameter configuration from a JSON or CSV file in the config_files subfolder.  
For instance, to run TreeKvD over the test set of arXiv with the hyper-parameters reported in the paper, use
```
cd treekvd/
python run.py -d arxiv -s test -nj <number-of-cpus> -eid <experiment-name> -c conf_files/recommended.json
```

The predictions will be saved in folder `treekvd/exps/<experiment-name>/arxiv-test/`.  
Use the same command from the `graphkvd` subfolder to run GraphKvD instead.  
If you wish to run more configurations at a time, you can add more rows to the CSV file or more elements to the corresponding list in the JSON file.


## Evaluation

We provide an evaluation pipeline that obtains relevance, redundancy, and local coherence metrics.
For instance, to evaluate the recently run TreeKvD over arXiv test set, run

```
bash pipeline_kvd.sh -d arxiv -s test -n <number-of-cpus> -g <gpu id> -k treekvd
```
To evaluate a GraphKvD model, use `-k graphkvd` instead.  

