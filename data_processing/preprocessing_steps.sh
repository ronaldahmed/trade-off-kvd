

export dataset="arxiv"
export split="test"
export njobs=20

## Tokenization and cleaning

python retokenize_cohan.py -d ${dataset} -s ${split} -n ${njobs}


## Running dependency parser
# add udpipe available to call from terminal
export PATH=PATH:<path-to-udpipe>/bin-linux64

python run_udparser.py -d ${dataset} -s ${split} -n ${njobs} -parser <path-to-udpipe-model>

# For instance, the UDPipe model for english is named `english-ewt-ud-2.5-191206.udpipe'


## Extract propositions from each sentence in source and target texts

python extract_propositions.py -d ${dataset} -s ${split} -n ${njobs}

