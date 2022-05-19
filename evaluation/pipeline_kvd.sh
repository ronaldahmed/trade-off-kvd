#!/bin/bash

set -e

DATASET="arxiv"
SPLIT="valid"
NJOBS=28
GPU=0
KVD="treekvd"

while [ $# -gt 1 ]
do
key="$1"
case $key in
    -d|--dataset)
    DATASET="$2"
    shift # past argument
    ;;
    -s|--split)
    SPLIT="$2"
    shift # past argument
    ;;
    -n|--njobs)
    NJOBS="$2"
    shift # past argument
    ;;
    -g|--gpu)
    GPU="$2"
    shift # past argument
    ;;
    -k|--kvd)
    KVD="$2"
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift
done 


CUDA_VISIBLE_DEVICES=${GPU} python run_srouge.py -d ${DATASET} -s ${SPLIT} -nj ${NJOBS} --pred ../${KVD}/exps/${DATASET}-${SPLIT}
CUDA_VISIBLE_DEVICES=${GPU} python run_scibert.py -d ${DATASET} -s ${SPLIT} -nj ${NJOBS} --pred ../${KVD}/exps/${DATASET}-${SPLIT}

python run_summeval.py -d ${DATASET} -s ${SPLIT} -nj ${NJOBS} --pred ../${KVD}/exps/${DATASET}-${SPLIT}

python report_evaluation.py -d ${DATASET} -s ${SPLIT} -nj ${NJOBS} --pred_file ../${KVD}/exps/${DATASET}-${SPLIT} --output results/${DATASET}-${SPLIT}-${KVD}.csv


