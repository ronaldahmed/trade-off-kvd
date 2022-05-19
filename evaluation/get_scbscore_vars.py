"""
Preprocess vars for SciBert-Score
- IDF (isf / sent lvl) on training set
- Baseline
"""
import os,sys
import argparse
import json
import pickle
import pandas as pd
import numpy as np
import glob as gb
import pdb
from collections import defaultdict
from math import log

MYTOOLS = os.environ.get("MYTOOLS",None)
np.random.seed(42)

if MYTOOLS is not None:
  sys.path.append(MYTOOLS)
from scibert_score.scorer import SciBertScorer
from scibert_score.utils import get_tokenizer,get_idf_dict, model2layers



if __name__ == '__main__':
  parser = argparse.ArgumentParser() 
  parser.add_argument("--dataset", "-d", type=str, help="dataset name [cnn,dm]", default="pubmed")
  parser.add_argument("--split", "-s", type=str, help="split [strain,train,valid,test]", default="valid")
  parser.add_argument("--njobs", "-nj", type=int, help="njobs", default=28)

  args = parser.parse_args()
  sents = []
  cnt = 0
  for line in open(f"../../datasets/{args.dataset}/train-retok.jsonl"):
    line = line.strip("\n")
    if line=="": continue
    item = json.loads(line)
    sents.extend([x for sec in item["sections"] for x in sec])
    if cnt % 10000 == 0:
      print(">",cnt)
    cnt += 1

    if cnt > 120000: break
  #
  idf_fn = f"../../datasets/{args.dataset}/idf_dict.pkl"

  print("N sents=",len(sents))
  # sys.exit(0)

  if not os.path.exists(idf_fn):
    print("calc idf...")
    tokenizer = get_tokenizer("allenai/scibert_scivocab_uncased", True)
    idf_dict = get_idf_dict(sents, tokenizer, nthreads=args.njobs)

    with open(idf_fn,"wb") as outfile:
      pickle.dump({k:v for k,v in idf_dict.items()},outfile)
    print("IDF Done!")

  else:
    print("idf loaded...")
    num_docs = len(sents)
    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    with open(idf_fn,"rb") as infile:
      obj = pickle.load(infile)
      for k,v in obj.items():
        idf_dict[k] = v

  ## baseline
  bscore = SciBertScorer(model_type="allenai/scibert_scivocab_uncased",nthreads=args.njobs,idf=True,idf_dict=idf_dict)
  block = 10000

  idxs = np.arange(len(sents))
  np.random.shuffle(idxs)
  N = 100000
  idxs = idxs[:N]
  rr,pp,f1 = 0.,0.,0.
  total = 0
  summs = []
  refs = []

  for i in range(0,N,2):
    summs.append(sents[i])
    refs.append([sents[i+1]])
    if len(summs)==block:
      bsc_list = bscore.score_all(summs,refs)
      rr += sum([x["bertscore"]["recall"] for x in bsc_list])
      pp += sum([x["bertscore"]["precision"] for x in bsc_list])
      f1 += sum([x["bertscore"]["f1"] for x in bsc_list])
      total += block

      summs = []
      refs = []
    #
  #
  if len(summs) > 0:
    bsc_list = bscore.score_all(summs,refs)
    rr += sum([x["bertscore"]["recall"] for x in bsc_list])
    pp += sum([x["bertscore"]["precision"] for x in bsc_list])
    f1 += sum([x["bertscore"]["f1"] for x in bsc_list])
    total += len(summs)
  #
  base_rr = rr / total
  base_pp = pp / total
  base_f1 = f1 / total

  n_layers = model2layers["allenai/scibert_scivocab_uncased"]

  with open(f"../../datasets/{args.dataset}/scb_baseline.tsv","w") as outfile:
    print("LAYER","P","R","F",sep=",",file=outfile)
    for i in range(n_layers ):
      print(str(i),"0.","0.","0.",file=outfile,sep=",")
    print(str(n_layers - 1),base_rr,base_pp,base_f1,file=outfile,sep=",")
  #