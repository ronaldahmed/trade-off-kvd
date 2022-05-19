import os
import sys

from pprint import pprint
import json
import pickle
import argparse
import numpy as np
import glob as gb

import matplotlib
import matplotlib.pyplot as plt

from collections import OrderedDict,defaultdict, Counter

sys.path.append("../common")
from prop_utils import Proposition
from utils import budgets
from eval_utils import *

import pdb

import torch
from transformers import BertTokenizer, BertForMaskedLM
from transformers import AutoTokenizer

import cProfile
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_data_block(fn,preds):
  data = []
  with open(fn,"rb") as infile:
    pkl = pickle.Unpickler(infile)
    while infile.peek(1):
      item = pkl.load()
      did = item["id"]
      if did not in preds: continue
      item["preds"] = preds[did]
      data.append(item)
      if len(data)==100:
        yield data
        data = []
    yield data
  #

if __name__ == '__main__':
  parser = argparse.ArgumentParser() 
  parser.add_argument("--dataset", "-d", type=str, help="dataset name [cnn,dm]", default="pubmed")
  parser.add_argument("--split", "-s", type=str, help="split [strain,train,valid,test]", default="valid")
  parser.add_argument("--njobs", "-nj", type=int, help="njobs", default=28)
  parser.add_argument("--mode", "-m", type=str, help="mode", default="dr")

  parser.add_argument("--pred", type=str, help="prediction file/folder (rel path)", default="debug")

  args = parser.parse_args()
  fn = f"../../datasets/{args.dataset}/{args.split}-props.pkl"
  predictions = json.load(open(args.pred,"r"))[0]

  dev = "cuda" if torch.cuda.is_available() else "cpu"

  model = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased').to(dev)
  tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

  sys_ppl = {}
  cnt = 0
  for block in load_data_block(fn,predictions):
    for item in block:
      cand = [item["section_sents"][x][y] for x,y in item["preds"]]
      sys_ppl[item["id"]] = calc_ppl(model,tokenizer," ".join(cand),tokenizer.mask_token_id)

      if cnt % 500 == 0:
        print(">",cnt,sys_ppl[item["id"]])
      cnt += 1
  #
  ofn = args.pred[:-4] + "ppl"
  with open(ofn,"w") as out:
    json.dump(sys_ppl,out)

