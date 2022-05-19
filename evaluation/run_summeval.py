"""
Gets SummEval metrics, redundancy
"""
import os,sys
import argparse
import json
import pickle
import pandas as pd
import glob as gb
import pdb

from multiprocessing import Pool
from summ_eval.data_stats_metric import DataStatsMetric
from collections import Counter, defaultdict
from scipy.stats import entropy
import numpy as np


sys.path.append("../common")
from prop_utils import Proposition
from eval_utils import *

##################################################

def load_data_block(fn):
  data = []
  with open(fn,"rb") as infile:
    pkl = pickle.Unpickler(infile)
    while infile.peek(1):
      item = pkl.load()
      data.append(item)
      if len(data)==100:
        yield data
        data = []
    yield data
  #


def run_parallel(bundle):
  item,preds = bundle
  did = item["id"]
  evaluator = DataStatsMetric(n_gram=3,tokenize=False)
  doc = [x for sec in item["section_sents"] for x in sec]
  cand = [item["section_sents"][sec][sid] for sec,sid in preds]
  rdict = evaluator.evaluate_example(" ".join(cand)," ".join(doc))

  red_dict = cand_redundancy(cand)
  res = {}
  for k,v in red_dict.items():
    res[k] = v
    # res["rsc_" + k] = rescale(v,baseline[k])
  
  for k in ['coverage', 'density', 'compression', 'summary_length']:
    res[k] = rdict[k]

  return did,res

def rescale(val,baseline):
    return (val - baseline + 1e-15) / (1.0 - baseline + 1e-15)


# def get_red_baselines(args):
#   abs_red_scores = json.load(open(f"analysis/{args.dataset}-{args.split}_abs-red.json"))
#   # ured = np.array([x["ref_u"] for x in abs_red_scores.values()])
#   # nidred = np.array([x["ref_nid"] for x in abs_red_scores.values()])
#   # rep = np.array([x["ref_rep-red"] for x in abs_red_scores.values()])

#   baselines = {
#       "u-u": np.mean([x["ref_u-u"] for x in abs_red_scores.values()]),
#       "u-b": np.mean([x["ref_u-b"] for x in abs_red_scores.values()]),
#       "u-t": np.mean([x["ref_u-t"] for x in abs_red_scores.values()]),
#       "u": np.mean([x["ref_u"] for x in abs_red_scores.values()]),
#       "nid-u": np.mean([x["ref_nid-u"] for x in abs_red_scores.values()]),
#       "nid-b": np.mean([x["ref_nid-b"] for x in abs_red_scores.values()]),
#       "nid-t": np.mean([x["ref_nid-t"] for x in abs_red_scores.values()]),
#       "nid": np.mean([x["ref_nid"] for x in abs_red_scores.values()]),
#       # "rep-u": np.mean([x["ref_rep-u"] for x in abs_red_scores.values()]),
#       # "rep-b": np.mean([x["ref_rep-b"] for x in abs_red_scores.values()]),
#       # "rep-t": np.mean([x["ref_rep-t"] for x in abs_red_scores.values()]),
#       # "rep": np.mean([x["ref_rep"] for x in abs_red_scores.values()]),
#       "r1": np.mean([x["ref_r1"] for x in abs_red_scores.values()]),
#       "r2": np.mean([x["ref_r2"] for x in abs_red_scores.values()]),
#       "rl": np.mean([x["ref_rl"] for x in abs_red_scores.values()]),
#   }
#   # return tuple(list(baselines.items()))
#   return baselines

############################################################################


if __name__ == '__main__':
  parser = argparse.ArgumentParser() 
  parser.add_argument("--dataset", "-d", type=str, help="dataset name [cnn,dm]", default="pubmed")
  parser.add_argument("--split", "-s", type=str, help="split [strain,train,valid,test]", default="valid")
  parser.add_argument("--njobs", "-nj", type=int, help="njobs", default=28)

  parser.add_argument("--pred", type=str, help="prediction file/folder (rel path)", default="debug")
  parser.add_argument('--force', action="store_true",help="force running")

  args = parser.parse_args()

  print("[EVALUATION] SummEval - Data stats ----------------------------")
  print(args)

  pred_files = [args.pred]
  if os.path.isdir(args.pred):
    pred_files = gb.glob(os.path.join(args.pred,"*.json"))

  # baselines = get_red_baselines(args)

  print("Loaded:",len(pred_files),"files")
  for pred_file in pred_files:
    print("[main] processing {%s}" % pred_file)
    outfn = pred_file[:-4] + "dstats"
    if not args.force and os.path.exists(outfn):
      continue

    # load dataset every time a system is evaluated
    metrics = {}
    try:
      cand_idxs,_ = json.load(open(pred_file,"r"))
    except:
      cand_idxs = json.load(open(pred_file,"r"))
    fn = f"../../datasets/{args.dataset}/{args.split}-props.pkl"
    
    with Pool(args.njobs) as pool:
      for block in load_data_block(fn):
        bundle = [[item,cand_idxs[item["id"]]] for item in block if item["id"] in cand_idxs]
        res = pool.map(run_parallel,bundle)
        for did,metr in res:
          metrics[did] = {k:v for k,v in metr.items()}
    #
    
    with open(outfn,"w") as outfile:
      json.dump(metrics,outfile)
    #
  print("Done!")
