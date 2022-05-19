import os,sys
import argparse
import json
import pickle
import pandas as pd
import glob as gb
import pdb
from collections import defaultdict
from math import log

MYTOOLS = os.environ.get("MYTOOLS",None)
sys.path.append("../common")
if MYTOOLS is not None:
  sys.path.append(MYTOOLS)
from scibert_score.scorer import SciBertScorer
from eval_utils import load_data_block


N_DOCS = {
  "arxiv": 23532536,
  "pubmed": 10894852,
}

############################################################################


if __name__ == '__main__':
  parser = argparse.ArgumentParser() 
  parser.add_argument("--dataset", "-d", type=str, help="dataset name [cnn,dm]", default="pubmed")
  parser.add_argument("--split", "-s", type=str, help="split [strain,train,valid,test]", default="valid")
  parser.add_argument("--njobs", "-nj", type=int, help="njobs", default=28)

  parser.add_argument("--pred", type=str, help="prediction file/folder (rel path)", default="debug")
  parser.add_argument('--force', action="store_true",help="force running")

  args = parser.parse_args()

  print("[EVALUATION] SCIBERT----------------------------")
  print(args)

  pred_files = [args.pred]
  if os.path.isdir(args.pred):
    pred_files = gb.glob(os.path.join(args.pred,"*.json"))
  # load preprocessed vars
  idf_dict = defaultdict(lambda: log((N_DOCS[args.dataset] + 1) / (1)))
  with open(f"../../datasets/{args.dataset}/idf_dict.pkl","rb") as infile:
    obj = pickle.load(infile)
    for k,v in obj.items():
      idf_dict[k] = v
  baseline_fn = f"/disk/ocean/rcardenas/datasets/{args.dataset}/scb_baseline.tsv"
  bscore = SciBertScorer(model_type="allenai/scibert_scivocab_uncased",
                         idf=True, idf_dict=idf_dict,
                         nthreads=args.njobs)
                         # rescale_with_baseline=True,
                         # baseline_path=baseline_fn,
  
  print("Loaded:",len(pred_files),"files")
  for pred_file in pred_files:
    print("[main] processing {%s}" % pred_file)
    #
    outfn = pred_file[:-4] + "scibert"
    if not args.force and os.path.exists(outfn):
      continue

    # load dataset every time a system is evaluated
    summaries = []
    references = []
    metrics = {}
    try:
      cand_idxs,_ = json.load(open(pred_file,"r"))
    except:
      cand_idxs = json.load(open(pred_file,"r"))
    fn = f"../../datasets/{args.dataset}/{args.split}-props.pkl"
    did_seq = []
    for block in load_data_block(fn):
      for item in block:
        did = item["id"]
        summ = [item["section_sents"][sec][sid] for sec,sid in cand_idxs[did]]
        did_seq.append(did)
        summaries.append(summ)
        references.append([item["abs_sents"]])
    #
    bsc_list = bscore.score_all(summaries,references)
    for did,bsc in zip(did_seq,bsc_list):
      metrics[did] = {k:v*100.0 for k,v in bsc["bertscore"].items()}
    #
    with open(outfn,"w") as outfile:
      json.dump(metrics,outfile)
    #
  print("Done!")
