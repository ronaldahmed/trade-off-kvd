import os,sys
import argparse
import json
import pickle
import pandas as pd
import glob as gb
import pdb

from sacrerouge.metrics import Rouge,Meteor

sys.path.append("../common")
from prop_utils import Proposition


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


############################################################################


if __name__ == '__main__':
  parser = argparse.ArgumentParser() 
  parser.add_argument("--dataset", "-d", type=str, help="dataset name [cnn,dm]", default="pubmed")
  parser.add_argument("--split", "-s", type=str, help="split [strain,train,valid,test]", default="valid")
  parser.add_argument("--njobs", "-nj", type=int, help="njobs", default=28)

  parser.add_argument("--pred", type=str, help="prediction file/folder (rel path)", default="debug")
  parser.add_argument('--force', action="store_true",help="force running")

  args = parser.parse_args()

  print("[EVALUATION] Sacrerouge - ROUGE ----------------------------")
  print(args)


  rouge =  Rouge(max_ngram=4,
                 use_porter_stemmer=True,
                 compute_rouge_l=True,
                 wlcs_weight=1.2,
                 scoring_function='average')

  pred_files = [args.pred]
  if os.path.isdir(args.pred):
    pred_files = gb.glob(os.path.join(args.pred,"*.json"))

  print("Loaded:",len(pred_files),"files")
  for pred_file in pred_files:
    outfn = pred_file[:-4] + "srouge"
    if not args.force and os.path.exists(outfn):
      continue
    print("[main] processing {%s}" % pred_file)
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
    rsc_list = rouge.score_all(summaries,references)

    for _id,rsc in zip(did_seq,rsc_list):
      metrics[_id] = {}
      for k in ["rouge-1","rouge-2","rouge-4","rouge-l"]:
        metrics[_id][k] = rsc[k]
    
    with open(outfn,"w") as outfile:
      json.dump(metrics,outfile)
    #
  #

  print("Done!")
