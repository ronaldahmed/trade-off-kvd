import os,sys
import argparse
import json
import pickle
import pandas as pd
import glob as gb
import pdb
from collections import defaultdict
from math import log
import logging
import transformers
from sacrerouge.common.util import flatten
from sacrerouge.data import MetricsDict
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
from bert_score import BERTScorer

sys.path.append("../common")
from eval_utils import load_data_block


############################################################################


def score_all(bscore, summaries, references_list):
  summaries = [flatten(summary) for summary in summaries]
  references_list = [[flatten(reference) for reference in references] for references in references_list]

  # Create the candidate and reference lists for passing to the scoring function
  input_candidates = []
  input_references = []
  empty_inputs = set()
  for i, (summary, references) in enumerate(zip(summaries, references_list)):
      if len(summary) == 0:
          empty_inputs.add(i)
      else:
          input_candidates.append(summary)
          input_references.append(references)

  # Score the summaries
  precisions, recalls, f1s = bscore.score(input_candidates,input_references)
  # Remap the scores to the summaries
  index = 0
  metrics_lists = []
  for i, summary in enumerate(summaries):
      if i in empty_inputs:
          precision, recall, f1 = 0.0, 0.0, 0.0
      else:
          precision = precisions[index].item()
          recall = recalls[index].item()
          f1 = f1s[index].item()
          index += 1
      metrics_lists.append(MetricsDict({
          'bertscore': {
              'precision': precision,
              'recall': recall,
              'f1': f1,
          }
      }))
  return metrics_lists


"""
loads 100k sentences from the training set
"""
def get_idf_sentences(dataset):
  sents = []
  cnt = 0
  for line in open(f"../../datasets/{dataset}/train-retok.jsonl"):
    line = line.strip("\n")
    if line=="": continue
    item = json.loads(line)
    sents.extend([x for sec in item["sections"] for x in sec])
    if len(sents) >=100000:
      break
  return sents[:100000]


################################################################################


if __name__ == '__main__':
  parser = argparse.ArgumentParser() 
  parser.add_argument("--dataset", "-d", type=str, help="dataset name [cnn,dm]", default="pubmed")
  parser.add_argument("--split", "-s", type=str, help="split [strain,train,valid,test]", default="valid")
  parser.add_argument("--njobs", "-nj", type=int, help="njobs", default=28)

  parser.add_argument("--pred", type=str, help="prediction file/folder (rel path)", default="debug")
  parser.add_argument('--force', action="store_true",help="force running")

  args = parser.parse_args()

  print("[EVALUATION] SCIBERT-SCORE ----------------------------")
  print(args)

  pred_files = [args.pred]
  if os.path.isdir(args.pred):
    pred_files = gb.glob(os.path.join(args.pred,"*.json"))
  
  bscore = BERTScorer(lang="en-sci",
                      idf=True,
                      idf_sents=get_idf_sentences(args.dataset),
                      rescale_with_baseline=False,
                      nthreads=args.njobs)

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
    bsc_list = score_all(bscore,summaries,references)
    for did,bsc in zip(did_seq,bsc_list):
      metrics[did] = {k:v*100.0 for k,v in bsc["bertscore"].items()}
    #
    with open(outfn,"w") as outfile:
      json.dump(metrics,outfile)
    #
  print("Done!")
