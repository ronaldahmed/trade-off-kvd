import os,sys
import argparse
import json
import pickle
import pandas as pd
import glob as gb
import numpy as np
import pdb

# DSTATS = [
# 'coverage', 'density', 'compression', 'summary_length',
# "u-red","nid-red","rep-red",
# "rsc_u-red","rsc_nid-red","rsc_rep-red",
# # 'percentage_novel_1-gram',
# # 'percentage_novel_2-gram',
# # 'percentage_novel_3-gram',
# # 'percentage_repeated_1-gram_in_summ',
# # 'percentage_repeated_2-gram_in_summ',
# # 'percentage_repeated_3-gram_in_summ',
# ]


def get_str(values):
  if len(values)==0: return "-"
  return "%.2f(%.2f)" % (np.mean(values),np.std(values))


if __name__ == '__main__':
  parser = argparse.ArgumentParser() 
  parser.add_argument("--dataset", "-d", type=str, help="dataset name [cnn,dm]", default="pubmed")
  parser.add_argument("--split", "-s", type=str, help="split [strain,train,valid,test]", default="valid")
  parser.add_argument("--njobs", "-nj", type=int, help="njobs", default=28)

  parser.add_argument("--pred_conf", type=str, help="list of files/folders to predict (rel path)", default=None)
  parser.add_argument("--pred_file", type=str, help="list of files/folders to predict (rel path)", default=None)
  parser.add_argument("--output", type=str, help="output csv file", default="debug.csv")
  parser.add_argument("--sort", action="store_true",help="sort output results")

  args = parser.parse_args()

  print("[AGGREGATION] ---------------------------------------")
  print(args)

  res_table = pd.DataFrame()
  
  if args.pred_conf is not None and args.pred_file is not None:
    print("[err] Specify prediction conf file or prediction file, but not both")
    sys.exit(1)

  pred_list = []
  if args.pred_conf is not None:
    pred_list = open(args.pred_conf,"r").read().strip("\n").split("\n")
  elif args.pred_file is not None:
    pred_list = [args.pred_file]
  else:
    print("[err] Specify one of the following:prediction conf file,prediction file")
    sys.exit(1)

  for pred_item in pred_list:
    pred_files = [pred_item]
    if os.path.isdir(pred_item):
      pred_files = gb.glob(os.path.join(pred_item,"*.json"))

    for pred_file in pred_files:
      print("[main] processing {%s}" % pred_file)
      base = pred_file[:-4]
      try:  
        assert os.path.exists(base+"srouge")
        assert os.path.exists(base+"scibert")
        assert os.path.exists(base+"dstats")
      except:
        pdb.set_trace()

      bscore = json.load(open(base+"scibert","r"))
      rouge  = json.load(open(base+"srouge","r"))
      dstats = json.load(open(base+"dstats","r"))
      dstats_keys = list((list(dstats.values())[0]).keys())

      item = {
        "System": os.path.basename(base.rstrip(".")),
        "ScBert-F1": get_str([x["f1"] for x in bscore.values()])
      }
      for k in ["rouge-1","rouge-2","rouge-l"]:
        item[k] = get_str([x[k]["f1"] for x in rouge.values()])
      for k in dstats_keys:
        if k in ["coverage","density","compression","summary_length"]:
          vals = [x.get(k,0) for x in dstats.values()]
        else:
          vals = [x.get(k,0)*100.0 for x in dstats.values()]
        if "summary_length" == k:
          item[k] = get_str(vals)
        else:
          item[k] = "%.2f" % (np.mean(vals))
      res_table = res_table.append(pd.Series(item,name="."))
      if args.sort:
        res_table = res_table.sort_values(by=["rouge-l"],ascending=False)
      res_table.to_csv(args.output)
    #
  #
  print("Done!")
      