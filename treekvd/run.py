import sys,os
import argparse
import json
import pandas as pd
from sklearn.model_selection import ParameterGrid

from runners import SentTreeKvDRunner

def generate_system_id(params):
  return f'wm{params["wm_size"]}_c.{params["scale1"]}_r.{params["num_recall"]}_{params["selection"]}'


def serve_params(args):
  if args.conf.endswith(".json"):
    param_grid = json.load(open(args.conf,"r"))
    for params in ParameterGrid(param_grid):
      yield params,generate_system_id(params)

  elif args.conf.endswith(".csv"):
    configs = pd.read_csv(args.conf) # check if DATAFRAME row can be treated as dict
    for i,params in configs.iterrows():
      yield dict(params),generate_system_id(dict(params))



if __name__=="__main__":
  parser = argparse.ArgumentParser() 
  # main config
  parser.add_argument("--dataset", "-d", type=str, help="dataset name [cnn,dm,books]", default="pubmed")
  parser.add_argument("--split", "-s", type=str, help="split [strain,train,valid,test]", default="valid")
  parser.add_argument("--njobs",'-nj', default=4, type=int, help="num jobs")
  parser.add_argument("--data_block_size",'-dbs', default=100, type=int, help="size of data block to load")

  parser.add_argument("--log_dir",type=str, help="log dir", default="exps")
  parser.add_argument("--exp_id", type=str, help="experiment dir", default="dataset-split-sel")

  # selection
  parser.add_argument('--force', action="store_true",help="force running")

  # use param file
  parser.add_argument("--conf", type=str, help="params as space or list(.json,.csv)", default="conf_files/default.csv")

  args = parser.parse_args()

  exp_dir = os.path.join(args.log_dir,args.exp_id)
  os.system("mkdir -p " + exp_dir)
  cnt = 0
  for params,system_id in serve_params(args):
    print(">[pipe]  (%d) %s " % (cnt,system_id))
    print(">[pipe]       ",params)
    runner = SentTreeKvDRunner(args,params,system_id)
    runner.run()
    cnt += 1
  #
