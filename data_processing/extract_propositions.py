import os,sys

sys.path.append("../common")

from prop_utils import *
from pprint import pprint

import json
import argparse

import time
import pdb

import pandas as pd
import pickle
from collections import OrderedDict,defaultdict
import multiprocessing
from multiprocessing import Pool

DATASET_BASEDIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"datasets")

class Processor:
  def __init__(self,in_fn,out_fn,args):
    self.input_fn = in_fn
    self.output_fn = out_fn
    self.in_block = 100
    self.njobs = args.njobs
    self.args = args

  # reads in blocks
  # def read_lines(self,):
  #   block = []
  #   for line in open(self.input_fn,"r"):
  #     line=line.strip("\n")
  #     if line=="": continue
  #     block.append(json.loads(line))
  #     if len(block)==self.in_block:
  #       yield block
  #       block = []
  #   #
  #   if len(block)>0:
  #     yield block 

  def read_lines(self,):
    id_len = []
    data = {}
    for line in open(self.input_fn,"r"):
      line=line.strip("\n")
      if line=="": continue
      item = json.loads(line)
      did = item["id"]
      data[did] = item
      id_len.append( (did,sum([len(x.strip("\n").split("\n\n")) for x in item["parsed_sections"]])) )
    id_len.sort(key=lambda x:x[1])
    for i in range(0,len(data),self.in_block):
      yield [data[x] for x,_ in id_len[i:i+self.in_block]]



  def run_parallel(self,item):
    did = item["id"]
    section_builder = PropositionExtractor()
    abstract_builder = PropositionExtractor()
    for sec_cnllu_line in item["parsed_sections"]:
      section_builder.update(sec_cnllu_line)
    abstract_builder.update(item["parsed_abstract"])

    if self.args.dataset in ["arxiv","pubmed"]:
      new_item = {
        "id" : did,
        "section_names" : item["section_names"],
        "doc_props" : section_builder.D,                # proposition dict
        "section_sents": section_builder.section_sents,   # sents by section
        "section_ptrees": section_builder.section_ptrees, # prop trees by section

        "abs_props" : abstract_builder.D,
        "abs_sents": abstract_builder.section_sents[0],
        "abs_ptrees": abstract_builder.section_ptrees[0],

      }
    else:
      new_item = {
        "id" : did,
        "doc_props" : section_builder.D,                # proposition dict
        "section_sents": section_builder.section_sents,   # sents by section
        "section_ptrees": section_builder.section_ptrees, # prop trees by section

        "abs_props" : abstract_builder.D,
        "abs_sents": abstract_builder.section_sents[0],
        "abs_ptrees": abstract_builder.section_ptrees[0],
      }

    return new_item



  def main(self,):
    total_time = 0.0
    total_props = 0.0
    cnt = 0
    with open(self.output_fn,"wb") as outfile:
      pkler = pickle.Pickler(outfile)
      if self.njobs==1:
        for block in self.read_lines():
          for item in block:
            # start = time.time() ##
            clean_item = self.run_parallel(item)
            # done = time.time()
            # nprops = len(clean_item["section_props"]) + len(clean_item["abstract_props"])
            # print("[item] Duration: ",done - start)
            # print("[item] Nprops=",nprops)
            # print()
            # total_time += done - start
            # total_props += nprops
            pkler.dump(clean_item)

            # if cnt>5: break
            cnt += 1
          #
        #
        # print("[main] Average Prop extraction time per prop:",total_time / total_props)
        # print("[main] Average Prop extraction time per doc:",total_time / cnt)
      #
          #
      #
      else:
        with Pool(processes=self.njobs) as pool:
          for block in self.read_lines():
            clean_block = pool.map(self.run_parallel,block)
            for clean_item in clean_block:
              pkler.dump(clean_item)
          #
        #
      #
    #


if __name__ == '__main__':
  parser = argparse.ArgumentParser() 
  parser.add_argument("--dataset", "-d", type=str, help="dataset name [cnn,dm]", default="pubmed")
  parser.add_argument("--split", "-s", type=str, help="split [strain,train,valid,test]", default="valid")
  parser.add_argument("--njobs", "-nj", type=int, help="n cpus", default=4)

  args = parser.parse_args()

  in_fname = os.path.join(DATASET_BASEDIR,args.dataset,"%s-ud.jsonl" % args.split)
  out_fname = os.path.join(DATASET_BASEDIR,args.dataset,"%s-props.pkl" % args.split)
  # out_fname = "timing-test.pkl"
  proc = Processor(in_fname,out_fname,args)
  proc.main()
