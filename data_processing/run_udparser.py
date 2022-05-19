"""
Applies the cohan's discard filter
- abstract min of 5 tokens
- each sections min of 5 tokens
- each sent truncated to 200 toks
"""

import os,sys
import json
import random
import numpy as np
import argparse
import re
import pdb

import subprocess as sp
from subprocess import Popen
import nltk
import multiprocessing
from multiprocessing import Pool
import time

from utils import *

DATASET_BASEDIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"datasets")


class Processor:
  def __init__(self,in_fn,out_fn,args):
    self.input_fn = in_fn
    self.output_fn = out_fn
    self.in_block = 500
    self.njobs = args.njobs
    self.args = args
    self.tmp_dir = "tmp-%s-%s" % (self.args.dataset,self.args.split)
    # regexs
    self.reg_citations = re.compile(r"(@\s*xcite[0-9]*)|(\[[,0-9]+\])")
    self.reg_eqs = re.compile(r"@\s*xmath[0-9]*")
    self.reg_empty = re.compile(r"(\(\s*\)|\[\s*\])")
    self.id_list = None
    if args.id_list != "ALL":
      self.id_list = [x for x in open(args.id_list,"r").read().strip("\n").split("\n") if x!=""]
      print("[Processor] loaded ids=",len(self.id_list))
    else:
      self.id_list = args.id_list

  # reads in blocks
  def read_lines(self,):
    block = []
    for line in open(self.input_fn,"r"):
      line=line.strip("\n")
      if line=="": continue
      item = json.loads(line)
      if self.args.dataset in ["arxiv","pubmed"]:
        item = doc_filter_latex(item)
      if item is None: continue
      block.append(item)
      if len(block)==self.in_block:
        yield block
        block = []
    #
    if len(block)>0:
      yield block    

  def filter(self,text):
    text = self.reg_citations.sub("",text)
    text = self.reg_eqs.sub("",text)
    text = self.reg_empty.sub("",text)
    text = " ".join(text.split()[:MAX_TOKENS])
    return text

  def run_parser(self,did,idx,sents):
    sents = [self.filter(x) for x in sents]
    fn = os.path.join(self.tmp_dir,"%s.%d" % (did,idx))
    with open(fn,"w") as outfn:
      outfn.write("\n".join(sents)+"\n")

    with Popen(["udpipe","--tokenizer=presegmented","--tag","--parse","--output","conllu=v2","../../tools/english-ewt-ud-2.5-191206.udpipe",fn],\
      stdout=sp.PIPE) as pobj:
      parsed = pobj.stdout.read().decode("utf-8")
    try:
      os.system("rm -r "+fn)
    except:
      print(">> couldnt remove folder ",fn)
    return parsed


  def run_parallel(self,item):
    if self.args.dataset in ["arxiv","pubmed"]:
      return self.process_cohan(item)
    elif self.args.dataset == "bigpat":
      return self.process_bigpat(item)


  def process_cohan(self,item):
    did = item["article_id"]
    idxs = []
    sec_names = [x.lower() for x in item["section_names"]]
    idxs = list(range(len(sec_names)))
    # if   self.args.doc_type == "doc":
    # elif self.args.doc_type == "idc":
    #   idxs = [i for i,x in enumerate(sec_names) \
    #       if any(["introduction" in x, "discussion" in x, "conclusion" in x, "summary " in x])]
    if len(idxs)==0:
      return None
    sections = [item["sections"][i] for i in idxs]
    sec_names = [sec_names[i] for i in idxs]

    new_item = {
      "id" : did,
      "section_names": sec_names,
      "parsed_sections": [self.run_parser(did,i,x) for i,x in enumerate(sections)],
      "parsed_abstract": self.run_parser(did+"-abs",0,item["abstract_text"])
    }
    return new_item

  def process_bigpat(self,item):
    did = item["id"]
    sections = item["sections"]

    new_item = {
      "id" : did,
      "parsed_sections": [self.run_parser(did,i,x) for i,x in enumerate(sections)],
      "parsed_abstract": self.run_parser(did+"-abs",0,item["abstract_text"])
    }

    return new_item


  def main(self,):
    os.system("mkdir -p " + self.tmp_dir)
    total_time = 0.0
    total_sents = 0.0
    cnt = 0
    if self.njobs==1:
      with open(self.output_fn,"w") as outfile:
        for block in self.read_lines():
          for item in block:
            # start = time.time() ##
            clean_item = self.run_parallel(item)
            if item is not None:
              # done = time.time()
              # nsents = sum(len(x) for x in item["sections"]) + len(item["abstract_text"])
              # print("[item] Duration: ",done - start  )
              # print("[item] Nsents=",nsents)
              # total_time += done - start
              # total_sents += nsents
              # pdb.set_trace()
              outfile.write(json.dumps(clean_item) + "\n")
              cnt += 1
            # if cnt>5: break
          #
      #
      # print("[main] Average UD time per sent:",total_time / total_sents)
      # print("[main] Average UD time per sent:",total_time / cnt)

    else:
      with Pool(processes=self.njobs) as pool:
        with open(self.output_fn,"w") as outfile:
          for block in self.read_lines():
            clean_block = pool.map(self.run_parallel,block)
            for item in clean_block:
              if item is not None:
                outfile.write(json.dumps(item) + "\n")
        #
      #



if __name__ == '__main__':
  parser = argparse.ArgumentParser() 
  parser.add_argument("--dataset", "-d", type=str, help="dataset name [cnn,dm]", default="pubmed")
  parser.add_argument("--split", "-s", type=str, help="split [strain,train,valid,test]", default="valid")
  # parser.add_argument("--doc_type", "-dt", type=str, help="whole article or intro [doc,idc,intersec]", default="idc")
  parser.add_argument("--id_list", "-ilist", type=str, help="list of ids to include", default="ALL")
  parser.add_argument("--njobs", "-nj", type=int, help="njobs", default=28)

  args = parser.parse_args()

  in_fname = os.path.join(DATASET_BASEDIR,args.dataset,"%s-retok.jsonl" % args.split)
  out_fname = os.path.join(DATASET_BASEDIR,args.dataset,"%s-ud.jsonl" % (args.split))
  # out_fname = "timing-test.jsonl"
  proc = Processor(in_fname,out_fname,args)
  proc.main()
