"""
Re-process / tokenize PubMed and ArXiv
(thank you Cohan)
"""
import os
import pdb
import numpy as np
from collections import defaultdict, OrderedDict

import argparse
import pickle
import json

import pprint as pp
import warnings
import copy as cp
import re
from pprint import pprint

from sacremoses import MosesTokenizer, MosesDetokenizer
from sacremoses import MosesPunctNormalizer
import nltk
import multiprocessing
from multiprocessing import Pool
from utils import *


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class Processor:
  def __init__(self,input_fn,output_fn,n_cpus,block_size,args):
    self.args = args
    self.input_fn = input_fn
    self.output_fn =output_fn
    self.in_block = block_size
    self.mpn = MosesPunctNormalizer()
    self.mt = MosesTokenizer(lang='en')
    self.md = MosesDetokenizer(lang='en')
    self.n_cpus = n_cpus
    self.re_fix_beg = re.compile(r"|".join(["(%s)"%x for x in fix_beg]) + r"|([([]{,2}[figeqta]+)" + r'|(\(?\s*[0-9]+)')
    self.re_replace_sec_in = re.compile(r"\[\s*sec\s*:\s*[a-z0-9\s-]+\]") # .. -> @sec
    self.re_replace_ref_in = re.compile(r"\[\s*[figeq]{2,3}\s*:\s*[a-z0-9\s-]+\]") # .. -> @xref
    self.re_replace_sec_compl = [ # .. -> sec. @sec
      re.compile(r"sec\s*[.]\s*\[\s*[a-z0-9-]+\s*\]"),
      # re.compile(r"sec(t)?(ion)?\s*[.]?\s*[0-9ivx]{1,3}"),
    ]
    self.re_replace_nocite = re.compile(r"[*]\s*[?\s]+\s*[*]?")
    self.re_repeated = re.compile(r'[-_+ ]{3,}')
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
      if self.id_list != "ALL" and item["article_id"] not in self.id_list:
        continue
      block.append(json.loads(line))
      if len(block)==self.in_block:
        yield block
        block = []
    #
    yield block

  def sent_tokenize(self,text):
    sents = text.split("\n")
    n = len(sents)
    new_sents = [sents[0]]
    if len(sents)==1:
      return new_sents
    ini = 1
    if has_special_fix(sents[0]) or has_too_few_toks(sents[0]):
      new_sents = [sents[0] + " " + sents[1]]
      ini = 2
    for i in range(ini,n):
      if has_special_fix(sents[i]) or has_too_few_toks(sents[i]):
        new_sents[-1] = new_sents[-1] + " " + sents[i]
      elif new_sents[-1].endswith("i.e.") or new_sents[-1].endswith("e.g."):
        new_sents[-1] = new_sents[-1] + " " + sents[i]
      elif any([new_sents[-1].endswith(x) for x in fix_endings]):
        match = self.re_fix_beg.match(sents[i])
        if match is None:
          new_sents.append(sents[i])
        else:
          new_sents[-1] = new_sents[-1] + " " + sents[i]
      else:
        new_sents.append(sents[i])
    #
    ## another round of join based on has_
    if any([has_too_few_toks(x) or has_special_fix(x) for x in new_sents]):
      new_sents = self.sent_tokenize("\n".join(new_sents))
    return new_sents
  

  def run_parallel(self,item):
    idxs = [i for i,x in enumerate(item["sections"]) if x!=[""]]
    clean_secs = [self.clean_text(" ".join(item["sections"][i])) for i in idxs]
    # split sents
    clean_secs = [self.sent_tokenize(x) for x in clean_secs]
    # add reference filter here###
    abstract = [x.replace("<S>","").replace("</S>","") for x in item["abstract_text"]]
    abstract = [self.clean_text(x,do_sent_tok=False) for x in abstract]
    abstract = [x for x in abstract if x!=""]
    abstract = self.sent_tokenize("\n".join(abstract))
    clean_item = {
      "article_id" : item["article_id"],
      "section_names" : [self.clean_text(item["section_names"][i],do_sent_tok=False) for i in idxs],
      "sections": clean_secs,
      "abstract_text" : abstract,
    }
    return clean_item

  def main(self,):
    if self.n_cpus==1:
      with open(self.output_fn,"w") as outfile:
        for block in self.read_lines():
          for item in block:
            clean_item = self.run_parallel(item)

            # print("Original-----------------------------------------")
            # pprint(item["sections"][0])
            # print()
            # print("Cleaned-----------------------------------------")
            # pprint(clean_item["sections"][0])
            # print("[[[check abstract")
            # pdb.set_trace()
            # print()

            outfile.write(json.dumps(clean_item) + "\n")

    else:
      with Pool(processes=self.n_cpus) as pool:
        with open(self.output_fn,"w") as outfile:
          for block in self.read_lines():
            clean_block = pool.map(self.run_parallel,block)
            for item in clean_block:
              outfile.write(json.dumps(item) + "\n")
        #
      #


  def clean_text(self,text,do_sent_tok=True):
    text = text.replace(":"," : ").replace(","," , ")
    pre = None
    while True:
      pre = self.re_replace_ref_in.sub("@xref",text)
      pre = self.re_replace_sec_in.sub("@sec",pre)
      for pat in self.re_replace_sec_compl:
        pre = pat.sub("section @sec",pre)
      pre = self.re_replace_nocite.sub("@xcite",pre)
      pre = self.re_repeated.sub("_",pre)
      if pre==text: break
      text = pre
    #
    text = pre
    # text = self.md.detokenize(self.mt.tokenize(self.md.detokenize(self.mpn.normalize(text).split())))
    raw = self.md.detokenize(self.mpn.normalize(text).split())
    if do_sent_tok:
      text = []
      for line in nltk.sent_tokenize(raw):
        line = line.strip(" ")
        if line=="": continue
        line = " ".join(self.mt.tokenize(line))
        line = line.replace("@ xcite","@xcite").replace("@ xmath","@xmath").replace("@ sec","@sec") \
                  .replace("ref .","ref.").replace("fig .","fig.").replace("eq .","eq.").replace("sec .","sec.") \
                  .replace("e.g .","e.g.").replace("i.e .","i.e.")
        text.append(line)
      text = "\n".join(text)
    else:
      text = " ".join(self.mt.tokenize(raw))
      text = text.replace("@ xcite","@xcite").replace("@ xmath","@xmath").replace("@ sec","@sec") \
                  .replace("ref .","ref.").replace("fig .","fig.").replace("eq .","eq.").replace("sec .","sec.") \
                  .replace("e.g .","e.g.").replace("i.e .","i.e.")
    return text



if __name__=="__main__":
  parser = argparse.ArgumentParser() 
  # main config
  parser.add_argument("--dataset", "-d", type=str, help="dataset name [cnn,dm,books]", default="pubmed")
  parser.add_argument("--split", "-s", type=str, help="split [strain,train,valid,test]", default="valid")
  parser.add_argument("--block_size", "-bs", type=int, help="block size for loading data", default=100)
  parser.add_argument("--id_list", "-ilist", type=str, help="list of ids to include", default="ALL")
  parser.add_argument("--n_cpus", "-n", type=int, help="n cpus", default=4)

  args = parser.parse_args()
  in_split = "train" if args.split == "strain" else args.split
  in_fname = os.path.join(DATASET_BASEDIR,args.dataset,"%s.jsonl" % in_split)
  out_fname = os.path.join(DATASET_BASEDIR,args.dataset,"%s-retok.jsonl" % args.split)
  proc = Processor(in_fname,out_fname,args.n_cpus,args.block_size,args)
  proc.main()
