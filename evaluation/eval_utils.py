import os,sys
import re
import numpy as np
import json
import pickle
from scipy.stats import pearsonr
from scipy.stats import entropy

from collections import OrderedDict,defaultdict, Counter
from summ_eval.data_stats_metric import DataStatsMetric
from intr_rouge.rouge import rouge_n_sentence_level as ROUGE_N
from intr_rouge.rouge import rouge_l_sentence_level as ROUGE_L
from summ_eval.data_stats_metric import DataStatsMetric
import torch

sys.path.append("../common")
from prop_utils import Proposition
from utils import budgets, STOPWORDS
import torch


DSTATS = [
'coverage', 'density', 'compression', 'summary_length',
'percentage_novel_1-gram',
'percentage_novel_2-gram',
'percentage_novel_3-gram',
'percentage_repeated_1-gram_in_summ',
'percentage_repeated_2-gram_in_summ',
'percentage_repeated_3-gram_in_summ',
]

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


##########################################################################################################

def _rouge_clean(s):
  return re.sub(r'[^a-zA-Z0-9 ]', '', s)

def calc_reds(sents):
  def unq_rat(distr,den):
    if den==0:  return 1e-12
    return max(1e-12,1.0 - (len(distr)/den) )
  def nid_red(distr,den):
    if den==0:  return 1e-12
    return max(1e-12,1.0 - entropy(distr)/np.log(den))
  ###
  ung = Counter()
  big = Counter()
  trg = Counter()
  for s in sents:
      s = _rouge_clean(" ".join(s)).split()
      ns = len(s)
      ung.update([x for x in s if x not in STOPWORDS])
      big.update([tuple(s[i:i+2]) for i in range(0,ns,2) if len(s[i:i+2])==2 and s[i] not in STOPWORDS and s[i+1] not in STOPWORDS])
      trg.update([tuple(s[i:i+3]) for i in range(0,ns,3) if len(s[i:i+3])==3 and s[i] not in STOPWORDS and s[i+2] not in STOPWORDS])
  #
  # unq_ratio
  us = sum(list(ung.values()))
  bs = sum(list(big.values()))
  ts = sum(list(trg.values()))
  urat = unq_rat(ung,us)
  brat = unq_rat(big,bs)
  trat = unq_rat(trg,ts)
  uni_red = (urat * brat * trat) ** (1.0/3.0) # higher -> more redundant
  
  # entropy red
  pu = [x/us if us!=0 else 1e-12 for x in ung.values()]
  pb = [x/bs if bs!=0 else 1e-12 for x in big.values()]
  pt = [x/ts if ts!=0 else 1e-12 for x in trg.values()]
  uentr = nid_red(pu,us)
  bentr = nid_red(pb,bs)
  tentr = nid_red(pt,ts)
  entr_red = (uentr * bentr * tentr)**(1.0/3.0) # higher -> more redundant

  return [urat,brat,trat,uni_red],[uentr,bentr,tentr,entr_red]
  

""" mode: [mean, mean-max]
"""
def get_rouge_red(sentences,mode="mean"):
    N = len(sentences)
    r1,r2,rl = [],[],[]
    if mode=="mean":
      for i in range(N):
        for j in range(i+1,N):
          r1.append(ROUGE_N(sentences[i], sentences[j], 1).f1_measure)
          r2.append(ROUGE_N(sentences[i], sentences[j], 2).f1_measure)
          rl.append(ROUGE_L(sentences[i], sentences[j]).f1_measure)
      #
    else:
      for i in range(N):
        rr1,rr2,rrl = [],[],[]
        for j in range(N):
          if j==i: continue
          rr1.append(ROUGE_N(sentences[i], sentences[j], 1).f1_measure)
          rr2.append(ROUGE_N(sentences[i], sentences[j], 2).f1_measure)
          rrl.append(ROUGE_L(sentences[i], sentences[j]).f1_measure)
        #
        rr1 = list(filter(lambda x: not np.isnan(x),rr1))
        rr2 = list(filter(lambda x: not np.isnan(x),rr2))
        rrl = list(filter(lambda x: not np.isnan(x),rrl))
        if len(rr1)>0: r1.append(max(rr1))
        if len(rr2)>0: r2.append(max(rr2))
        if len(rrl)>0: rl.append(max(rrl))
      #
    #
    r1 = list(filter(lambda x: not np.isnan(x),r1))
    r2 = list(filter(lambda x: not np.isnan(x),r2))
    rl = list(filter(lambda x: not np.isnan(x),rl))
    m1 = 0. if len(r1)==0 else np.mean(r1)
    m2 = 0. if len(r2)==0 else np.mean(r2)
    ml = 0. if len(rl)==0 else np.mean(rl)
    return m1,m2,ml

def get_single_rouge_red(s1_s2,):
  s1,s2 = s1_s2
  r1 = ROUGE_N(s1, s2, 1).f1_measure
  r2 = ROUGE_N(s1, s2, 2).f1_measure
  rl = ROUGE_L(s1,s2).f1_measure
  return r1,r2,rl

def cand_redundancy(sents):
  res_dict = {}
  reds = calc_reds([x.split() for x in sents])
  res_dict["u-u"] = reds[0][0]
  res_dict["u-b"] = reds[0][1]
  res_dict["u-t"] = reds[0][2]
  res_dict["u"] = reds[0][3]
  res_dict["nid-u"] = reds[1][0]
  res_dict["nid-b"] = reds[1][1]
  res_dict["nid-t"] = reds[1][2]
  res_dict["nid"] = reds[1][3]

  # evaluator = DataStatsMetric(n_gram=3,tokenize=False)
  # rdict = evaluator.evaluate_example(" ".join(sents)," ".join(sents))
  # rep = ((rdict["percentage_repeated_1-gram_in_summ"] + 1e-12) * \
  #          (rdict["percentage_repeated_2-gram_in_summ"] + 1e-12) * \
  #          (rdict["percentage_repeated_3-gram_in_summ"] + 1e-12) ) ** (1.0/3.0)
  # res_dict["rep-u"] = rdict["percentage_repeated_1-gram_in_summ"]
  # res_dict["rep-b"] = rdict["percentage_repeated_2-gram_in_summ"]
  # res_dict["rep-t"] = rdict["percentage_repeated_3-gram_in_summ"]
  # res_dict["rep"] = rep

  mode = "mean"
  r1,r2,rl = get_rouge_red(sents,mode)
  res_dict["r1"] = r1
  res_dict["r2"] = r2
  res_dict["rl"] = rl
  
  return res_dict

## CAPACITY: batch x seq len :: max = 4k
def calc_ppl(model, tokenizer, sentence,  mask_tok_id=104):
  dev = "cuda" if torch.cuda.is_available() else "cpu"
  tensor_input = tokenizer.encode(sentence, return_tensors='pt',truncation=True,max_length=510).to(dev)
  ntoks = tensor_input.size(-1)-2
  # block_size = min(75 // ntoks,ntoks)
  block_size = min(3200 // (ntoks+2),ntoks)
  nblocks = 1 if block_size==ntoks else (ntoks // block_size) + 1
  # print("ntoks | block size | n blocks")
  # print(ntoks,block_size,nblocks)

  tot_loss = 0.
  for i in range(nblocks):
    bsize = block_size
    if i==nblocks-1 and block_size!=ntoks:
      bsize = ntoks - i*block_size
    repeat_input = tensor_input.repeat(bsize, 1)
    mask = torch.zeros([bsize,ntoks+2]).to(dev)
    for j in range(bsize):
      mask[j,i*block_size + j+1] =  1
    masked_input = repeat_input.masked_fill(mask == 1, mask_tok_id)
    _labels = repeat_input.masked_fill( masked_input != mask_tok_id, -100)
    output = model(masked_input, labels=_labels)
    tot_loss += output.loss.item() * bsize

    # print(">rep in",repeat_input.shape)
    # print(">mask ",mask.shape,mask)
    # print(">masked in",masked_input.shape)
    # print(">labels",_labels.shape,_labels)
    # print(">loss=",output.loss.item() * bsize)
  #
  ppl = np.exp(tot_loss / ntoks)
  return ppl