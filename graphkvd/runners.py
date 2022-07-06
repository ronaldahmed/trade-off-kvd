import os, sys
import numpy as np
import random
import json
import pickle
import glob as gb
from collections import defaultdict, Counter, OrderedDict
from multiprocessing import Pool
import traceback
import bisect

import pdb
import time
from graphkvd import GraphKvD

sys.path.append("../common")
from prop_utils import Proposition
from base_runner import BaseRunner
from selection import *


np.random.seed(42)
random.seed(42)


############################################################################################################


class SentGraphKvDRunner(BaseRunner):
  def __init__(self,args,params,
                system_id):
    super(SentGraphKvDRunner, self).__init__(args,params,system_id=system_id)
    

  def select(self,sent_scores,item):
    sent_lens = {}
    for sec_id,sent_id in sent_scores.keys():
        sent_lens[(sec_id,sent_id)] = len(item["section_sents"][sec_id][sent_id].split())
    if   self.params["selection"] == "greedy":
      selected = greedy_selection_sent(sent_scores,sent_lens,budget=self.budget,w_hard_thr=self.wthr)
    elif self.params["selection"] == "knp":
      selected = knp_selection_sent(sent_scores,sent_lens,budget=self.budget,w_hard_thr=self.wthr)
    return selected

  """
  Runs GraphKvD for a sample item
  returns [
    id : Str  # original Cohan ID,
    selected: List[(int,int)] # list of selected sentence IDs in (section id, relative position of sentence in section)
    p_scores: Dict{int: float} # final score of propositions mapped by proposition ID
    ]
  """
  def run_parallel(self,item):
    proc = GraphKvD(item["doc_props"],self.params)
    proc.run(item["section_ptrees"],item["section_sents"])

    p_scores = proc.node_scores
    s_scores = defaultdict(float)
    for pid,sc in p_scores.items():
      prop = proc.D[pid]
      sec_id,sent_id = prop.section_id,prop.section_sent_id
      s_scores[(sec_id,sent_id)] += sc
    selected = self.select(s_scores,item)

    return item["id"],selected,p_scores

