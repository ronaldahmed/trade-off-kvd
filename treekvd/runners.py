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

from treekvd import TreeKvd

sys.path.append("../common")
from prop_utils import Proposition
from base_runner import BaseRunner
from selection import *


np.random.seed(42)
random.seed(42)


############################################################################################################


class SentTreeKvDRunner(BaseRunner):
  def __init__(self,args,params,
                system_id):
    super(SentTreeKvDRunner, self).__init__(args,params,system_id=system_id)
    

  def select(self,scores,item):
    sent_lens = {}
    for sec_id,sent_id in scores.keys():
      sent_lens[(sec_id,sent_id)] = len(item["section_sents"][sec_id][sent_id].split())
    if   self.params["selection"] == "greedy":
      selected = greedy_selection_sent(scores,sent_lens,budget=self.budget,w_hard_thr=self.wthr)
    elif self.params["selection"] == "knp":
      selected = knp_selection_sent(scores,sent_lens,budget=self.budget,w_hard_thr=self.wthr)
    return selected


  def run_parallel(self,item):
    proc = TreeKvd(item["doc_props"],self.params)
    for psec in item["section_ptrees"]:
      proc.run(psec)

    p_scores = proc.node_scores
    s_scores = defaultdict(float)
    for pid,sc in p_scores.items():
      prop = proc.D[pid]
      sec_id,sent_id = prop.section_id,prop.section_sent_id
      s_scores[(sec_id,sent_id)] += sc
    selected = self.select(s_scores,item)

    return item["id"],selected,p_scores


############################################################################################################
# Prop based extraction
#