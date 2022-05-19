"""
# SELECTION ROUTINES
- greedy
- knapsack selection
- red-aware selection

"""
import re
import copy 
import pdb
import numpy as np
from utils import MaxPriorityQueue


###############################################################################

def greedy_selection_sent(item_scores,sent_lens,budget=190,w_hard_thr=-1):
  i_v = list(item_scores.items())
  i_v.sort(key=lambda x:x[1],reverse=True)
  wc = 0.0
  WMAX = w_hard_thr if w_hard_thr!=-1 else 100000
  selected = []
  for idx,sc in i_v:
    ntoks = sent_lens[idx]
    if wc + ntoks > WMAX:  break
    selected.append(idx)
    wc += ntoks
    if wc >= budget:  break
  if len(selected)==0:
    return [i_v[0][0]]
  selected.sort()
  return selected



# def knp_selection_sent(item_scores,sent_lens,budget=205,w_hard_thr=-1):
#   weight = 0.0
#   value = 0.0
#   i_v = list(item_scores.items())
#   i_v.sort(key=lambda x:x[1],reverse=True)
#   idxs = [x for x,y in i_v]
#   maxPossibleWeight = sum(list(sent_lens.values()))
#   maxWeight = budget
#   w_hard_thr = maxPossibleWeight if w_hard_thr==-1 else w_hard_thr
#   cap = -1
#   for i,idx in enumerate(idxs):
#     new_weight = weight + sent_lens[idx]
#     new_value = value + item_scores[idx]
#     if new_weight <= maxWeight:
#       weight = new_weight
#       value = new_value
#       cap = i
#     else:
#       new_ratio = 0.0 if new_weight==0 else new_value/new_weight
#       ratio = 0.0 if weight==0 else value/weight
#       if abs(maxWeight-weight) > abs(maxWeight-new_weight) and new_weight<=w_hard_thr:
#         weight = new_weight
#         value = new_value
#         cap = i
#       break # test only the immediately greater
#     #
#   #
#   if cap==-1: cap=0 # first drawn sentence was too long
#   idxs = list(idxs[:(cap+1)])
#   idxs.sort()
#   if len(idxs)==0:
#     return [i_v[0][0]]
#   return idxs

def knp_selection_sent(item_scores,sent_lens,budget=205,w_hard_thr=-1):
  weight = 0.0
  value = 0.0
  i_v = list(item_scores.items())
  idx_max_values = max(i_v,key=lambda x:x[1])[0]
  # i_v.sort(key=lambda x:x[1],reverse=True)
  maxPossibleWeight = sum(list(sent_lens.values()))
  maxWeight = budget
  w_hard_thr = maxPossibleWeight if w_hard_thr==-1 else w_hard_thr
  # solution[w] is the optimal solution within weight w, as (total val, total weight, set of items)
  solutions = [(0.0, 0, set())]
  for weight in range(1, maxPossibleWeight + 1):
    best = solutions[-1]
    for idx,v in i_v:
      w = sent_lens[idx]
      if w > weight or w==0: continue
      if idx in solutions[weight - w][2]:  continue
      newVal = solutions[weight - w][0] + v
      if newVal > best[0]:
        best = newVal, solutions[weight - w][1] + w, solutions[weight - w][2] | {idx}
    solutions.append(best)
    if weight > maxWeight:
      if best != solutions[weight-1]:
        assert solutions[weight-1][1] <= maxWeight
        if np.fabs(maxWeight-solutions[weight-1][1]) > np.fabs(maxWeight-best[1]) and weight<=w_hard_thr:
          best_set = list(best[2])
          return sorted(best_set) if len(best_set)>0 else [idx_max_values]
        break
    #
  #

  idxs = []
  try:
    idxs = list(solutions[maxWeight][2])
  except IndexError:
    idxs = list(solutions[maxPossibleWeight][2])
  idxs.sort()
  if len(idxs)==0:
    return [idx_max_values]
  return idxs



## TODO - ADAPT
def prim_selection_sent(node_scores,pgraph,w_hard_thr=-1):
  def get_n_tokens(id_list):
    return sum([pgraph.sent_lens[x] for x in id_list])
    
  G = pgraph.sent_graph
  node_list = list(node_scores.keys())
  N = len(node_list)
  nd = len(node_scores)

  maxPossibleWeight = sum(pgraph.sent_lens)
  w_hard_thr = maxPossibleWeight if w_hard_thr==-1 else w_hard_thr

  Q = MaxPriorityQueue(zip(node_list,[-1]*N))
  max_node,max_sc = max(list(node_scores.items()),key=lambda x:x[1])
  Q.increase_key(max_node,max_sc)

  prim_set = set()
  visited = set()
  while Q.size > 0 and pgraph.check_budget_sent(prim_set):
    unode = Q.extract_max()
    visited.add(unode.id)
    if unode.wgt == -1: break
    test_set = set(list(prim_set)+[unode.id])
    if get_n_tokens(test_set) > w_hard_thr:
      continue

    prim_set.add(unode.id)
    for v in G[unode.id]:
      if v not in visited:
        Q.increase_key(v,node_scores[v])
  #
  selected = list(prim_set)
  selected.sort()
  return selected

