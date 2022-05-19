import os,sys
import getpass
import string
import numpy as np
import random
from collections import OrderedDict,defaultdict,namedtuple
import bisect
from nltk.corpus import stopwords
from pprint import pprint
import pdb

np.random.seed(42)

DATASET_BASEDIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"datasets")

PQNode = namedtuple("PQNode","id wgt")
STOPWORDS = stopwords.words('english')

# allowed POS during argument overlap
OVERLAP_POS = [
  "NOUN","PROPN","VERB","NUM","PRON"
]

sys.path.append("../common")
from base_runner import budgets


MAX_T_PERSISTANCE = 5 # maximum number of iterations T can remain unchanged

###############################################

"""
overlap score = intersect of lemmas / total lemmas in both props
overlap score = pred_score + args_score
normalize score to [0,1]
S[p2] ; D[p1]
"""
# def calc_overlap(p1x,p2x,D):
#   def compare_args(a1,a2):
#     lem1 = set([x.lemma for x in a1 if x.lemma not in STOPWORDS and x.lemma not in string.punctuation])
#     lem2 = set([x.lemma for x in a2 if x.lemma not in STOPWORDS and x.lemma not in string.punctuation])
#     joint = set()
#     joint.update(lem1)
#     joint.update(lem2)
#     sc = (len(lem1 & lem2) / len(joint)) if len(joint) > 0 else 0.0
#     return sc
#   p1 = D[p1x]
#   p2 = D[p2x]
#   # compare predicates
#   score_pred = compare_args(p1.predicate,p2.predicate)
#   # p1 args -> p2.pred
#   score_12 = 0
#   for ic1 in p1.arguments:
#     if type(ic1)==int:  ic1 = D[ic1].predicate
#     score_12 = max(score_12,compare_args(ic1,p2.predicate))
#   # p2.args -> p1.pred
#   score_21 = 0
#   for ic2 in p2.arguments:
#     if type(ic2)==int:  ic2 = D[ic2].predicate
#     score_21 = max(score_21,compare_args(ic2,p1.predicate))
#   # p1.args -- p2.args
#   scores_cc = 0
#   for ic1 in p1.arguments:
#     if type(ic1)==int:  ic1 = D[ic1].predicate
#     for ic2 in p2.arguments:
#       if type(ic2)==int:  ic2 = D[ic2].predicate
#       score_cc = max(scores_cc,compare_args(ic1,ic2))

#   return max([score_pred,score_12,score_21,score_cc])


def calc_overlap(x,y,D):
  def is_allowed(tok):
    return tok.lemma not in STOPWORDS and \
           tok.lemma not in string.punctuation and tok.pos in OVERLAP_POS
  def compare_args(a1,a2):
      lem1 = set([x.lemma for x in a1 if is_allowed(x)])
      lem2 = set([x.lemma for x in a2 if is_allowed(x)])
      joint = set()
      joint.update(lem1)
      joint.update(lem2)
      sc = (len(lem1 & lem2) / len(joint)) if len(joint) > 0 else 0.0
      return sc
  ###
  p1,p2 = D[x],D[y]
  id2toks = {}
  side1 = {0:p1.predicate}
  for i,ch in enumerate(p1.arguments):
    if type(ch)==int: side1[i+1] = D[ch].predicate
    else:       side1[i+1] = ch
    
  side2 = {0:p2.predicate}
  for i,ch in enumerate(p2.arguments):
    if type(ch)==int: side2[i+1] = D[ch].predicate
    else:       side2[i+1] = ch
  scores = []
  for i,itoks in side1.items():
    for j,jtoks in side2.items():
      sc = compare_args(itoks,jtoks)
      if sc == 0: continue
      scores.append([i,j,sc])
  if len(scores)==0: return 0.0
  scores.sort(key=lambda x:x[2],reverse=True)
  c_p,c_pa,c_a = 0,[],[]
  matching = []
  vis1=set(); vis2=set()
  for i,j,sc in scores:
    if i in vis1 or j in vis2: continue
    matching.append(sc)
    vis1.add(i); vis2.add(j)
  return np.mean(matching)
  # for i,j,sc in scores:
  #   if i in vis1 or j in vis2: continue
  #   if   i==0 and j==0:   c_p = sc
  #   elif i==0 or j==0:    c_pa.append(sc)
  #   else:                 c_a.append(sc)
  #   vis1.add(i); vis2.add(j)
  # c_pa = 0 if len(c_pa)==0 else np.mean(c_pa)
  # c_a = 0 if len(c_a)==0 else np.mean(c_a)
  # tot = c_p + c_pa + c_a
  # return 0 if tot==0 else tot/3.0


###############################################
def dfs_search_node(u,tgt,depth,max_len,G,visited):
  visited.add(u)
  if u == tgt: return depth
  if depth>=max_len: return -2
  dep = -1
  for v in G[u]:
    if v not in visited:
      dep = dfs_search_node(v,tgt,depth+1,max_len,G,visited)
      if dep==-2: return dep
      if dep!=-1: return dep
  return -1

def dfs_retrieve_path(u,tgt,G,visited,path):
  visited.add(u)
  if u == tgt: return path,True
  p = []
  for v in G[u]:
    if v not in visited:
      p = [x for x in path] + [v]
      p,f = dfs_retrieve_path(v,tgt,G,visited,p)
      if f: return p,f
  return path,False

def dfs_color(u,G,visited):
  visited.add(u)
  for v in G[u]:
    if v not in visited:
      visited = dfs_color(v,G,visited)
  return visited

def get_node_level(root,T):
  visited = set()
  dist = {root:1}
  Q = [root]
  while len(Q)>0:
    u = Q.pop()
    visited.add(u)
    for v in T[u]:
      if v not in visited:
        dist[v] = dist[u] + 1
        Q = [v] + Q
  assert len(dist) == len(T)
  return dist


def get_subtree_size(u,T,st_size_list,visited):
  st_size_list[u] = 1
  visited.add(u)
  for v in T[u]:
    if v not in visited:
      st_size_list,visited = get_subtree_size(v,T,st_size_list,visited)
      st_size_list[u] += st_size_list[v]
  return st_size_list,visited

"""
traverses tree bread-first manner
"""
def iter_tree(root,T):
  visited = set()
  Q = [root]
  while len(Q)>0:
    u = Q.pop()
    visited.add(u)
    yield u
    for v in T[u]:
      if v not in visited:  Q = [v] + Q



import networkx as nx

def get_centrality(u,T,edge_wgt):
  G = nx.Graph()
  for x in T.keys():
    G.add_node(x)
  for u,vv in T.items():
    for v in vv: 
      if u<=v: x,y = u,v
      else:    x,y = v,u
      wgt = 10000 if (x,y) not in edge_wgt else 1.0/edge_wgt[(x,y)]
      G.add_edge(x,y,weight=wgt)
  try:
    sc = nx.closeness_centrality(G)
    return sc[u]
  # if not doable, resort to highest degree
  except:
    return len(T[u])
    

# defines new root according to centrality 
def get_new_root(root,T,edge_wgt):
  G = nx.Graph()
  for x in T.keys():
    G.add_node(x)
  for u,vv in T.items():
    for v in vv: 
      if u<=v: x,y = u,v
      else:    x,y = v,u
      wgt = 10000 if (x,y) not in edge_wgt else 1.0/edge_wgt[(x,y)]
      G.add_edge(x,y,weight=wgt)
  nsc = []
  try:
    sc = nx.closeness_centrality(G)
    nsc = list(sc.items())
  # if not doable, resort to highest degree
  except:
    # print("[get_root] backing up to highest degree")
    nsc = [(u,len(vl)) for u,vl in T.items()] # node with highest degree
  new_root,max_sc = max(nsc,key=lambda x:x[1])
  nmax = len([x for x in nsc if x[1]==max_sc])
  if nmax==1:
    return new_root
  distances = get_node_level(root,T)
  # in case of tie, choose chosest to former root
  new_root = min([(u,distances[u]) for u,sc in nsc if sc==max_sc])[0]

  return new_root


def unwgtd_centrality_scorer(T,method="eigen"):
  G = nx.Graph()
  for x in T.keys():
    G.add_node(x)
  for x,vv in T.items():
    for v in vv:  G.add_edge(x,v)
  sc = {}
  try:
    if   method=="eigen":
      sc = nx.eigenvector_centrality(G,max_iter=500)
    elif method=="btw":
      sc = nx.betweenness_centrality(G)
    elif method=="cls":
      sc = nx.closeness_centrality(G)
  except:
    sc = {x:0.0 for x in T.keys()}
  return sc

###############################################


# def fix_prop_tree(P,D):
#   try:
#     root = [x for x in P.keys() if D[u].parent==-1][0]
#   except:
#     root = list(P.keys())[0]
#   edges = [(u,v) for u,vl in P.items() for v in vl if u < v]

#   if len(P)-1 == len(edges):
#     if not is_tree(P):
#       print("[fix_prop_tree].."); pdb.set_trace(); print(">>")
#     return P

#   # get distances from root
#   num_nodes = len(P)
#   Q = [root]
#   dist = defaultdict(lambda:10000)
#   dist[root] = 0
#   visited = set()
#   while len(Q)>0:
#     u = Q.pop()
#     visited.add(u)
#     for v in P[u]:
#       if v not in visited and dist[v] >= dist[u] + 1:
#         dist[v] = dist[u] + 1
#         Q = [v] + Q
#   try:
#     assert len(dist) == num_nodes
#   except:
#     pdb.set_trace()
#   # get edges wih wgt = sum lvl
#   edges = [(u,v,dist[v]+dist[u]) for u,vl in P.items() for v in vl if u < v]
#   edges.sort(key=lambda x:x[2])
#   union_set = {u:u for u in P.keys()}
#   newP = {u:set() for u in P.keys()}
#   ned = 0
#   for u,v,d in edges:
#     uV = union_set[v]
#     if union_set[u] == uV:
#       continue
#     for x in P.keys():
#       if union_set[x]==uV:
#         union_set[x] = union_set[u]
#     newP[u].add(v)
#     newP[v].add(u)
#     ned +=1
#   #
#   assert len(set(list(union_set.values())))<=1
#   assert ned == num_nodes-1
#   assert len(newP) == len(P)

#   if not is_tree(newP):
#     print("[fix_prop_tree].."); pdb.set_trace(); print(">>")

#   return newP


def is_tree(tree):
  def dfs_local(u):
    visited.add(u)
    for v in tree[u]:
      if v not in visited: dfs_local(v)
  if len(tree)==0:
    return True
  ed = [(u,v) for u,vl in tree.items() for v in vl if u < v]
  not_bidir = False
  for u,vl in tree.items():
    for v in vl:
      if u not in tree[v]:
        not_bidir = True; break

  visited = set()
  start = list(tree.keys())[0]
  dfs_local(start)
  if len(ed) == len(tree)-1 and len(visited)==len(tree) and not not_bidir:
    return True
  return False


###############################################
## maximum priority queue

class MaxPriorityQueue:
  def __init__(self,data=[]):
    self.A = [-1]
    self.id2pos = {}
    for d,w in data:
      self.insert(d,w)

  @property
  def size(self):
    return len(self.A)-1

  def insert(self,node,wgt):
    self.A.append(PQNode(node,-1))
    self.id2pos[node] = len(self.A)-1
    self.increase_key(node,wgt)

  def increase_key(self,node,key):
    pos = self.id2pos[node]
    if key <= self.A[pos].wgt:
      return
    self.A[pos] = PQNode(self.A[pos].id,key)
    while pos > 1 and self.A[pos//2].wgt < self.A[pos].wgt:
      self.id2pos[self.A[pos].id],self.id2pos[self.A[pos//2].id] = \
            self.id2pos[self.A[pos//2].id],self.id2pos[self.A[pos].id]
      self.A[pos],self.A[pos//2] = self.A[pos//2],self.A[pos]
      pos = pos//2
    return

  def is_empty(self,):
    return len(self.A) == 0

  def extract_max(self,):
    # if self.is_empty(): return None
    _max = self.A[1]
    self.A[1] = self.A[-1]
    self.id2pos[self.A[1].id] = 1
    self.A.pop()
    if len(self.A) > 1:
      self.max_heapify(1)
    return _max

  def is_leaf(self,pos):
    size = len(self.A)-1
    return pos >= (size//2) and pos <= size

  def max_heapify(self,pos):
    l = 2*pos
    r = 2*pos + 1
    if self.is_leaf(pos): return
    if self.A[pos].wgt < self.A[l].wgt or self.A[pos].wgt < self.A[r].wgt:
      if self.A[l].wgt > self.A[r].wgt:
        self.id2pos[self.A[pos].id],self.id2pos[self.A[l].id] = \
            self.id2pos[self.A[l].id],self.id2pos[self.A[pos].id]
        self.A[pos],self.A[l] = self.A[l],self.A[pos]
        self.max_heapify(l)
      else:
        self.id2pos[self.A[pos].id],self.id2pos[self.A[r].id] = \
            self.id2pos[self.A[r].id],self.id2pos[self.A[pos].id]
        self.A[pos],self.A[r] = self.A[r],self.A[pos]
        self.max_heapify(r)

##############################################################################
# sentence distance (in position) from oracle to sent

def sent_dist(orc,rcl):
  if orc[0] != rcl[0]: return 1000
  return rcl[1] - orc[1]

def get_props_by_sent(sid,D):
  return set([pid for pid,p in D.items() if tuple([p.section_id,p.section_sent_id])==tuple(sid)])