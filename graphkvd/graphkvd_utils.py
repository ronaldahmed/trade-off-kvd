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
STOPWORDS = stopwords.words('english')
PQNode = namedtuple("PQNode","id wgt")

# allowed POS during argument overlap
OVERLAP_POS = [
  "NOUN","PROPN","VERB","NUM","PRON"
]

sys.path.append("../common")
from base_runner import budgets


CAND_BEAM_SIZE_COEF = 10  #  BEAM SIZE = COEF * NUM_BWD_LINKS PER NODE
MAX_T_PERSISTANCE = 5 # maximum number of iterations T can remain unchanged

############################################################################

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

############################################################################

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
  # in case of tie, choose closest to former root
  new_root = min([(u,distances[u]) for u,sc in nsc if sc==max_sc])[0]

  return new_root


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


############################################################################
# recall mech utils

""" Find subgraph G' that contains T
"""
def find_support_subgraph(T,G):
  # def dfs_local(u,):
  #   visited.add(u)
  #   for v in G[u]:
  #     if v not in visited:
  #       dfs_local(v)
  ####
  if len(T)==0: return set()
  visited = set()
  s = list(T.keys())[0]
  # bfs local
  Q = [s]
  dist = defaultdict(lambda: 10000)
  dist[s] = 0
  while len(Q)>0:
    u = Q.pop()
    visited.add(u)
    for v in G[u]:
      if v not in visited and dist[v] > dist[u] + 1:
        dist[v] = dist[u] + 1
        Q = [v] + Q
  #
  
  assert all([x in visited for x in T.keys()])
  return visited


""" Finds best scoring paths from 's' to any t \\in T through G
- s: candidate connecting point for p \\in P (p,s)
"""
def find_bestsc_paths(s,T,supV,G):
  G = gobj.G
  assert s in supV
  Q = MinPriorityQueue([(x,100000) if x!=s else (x,0) for x in G.keys()])
  dist = defaultdict(lambda: 100000)
  dist[s] = 0
  backprop = {x:-1 for x in G.keys()}

  while Q.size() > 0:
    u = Q.extract_min()
    for v in G[u]:
      if not Q.is_here(v): continue
      if dist[v] > dist[u] + 1:
        dist[v] = dist[u] + 1
        backprop[v] = u
        Q.decrease_key(v,dist[v])
  #
  paths_per_t = {}
  for u in T:
    assert u in backprop
    




############################################################################
# assertions

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


def get_num_subgraphs(G):
  def bfs_local(u):
    Q = [u]
    cnt = 0
    while len(Q)>0:
      ss = Q.pop()
      visited.add(ss)
      cnt += 1
      for v in G[ss]:
        if v not in visited and v not in Q:
          Q = [v] + Q
    #
    return cnt
  #
  visited = set()
  sg_sizes = []
  for x in G.keys():
    if x not in visited:
      size = bfs_local(x)
      sg_sizes.append(size)
  total = len(G)
  sg_sizes = [y/total for y in sg_sizes]
  assert np.fabs(sum(sg_sizes)-1.0) < 1e-8
  return sg_sizes


def get_graph_stats(Gdict,edge_wgt):
  stats = {}
  G = nx.Graph()
  edges = set()
  for x in Gdict.keys():
    G.add_node(x)
  for u,vv in Gdict.items():
    for v in vv: 
      if u<=v: x,y = u,v
      else:    x,y = v,u
      if (x,y) in edges: continue
      wgt = 10000 if (x,y) not in edge_wgt else 1.0/edge_wgt[(x,y)]
      G.add_edge(x,y,weight=wgt)
      edges.add((x,y))
  
  sc = nx.closeness_centrality(G)
  sc = list(sc.items())
  sc.sort(key=lambda x:x[0])
  if len(sc)>0:
    stats["cls"] = [x[1] for x in sc]
  stats["deg"] = [len(x) for x in Gdict.values()]
  stats["n_nodes"] = len(Gdict)
  stats["n_edges"] = len(edges)
  stats["edge_wgt"] = np.mean([edge_wgt[(x,y)] for x,y in edges])
  return stats


###############################################
## maximum priority queue

class MinPriorityQueue:
  def __init__(self,data=[]):
    self.A = [-1]
    self.id2pos = {}
    for d,w in data:
      self.insert(d,w)

  @property
  def size(self):
    return len(self.A)-1

  def is_here(self,u):
    return u in self.id2pos

  def insert(self,node,wgt):
    self.A.append(PQNode(node,-1))
    self.id2pos[node] = len(self.A)-1
    self.decrease_key(node,wgt)

  def decrease_key(self,node,key):
    pos = self.id2pos[node]
    if key >= self.A[pos].wgt:
      return
    self.A[pos] = PQNode(self.A[pos].id,key)
    while pos > 1 and self.A[pos//2].wgt > self.A[pos].wgt:
      self.id2pos[self.A[pos].id],self.id2pos[self.A[pos//2].id] = \
            self.id2pos[self.A[pos//2].id],self.id2pos[self.A[pos].id]
      self.A[pos],self.A[pos//2] = self.A[pos//2],self.A[pos]
      pos = pos//2
    return

  def is_empty(self,):
    return len(self.A) == 0

  def extract_min(self,):
    # if self.is_empty(): return None
    _min = self.A[1]
    del self.id2pos[self.A[1].id]
    self.A[1] = self.A[-1]
    self.id2pos[self.A[1].id] = 1
    self.A.pop() # remove last
    if len(self.A) > 1:
      self.min_heapify(1)
    return _min

  def is_leaf(self,pos):
    size = len(self.A)-1
    return pos >= (size//2) and pos <= size

  def min_heapify(self,pos):
    l = 2*pos
    r = 2*pos + 1
    if self.is_leaf(pos): return
    if self.A[pos].wgt > self.A[l].wgt or self.A[pos].wgt > self.A[r].wgt:
      if self.A[l].wgt < self.A[r].wgt:
        self.id2pos[self.A[pos].id],self.id2pos[self.A[l].id] = \
            self.id2pos[self.A[l].id],self.id2pos[self.A[pos].id]
        self.A[pos],self.A[l] = self.A[l],self.A[pos]
        self.min_heapify(l)
      else:
        self.id2pos[self.A[pos].id],self.id2pos[self.A[r].id] = \
            self.id2pos[self.A[r].id],self.id2pos[self.A[pos].id]
        self.A[pos],self.A[r] = self.A[r],self.A[pos]
        self.min_heapify(r)

###############################################################################################

def sent_dist(orc,rcl):
  if orc[0] != rcl[0]: return 1000
  return rcl[1] - orc[1]

def get_props_by_sent(sid,D):
  return set([pid for pid,p in D.items() if tuple([p.section_id,p.section_sent_id])==tuple(sid)])


############################################################################################
## backup from graphkvd

# def get_max_overlap_tree(self,nodes):
#   edges = []
#   for u in nodes:
#     for v in self.G[u]:
#       if v not in nodes: continue
#       x,y = u,v
#       if x>y: x,y = v,u
#       if (x,y) in self.overlap_memo:
#         edges.append([x,y,self.overlap_memo[(x,y)]])
#   #
#   edges.sort(key=lambda x: x[2],reverse=True)
#   union_set = {u:u for u in nodes}
#   T = {u:set() for u in nodes}
#   ned = 0
#   for u,v,d in edges:
#     uV = union_set[v]
#     if union_set[u] == uV:  continue
#     for x in nodes:
#       if union_set[x]==uV:
#         union_set[x] = union_set[u]
#     T[u].add(v)
#     T[v].add(u)
#     ned +=1
#   #
  
#   assert len(set(list(union_set.values())))<=1
#   assert ned == len(nodes)-1
#   assert len(T) == len(nodes)
#   return T

  # def run(self,section_prop_trees,section_sents):
    
  #   nsections = len(section_prop_trees)
  #   for sec_id in range(nsections):
  #     nsents = len(section_prop_trees[sec_id])
  #     T = {} # memory tree
  #     root = None
  #     for sent_id in range(nsents):
  #       proot,P = section_prop_trees[sec_id][sent_id]
  #       sent = section_sents[sec_id][sent_id]
  #       P = {x:set(y) for x,y in P.items()}
  #       self.update_overlap_memo(P)
  #       # add P to document graph
  #       for u,vl in P.items():  self.G[u] = set([x for x in vl])
  #       ## find best attachments and populate Q
  #       bwd_nodes = {}
  #       for p in iter_tree(proot,P):
  #         Q = {}
  #         # 0. atachments in prev memory tree
  #         Q = self.find_best_attachments(p,[(root,T)],Q)
  #         # 1. atachments in current section (check complete)
  #         if sent_id>0:
  #           Q = self.find_best_attachments(p,section_prop_trees[sec_id][:sent_id],Q)
  #         # 2. attachments in previous sections (stop when |Q'| > COEF*n_bwd)
  #         nq = len(Q)
  #         for t in range(sec_id-1,-1,-1):
  #           Q = self.find_best_attachments(p,section_prop_trees[t],Q)
  #           if len(Q) - nq > CAND_BEAM_SIZE_COEF * self.num_bwd_links:
  #             break
  #           nq = len(Q)
  #         # keep only top 1 - add edge
  #         Q = list(Q.items())
  #         Q.sort(key=lambda x:x[1],reverse=True)

  #         # update document graph with new backward links
  #         if len(Q)>0:
  #           t,sc = Q[0]
  #           if t not in bwd_nodes: bwd_nodes[t] = {}
  #           bwd_nodes[t][p] = sc
  #       #
  #       bwd_nodes = [(x,y,sc) for x,vl in bwd_nodes.items() for y,sc in vl.items()]
  #       bwd_nodes.sort(key=lambda x:x[2],reverse=True)
  #       bwd_nodes = bwd_nodes[:self.num_bwd_links]

  #       for t,p,sc in bwd_nodes:
  #         self.G[t].add(p); self.G[p].add(t)
  #       nodes = set(list(P.keys()) + [x[0] for x in bwd_nodes])
  #       T = self.get_max_overlap_tree(nodes)
  #       new_root = get_new_root(proot,T,self.get_edge_wgt(T))

  #       T = self.kvd_memory_select(new_root,T)
  #       diff_frontier = self.scorer(new_root,T)

  #       root = new_root
  #     #END-SEC
  #   #END-DOC

  #   return

