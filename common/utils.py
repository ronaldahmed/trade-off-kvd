import os,sys
import string
from collections import OrderedDict,defaultdict,namedtuple
from nltk.corpus import stopwords
import pdb

budgets = {
  "pubmed" : 205,
  "arxiv" : 190,
}

MERGE_RELATIONS_NO_COND = [
  "fixed","flat","compound",
  "det","clf","case",
  "aux",
  "nummod",
  "goeswith",
  "discourse",
]

MERGE_RELATIONS_COND = [
  'advmod',
  'amod',
]

# MERGE_RELATIONS_COND = {
#   # "det" : open("resources/det","r").read().split("\n"),
#   "advmod": ["n't","not"],
#   "amod" : open("resources/amod","r").read().split("\n"),
# }

NOUN_POS = ["NOUN","PROPN"]
SPECIAL_NODE_FORM = [
  "CONJ", "BE", "RELATED"
]

STOPWORDS = stopwords.words('english')
LATEX_KW = [
  "\\usepackage","\\begin", "\\end", "\\documentclass", "\\setlength","\\oddsidemargin","\\setlength"
]

TokenTup = namedtuple("TokenTup","id form lemma pos")
MAX_TOKENS = 200        # for parser
MIN_TOKS_PER_SENT = 3   # for prop extractor

##############################################################################################################

DATASET_BASEDIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"datasets")
fix_endings = ["ref.", "eq.", "fig.","sec."]
fix_beg = ["@xcite","@xref","@xmath","@sec",","]

# if sent has too many special tags(@X..)
def has_special_fix(text):
  toks = text.split()
  nsp = 0
  for x in toks:
    if x in string.punctuation:
      nsp += 1
    else:
      for kw in fix_beg:
        if kw in x: nsp+=1; break

  return (nsp / len(toks)) > 0.9

def has_too_few_toks(text):
  return len([x for x in text.split() if x not in string.punctuation]) < 5

##############################################################################################################

# def get_local_p_graph(pids,D):
#   sG = defaultdict(set)
#   for pid in pids:
#     sG[pid] = set([u for u in D[pid].children if type(u)==int])
#     if -1 in sG[pid]:
#       print("[get_local_p_graph] found c=-1"); pdb.set_trace()
#     for v in sG[pid]:  sG[v].add(pid)
#     if D[pid].parent!=-1:
#       sG[pid].add(D[pid].parent)
#       sG[D[pid].parent].add(pid)
#   G = {x:y for x,y in sG.items()}
#   return G
#   # return fix_prop_tree(G,D)

def get_dep_tree(T):
  sG = defaultdict(set)
  root = -1
  for pid in T.keys():
    sG[pid] = set([u for u in T[pid].children if type(u)==int])
    if -1 in sG[pid]:  sG.remove(-1)
    for v in sG[pid]:  sG[v].add(pid)
    if T[pid].parent!=-1:
      sG[pid].add(T[pid].parent)
      sG[T[pid].parent].add(pid)
    else:
      root = pid
  G = {x:y for x,y in sG.items()}
  return root,G


def find_tripa(root,T,dP):
  def is_link(u):
    if deg[u]==1 and len([x for x in dP[u].children if x!=-1])==1:
      return True
    elif deg[u]==0:
      return True
    return False
  deg = {u:len(T[u])-1 for u in T.keys()}
  deg[root] = len(T[root])
  # get parents
  par = {root:-1}
  visited = set()
  Q = [root]
  while len(Q)>0:
    u = Q.pop()
    visited.add(u)
    for v in T[u]:
      if v not in visited:
        par[v] = u
        Q = [v] + Q
  # get path from leave
  leaves = [u for u in T.keys() if deg[u]==0]
  total_seqs = []
  for lf in leaves:
    path = []
    u = lf
    while u!=-1:
      path.append(u)
      u = par[u]
    path = path[::-1]

    # print(f"[find_tripa] leaf={lf}, path={path}")
    # pdb.set_trace()

    # find list tripa
    seq = [path[0]] # start with root
    for i in range(1,len(path)):
      if is_link(path[i]):
        seq.append(path[i])
      else:
        if len(seq)>1:
          total_seqs.append(seq)
        seq = []
    #
    if len(seq)>1:
      total_seqs.append(seq)
  #

  # find non overlapping tripas
  total_seqs.sort(key=lambda x: len(x),reverse=True)
  non_ovlp = []
  nt = len(total_seqs)
  for x in total_seqs:
    xs = set(x)
    is_ov = False
    for y in non_ovlp:
      if len(xs & set(y))>0:
        is_ov = True; break
    if not is_ov:
      non_ovlp.append(x[::-1]) # inversed
  #
  # print("[find_tripa] non overlp=",non_ovlp)
  # pdb.set_trace()
  # print(">>")

  return non_ovlp



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

# def get_prop_str(self,prop):
#   pred = prop.form_seq
#   args = [str(x) if type(x)==int else x.form_seq for x in prop.children]
#   return "%s (%s)" % (pred,"; ".join(args))


def is_latex(txt):
  nkw = 0
  for kw in LATEX_KW:
    if kw in txt: nkw += 1
  if nkw >= 3: return True
  return False


"""
filters out sentences that might have latex code.
filters out abstracts/sections with less than 5 toks, empty docs
"""
def doc_filter_latex(item):
  abstract = [x for x in item["abstract_text"] if not is_latex(x)]
  ntoks_abs = sum([len(y.split()) for y in abstract])
  if ntoks_abs < 5:
    return None
  ns = len(item["sections"])
  idxs = []
  sections = []
  for i,osec in zip(range(ns),item["sections"]):
    sec = [x for x in osec if not is_latex(x)]
    ntoks = sum([len(x.split()) for x in sec])
    if ntoks < 5 or is_latex(" ".join(sec)): continue
    idxs.append(i)
    sections.append(sec)

  if len(idxs)==0:
    return None
  snames = [item["section_names"][i] for i in idxs]
  assert len(sections) == len(snames)

  item["sections"] = sections
  item["section_names"] = snames
  return item



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


