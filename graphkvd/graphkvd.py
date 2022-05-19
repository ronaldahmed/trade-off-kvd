from collections import defaultdict,OrderedDict
import numpy as np
import pdb
from pprint import pprint
import networkx as nx
from networkx.algorithms import tree
import copy as cp

from graphkvd_utils import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class GraphKvD:
  def __init__(self,prop_dict,inf_params):
    self.D = prop_dict # dict of proposition objects
    self.G = {} # proposition graph
    
    self.node_scores = defaultdict(float)
    # memoization of overlap calculation
    self.overlap_memo = {}
    # params
    self.wm_size = inf_params.get("wm_size",5)
    self.num_recall = inf_params.get("num_recall",5) # max length of recall path
    self.c_score = inf_params.get("scale1","none") # c(x)
    self.gamma = 0.01
    self.on_hold = False
    self.t_persistance_cnt = 0 # how many iters T remains unchanged

  def get_overlap(self,x,y):
    if x>y: x,y=y,x
    if (x,y) not in self.overlap_memo:
      self.overlap_memo[(x,y)] = calc_overlap(x,y,self.D)
    return self.overlap_memo[(x,y)]

  def get_edge_wgt(self,T):
    wgts = {}
    for u,vl in T.items():
      for v in vl:
        if u<=v: x,y = u,v
        else:    x,y = v,u
        wgts[(x,y)] = self.get_overlap(x,y)
    #
    return wgts

  """
  all overlap edges within P (prop tree from dep tree) are 1.0
  """
  def update_overlap_memo(self,P):
    for u,vl in P.items():
      for v in vl:
        if u<=v:  x,y = u,v
        else:     x,y = v,u
        self.overlap_memo[(x,y)] = 1.0

  """ Finds best scoring paths from 's' to any t \\in T through G
  - cand_atc =  [s]
  - s: candidate connecting point for p \\in P (p,s)
  - sc: attchm score (p,s)
  - path returned includes connection t \\in T
  """
  def find_bestsc_path(self,cand_atc,root,T,supV): 
    def get_path_wgt(u):
      path = [u]; v = u
      while v!=s and v!=-1:
        path.append(backprop[v])
        v = backprop[v]
      #
      path = path[::-1]
      wgt = sum([self.get_overlap(x,path[i+1]) for i,x in enumerate(path[:-1])])
      return path,wgt
    ############
    # 0. get tree-based node scores for T
    depth_map = get_node_level(root,T)
    st_map = {x:0 for x in T.keys()}      
    st_map,_ = get_subtree_size(root,T,st_map,set())
    n_nodes = len(T)
    tscore = {}
    for u in T.keys():
      tscore[u] = (st_map[u] / n_nodes) * np.exp(1.0/depth_map[u] + 1e-12)

    path_scores = []
    for s in cand_atc:
      assert s in supV
      # 1. run local bfs on supV, get min distances from source S
      Q = [s]
      dist = defaultdict(lambda: 100000)
      dist[s] = 0
      backprop = {x:-1 for x in supV}
      visited = set()
      while len(Q) > 0:
        u = Q.pop()
        visited.add(u)
        for v in self.G[u]:
          if v not in supV or v in visited: continue
          if dist[v] > dist[u] + 1:
            dist[v] = dist[u] + 1
            backprop[v] = u
            Q = [v] + Q
      #
      assert all([dist[t]>-1 for t in T.keys()])

      # 2. get path scores for each t \\in T
      for t in T.keys():
        if dist[t] > self.num_recall: continue
        path,wgt = get_path_wgt(t)
        psc = tscore[t] * wgt * np.exp(-dist[t])
        path_scores.append([psc,path])
      #
    #END-CANDIDATES

    # case all candidate paths where longer than |num_recall|
    if len(path_scores)==0:
      return None,None

    path_scores.sort(key=lambda x:x[0],reverse=True)
    return path_scores[0]

  def get_max_overlap_tree(self,text_base):
    G = nx.Graph()
    for x in text_base.keys():  G.add_node(x)
    for x,vv in text_base.items():
      for v in vv:  G.add_edge(x,v,weight=self.get_overlap(x,v))
    mst = tree.maximum_spanning_edges(G, algorithm="kruskal", data=False)
    Tp = {k:set() for k in text_base.keys()}
    for u,v in mst:
      Tp[u].add(v)
      Tp[v].add(u)
    assert is_tree(Tp)
    return dict(Tp)

  ##################################################################################

  
  # finds candidates in memory tree, returns T + P + links
  def find_cands_in_mem(self,proot,P,troot,T):
    found = False
    res = cp.deepcopy(P)
    for t,vl in T.items():
      if t not in res: res[t] = set()
      for v in vl:
        if v not in res: res[v] = set()
        res[t].add(v); res[v].add(t)
    #
    for p in iter_tree(proot,P):
      best = [-1,0]
      for t in iter_tree(troot,T):
        sc = self.get_overlap(p,t)
        if sc > best[1]:  best = [t,sc]
      if best[1] > 0:
        res[p].add(best[0])
        res[best[0]].add(p)
        found = True
    #
    if not found: return {}
    return res

  # finds candidates in entire document, returns P + candidates
  def find_cand_doc(self,proot,P,bundle):
    def local_search(p,ptrees,QQ):
      candidates = {}
      for xroot,X in ptrees:
        if len(X)==0: continue
        for u in iter_tree(xroot,X):
          sc = self.get_overlap(p,u)
          if sc > 0:
            if u not in candidates: candidates[u] = 0.0
            candidates[u] = max(candidates[u],sc)
        if len(candidates) > CAND_BEAM_SIZE_COEF * self.num_recall:
          break
      if len(candidates)==0:
        return QQ
      candidates = list(candidates.items())
      candidates.sort(key=lambda x:x[1],reverse=True)
      for u,sc in candidates[:self.num_recall]:
        QQ[u] = max(QQ.get(u,0.0),sc)
      return QQ
    ##########
    res = cp.deepcopy(P)
    cand_connections = defaultdict(list)
    section_prop_trees,sec_id,sent_id = bundle
    for p in iter_tree(proot,P):
      Q = {}
      # a. atachments in current section (check complete)
      if sent_id>0:
        Q = local_search(p,section_prop_trees[sec_id][:sent_id],Q)
      # b. attachments in previous sections (stop when |Q'| > COEF*n_bwd)
      nq = len(Q)
      for t in range(sec_id-1):
        Q = local_search(p,section_prop_trees[t],Q)
        if len(Q) - nq > CAND_BEAM_SIZE_COEF * self.num_recall:
          break
        nq = len(Q)
      #
      Q = list(Q.items())
      Q.sort(key=lambda x:x[1],reverse=True)
      # update document graph with new backward links
      if len(Q)>0:
        for x,_ in Q[:5]:
          if x not in res: res[x] = set()
          res[p].add(x); res[x].add(p)
    #
    return res


  # find candidates in supporting graph
  def find_cand_support(self,P,supV):
    cand_connections = {}
    for p in P.keys():
      conn = []
      for u in supV:
        sc = self.get_overlap(p,u)
        if sc>0: conn.append([u,sc])
      if len(conn)>0:
        conn.sort(key=lambda x:x[1],reverse=True)
        cand_connections[p] = [x for x,_ in conn[:5]]
    return cand_connections


  # returns connection graph P -- G' -- T, length of paths
  def recall_mechanism(self,proot,P,troot,T):
    # 0. find supporting graph G' (G' incl G, s.t. T incl G')
    supV = find_support_subgraph(T,self.G) # set of nodes in G'
    
    # 1. for each p in P, find K candidate connections to G'
    cand_connections = self.find_cand_support(P,supV)
    # 2. if supV is empty: connect to rest of graph, return
    if len(supV)==0 or len(cand_connections)==0:
      return {},[]

    # 3. for each (p,u) in Cand, find best_scoring path (p,u,-G-,a) where a in T
    rec_base = defaultdict(set) # recall base: subgraph with all paths
    path_lens = []
    found = False
    for p,cand in cand_connections.items():
      wgt,path = self.find_bestsc_path(cand,troot,T,supV)
      if path is None: continue
      path_lens.append(len(path))
      rec_base[p].add(path[0])
      rec_base[path[0]].add(p)
      for i,x in enumerate(path[:-1]):
        rec_base[x].add(path[i+1])
        rec_base[path[i+1]].add(x)
      #
    #

    res = cp.deepcopy(P)
    if len(path_lens) > 0:
      for t,vl in T.items():
        if t not in res: res[t] = set()
        for v in vl:
          if v not in res: res[v] = set()
          res[t].add(v); res[v].add(t)
      #
      for u,vl in rec_base.items():
        if u not in res: res[u] = set()
        for v in vl:
          if v not in res: res[v] = set()
          res[u].add(v); res[v].add(u)
      #
    return res,path_lens
    

  """
  Finds connection for P in G, returns text base (subgraph G connected to P)
  """
  def find_best_attachment(self,p_root,P,t_root,T,bundle):

    # trivial case: |T|==0
    if len(T)==0:
      # find connections in all prev graph
      richP = self.find_cand_doc(p_root,P,bundle)
      return p_root,richP,[]
    
    # case 1: P - T
    newT = self.find_cands_in_mem(p_root,P,t_root,T)
    if len(newT)>0:
      return t_root,newT,[]
          
    # case 2: recall mechanism T -*- P
    newT,path_lens = self.recall_mechanism(p_root,P,t_root,T)
    if len(path_lens)>0:
      return t_root,newT,path_lens

    # case 3: replacement. choose tree A st |A|>B and clos(A)>clos(B)
    p_cls = get_centrality(p_root,P,self.get_edge_wgt(P))
    t_cls = get_centrality(t_root,T,self.get_edge_wgt(T))
    if len(P)>len(T) and p_cls > t_cls:
      richP = self.find_cand_doc(p_root,P,bundle)
      return p_root,richP,[]

    self.on_hold = True # hold on simulation scoring since T wasnt changed and P was discarded
    # P is about to be discarded, hence save it on G for future connections
    for u,vl in P.items():
      if u not in self.G: self.G[u] = set()
      for v in vl:
        if v not in self.G: self.G[v] = set()
        self.G[u].add(v);self.G[v].add(u)

    return t_root,T,[]


  def kvd_memory_select(self,root,T):
    def kvd_dfs(u):
      if len(mem)>=self.wm_size:
        return
      visited.add(u)
      mem.add(u)
      v_l = list(T[u])
      v_l.sort(reverse=True)
      for v in v_l:
        if v not in visited:
          kvd_dfs(v)
    #
    def kvd_bfs(node):
      Q = [node]
      while len(Q) > 0:
        u = Q.pop()
        visited.add(u)
        mem.add(u)
        if len(mem) >= self.wm_size:
          break
        v_l = list(T[u])
        v_l.sort(reverse=True)
        for v in v_l:
          if v not in visited:
            Q = [v] + Q
      #
      return
    ## explore recent -dfs, visited, buffer,
    visited = set(); mem = set()
    kvd_dfs(root)
    if len(mem)<self.wm_size:
      visited = set()
      kvd_bfs(root)
    #
    newT = {}
    for u in mem:
      newT[u] = set([x for x in T[u] if x in mem])
    return newT

  def scorer(self,root,T):
    score = {}
    if   self.c_score=="lvl":
      depth_map = get_node_level(root,T)
      for u,dp in depth_map.items():
        score[u] = np.exp(1.0/dp + 1e-12)

    elif self.c_score=="subtree":
      st_map = {x:0 for x in T.keys()}      
      st_map,_ = get_subtree_size(root,T,st_map,set())
      n_nodes = len(T)
      for u,sz in st_map.items():
        score[u] = sz / n_nodes

    elif self.c_score=="comb":
      depth_map = get_node_level(root,T)
      st_map = {x:0 for x in T.keys()}      
      st_map,_ = get_subtree_size(root,T,st_map,set())
      n_nodes = len(T)
      for u in T.keys():
        score[u] = (st_map[u] / n_nodes) * np.exp(1.0/depth_map[u] + 1e-12)

    elif self.c_score=="cnt":
      for u in T.keys():
        score[u] = 1.0

    # graph centrality scaling
    elif self.c_score in ["btw","cls","eigen"]:
      sc = unwgtd_centrality_scorer(T,self.c_score)
      for u,s in sc.items():
        score[u] = s
    
    # get diffusion frontier
    frontier = {}
    for u in T.keys():
      for v in self.G[u]:
        if v in T: continue
        if v not in frontier: frontier[v] = 0.0
        frontier[v] = max(frontier[v],self.gamma * score[u])
    # update score for active memory nodes
    for u,sc in score.items():
      self.node_scores[u] += sc
    # update score for frontier
    for u,sc in frontier.items():
      self.node_scores[u] += sc

    return list(frontier.keys())



  def run(self,section_prop_trees,section_sents):
    
    nsections = len(section_prop_trees)
    for sec_id in range(nsections):
      nsents = len(section_prop_trees[sec_id])
      T = {} # memory tree
      root = None

      for sent_id in range(nsents):
        proot,P = section_prop_trees[sec_id][sent_id]
        sent = section_sents[sec_id][sent_id]
        P = {x:set(y) for x,y in P.items()}
        self.update_overlap_memo(P)
        self.on_hold = False
        
        # find connections to current Tree / prev doc
        bundle = section_prop_trees,sec_id,sent_id
        new_root,Tbase,path_lens = self.find_best_attachment(proot,P,root,T,bundle)
        ## add text base to main graph G
        for u,vl in Tbase.items():
          if u not in self.G: self.G[u] = set()
          for v in vl:
            if v not in self.G: self.G[v] = set()
            self.G[u].add(v);self.G[v].add(u)

        T = self.get_max_overlap_tree(Tbase)
        new_root = get_new_root(new_root,T,self.get_edge_wgt(T))
        T = self.kvd_memory_select(new_root,T)

        if not self.on_hold:
          diff_frontier = self.scorer(new_root,T)
          self.t_persistance_cnt = 0
        else:
          self.t_persistance_cnt += 1
        root = new_root
        if self.t_persistance_cnt == MAX_T_PERSISTANCE:
          del T,root
          T = {}; root = None
      #END-SEC
    #END-DOC
    return



  def run_debug(self,section_prop_trees,section_sents,\
                debug=False,assertions=False,oracle=None):
    stats = {
      "new_roots":0,
      "recall_path_lens":[],
      "pos_recalled":{},
      "t_persistance_cnt":[],
      "mt_props":set(),
      "mt_and_frontier":set(),
    }
    oracle_props = set()
    if oracle is not None:
      for osnt in oracle:
        oracle_props.update(get_props_by_sent(osnt,self.D))

    nsections = len(section_prop_trees)
    for sec_id in range(nsections):
      nsents = len(section_prop_trees[sec_id])
      T = {} # memory tree
      root = None
      self.t_persistance_cnt = 0

      for sent_id in range(nsents):
        proot,P = section_prop_trees[sec_id][sent_id]
        sent = section_sents[sec_id][sent_id]
        P = {x:set(y) for x,y in P.items()}
        self.update_overlap_memo(P)
        self.on_hold = False
        # add P to document graph
        # for u,vl in P.items():  self.G[u] = set([x for x in vl])

        if assertions: assert is_tree(P)
        if debug:
          pid = list(P.keys())[0]
          print("\n#############################################################")
          print("[main iter] section=%d, sent_in_sec=%d, section_sent_id_orig=%d | |P|=%d | idx=(%d,%d)" % \
            (self.D[pid].section_id,self.D[pid].section_sent_id,self.D[pid].section_sent_id_orig,len(P),sec_id,sent_id) )
          print("\tsent= ",sent)
          print("\t local graph P; p_root=",proot)
          pprint(P)
          print("\tInitial T; t_root=",root)
          pprint(T)
          for k in P.keys(): print("\t",self.D[k])
          for k in T.keys(): print("\t",self.D[k])
          # pdb.set_trace()
        #
        
        # find connections to current Tree / prev doc
        bundle = section_prop_trees,sec_id,sent_id
        new_root,Tbase,path_lens = self.find_best_attachment(proot,P,root,T,bundle)
        if len(path_lens)>0:
          stats["recall_path_lens"].extend(path_lens)
          stats["pos_recalled"][(sec_id,sent_id)] = (max(path_lens)-1,)
          # case 1: recall after oracle
          if oracle is not None:
            if len(set(T.keys()) & oracle_props) > 0:
              stats["pos_recalled"][(sec_id,sent_id)] = (max(path_lens)-1,"Orc1")

        if debug:
          print("[main iter] find best attachmt | Tbase | new root=",new_root)
          pprint(Tbase)
          print("\t len of connection paths::",path_lens)
          # pdb.set_trace()

        ## add text base to main graph G
        for u,vl in Tbase.items():
          if u not in self.G: self.G[u] = set()
          for v in vl:
            if v not in self.G: self.G[v] = set()
            self.G[u].add(v);self.G[v].add(u)

        T = self.get_max_overlap_tree(Tbase)
        new_root = get_new_root(new_root,T,self.get_edge_wgt(T))

        if debug:
          print("[max overlap over text base] Max spanng tree | new root=",new_root)
          pprint(T)
          print("\t graph G::",)
          pprint(self.G)
          print()
          # pdb.set_trace()
        if new_root != root and root is not None:
          stats["new_roots"] += 1

        T = self.kvd_memory_select(new_root,T)

        ## case 2: recall == oracle
        if len(path_lens)>0 and oracle is not None:
          if len(set(T.keys()) & oracle_props) > 0:
            stats["pos_recalled"][(sec_id,sent_id)] = (max(path_lens)-1,"Orc2")
          

        if not self.on_hold:
          diff_frontier = self.scorer(new_root,T)
          if self.t_persistance_cnt > 0:
            stats["t_persistance_cnt"].append(self.t_persistance_cnt)
          self.t_persistance_cnt = 0
        else:
          self.t_persistance_cnt += 1

        if assertions:
          assert is_tree(T)
          assert len(T) <= self.wm_size

        root = new_root

        if debug:
          print(f"[kvd-select] root={new_root}, |T|={len(T)}, on_hold={self.on_hold}, persistance_cnt={self.t_persistance_cnt}")
          pprint(T)
          pprint(self.node_scores)
          # pdb.set_trace()
        stats["mt_props"].update(T.keys())
        stats["mt_and_frontier"].update(diff_frontier + list(T.keys()))

        if self.t_persistance_cnt == MAX_T_PERSISTANCE:
          del T,root
          T = {}; root = None
      #END-SEC
    #END-DOC
    if assertions:
      # make sure edges in graph are bidirectional
      for u,vl in self.G.items():
        assert u not in vl
        for v in vl:
          assert u in self.G[v]

    stats["sgraph_rel_size"] = get_num_subgraphs(self.G)
    gstats = get_graph_stats(self.G,self.overlap_memo)
    for k,v in gstats.items():
      stats[k] = v

    return stats


