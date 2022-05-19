from collections import defaultdict,OrderedDict
import numpy as np
import pdb
from pprint import pprint

from treekvd_utils import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TreeKvd:
  def __init__(self,prop_dict,inf_params):
    self.D = prop_dict # dict of proposition objects
    self.F = [] # forget pool
    self.distF = [] # preprocessed dist on subgraphs in F, [ {(x,y): distance x-y, if dist <= _num_recall} ]
    self.node2F = {} # node id to pos in F, subgraph F[idx] it belongs
    self.node_scores = defaultdict(float)
    # memoization of overlap calculation
    self.recall_memo = {}
    # control score update
    self.on_hold = False
    self.t_persistance_cnt = 0 # how many iters T remains unchanged
    # params
    self.wm_size = inf_params.get("wm_size",5)
    self.num_recall = inf_params.get("num_recall",1)
    self.c_score = inf_params.get("scale1","none") # c(x)


  def flush(self):
    del self.F, self.distF
    self.F = []
    self.distF = []

  def get_overlap(self,x,y):
    if x>y: x,y=y,x
    if (x,y) not in self.recall_memo:
      self.recall_memo[(x,y)] = calc_overlap(x,y,self.D)
    return self.recall_memo[(x,y)]
  
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
        self.recall_memo[(x,y)] = 1.0


  def find_best_attachment(self,p_root,P,t_root,T):
    # trivial case: |T|==0
    if len(T)==0:
      return p_root,P,0
    # case 1: P - T
    best = []; best_sc = 0
    for t in iter_tree(t_root,T):
      for p in iter_tree(p_root,P):
        sc = self.get_overlap(p,t)
        if sc > best_sc:
          best_sc = sc; best = [p,t]
    #
    if best_sc > 0:
      p,t = best
      # join trees
      T[t].add(p)
      for k,l in P.items(): T[k] = set(list(l))
      T[p].add(t)
      return t_root,T,0
    # case 2: recall mechanism T -*- P
    T,nrecalled = self.recall_mechanism(p_root,P,t_root,T)
    if self.num_recall>0 and nrecalled>0:
      return t_root,T,nrecalled
    # case 3: replacement. choose tree A st |A|>B and clos(A)>clos(B)
    p_cls = get_centrality(p_root,P,self.get_edge_wgt(P))
    t_cls = get_centrality(t_root,T,self.get_edge_wgt(T))
    if len(P)>len(T) and p_cls > t_cls:
      self.update_forget_pool(T)
      return p_root,P,0
    self.on_hold = True # hold on simulation scoring since T wasnt changed and P was discarded
    return t_root,T,0

  def update_forget_pool(self,sF):
    def join_blops(t1,t2):
      jt = defaultdict(set)
      keys = set(list(t1.keys()) + list(t2.keys()))
      for k in keys:
        if k in t1: jt[k].update(t1[k])
        if k in t2: jt[k].update(t2[k])
      return {k:v for k,v in jt.items()}
    def get_distances(blop):
      blop_keys = list(blop.keys())
      edge_wgt = self.get_edge_wgt(blop)
      G = {}      # traverse blop following heaviest edge first
      for u in blop_keys:
        vl = list(blop[u])
        vl.sort(key=lambda x: edge_wgt[(u,x)] if (u,x) in edge_wgt else edge_wgt[(x,u)],reverse=True)
        G[u] = vl
      nnd = len(blop)
      new_dist = {}
      for x in range(nnd):
        u = blop_keys[x]
        for y in range(x,nnd):
          v = blop_keys[y]
          vis = set()
          dist = dfs_search_node(u,v,0,self.num_recall-1,G,vis)
          if dist >= 0:
            assert dist <= self.num_recall-1
            new_dist[(u,v)] = dist
      #
      return new_dist
    ###
    visited = set()
    for u in sF.keys():
      if u in visited: continue
      prev = set([x for x in visited])
      visited = dfs_color(u,sF,visited)
      # update F: forest sF -> F
      blop_keys = list(visited - prev)
      blop = {x:set() for x in blop_keys}
      link = None
      for u in blop_keys:
        blop[u].update([x for x in sF[u] if x in blop_keys])
        if u in self.node2F:  link = u

      if link is None:
        self.F.append(blop)
        for u in blop_keys:
          self.node2F[u] = len(self.F)-1
      else:
        self.F[self.node2F[link]] = join_blops(self.F[self.node2F[link]],blop)
        for u in blop_keys:
          self.node2F[u] = self.node2F[link]
      
      if self.num_recall==0: continue
      
      # update distF
      if link is None:
        self.distF.append(get_distances(blop))
      else:
        self.distF[self.node2F[link]] = get_distances(self.F[self.node2F[link]])
    #
    return


  # num: number of forgotten nodes to retrieve
  def recall_mechanism(self,p_root,P,t_root,T):
    def find_best_F():
      best_sc = -1
      best_f = [] # idx in F, x,y
      best_in = [] # t,p
      df_idx = range(len(self.distF)-1,-1,-1)
      
      nodes_in_pool = [x for x in T.keys() if x in self.node2F]
      # case 1. t already in forgetting pool, retrieve n-1 nodes
      if len(nodes_in_pool)>0:
        for t in iter_tree(t_root,T):
          if t not in nodes_in_pool: continue
          lnk = self.node2F[t]
          sf = self.distF[lnk]
          for (x,y),dist in sf.items():
            if x!=t and y!=t: continue
            for p in iter_tree(p_root,P):
              if x==t:
                sc1 = 1.0
                sc2 = self.get_overlap(p,y)
                c_f = [lnk,x,y]
              else:
                sc1 = self.get_overlap(p,x)
                sc2 = 1.0
                c_f = [lnk,y,x]
              if sc1 > 0 and sc2 > 0 and best_sc < sc1 + dist + sc2:
                best_sc = sc1 + dist + sc2; best_f = c_f; best_in = [t,p]
              if best_sc > 0.5:
                return best_sc,best_f,best_in
        return best_sc,best_f,best_in

      # case 2. t not in F, retrieve n nodes
      for t in iter_tree(t_root,T):
        for p in iter_tree(p_root,P):
          # each subgraph
          for i in df_idx:
            sf = self.distF[i]
            # each pair of nodes in subgraph
            for (x,y),dist in sf.items():
              if x in T or y in T: continue
              sc1 = self.get_overlap(t,x)
              sc2 = self.get_overlap(p,y)
              if sc1 > 0 and sc2 > 0 and best_sc < sc1 + dist + sc2:
                best_sc = sc1 + dist + sc2; best_f = [i,x,y]; best_in = [t,p]
              sc1 = self.get_overlap(p,x)
              sc2 = self.get_overlap(t,y)
              if sc1 > 0 and sc2 > 0 and best_sc < sc1 + dist + sc2:
                best_sc = sc1 + dist + sc2; best_f = [i,y,x]; best_in = [t,p]
              if best_sc > 0.5:
                return best_sc,best_f,best_in
            #
      #
      return best_sc,best_f,best_in
    ###
    if self.num_recall==0:  return T,0
    # find best connection point, the more recent the better
    best_sc,best_f,best_in = find_best_F()
    
    # case when F/DistF is empty, no connection was found, discard curr sentence
    if best_f==[]:  return T,0
    
    # retrieve connection path from F
    idx,x,y = best_f
    t,p = best_in
    visited = set()
    path,found = dfs_retrieve_path(x,y,self.F[idx],visited,[x])
    if not found: print("[recall_mechanism] path not found!");pdb.set_trace()
    
    if t==path[0]:
      path = path[1:];
    if t==path[-1]: path = path[:-1]
    x=path[0];y=path[-1]

    try:
      assert x in path and y in path and len(path)<=self.num_recall
      assert t not in path
    except:
      print("[recall_mechanism] ...")
      pdb.set_trace()


    if not is_tree(T): print("[recall_mechanism] not tree"); pdb.set_trace()
  
    # add path to connection point
    T[t].add(x)
    for a in path:
      if a not in T:  T[a] = set()
    T[x].add(t)
    for i in range(1,len(path)):
      T[path[i]].add(path[i-1])
      T[path[i-1]].add(path[i])
    #
    T[y].add(p)
    for k,vl in P.items():
      T[k] = set(list(vl))
    T[p].add(y)
    #
    if not is_tree(T): print("[recall_mechanism] not tree"); pdb.set_trace(); print(">>")

    return T,len(path)

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
    sF = {}
    non_mem = set([x for x in T if x not in mem])
    for u in non_mem:
      sF[u] = set([x for x in T[u] if x in non_mem])
    #
    return newT,sF

  def scorer(self,root,T):
    if   self.c_score=="lvl":
      depth_map = get_node_level(root,T)
      for u,dp in depth_map.items():
        self.node_scores[u] += 1.0/dp

    elif self.c_score=="subtree":
      st_map = {x:0 for x in T.keys()}      
      st_map,_ = get_subtree_size(root,T,st_map,set())
      n_nodes = len(T)
      for u,sz in st_map.items():
        self.node_scores[u] += (sz / n_nodes)

    elif self.c_score=="comb":
      depth_map = get_node_level(root,T)
      st_map = {x:0 for x in T.keys()}      
      st_map,_ = get_subtree_size(root,T,st_map,set())
      n_nodes = len(T)
      for u in T.keys():
        self.node_scores[u] += (st_map[u] / n_nodes) * np.exp(1.0/depth_map[u] + 1e-12)

    elif self.c_score=="cnt":
      for u in T.keys():
        self.node_scores[u] += 1.0

    # graph centrality scaling
    elif self.c_score in ["btw","cls","eigen"]:
      sc = unwgtd_centrality_scorer(T,self.c_score)
      for u,s in sc.items():
        self.node_scores[u] += s
    return

  def run(self,prop_trees):
    # self.flush() # reset forgetting pool
    T = {} # memory tree
    root = None
    self.t_persistance_cnt = 0
    for proot,P in prop_trees:
      P = {x:set(y) for x,y in P.items()}
      self.update_overlap_memo(P)
      self.on_hold = False
      # add new nodes to mem tree
      root,T,nrecalled = self.find_best_attachment(proot,P,root,T)
      # adjust root
      new_root = get_new_root(root,T,self.get_edge_wgt(T))
      # prune tree by KvD select
      T,sF = self.kvd_memory_select(new_root,T)
      self.update_forget_pool(sF)
      if not self.on_hold:
        self.scorer(new_root,T)
        self.t_persistance_cnt = 0
      else:
        self.t_persistance_cnt += 1
      root = new_root
      if self.t_persistance_cnt == MAX_T_PERSISTANCE:
        del T,root
        T = {}; root = None
    #
    return

  def run_debug(self,section_prop_trees,section_sents, \
                debug=False,assertions=False,oracle=None):
    stats = {
      "new_roots":0,
      "n_recalled":[],
      "pos_recalled":{},
      "t_persistance_cnt":[],
      "mt_props":set(),
      "forgotten": []
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
        if assertions: assert is_tree(P)
        if debug:
          pid = list(P.keys())[0]
          print("\n#############################################################")
          print("[main iter] section=%d, sent_in_sec=%d, section_sent_id_orig=%d | |P|=%d" % \
            (self.D[pid].section_id,self.D[pid].section_sent_id,self.D[pid].section_sent_id_orig,len(P)))
          print("\tsent= ",sent)
          print("\t local graph P; p_root=",proot)
          pprint(P)
          print("\tInitial T; t_root=",root)
          pprint(T)
          for k in P.keys(): print("\t",self.D[k])
          for k in T.keys(): print("\t",self.D[k])
          # print("\tInitial F: (last 10)")
          # pprint(self.F[-10:])
          # print("\tInitial distance in F: (last 10)")
          # pprint(self.distF[-10:])
          # pdb.set_trace()
        root,T,nrecalled = self.find_best_attachment(proot,P,root,T)
        try:
          if assertions: assert is_tree(T)
        except:
          print("[main] no tree")
          # pdb.set_trace()
          
        if nrecalled>0:
          stats["n_recalled"].append(nrecalled)
          stats["pos_recalled"][(sec_id,sent_id)] = (nrecalled,)
          # case 1: recall after oracle
          if oracle is not None:
            if len(set(T.keys()) & oracle_props) > 0:
              stats["pos_recalled"][(sec_id,sent_id)] = (nrecalled,"Orc1")
            
        if debug:
          print("[after attachmnt]",T)
        # adjust root
        try:
          wgts = self.get_edge_wgt(T)
          new_root = get_new_root(root,T,wgts)
        except:
          print("[main] wgt =0 ,",[x for x,y in wgts.items() if y==0])
          pdb.set_trace()

        if new_root != root and root is not None:
          stats["new_roots"] += 1
        # prune tree by KvD select
        T,sF = self.kvd_memory_select(new_root,T)

        ## case 2: recall == oracle
        if nrecalled>0 and oracle is not None:
          if len(set(T.keys()) & oracle_props) > 0:
            stats["pos_recalled"][(sec_id,sent_id)] = (nrecalled,"Orc2")
          
        if assertions: assert is_tree(T)
        self.update_forget_pool(sF)
        if not self.on_hold:
          self.scorer(new_root,T)
          if self.t_persistance_cnt > 0:
            stats["t_persistance_cnt"].append(self.t_persistance_cnt)
          self.t_persistance_cnt = 0
        else:
          self.t_persistance_cnt += 1

        if assertions:
          assert is_tree(T)
          assert len(T) <= self.wm_size

        root = new_root
        stats["mt_props"].update(T.keys())

        if debug:
          print(f"[kvd-select] root={new_root}, |T|={len(T)}, on_hold={self.on_hold}, persistance_cnt={self.t_persistance_cnt}")
          print("[kvd-select] T")
          pprint(T)
          print("\t discarded...")
          pprint(sF)
          print("\tnode scores")
          pprint(self.node_scores)
          print(f'\t#recalls={len(stats["n_recalled"])}, #avg nodes recalled={np.mean(stats["n_recalled"])}')
          # pdb.set_trace()

        if self.t_persistance_cnt == MAX_T_PERSISTANCE:
          del T,root
          T = {}; root = None
      #
      stats["forgotten"].append(sum([len(x) for x in self.F])) # blop sizes
      # stats["blop_dists"] = [d for dF in self.distF for xy,d in dF.items()] # blop dists

    #
    return stats