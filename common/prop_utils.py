from collections import OrderedDict,defaultdict,namedtuple
import bisect
import pdb
import copy
from pprint import pprint
import numpy as np
from utils import *
import string

class DepNode:
  def __init__(self,_id,tok_list,parent,rel,children,merge_type):
    self.id = _id
    self.token_list = tok_list
    # local connections
    self.rel = rel
    self.parent = parent
    self.children = children
    # global connecitons
    self.merge_type = merge_type
    self.sub_rel = ""
    self.main_token = None

  def __repr__(self):
    return "id:%2d | Tokens: [%35s] | h-rel: %8s | %15s | mtype: %s" % \
    ( self.id,
      # ",".join(["(%d,%s,%s)"%(x.id,x.form,x.pos) for x in self.token_list]),
      ",".join(["(%d,%s)"%(x.id,x.form) for x in self.token_list])[:35],
      "%d-%s" % (self.parent, self.rel),
      "ch:" + ",".join([str(x) if type(x)==int else str(x.id) for x in self.children]),
      self.merge_type)


  def calc_main_token(self,):
    id_list = [x for x in self.token_list if x.id==self.id]
    if len(id_list)>0:
      self.main_token = id_list[0]
    else:
      self.main_token = self.token_list[0]


class Proposition:
  def __init__(self,pid,depnode,\
               section_id,section_sent_id,
               section_sent_id_orig):
    self.id = pid
    self.section_sent_id = section_sent_id # id inside section
    self.section_sent_id_orig = section_sent_id_orig # id inside section before filtering
    self.section_id = section_id
    self.predicate = depnode.token_list
    self.arguments = [u if type(u)==int else u.token_list for u in depnode.children]
  
  def form_seq(self,tok_list):
    return "_".join([x.form for x in tok_list])

  def __repr__(self):
    pred = self.form_seq(self.predicate)
    args = [f"${x}" if type(x)==int else self.form_seq(x) for x in self.arguments]
    return "[%d] %s (%s)" % (self.id,pred,"; ".join(args))

  def text(self):
    pred = self.form_seq(self.predicate)
    args = [f"${x}" if type(x)==int else self.form_seq(x) for x in self.arguments]
    return "%s (%s)" % (pred,"; ".join(args))



class Argument:
  def __init__(self,aid,depnode,section_id,section_sent_id,section_sent_id_orig):
    self.id = aid
    self.section_sent_id = section_sent_id # id inside section
    self.section_sent_id_orig = section_sent_id_orig # id inside section before filtering
    self.section_id = section_id
    self.tokens = depnode.token_list
    self.deprel = depnode.rel # dependency relation of head token in argument wrt its parent node


"""
Wrapper class for propositions, arguments, prop trees,...
useful for source doc, summary, ...
"""
class StructWrapper:
  def __init__(self,):
    self.propositions = OrderedDict()
    self.arguments = OrderedDict()
    self.prop_trees = []



############################################################################################################

def textualize_prop(p,D):
  toks = [x for x in p.predicate]
  for a in p.arguments:
    if type(a)==int:
      toks.extend(D[a].predicate)
    else:
      toks.extend(a)
  toks.sort(key=lambda x:x.id)
  return [x.form for x in toks]


############################################################################################################

## extracts propositions by section, for entire document
## D: dict of tree nodes
## sections: [{node id dict} ]; List of sectionss each section a list of trees

class PropositionExtractor:
  def __init__(self,):
    self.global_node_cnt = 0
    self.section_sent_cnt_orig = -1
    self.D = OrderedDict()
    self.section_sents = []
    self.section_ptrees = []


  """
  block of sentences in CONLLU format
  """
  def read_conllu_from_str(self,text):
    text = text.strip("\n").split("\n\n")
    self.section_sent_cnt_orig = -1
    for i,sent_text in enumerate(text):
      tree = {}
      sent = []
      lines = [x for x in sent_text.split("\n") if not x.startswith("#")]
      for line in lines:
        line = line.strip("\n")
        cols = line.split("\t")
        if "." in cols[0]: continue
        _id = int(cols[0]) - 1
        cols = line.split("\t")
        sent.append(cols[1])
        par = int(cols[6]) - 1
        drel = cols[7].split(":")[0]
        tnode = DepNode(_id,[TokenTup(_id,cols[1],cols[2],cols[3])],\
                          par,drel,[],None)
        tree[_id] = tnode
      #
      self.section_sent_cnt_orig += 1 # increased always, even if sent is discarded
      if len([x for x in sent if x not in string.punctuation]) < MIN_TOKS_PER_SENT:
        continue
      for item in tree.values():
        if item.parent == -1: continue
        tree[item.parent].children.append(item.id)
      for item in tree.values():
        item.children.sort()
        self.global_node_cnt = max(self.global_node_cnt,item.id)
      #
      yield i,sent,tree
    #

  """
  n1 is assumed to be parent, n1->n2
  """
  def merge_tnodes(self,n1,n2,T,force_mtype=None):
    new_form_list = n1.token_list + n2.token_list
    new_form_list.sort(key=lambda x:x.id)
    new_children = [x for x in n1.children if x!=n2.id] + n2.children
    new_mtype = force_mtype
    if force_mtype is None:
      if n2.merge_type is None and n1.merge_type is None:
        new_mtype = n2.rel if n2.rel!="punct" else None
      else:
        new_mtype = n1.merge_type or n2.merge_type

    new_tnode = DepNode(n1.id,
                    new_form_list,
                    n1.parent,
                    n1.rel,
                    new_children,
                    new_mtype)  # rel assumed to be eq or one of them is None
    # update n2 children's parent
    for u in n2.children:
      if type(u)==int and u!=-1:
        T[u].parent = n1.id
    # update new tree node in structure
    T[n1.id] = new_tnode
    # delete old node from strct
    del T[n2.id]
    return new_tnode,T


  def is_mergeable(self,tchild,tnode,T):
    not_allwd_punct = "()[]{},"
    if tchild.rel == "punct":
      if tchild.token_list[0].form in not_allwd_punct:
        return False
      v_id = tnode.children.index(tchild.id)
      rb = v_id
      while rb < len(tnode.children) and T[tnode.children[rb]].rel == "punct":
        rb += 1
      if rb==len(tnode.children):
        # case: last tok of compound is head
        if tchild.token_list[0].id < tnode.token_list[-1].id:
          return True
        return False
      # case: first tok of compound is head
      else:
        return self.is_mergeable(T[tnode.children[rb]],tnode,T)

    # if both types are not None, don't consider det
    if tnode.merge_type is not None and tchild.merge_type is not None:
      return tnode.merge_type == tchild.merge_type and tchild.merge_type!="det"

    # when parser fails to clsf
    if tnode.merge_type is not None and tchild.rel == "dep":
      return True

    # Case: aux + cop (e.g. will be / have been)
    if tchild.rel=="cop":
      for ch in tnode.children:
        if T[ch].rel=="aux": return True
      return False

    if tchild.rel in MERGE_RELATIONS_NO_COND:
      return True
    elif tchild.rel in MERGE_RELATIONS_COND:
      if len(tchild.token_list)==1:
        return True
      return False
    return False


  def flat_phrase(self,u,T):
    ini = -1; end = -1
    allwd = ["fixed","flat","compound",]
    not_allwd_punct = "()[]{}"
    for i,v in enumerate(T[u].children):
      if T[v].rel in allwd:
        ini = i; break
      if T[v].rel == "punct":
        if T[v].token_list[0].form not in not_allwd_punct:
          ini = i; break
    if ini==-1:
      return []
    flat_found = False
    for j,v in enumerate(T[u].children[ini+1:]):
      if T[v].rel in ["fixed","flat","compound"]:
        flat_found = True
      if T[v].rel not in allwd:
        break
      end = j + ini+1
    #
    if not flat_found:
      return []
    if T[T[u].children[end]].rel == "punct":
      end -= 1
    return T[u].children[ini:end+1]


  """
  Step A: simplify <set1> types & create coordination nodes
  """
  def pre_traversal(self,u,T):
    # merge
    flat_phr = self.flat_phrase(u,T)
    for v in flat_phr:
      T[u],T = self.merge_tnodes(T[u],T[v],T,force_mtype="flat")
    
    ch_list = list(T[u].children)
    for v in ch_list:
      T = self.pre_traversal(v,T)
      if self.is_mergeable(T[v],T[u],T):
        T[u],T = self.merge_tnodes(T[u],T[v],T)

    return T


  def fix_punct_nodes(self,T):
    punct_ids = [x for x,u in T.items() if u.rel=="punct"]
    n_t = len(T)

    for _ in range(n_t):
      curr = 0
      for u in punct_ids:
        if len(T[u].children)==0: continue
        p = T[u].parent
        if p!=-1:
          T[p].children.extend(T[u].children)
          T[p].children.sort()
          for v in T[u].children:
            T[v].parent = p
          T[u].children = []
        # tricky case: u is root with rel == punct
        else:
          T[u].rel = "root"
        curr +=1
      #
      if curr == 0:  break
    #

    # if curr!=0:
    #   print(">>[fix punct] still punct to fix!!")
    #   pdb.set_trace()
    return T


  def create_conj_traversal(self,u,T,pool_cjtors):
    # pool_cjtors: T ids of conjunctors in global tree

    def relocate(mod_node,conj_node):
      # add CONJ child pnt, erases child from original parent, sets new parent, does not 
      conj_node.children.append(mod_node.id)
      conj_node.rel = mod_node.rel
      mod_node.rel = "conj"
      if mod_node.parent != -1:
        T[mod_node.parent].children.remove(mod_node.id)
        # T[mod_node.parent].children.append(conj_node.id)
      mod_node.parent = conj_node.id
      return mod_node,conj_node


    ###############################################################

    # pnode = T[T[u].parent] # head of first conjunctor
    pnode_id = T[u].parent

    # identify siblings and CC
    conjunctors_id = []
    cc_node_id = -1
    for c in T[u].children:
      cnode = T[c]
      if cnode.rel != "conj":
        if cnode.rel == "cc":
          cc_node_id = c
        continue
      conjunctors_id.append(c)
      # search for "cc" node
      for gc in cnode.children:
        if T[gc].rel =="cc":
          cc_node_id = gc; break
      # 
    #
    conjunctors_id.sort()
    conjunctors_id_parents = [T[x].parent for x in conjunctors_id]
    conj_node = None
    if cc_node_id!=-1:
      conj_node = T[cc_node_id]
      conj_node_id = cc_node_id
      # extirpar CC node
      T[conj_node.parent].children.remove(conj_node_id)
      conj_node.parent = pnode_id
      conj_node.rel = T[u].rel

    else:
      conj_node_id = self.global_node_cnt + 1
      conj_node = DepNode(conj_node_id,[TokenTup(conj_node_id,"CCONJ","CCONJ", None)],
                    pnode_id,T[u].rel,[],None)
      T[conj_node_id] = conj_node
      self.global_node_cnt = max(self.global_node_cnt,conj_node_id)
    #
    # exchange conj head with new CONJ node
    T[u],conj_node = relocate(T[u],conj_node)
    if pnode_id!=-1:
      T[pnode_id].children.append(conj_node_id)
      T[pnode_id].children.sort()
      
    
    # case where might not be found: A,B <>, where B->conj->A (or vicev)
    # if cc_node_id!=-1:
    #   T[cc_node_id],conj_node = relocate(T[cc_node_id],conj_node)

    ## Resolving scope ambiguity
    # Type heuristic: [preproc] obtain relation types by conjunctor
    rel_types_per_conj = {}
    for cnjt in conjunctors_id:
      if cnjt not in rel_types_per_conj:
        rel_types_per_conj[cnjt] = set()
      for mod_id in T[cnjt].children:
        if T[mod_id].rel != "conj":
          rel_types_per_conj[cnjt].add(T[mod_id].rel)
    #

    # subtree heuristic
    for i,c_id in enumerate(conjunctors_id):
      tnode = T[c_id]
      chldrn = tnode.children.copy()
      for m_id in chldrn:
        # case: minimum tree covers all conjt, apply heuristics
        ## 1. Position heuristic
        pos_narrow = False
        pos_mod = bisect.bisect_left(conjunctors_id,m_id)
        if pos_mod==i+1 or pos_mod==i:
          pos_narrow = True
        ## 2. Type heuristic
        mod_type = T[m_id].rel
        is_subj_type = (mod_type=="nsubj" or mod_type=="csubj")
        if i==0 and not pos_narrow:
          for ccj_id,types in rel_types_per_conj.items():
            if any([
                mod_type in types,
                all([is_subj_type,
                    "nsubj" in types or "csubj" in types])
              ]):
              pos_narrow = True
              break
          #
        ## relocate if scope is wide
        if not pos_narrow:
            T[m_id],conj_node = relocate(T[m_id],conj_node)
        #
      # relocate conjunctors
        T[c_id],conj_node = relocate(tnode,conj_node)
    # re-sort conjunctors wrt sent order
    conj_node.children.sort(key=lambda x:T[x].token_list[0].id)
        
    # update pool of conj in global tree
    pool_cjtors -= set(conjunctors_id_parents)
    
    return pool_cjtors


  def collapse_dep_tree(self,T):
    def get_root(TT):
      root = -1
      for _id,tnode in TT.items():
        if tnode.parent==-1: root = _id; break
      return root

    """
    @param T: dependency tree
    """
    # A.0. fix punctuation nodes (parser errors), case when punct is not a leaf
    T = self.fix_punct_nodes(T)

    # A.1. Merge/join non-cnt/mod/funct nodes
    root = get_root(T)
    T = self.pre_traversal(root,T)

    # r,tree = get_dep_tree(T)
    # if not is_tree(tree):
    #   print("\t post pretraversal | not a tree",root)
    #   pprint(tree)
    #   pdb.set_trace()

    # A.2. Create CONJ nodes
    conj_d_nodes = set([x.parent for ti,x in T.items() if ti!=root and x.rel=="conj"])
    visited = set()
    while len(conj_d_nodes) > 0:
      start = conj_d_nodes.pop()
      visited.add(start)
      conj_d_nodes = self.create_conj_traversal(start,T,conj_d_nodes)
      conj_d_nodes = conj_d_nodes - visited
    # print("--> step A finished!")

    # Remove end-of-phrase punct
    tmpT = copy.deepcopy(T)
    keys = list(T.keys())
    for x in keys:
      if tmpT[x].rel == "punct":
        if x in tmpT[tmpT[x].parent].children:
          tmpT[tmpT[x].parent].children.remove(x)
        del tmpT[x]
    #

    # Collapse leaves without arguments
    newT = {}
    keep_ids = [y for y,x in tmpT.items() if len(x.children)>0 or x.parent==-1]
    for _id in keep_ids:
      tnode = tmpT[_id]
      chldren = []
      for xid in tnode.children:
        if T[xid].rel=="punct": continue
        if xid not in keep_ids:
          otnode = T[xid]
          otnode.id = -1
          chldren.append(otnode)
        else:
          chldren.append(xid)
      tnode.children = chldren
      newT[_id] = tnode
    #
    del tmpT,T
    T = newT

    # root,tree = get_dep_tree(T)
    # print("[collapse_dep_tree] check tree")
    # print("\tIs tree=",is_tree(tree))
    # pprint(T)
    # pdb.set_trace()

    # Collapse lists structs: a-b-c-..-d where a has 1 tok in pred, 1 arg int
    root,tree = get_dep_tree(T)
    tripas = find_tripa(root,tree,T)
    while len(tripas)>0:
      for tripa in tripas:
        # print("\tlist=",tripa)
        for i in range(1,len(tripa)):
          v = tripa[i]
          v_1 = tripa[i-1]
          # print("\t  join=",v,v_1)
          T[v],T = self.merge_tnodes(T[v],T[v_1],T)
        ## case: joint tripa prop has no args
        # add as arg into parent
        tnode = T[tripa[-1]]
        if len(tnode.children)==0 and tnode.parent != -1:
          idx = T[tnode.parent].children.index(tnode.id)
          tnode.id = -1
          T[tnode.parent].children[idx] = tnode
          del T[tripa[-1]]

        # root,tree = get_dep_tree(T)
        # pprint(tree)
        # pdb.set_trace()
        
      root,tree = get_dep_tree(T)
      tripas = find_tripa(root,tree,T)

    for _id,tnode in T.items():
      tnode.calc_main_token()
      for v in tnode.children:
        if type(v)!=int:
          v.calc_main_token()
    #
    for _id,tnode in T.items():
      tnode.children.sort(key=lambda x: T[x].main_token.id if type(x)==int else x.main_token.id)

    return T


  ###############################################################

  def get_propositions(self,T,section_sent_id_orig):
    keys = list(T.keys())
    offset_id = len(self.D)
    curr_section = len(self.section_sents)-1
    section_sent_id = len(self.section_sents[-1])
    root,tree = get_dep_tree(T)
    dep2prop_id = dict(zip(keys,range(offset_id,offset_id+len(T))))
    root = dep2prop_id[root]
    
    for _id in keys:
      tnode = T[_id]
      tnode.id = dep2prop_id[_id]
      if tnode.parent!=-1:
        tnode.parent = dep2prop_id[tnode.parent]
      chldren = [dep2prop_id[x] if type(x)==int else x for x in tnode.children]
      tnode.children = chldren
      # build local tree
      pnode = Proposition(pid=tnode.id,
                  depnode=tnode,
                  section_id=curr_section,
                  section_sent_id=section_sent_id,
                  section_sent_id_orig=section_sent_id_orig)

      self.D[pnode.id] = pnode

    #
    ptree = {dep2prop_id[k]:[dep2prop_id[y] for y in v] for k,v in tree.items()}

    if not is_tree(ptree):
      print(">newT not tree!"); pdb.set_trace()

    return root,ptree



  ###############################################################


  def update(self,cnllu_line):
    self.section_sents.append([])
    self.section_ptrees.append([])
    for orig_sent_id,tokens,dT in self.read_conllu_from_str(cnllu_line):
      # print("#"*100); pprint(dT); print()
      # print("\t|dT|=%d" % (len(dT)))

      dT = self.collapse_dep_tree(dT)
      root,ptree = self.get_propositions(dT,orig_sent_id)
      if len(ptree)==1 and \
         any([len(self.D[root].predicate)==0 or len(self.D[root].arguments)==0]):
         del self.D[root],dT,ptree,root,tokens
         continue

      self.section_sents[-1].append(" ".join(tokens))
      self.section_ptrees[-1].append([root,ptree])

      # print("#"*100); 
      # print(f"[update] sid={local_sent_id}, |P|={len(ptree)}, root={root}")
      # pprint(ptree); pprint([self.D[i] for i in ptree.keys()]);
      # pdb.set_trace()
      # print()

      del dT,ptree,root,tokens


