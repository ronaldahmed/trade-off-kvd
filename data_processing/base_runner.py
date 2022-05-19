import os,sys
import pickle
import json
from multiprocessing import Pool

budgets = {
  "pubmed" : 205,
  "arxiv" : 190,
}

BASELINES = ["oracle","lead","rnd","rnd_wgt"]

DATASET_BASEDIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"datasets")

######################################################################################################################################

class BaseRunner:
  def __init__(self,args,params,system_id):
    self.args = args
    self.params = params
    self.system_id = system_id
    self.predictions_by_docname = {}
    self.prop_score_by_docname = {}
    self.budget = budgets[args.dataset]
    self.wthr = budgets[args.dataset] + 50 # extra slack is experimental, 50
    # check if can load preproc
    self.out_filename = os.path.join(self.args.log_dir,self.args.exp_id,"%s.json" % (self.system_id))

  def load_data_block(self,):
    fn = os.path.join(DATASET_BASEDIR,self.args.dataset,"%s-props.pkl" % (self.args.split))
    data = []
    with open(fn,"rb") as infile:
      pkl = pickle.Unpickler(infile)
      while infile.peek(1):
        item = pkl.load()
        data.append(item)
        if len(data)==self.args.data_block_size:
          yield data
          data = []
      #
      yield data
    #

  def load_preds(self,):
    if os.path.exists(self.out_filename):
      with open(self.out_filename,"r") as infile:
        tmp = json.load(infile)
        self.predictions_by_docname,self.prop_score_by_docname = tmp
      print("[Runner] predictions loaded {%s}..." % self.out_filename)
      return True
    return False

  def save(self):
    with open(self.out_filename,"w") as outfile:
        json.dump([self.predictions_by_docname,self.prop_score_by_docname], outfile)

  def update(self,did,preds,pscores):
    if did not in self.predictions_by_docname:
      self.predictions_by_docname[did] = preds
      self.prop_score_by_docname[did] = pscores

  def run(self,):
    if self.load_preds(): return
    cnt = 0
    if self.args.njobs==1:
      for block in self.load_data_block():
        for item in block:
          did,pred,pscores = self.run_parallel(item)
          self.update(did,pred,pscores)
          cnt += 1
    else:
      with Pool(self.args.njobs) as pool:
        for block in self.load_data_block():
          pack = pool.map(self.run_parallel,block)
          for did,pred,pscores in pack:
            self.update(did,pred,pscores)
    #
    # save results
    self.save()
    return

  def run_parallel(self,doc_id):
    return 0.0
