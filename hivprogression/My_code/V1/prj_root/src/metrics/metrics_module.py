
def calculate_f1_score(pred_vali,labels_oh_tc):
  preds=(pred_vali>0.0).int()
  # print("preds",preds)

  targs=labels_oh_tc.int()
  # print("targs",targs)


  single_TP=(preds*targs).float().sum(dim=0)
  single_FP=(preds>targs).float().sum(dim=0)
  single_FN=(preds<targs).float().sum(dim=0)
  
  return single_TP,single_FP,single_FN