
import torch.nn.functional as F

def FocalLoss(output,target):
  gamma=2
  if not (target.size()==output.size()):
    raise ValueError("Targetsize({})mustbethesameasinputsize({})".format(target.size(),output.size()))

  max_val=(-output).clamp(min=0)
  loss=output-output*target+max_val+((-max_val).exp()+(-output-max_val).exp()).log()

  invprobs=F.logsigmoid(-output*(target*2.0-1.0))
  loss=(invprobs*gamma).exp()*loss
    
  return loss.sum(dim=1).mean()
