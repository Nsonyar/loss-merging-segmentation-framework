import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_dice import DiceLoss
from loss_focal import FocalLossM
from loss_tversky import TverskyLoss
from loss_boundary import BoundaryLoss
from loss_hausdorff import HausdorffLoss

from torchmetrics import Dice
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassJaccardIndex

from inputClass import Input

#torch.set_printoptions(precision=2)

background=True
focal=FocalLossM(do_bg=background)
dice = DiceLoss(do_bg=background)
tversky=TverskyLoss(do_bg=background)
boundary=BoundaryLoss()
hausdorff=HausdorffLoss()

input = Input()
dice_mc = Dice(average='macro',num_classes=4).cuda()
accuracy_mc = MulticlassAccuracy(average='macro',num_classes=4).cuda()
jaccardIndex_mc = MulticlassJaccardIndex(average='macro',num_classes=4).cuda()
precision_mc = MulticlassPrecision(average='macro',num_classes=4).cuda()

torch.cuda.set_device(0)

#Can be used to provide weights to CE
weights=torch.tensor([1,1,1,0.1],dtype=torch.float64).cuda()

print('CE:',F.cross_entropy(input.y_hat_score[:,:2,:,:], input.y[:,:2,:,:]).item())
print('FL:', focal(input.y_hat_score[:,:2,:,:],input.y[:,:2,:,:]).item())

print('DL:',dice(input.y_hat_soft, input.y).item())
print('TL:',tversky(input.y_hat_soft,input.y).item())

print('BL:',boundary(input.y_hat_soft,input.y).item())
print('HL:',hausdorff(input.y_hat_soft,input.y).item())
