import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """ 
    Dice Loss. 
    """ 
    def __init__(self, eps=1e-6, do_bg=True):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.do_bg = do_bg
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Calculates the Dice Loss.
        :param y_pred: Predictions tensor (NxCxHxW)
        :param y_true: Ground truth tensor (NxCxHxW)
        :return: scalar representing Dice Loss
        """
        # Get the batch size and number of classes
        bs = y_true.size(0) #N
        cl = y_true.size(1) #C

        # Reshape the tensors
        y_true = y_true.view(bs, cl, -1) #NxCx(H*W)
        y_pred = y_pred.view(bs, cl, -1) #NxCx(H*W)

        # Calculate intersection and cardinality
        intersection = torch.sum(y_pred * y_true, (0, 2)) #C
        cardinality = torch.sum(y_pred + y_true, (0, 2)) #C

        # Calculate dice score
        dice_score = (2. * intersection + self.eps) / (cardinality + self.eps)
  
        # If do_bg is False, skip the background class
        if not self.do_bg:
            dice_score = dice_score[1:, ...]

        # Return Dice Loss
        return (1. - dice_score).mean()