import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLossM(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True, do_bg=True):
        super(FocalLossM, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.do_bg = do_bg
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha]) # Set alpha
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha) # Set alpha (alternative format)
        self.size_average = size_average # Flag to average the loss

    def forward(self, x, y):
        if self.do_bg == False:
            x = x[:,1:,...] # Exclude background from prediction
            if y.dim() > 1:
                y = y[:,1:,...] # Exclude background from ground truth
            else:
                raise Exception('Wrong gt value shape. Please provide gt as ([N,C,H,W])')

        # Reshape prediction tensor if needed
        if x.dim() > 2:
            x = x.view(x.size(0), x.size(1), -1) # Reshape to N,C,H*W
            x = x.transpose(1, 2) # Transpose to N,H*W,C
            x = x.contiguous().view(-1, x.size(2)) # Flatten to N*H*W,C

        # Reshape ground truth tensor if needed
        if y.dim() > 1:
            y = torch.argmax(y, dim=1) # Convert one-hot to labels
            y = torch.flatten(y) # Flatten the tensor

        y = y.view(-1,1) # Reshape for gather operation
        logpt = F.log_softmax(x, 1) # Calculate log softmax of prediction
        logpt = logpt.gather(1, y.long()) # Gather values corresponding to ground truth labels
        logpt = logpt.view(-1) # Reshape logpt
        pt = logpt.data.exp() # Get probabilities

        # Calculate final loss
        if self.alpha is not None:
            if self.alpha.type() != x.data.type(): # Type check
                self.alpha = self.alpha.type_as(x.data) # Convert alpha to same type as x.data
            at = self.alpha.gather(0, y.data.view(-1)) # Gather alpha values
            logpt = logpt * Variable(at) # Multiply logpt with alpha

        # Calculate focal loss
        loss = -1 * (1 - pt) ** self.gamma * logpt

        # Return mean or sum of loss based on size_average flag
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()
