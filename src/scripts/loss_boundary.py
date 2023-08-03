import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Tuple, Iterable, Set
from scipy.ndimage import distance_transform_edt

# Set the softmax function with dimension=1
softmax = nn.Softmax(dim=1)

def check_if_one_hot(t: Tensor, axis=1) -> bool:
    """
    Check if a tensor is one-hot encoded.
    This function checks if all elements are either 0 or 1 and 
    if the sum along the specified axis is 1 (condition for one-hot encoded tensor).
    """
    return simplex(t, axis) and sset(t, [0, 1])

def sset(a: Tensor, sub: Iterable) -> bool:
    """
    Check if all unique elements of tensor 'a' are in 'sub'.
    """
    return uniq(a).issubset(sub)

def uniq(a: Tensor) -> Set:
    """
    Get the unique elements of tensor 'a'.
    """
    return set(torch.unique(a.cpu()).numpy())

def simplex(t: Tensor, axis=1) -> bool:
    """
    Check if the tensor 't' is a one-hot encoded tensor.
    The function checks this by summing the tensor elements along a given axis.
    The sum should be 1 for a one-hot encoded tensor.
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def distance_map_calculation(seg: np.ndarray, threshold: float, resolution: Tuple[None, float, float] = None, dtype=None) -> np.ndarray:
    """
    Calculate the distance map from a one-hot encoded segmentation mask.
    """
    assert check_if_one_hot(seg, axis=1)
    K = seg.shape[1]
    seg_np = seg.cpu().detach().numpy()

    res = np.zeros_like(seg_np, dtype=dtype)
    res = np.squeeze(res, axis=0)
    seg_np = np.squeeze(seg_np, axis=0)

    for k in range(K):
        posmask = seg_np[k] > threshold
        if posmask.any():
            negmask = ~posmask
            negmask_dt = distance_transform_edt(negmask, sampling=resolution)
            posmask_dt = distance_transform_edt(posmask, sampling=resolution)

            res[k] = negmask_dt * negmask - (posmask_dt - 1) * posmask
    return res

class BoundaryLoss(nn.Module):
    """
    Implementation of the Boundary Loss for highly unbalanced segmentation.
    """
    def __init__(self, do_bg=True, threshold=0.5, gpu=0):
        super(BoundaryLoss, self).__init__()
        self.do_bg = do_bg
        self.threshold = threshold
        self.gpu = gpu

    def forward(self, y_pred, y_true):
        """
        Calculate the boundary loss between the predicted labels and the ground truth labels.
        """
        distance_map = distance_map_calculation(y_true, self.threshold)
        distance_map_torch = torch.tensor(distance_map, requires_grad=True, device=torch.device('cuda:' + str(self.gpu)))[None]

        y_pred = (y_pred > self.threshold).float()

        if not self.do_bg:
            y_pred = y_pred[:, 1:, ...].type(torch.float32)
            y_true = y_true[:, 1:, ...].type(torch.float32)
            distance_map_torch = distance_map_torch[:, 1:, ...].type(torch.float32)

        temp_1 = torch.einsum("bcxy,bcxy->bcxy", y_pred, distance_map_torch)
        temp_2 = torch.einsum("bcxy,bcxy->bcxy", y_true, distance_map_torch)
        difference = temp_1 - temp_2

        return difference.mean()
