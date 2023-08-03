import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Tuple
from scipy.ndimage import distance_transform_edt

def distance_map_calculation(seg: np.ndarray, threshold: float, resolution: Tuple[float, float] = None,  dtype=None) -> np.ndarray:
    """
    Calculate the distance map for a given segmentation.
    
    Args:
        seg: The segmentation for which to calculate the distance map.
        threshold: The threshold value to use in the distance calculation.
        resolution: The resolution to use in the distance calculation.
        dtype: The desired data type of the output.

    Returns:
        res: The calculated distance map.
    """
    K: int = seg.shape[1]
    seg_np = seg.cpu().detach().numpy()
    res = np.zeros_like(seg_np, dtype=dtype)
    for k in range(K):
        posmask = seg_np[:,k] > threshold
        if posmask.any():
            negmask = ~posmask
            negmask_dt = distance_transform_edt(negmask, sampling=resolution)
            posmask_dt = distance_transform_edt(posmask, sampling=resolution)
            res[:,k] = posmask_dt + negmask_dt
    return res

class HausdorffLoss(nn.Module):
    """
    A class for calculating the Hausdorff loss. This is a loss function used in image segmentation tasks.
    """
    def __init__(self, do_bg=True, eps=1e-6, hd_alpha=2.0, threshold=0.5, gpu=0):
        """
        Constructor for the HausdorffLoss2 class.

        Args:
            do_bg: Whether to include the background in the loss calculation.
            eps: A small value to avoid division by zero.
            hd_alpha: The alpha value in the Hausdorff distance calculation.
            threshold: The threshold value to use in the distance calculation.
            gpu: The GPU to use for calculations.
        """
        super(HausdorffLoss2, self).__init__()
        self.do_bg = do_bg
        self.eps = eps
        self.alpha = hd_alpha
        self.threshold = threshold
        self.gpu = gpu

    def forward(self, y_hat, y):
        """
        The forward pass of the loss calculation.

        Args:
            y_hat: The predicted segmentation.
            y: The ground truth segmentation.

        Returns:
            The calculated loss.
        """
        # Calculate the distance maps for the ground truth and prediction
        distance_map_y = distance_map_calculation(y, self.threshold)
        distance_map_y_torch = torch.tensor(distance_map_y, requires_grad=True, device=torch.device('cuda:' + str(self.gpu)))
        distance_map_y_hat = distance_map_calculation(y_hat, self.threshold)
        distance_map_y_hat_torch = torch.tensor(distance_map_y_hat, requires_grad=True, device=torch.device('cuda:' + str(self.gpu)))

        # Calculate the prediction error and distance for either the whole image or the image without the background
        if self.do_bg:
            pred_error = (y_hat - y) ** 2
            distance = distance_map_y_hat_torch ** self.alpha + distance_map_y_torch ** self.alpha
        else:
            y_hat_wo_bg = y_hat[:, 1:, ...]
            y_wo_bg = y[:, 1:, ...]
            pred_error = (y_hat_wo_bg - y_wo_bg) ** 2
            distance = torch.abs(distance_map_y_hat_torch[:, 1:, ...]) + torch.abs(distance_map_y_torch[:, 1:, ...])

        # Calculate the final loss
        dt_field = pred_error * distance
        return dt_field.mean()
