import torch
import numpy as np
import torch.nn as nn

def sum_tensor(inp, axes, keepdim=False):
    """
    Sum over specific axes.
    """
    axes = np.unique(axes).astype(int)
    for ax in axes:
        inp = inp.sum(int(ax), keepdim=True)
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    Returns true positives, false positives and false negatives.
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    if len(net_output.shape) != len(gt.shape):
        gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

    if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
        y_onehot = gt
    else:
        gt = gt.long()
        y_onehot = torch.zeros(net_output.shape)
        if net_output.device.type == "cuda":
            y_onehot = y_onehot.cuda(net_output.device.index)
        y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

class TverskyLoss(nn.Module):
    """
    Tversky loss for image segmentation.
    """
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, alpha=0.3, beta=0.7):
        super(TverskyLoss, self).__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        """
        Compute Tversky loss.
        """
        axes = [0] + list(range(2, len(x.shape))) if self.batch_dice else list(range(2, len(x.shape)))
        x = self.apply_nonlin(x) if self.apply_nonlin is not None else x
        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        
        if not self.do_bg:  # Exclude background from loss calculation
            tversky = tversky[1:] if self.batch_dice else tversky[:,1:, ...]

        return 1 - tversky.mean()
