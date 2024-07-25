import torch.nn as nn
import torch

def rmse(targets: torch.Tensor, pred: torch.Tensor):
    assert targets.shape == pred.shape
    if targets.dim() == 3:
        dim2 = (0,1)
        dim1 = 2
    else:
        dim2 = 0
        dim1 = 1
    return torch.mean(torch.norm((pred - targets), p=2, dim=dim1), dim=dim2)

loss_fn_map = {
    "l1": nn.functional.l1_loss,
    "rmse": rmse,
    "l2": nn.functional.mse_loss,
}