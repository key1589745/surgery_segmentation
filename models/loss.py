import torch
import torch.nn as nn
import torch.nn.functional as F

class ELBoLoss(nn.Module):
    def __init__(self, nll_loss, alpha=None):
        super(ELBoLoss, self).__init__()
        self.nll_loss = nll_loss
        self.alpha = alpha

    def forward(self, outputs, target):
        mask_pred, kld = outputs
        ce_loss = self.nll_loss(mask_pred, target, ignore_index=255)
        return ce_loss + self.alpha * kld.mean()