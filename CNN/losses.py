"""Mixed loss used by the baseline CNN training pipeline."""

import torch
import torch.nn as nn


class MixedLoss(nn.Module):
    """Mixed loss combining weighted L1 loss and gradient loss."""
    def __init__(self, alpha=1.0, beta=0.5, split_idx=125):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.split_idx = split_idx

        weights = torch.ones(1, 1, 36, 250)
        weights[:, :, :, :split_idx] = 1
        weights[:, :, :, split_idx:] = 3
        self.weight_matrix = weights
        self.l1_loss = nn.L1Loss(reduction='none')

    def gradient_loss(self, pred, target, weights):
        pred_grad_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_grad_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_h = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_grad_w = target[:, :, :, 1:] - target[:, :, :, :-1]

        weights_h = weights[:, :, 1:, :]
        weights_w = weights[:, :, :, 1:]
        grad_loss_h = (self.l1_loss(pred_grad_h, target_grad_h) * weights_h).mean()
        grad_loss_w = (self.l1_loss(pred_grad_w, target_grad_w) * weights_w).mean()
        return (grad_loss_h + grad_loss_w) / 2

    def forward(self, pred, target):
        weights = self.weight_matrix.to(pred.device)
        l1_loss_elem = self.l1_loss(pred, target)
        l1 = (l1_loss_elem * weights).mean()
        grad = self.gradient_loss(pred, target, weights)
        return self.alpha * l1 + self.beta * grad
