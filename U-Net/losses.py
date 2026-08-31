import torch
import torch.nn as nn


class MixedLoss(nn.Module):
    """Weighted combination of L1 loss and gradient loss."""

    def __init__(self, alpha=1.0, beta=0.5, split_idx=125):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.split_idx = split_idx
        weights = torch.ones(1, 1, 36, 250)
        weights[:, :, :, :split_idx] = 1
        weights[:, :, :, split_idx:] = 2
        self.register_buffer("weight_matrix", weights, persistent=False)
        self.l1_loss = nn.L1Loss(reduction="none")

    def gradient_loss(self, pred, target, weights=None):
        if weights is None:
            weights = self.weight_matrix
        pred_grad_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_grad_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_h = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_grad_w = target[:, :, :, 1:] - target[:, :, :, :-1]
        weights_h = weights[:, :, 1:, :]
        weights_w = weights[:, :, :, 1:]
        grad_h = (self.l1_loss(pred_grad_h, target_grad_h) * weights_h).mean()
        grad_w = (self.l1_loss(pred_grad_w, target_grad_w) * weights_w).mean()
        return (grad_h + grad_w) / 2

    def forward(self, pred, target):
        l1 = (self.l1_loss(pred, target) * self.weight_matrix).mean()
        grad = self.gradient_loss(pred, target)
        return self.alpha * l1 + self.beta * grad
