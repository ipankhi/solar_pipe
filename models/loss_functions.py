import torch
import torch.nn as nn

class AsymSmoothL1_Physics(nn.Module):
    def __init__(self, beta=0.1, under_w=1.4, over_w=1.0, lambda_phys=0.05):
        super().__init__()
        self.beta, self.under_w, self.over_w, self.lambda_phys = beta, under_w, over_w, lambda_phys

    def forward(self, pred, target, ghi_tensor):
        diff = pred - target
        abs_diff = torch.abs(diff)
        base = torch.where(abs_diff < self.beta, 0.5 * diff**2 / self.beta, abs_diff - 0.5*self.beta)
        w = torch.where(diff < 0, self.under_w, self.over_w)
        asym_loss = (base * w).mean()
        ghi_norm = (ghi_tensor / (ghi_tensor.max() + 1e-6)).view(-1,1)
        phys_penalty = torch.relu(pred * ghi_norm - 1.0).pow(2).mean()
        return asym_loss + self.lambda_phys * phys_penalty
