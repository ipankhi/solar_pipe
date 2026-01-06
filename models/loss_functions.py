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
    

class DistillSmoothL1(nn.Module):
    """
    Teacher–Student loss for power prediction
    Target and predictions are normalized power (cf)
    """

    def __init__(
        self,
        beta=0.05,
        alpha=0.7,
        under_w=1.2,
        over_w=1.0
    ):
        """
        beta     : SmoothL1 transition
        alpha    : weight for true-label loss
        under_w  : penalty for under-prediction
        over_w   : penalty for over-prediction
        """
        super().__init__()

        self.beta = beta
        self.alpha = alpha
        self.under_w = under_w
        self.over_w = over_w

    def smooth_l1(self, diff):
        abs_diff = torch.abs(diff)
        return torch.where(
            abs_diff < self.beta,
            0.5 * diff**2 / self.beta,
            abs_diff - 0.5 * self.beta
        )

    def forward(self, pred_student, target, pred_teacher=None):
        """
        pred_student : student output (cf)
        target       : true cf
        pred_teacher : teacher output (cf) or None
        """

        # ---------------------------
        # 1️⃣ True label loss
        # ---------------------------
        diff_true = pred_student - target
        w_true = torch.where(diff_true < 0, self.under_w, self.over_w)
        loss_true = (self.smooth_l1(diff_true) * w_true).mean()

        # ---------------------------
        # 2️⃣ Distillation loss
        # ---------------------------
        if pred_teacher is not None:
            diff_dist = pred_student - pred_teacher.detach()
            loss_dist = self.smooth_l1(diff_dist).mean()
            return self.alpha * loss_true + (1 - self.alpha) * loss_dist

        # ---------------------------
        # 3️⃣ No teacher (pure student)
        # ---------------------------
        return loss_true

