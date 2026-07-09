import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from torch.nn import BCEWithLogitsLoss
import numpy as np
from typing import Optional
from scipy.ndimage import gaussian_filter

from utils.config import SAMFinetuneConfig


class CombinedLoss(torch.nn.Module):
    def __init__(
        self,
        config: SAMFinetuneConfig
    ):
        super().__init__()
        self.smooth = 1e-6
        self.lambda_dice = 1 - config.lambda_bce - config.lambda_kl 
        self.lambda_bce = config.lambda_bce
        self.lambda_kl = config.lambda_kl
        
        self.dice = DiceLoss(
            include_background=True,
            sigmoid=True,
            squared_pred=True,
            reduction='mean'
        )
        
        self.BCE = BCEWithLogitsLoss(reduction='mean')
        self.MSE = nn.MSELoss(reduction='mean')
        
        self.device = config.device
        self.sigma = config.sigma

    def soft_label(self, mask : torch.Tensor) -> torch.Tensor:
        """
        Generate soft label for KL divergence loss. (Gaussian filter)
        """
        mask_np = mask.cpu().numpy()
        soft_mask = gaussian_filter(mask_np.astype(float), sigma=self.sigma)
        soft_mask = torch.tensor(soft_mask).to(mask.device)
        if soft_mask.max() < 1e-8:
            return torch.zeros_like(mask)
        return soft_mask / (soft_mask.max() + 1e-8)
    
    def kl_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        KL divergence loss.
        """
        pred_sigmoid = torch.sigmoid(pred)
        target_soft = self.soft_label(target)
        
        if torch.all(target_soft == 0):
            return torch.tensor(0.0, device=pred.device)
        
        # Normalize per pixel
        pred_sigmoid = pred_sigmoid.view(-1, 1)
        target_soft = target_soft.view(-1, 1)
        
        zeros = torch.zeros_like(pred_sigmoid)
        pred_dist = torch.cat([1 - pred_sigmoid, pred_sigmoid], dim=1)
        target_dist = torch.cat([1 - target_soft, target_soft], dim=1)
                
        kl = F.kl_div(
            F.log_softmax(pred_dist, dim=1),
            target_dist,
            reduction='batchmean',
            log_target=False
        )
        
        return torch.clamp(kl, 0, 2)

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
    ) -> torch.Tensor:
        
        dice_loss = self.dice(pred, target)
        bce_loss = self.BCE(pred, target)
        kl = self.kl_loss(pred, target)
        
        lambdas = np.array([self.lambda_dice, self.lambda_bce, self.lambda_kl])
        
        if lambdas.sum() > 1:
            lambdas = lambdas / lambdas.sum()
            print(f"Warning: lambdas sum is greater than 1. lambdas: {lambdas}, Normalization applied.")
        

        total_loss = (
            lambdas[0] * dice_loss + 
            lambdas[1] * bce_loss + 
            lambdas[2] * kl 
        )
        
        return total_loss
    
if __name__ == "__main__":
    config = SAMFinetuneConfig(
        device='cpu',
        lambda_bce=0.2,
        lambda_kl=0.2,
        sigma=1
    )
    loss = CombinedLoss(config)
    pred = torch.randn(1, 1, 128, 128)
    target = torch.randn(1, 1, 128, 128)
    print(loss(pred, target))