import torch
from torch.utils.data import DataLoader

import os
import wandb
import argparse
import numpy as np
from tqdm import tqdm
from monai.metrics import DiceMetric

from models.sam import SAMModel
from utils.loss import CombinedLoss
from utils.dataset import SAMDataset
from utils.config import SAMFinetuneConfig, SAMDatasetConfig


class TrainSAM:
    def __init__(
        self,
        config: SAMFinetuneConfig,
        train_dataset: SAMDataset,
        val_dataset: SAMDataset,
    ):
        self.config = config
        self.device = torch.device(config.device)

        self.output_dir = self.config.output_path
        self.output_dir = "%s_Fr%d_Lr%f" % (
            self.output_dir, self.config.freeze, self.config.learning_rate,
        )

        self.run_number = 0
        ExistFlag = os.path.exists("%s_run%d" % (self.output_dir, self.run_number))
        if ExistFlag:
            while os.path.exists("%s_run%d" % (self.output_dir, self.run_number)):
                self.run_number += 1
        self.run_name = f'_run{self.run_number}'

        self.output_dir = "%s_run%d" % (self.output_dir, self.run_number)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize wandb
        self.init_wandb()
        
        # Initialize model and loss
        self.model = SAMModel(config)
        self.criterion = CombinedLoss(config)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0
        
        # metrics
        self.dice_metric = DiceMetric(
            include_background=True, 
            reduction="mean"
            )
        
    def init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.config.wandb_project,
            name=self.run_name,
            config={
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "model_type": self.config.model_type,
                "lambda_dice": 1 - self.config.lambda_bce - self.config.lambda_kl,
                "lambda_bce": self.config.lambda_bce,
                "lambda_kl": self.config.lambda_kl,
            },
            mode='disabled' if self.config.wandb_mode == 'disabled' else 'online'
        )
        
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
        }
        
        # Save regular checkpoint
        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.output_dir,
            f'checkpoint_epoch_{self.current_epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if this is the best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_dice = checkpoint['best_val_dice']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train_epoch(self) -> float:
        self.model.train()
        self.dice_metric.reset()
        epoch_loss = 0.0
        dice_scores = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Process each prompt in the batch
            batch_loss = 0
            batch_pred_masks = []
            
            for i, _ in enumerate(images):                
                prompt_data = {}
                if batch['points_coords'] is not None:
                    prompt_data['points'] = {
                        'coords': batch['points_coords'][i],
                        'labels': batch['points_labels'][i]
                    }
                if batch['boxes'] is not None:
                    prompt_data['boxes'] = batch['boxes'][i]

                    # Forward pass
                    pred_mask, _ = self.model.forward_one_image(
                        image=images[i:i+1],
                        points=prompt_data.get('points'),
                        bounding_box=prompt_data.get('boxes'),
                        is_train=True
                    )
                
                batch_pred_masks.append(pred_mask)
                
            pred_masks = torch.cat(batch_pred_masks, dim=0)

            pred_masks = torch.sigmoid(pred_masks)
            pred_masks_binary = (pred_masks > 0.5).float()
                        
            dice = self.dice_metric(y_pred=pred_masks_binary, y=masks)
            dice_scores.append(dice.mean().item())
            
            # Calculate loss
            loss = self.criterion(pred=pred_masks, target=masks)
            batch_loss = loss            
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            # Update progress
            epoch_loss += batch_loss.item()
            progress_bar.set_postfix({'loss': batch_loss.item()})
        
        epoch_loss /= len(self.train_loader)
        epoch_dice = sum(dice_scores) / len(dice_scores)
        
        wandb.log({
            "/train/loss": epoch_loss,
            "/train/dice": epoch_dice,
            "/train/learning_rate": self.scheduler.get_last_lr()[0]
        }, step=self.current_epoch)
        
        return epoch_loss, epoch_dice
    
    def validate(self) -> float:
        self.model.eval()
        self.dice_metric.reset()
        val_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].float().to(self.device)
                batch_size = batch['image'].shape[0]
                
                batch_loss = 0
                batch_pred_masks = []
                
                for i in range(batch_size):
                    prompt_data = {}
                    if batch['points_coords'] is not None:
                        prompt_data['points'] = {
                            'coords': batch['points_coords'][i],
                            'labels': batch['points_labels'][i]
                        }
                    if batch['boxes'] is not None:
                        prompt_data['boxes'] = batch['boxes'][i]
                        
                    pred_mask, _ = self.model.forward_one_image(
                        image=images[i:i+1],
                        points=prompt_data.get('points'),
                        bounding_box=prompt_data.get('boxes'),
                        is_train=False
                        )

                    batch_pred_masks.append(pred_mask)
                    
                pred_masks = torch.cat(batch_pred_masks, dim=0)
                
                pred_masks = torch.sigmoid(pred_masks)
                pred_masks_binary = (pred_masks > 0.5).float()
                
                dice = self.dice_metric(y_pred=pred_masks_binary, y=masks)
                dice_scores.append(dice.mean().item())
                
                loss = self.criterion(pred=pred_masks, target=masks)
                batch_loss = loss
            
                val_loss += batch_loss.item()
        
        val_loss /= len(self.val_loader)
        epoch_dice = sum(dice_scores) / len(dice_scores)
        
        wandb.log({
            "/val/loss": val_loss,
            "/val/dice": epoch_dice,
        }, step=self.current_epoch)
        
        return val_loss, epoch_dice
    
    def train(self, num_epochs: int):
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss, train_dice = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Dice = {train_dice:.4f}")
            
            # Validate
            val_loss, val_dice = self.validate()
            print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}, Validation Dice = {val_dice:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
            
            if is_best:
                self.save_checkpoint(is_best)
        
        wandb.finish()
        print(f"Training completed. Best validation dice: {self.best_val_dice:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM finetuning")

    parser.add_argument("--sam_path", type=str, default="facebook/sam-vit-base")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--dataset_path", type=str, default="./data")
    parser.add_argument("--freeze", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()
    
    finetune_config = SAMFinetuneConfig(
        device='cuda',
        wandb_project='SAM_finetune',
        wandb_name='test_run',
        model_type='vit_b',
        sam_path=args.sam_path,
        output_path=args.output_path,
        freeze=args.freeze,
        num_epochs=50,
        batch_size=2,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        lambda_bce=0.2,
        lambda_kl=0.2,
        sigma=1,
        wandb_mode='disabled',
        num_workers=0
    )
    train_dataset_config = SAMDatasetConfig(
        dataset_path='%s/train/' % args.dataset_path,
        remove_nonscar=True,
        sample_size=None,
        point_prompt=True,
        point_prompt_types=['positive'],
        num_points=3,
        box_prompt=True,
        enable_direction_aug=True,
        enable_size_aug=True,
        image_size=1024,
        train=True
    )
    
    val_dataset_config = SAMDatasetConfig(
        dataset_path='%s/val/' % args.dataset_path,
        remove_nonscar=True,
        sample_size=None,
        point_prompt=True,
        point_prompt_types=['positive'],
        num_points=3,
        box_prompt=True,
        enable_direction_aug=False,
        enable_size_aug=False,
        image_size=1024,
        train=False
    )

    train_dataset = SAMDataset(train_dataset_config)
    val_dataset = SAMDataset(val_dataset_config)

    trainer = TrainSAM(finetune_config, train_dataset, val_dataset)
    trainer.train(finetune_config.num_epochs)


if __name__ == "__main__":
    main()
