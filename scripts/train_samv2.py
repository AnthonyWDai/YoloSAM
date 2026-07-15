import os
import argparse

import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from models.sam import SAMModel
from utils.loss import CombinedLoss
from utils.datasetv2 import SAMDataset
from utils.configv2 import SAMFinetuneConfig, SAMDatasetConfig


def compute_seg_dice_stats(pred, target, num_classes=1, eps=1e-5):
    """
    Compute aggregated Dice statistics over valid (sample, class) pairs.

    Args:
        pred:   Tensor [B, ...] of predicted class ids
        target: Tensor [B, ...] of ground-truth class ids
        num_classes: number of foreground classes, assumes classes are 1..num_classes
        eps: numerical stability term

    Returns:
        dice_sum: sum of Dice scores over valid (sample, class) pairs
        valid_count: number of valid (sample, class) pairs

    Notes:
        - A (sample, class) pair is valid if that class appears in pred or target.
        - Absent-in-both cases are excluded.
        - For binary segmentation with labels {0,1}, use num_classes=1.
    """
    assert pred.shape == target.shape, f"Shape mismatch: pred={pred.shape}, target={target.shape}"

    reduce_dims = tuple(range(1, pred.ndim))
    dice_sum = 0.0
    valid_count = 0

    for cls in range(1, num_classes + 1):
        pred_c = (pred == cls).float()
        target_c = (target == cls).float()

        intersect = (pred_c * target_c).sum(dim=reduce_dims)
        pred_sum = pred_c.sum(dim=reduce_dims)
        target_sum = target_c.sum(dim=reduce_dims)
        denom = pred_sum + target_sum

        valid = denom > 0
        if valid.any():
            dice = (2.0 * intersect[valid] + eps) / (denom[valid] + eps)
            dice_sum += dice.sum().item()
            valid_count += valid.sum().item()

    return dice_sum, valid_count


class TrainSAM:
    def __init__(
        self,
        config: SAMFinetuneConfig,
        train_dataset: SAMDataset,
        val_dataset: SAMDataset,
    ):
        self.config = config
        self.device = torch.device(config.device)

        self.output_dir = "%s/Bs%d_Lr%f_Fr%d" % (
            self.config.output_path,
            self.config.batch_size,
            self.config.learning_rate,
            self.config.freeze,
        )

        self.run_number = 0
        while os.path.exists(f"{self.output_dir}_run{self.run_number}"):
            self.run_number += 1

        self.run_name = f"{getattr(self.config, 'wandb_name', 'run')}_run{self.run_number}"
        self.output_dir = f"{self.output_dir}_run{self.run_number}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.init_wandb()

        self.model = SAMModel(config).to(self.device)
        self.criterion = CombinedLoss(config)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.total_steps = config.num_epochs * max(len(self.train_loader), 1)
        self.warmup_steps = getattr(config, "warmup_epochs", 0) * max(len(self.train_loader), 1)

        base_lr = config.learning_rate
        min_lr = getattr(config, "min_lr", base_lr * 1e-3)
        warmup_lr = getattr(config, "warmup_lr", min_lr * 10)

        if self.warmup_steps > 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr

        def lr_lambda(current_step):
            if self.total_steps <= 0:
                return 1.0

            if self.warmup_steps > 0 and current_step < self.warmup_steps:
                alpha = current_step / max(self.warmup_steps, 1)
                current_lr = warmup_lr + alpha * (base_lr - warmup_lr)
                return current_lr / base_lr

            cosine_steps = self.total_steps - self.warmup_steps
            if cosine_steps <= 0:
                return 1.0

            progress = (current_step - self.warmup_steps) / max(cosine_steps, 1)
            progress = min(max(progress, 0.0), 1.0)

            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_lr = min_lr + (base_lr - min_lr) * cosine_decay
            return current_lr / base_lr

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.current_epoch = 0
        self.best_val_dice = 0.0

    def init_wandb(self):
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
                "weight_decay": self.config.weight_decay,
                "freeze": self.config.freeze,
                "num_epochs": self.config.num_epochs,
            },
            mode="disabled" if self.config.wandb_mode == "disabled" else "online",
        )

    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_dice": self.best_val_dice,
            "config": vars(self.config) if hasattr(self.config, "__dict__") else None,
        }

        os.makedirs(self.output_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            self.output_dir,
            f"checkpoint_epoch_{self.current_epoch}.pth",
        )
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_dice = checkpoint["best_val_dice"]
        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def _extract_prompt_data(self, batch, index):
        prompt_data = {}

        if batch.get("points_coords", None) is not None:
            prompt_data["points"] = {
                "coords": batch["points_coords"][index],
                "labels": batch["points_labels"][index],
            }

        if batch.get("boxes", None) is not None:
            prompt_data["boxes"] = batch["boxes"][index]

        return prompt_data

    def _forward_batch(self, images, batch, is_train):
        batch_pred_masks = []

        for i in range(images.shape[0]):
            prompt_data = self._extract_prompt_data(batch, i)

            pred_mask, _ = self.model.forward_one_image(
                image=images[i:i + 1],
                points=prompt_data.get("points"),
                bounding_box=prompt_data.get("boxes"),
                is_train=is_train,
            )
            batch_pred_masks.append(pred_mask)

        pred_masks = torch.cat(batch_pred_masks, dim=0)
        return pred_masks

    @staticmethod
    def _prepare_binary_masks(pred_probs, masks, threshold=0.5):
        pred_binary = (pred_probs > threshold).long()
        target_binary = (masks > threshold).long()

        if pred_binary.shape != target_binary.shape:
            raise ValueError(
                f"Binary mask shape mismatch: pred={pred_binary.shape}, target={target_binary.shape}"
            )

        return pred_binary, target_binary

    def train_epoch(self):
        self.model.train()

        epoch_loss = 0.0
        dice_sum_total = 0.0
        dice_valid_total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")

        for batch in progress_bar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].float().to(self.device, non_blocking=True)

            pred_logits = self._forward_batch(images, batch, is_train=True)
            pred_probs = torch.sigmoid(pred_logits)

            pred_binary, target_binary = self._prepare_binary_masks(pred_probs, masks)

            batch_dice_sum, batch_valid_count = compute_seg_dice_stats(
                pred=pred_binary,
                target=target_binary,
                num_classes=1,
            )
            dice_sum_total += batch_dice_sum
            dice_valid_total += batch_valid_count

            # Assumes CombinedLoss expects probabilities, as in the original script.
            # If CombinedLoss internally uses BCEWithLogitsLoss, pass pred_logits instead.
            loss = self.criterion(pred=pred_probs, target=masks)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if getattr(self.config, "grad_clip", 0.0) and self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.grad_clip,
                )

            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item()
            running_dice = dice_sum_total / max(dice_valid_total, 1)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{running_dice:.4f}",
            )

        epoch_loss /= max(len(self.train_loader), 1)
        epoch_dice = dice_sum_total / max(dice_valid_total, 1)

        wandb.log(
            {
                "/train/loss": epoch_loss,
                "/train/dice": epoch_dice,
                "/train/learning_rate": self.scheduler.get_last_lr()[0],
            },
            step=self.current_epoch,
        )

        return epoch_loss, epoch_dice

    def validate(self):
        self.model.eval()

        val_loss = 0.0
        dice_sum_total = 0.0
        dice_valid_total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")

            for batch in progress_bar:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].float().to(self.device, non_blocking=True)

                pred_logits = self._forward_batch(images, batch, is_train=False)
                pred_probs = torch.sigmoid(pred_logits)

                pred_binary, target_binary = self._prepare_binary_masks(pred_probs, masks)

                batch_dice_sum, batch_valid_count = compute_seg_dice_stats(
                    pred=pred_binary,
                    target=target_binary,
                    num_classes=1,
                )
                dice_sum_total += batch_dice_sum
                dice_valid_total += batch_valid_count

                # Assumes CombinedLoss expects probabilities, as in the original script.
                loss = self.criterion(pred=pred_probs, target=masks)
                val_loss += loss.item()

                running_dice = dice_sum_total / max(dice_valid_total, 1)
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    dice=f"{running_dice:.4f}",
                )

        val_loss /= max(len(self.val_loader), 1)
        epoch_dice = dice_sum_total / max(dice_valid_total, 1)

        wandb.log(
            {
                "/val/loss": val_loss,
                "/val/dice": epoch_dice,
            },
            step=self.current_epoch,
        )

        return val_loss, epoch_dice

    def train(self, num_epochs: int):
        print(f"Starting training for {num_epochs} epochs")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            train_loss, train_dice = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Dice = {train_dice:.4f}")

            val_loss, val_dice = self.validate()
            print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}, Validation Dice = {val_dice:.4f}")

            # self.scheduler.step()

            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
                self.save_checkpoint(is_best=True)

        wandb.finish()
        print(f"Training completed. Best validation dice: {self.best_val_dice:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM finetuning")
    parser.add_argument("--sam_path", type=str, default="facebook/sam-vit-base")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--dataset_path", type=str, default="./data")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--freeze", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--min_lr", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()

    finetune_config = SAMFinetuneConfig(
        device=args.device,
        wandb_project="SAM_finetune",
        wandb_name="test_run",
        model_type="vit_b",
        sam_path=args.sam_path,
        output_path=args.output_path,
        freeze=args.freeze,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        lambda_bce=0.2,
        lambda_kl=0.2,
        sigma=1,
        wandb_mode="disabled",
        num_workers=0,
        grad_clip=args.grad_clip,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        min_lr=args.min_lr,
    )

    train_dataset_config = SAMDatasetConfig(
        dataset_path=f"{args.dataset_path}/train/",
        remove_nonscar=True,
        sample_size=None,
        point_prompt=True,
        point_prompt_types=["positive"],
        num_points=3,
        box_prompt=True,
        enable_direction_aug=True,
        enable_size_aug=True,
        image_size=1024,
        train=True,
    )

    val_dataset_config = SAMDatasetConfig(
        dataset_path=f"{args.dataset_path}/val/",
        remove_nonscar=True,
        sample_size=None,
        point_prompt=True,
        point_prompt_types=["positive"],
        num_points=3,
        box_prompt=True,
        enable_direction_aug=False,
        enable_size_aug=False,
        image_size=1024,
        train=False,
    )

    train_dataset = SAMDataset(train_dataset_config)
    val_dataset = SAMDataset(val_dataset_config)

    trainer = TrainSAM(finetune_config, train_dataset, val_dataset)
    trainer.train(finetune_config.num_epochs)


if __name__ == "__main__":
    main()